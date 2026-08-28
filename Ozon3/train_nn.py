"""Temporal walk-forward training for the tabular neural network.

The module deliberately has no top-level PyTorch import.  This keeps audit and
ensemble environments import-safe; PyTorch is required only when ``main``
actually starts NN training.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import gc
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


KEY_COLUMNS = ["user_id", "fold", "cutoff_date"]
DEFAULT_SEEDS = (42, 1337, 2026)
CHECKPOINT_COLUMNS = ["user_id", "fold", "cutoff_date", "target", "pred", "pred_log"]
CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class Standardizer:
    """Small numpy-only standardizer with explicit serialized statistics."""

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, matrix: np.ndarray) -> "Standardizer":
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            raise ValueError("Standardizer requires a non-empty 2D matrix")
        # Chunked float64 moments avoid numpy's multi-gigabyte temporary when
        # the latest temporal split contains more than a million rows.
        total = np.zeros(matrix.shape[1], dtype=np.float64)
        total_square = np.zeros(matrix.shape[1], dtype=np.float64)
        chunk_rows = 32_768
        for start in range(0, len(matrix), chunk_rows):
            chunk = matrix[start : start + chunk_rows].astype(np.float64, copy=False)
            total += chunk.sum(axis=0, dtype=np.float64)
            total_square += np.square(chunk).sum(axis=0, dtype=np.float64)
        mean = total / len(matrix)
        variance = np.maximum(total_square / len(matrix) - np.square(mean), 0.0)
        scale = np.sqrt(variance)
        scale = np.where(np.isfinite(scale) & (scale >= 1e-6), scale, 1.0)
        mean = np.where(np.isfinite(mean), mean, 0.0)
        return cls(mean=mean.astype(np.float32), scale=scale.astype(np.float32))

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        if matrix.dtype != np.float32 or not matrix.flags.writeable:
            matrix = np.asarray(matrix, dtype=np.float32).copy()
        np.subtract(matrix, self.mean, out=matrix)
        np.divide(matrix, self.scale, out=matrix)
        return np.nan_to_num(
            matrix,
            copy=False,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32, copy=False)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean, scale=self.scale)


def _load_checked_fold(
    dataset_dir: Path,
    fold_index: int,
    expected_spec,
    train,
    cfg,
    expected_features: list[str] | None = None,
    expected_dtypes: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    """Load exactly one production snapshot and validate its fixed schema."""

    path = dataset_dir / f"{expected_spec.name}.parquet"
    if expected_features is None:
        frame = pd.read_parquet(path)
    else:
        required = [
            cfg.ID_COL,
            cfg.TARGET_COL,
            cfg.TARGET_LOG_COL,
            "fold",
            "cutoff_date",
        ]
        columns = list(dict.fromkeys([*expected_features, *required]))
        frame = pd.read_parquet(path, columns=columns)
    train._validate_loaded_fold(frame, expected_spec)
    features = list(train.get_features([frame]))
    dtypes = {column: str(frame[column].dtype) for column in features}
    if expected_features is not None and features != expected_features:
        raise AssertionError(f"Feature schema mismatch in {expected_spec.name}")
    if expected_dtypes is not None and dtypes != expected_dtypes:
        raise AssertionError(f"Feature dtype schema mismatch in {expected_spec.name}")
    return frame, features, dtypes


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise RuntimeError(
                "PyTorch is required for train_nn.py but is not installed. "
                "Install a build appropriate for your CPU/CUDA environment "
                "from https://pytorch.org/get-started/locally/."
            ) from exc
        raise
    return torch, nn, DataLoader, TensorDataset


def _load_project_api():
    import config as cfg
    import time_split
    import train

    required_train = (
        "load_folds",
        "get_features",
        "resolve_artifact_layout",
        "rmsle",
    )
    missing = [name for name in required_train if not hasattr(train, name)]
    if missing:
        raise RuntimeError(f"train.py is missing required API: {missing}")
    return cfg, time_split, train


def _parse_seeds(value: str | Iterable[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        seeds = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    else:
        seeds = tuple(int(seed) for seed in value)
    if not seeds:
        raise ValueError("At least one seed is required")
    return seeds


def _parse_final_epochs(value: str, seeds: Sequence[int]) -> dict[int, int]:
    """Parse an explicit post-CV epoch contract such as ``42:15,1337:6``."""

    parsed: dict[int, int] = {}
    for item in value.split(","):
        seed_text, separator, epoch_text = item.strip().partition(":")
        if not separator:
            raise ValueError(f"Invalid --final-epochs item: {item!r}")
        seed = int(seed_text)
        epochs = int(epoch_text)
        if seed in parsed or epochs < 1:
            raise ValueError(f"Invalid --final-epochs item: {item!r}")
        parsed[seed] = epochs
    if set(parsed) != set(seeds):
        raise ValueError(
            f"--final-epochs seeds={sorted(parsed)}, expected={sorted(seeds)}"
        )
    return parsed


def _set_seed(torch, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _resolve_device(torch, requested: str):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def _preflight_device(torch, requested: str):
    """Exercise CUDA once before any fold; auto mode falls back cleanly."""

    device = _resolve_device(torch, requested)
    if device.type != "cuda":
        return device
    try:
        probe = torch.randn((2048, 512), device=device, requires_grad=True)
        loss = (probe.square().mean() + probe.mean())
        loss.backward()
        torch.cuda.synchronize(device)
        del probe, loss
        torch.cuda.empty_cache()
        return device
    except Exception as exc:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        if requested == "auto":
            print(
                "CUDA preflight failed; falling back to CPU before CV: "
                f"{type(exc).__name__}: {exc}"
            )
            return torch.device("cpu")
        raise RuntimeError(f"CUDA preflight failed: {exc}") from exc


def _make_model(nn, input_dim: int, hidden_dim: int, num_blocks: int, dropout: float):
    class ResidualBlock(nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.block = nn.Sequential(
                nn.BatchNorm1d(dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(dim, dim),
                nn.BatchNorm1d(dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(dim, dim),
            )

        def forward(self, values):
            return values + self.block(values)

    class TabularResNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_layer = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.SiLU(),
            )
            self.blocks = nn.ModuleList(
                [ResidualBlock(hidden_dim) for _ in range(num_blocks)]
            )
            self.head = nn.Linear(hidden_dim, 1)

        def forward(self, values):
            values = self.input_layer(values)
            for block in self.blocks:
                values = block(values)
            return self.head(values)

    return TabularResNet()


def _matrix(frame: pd.DataFrame, feature_cols: Sequence[str]) -> np.ndarray:
    matrix = frame.loc[:, feature_cols].to_numpy(dtype=np.float32, copy=True)
    return np.nan_to_num(
        matrix,
        copy=False,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def _load_training_arrays(
    dataset_dir: Path,
    specs: Sequence[Any],
    val_index: int,
    row_counts: Sequence[int],
    feature_cols: list[str],
    feature_dtypes: dict[str, str],
    train,
    cfg,
    matrix_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build one float32 training matrix without retaining pandas folds."""

    total_rows = int(sum(row_counts[:val_index]))
    if matrix_path is None:
        matrix = np.empty((total_rows, len(feature_cols)), dtype=np.float32)
    else:
        matrix_path.parent.mkdir(parents=True, exist_ok=True)
        matrix = np.memmap(
            matrix_path,
            dtype=np.float32,
            mode="w+",
            shape=(total_rows, len(feature_cols)),
        )
    target_log = np.empty(total_rows, dtype=np.float32)
    offset = 0
    for index in range(val_index):
        frame, _, _ = _load_checked_fold(
            dataset_dir,
            index,
            specs[index],
            train,
            cfg,
            expected_features=feature_cols,
            expected_dtypes=feature_dtypes,
        )
        next_offset = offset + len(frame)
        matrix[offset:next_offset] = frame.loc[:, feature_cols].to_numpy(
            dtype=np.float32, copy=False
        )
        target_log[offset:next_offset] = frame[cfg.TARGET_LOG_COL].to_numpy(
            dtype=np.float32, copy=False
        )
        offset = next_offset
        del frame
        gc.collect()
    if offset != total_rows:
        raise AssertionError(f"Loaded training rows={offset}, expected={total_rows}")
    np.nan_to_num(matrix, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    if isinstance(matrix, np.memmap):
        matrix.flush()
    return matrix, target_log


def _fixed_common_features(folds: Sequence[pd.DataFrame], get_features) -> list[str]:
    """Return one target-independent feature list used by every NN fold."""

    if not folds:
        raise ValueError("No folds were loaded")
    candidates = list(get_features(folds))
    common = [
        column
        for column in candidates
        if all(
            column in frame.columns
            and pd.api.types.is_numeric_dtype(frame[column])
            for frame in folds
        )
    ]
    if not common:
        raise ValueError("No common numeric features are available across all folds")
    if len(common) != len(candidates):
        removed = sorted(set(candidates) - set(common))
        print(f"Dropped {len(removed)} non-common/non-numeric features: {removed[:10]}")
    if len(common) != len(set(common)):
        raise AssertionError("Duplicate feature names detected")
    return common


def _build_fold_specs(time_split, cfg, n_folds: int):
    """Use the canonical time_split contract, with legacy compatibility."""

    if hasattr(time_split, "build_cv_folds"):
        specs = time_split.build_cv_folds(
            n_folds=n_folds,
            step_days=cfg.STEP_DAYS,
        )
        test_spec = (
            time_split.build_test_fold()
            if hasattr(time_split, "build_test_fold")
            else None
        )
        time_split.validate_fold_contract(specs, test_spec)
        walk_forward = list(time_split.walk_forward_splits(specs))
        if len(walk_forward) != max(len(specs) - 1, 0):
            raise AssertionError(
                "time_split.walk_forward_splits returned an unexpected number of splits"
            )
        return list(specs)

    if not hasattr(time_split, "build_folds"):
        raise RuntimeError("time_split.py exposes neither build_cv_folds nor build_folds")
    legacy = time_split.build_folds(n_folds=n_folds, step_days=cfg.STEP_DAYS)
    specs = [fold for fold in legacy if getattr(fold, "name", None) != "test"]
    return specs


def _as_date(value: Any) -> dt.date:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError(f"Invalid date value: {value!r}")
    return timestamp.date()


def _validate_fold_frames(
    folds: Sequence[pd.DataFrame],
    specs: Sequence[Any],
    cfg,
) -> None:
    if len(folds) != len(specs):
        raise AssertionError(f"Loaded {len(folds)} folds but time_split defines {len(specs)}")

    required = {
        cfg.ID_COL,
        "fold",
        "cutoff_date",
        cfg.TARGET_COL,
        cfg.TARGET_LOG_COL,
    }
    for index, (frame, spec) in enumerate(zip(folds, specs)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise AssertionError(f"fold_{index} is missing required columns: {missing}")
        if frame.empty:
            raise AssertionError(f"fold_{index} is empty")
        cutoffs = {_as_date(value) for value in frame["cutoff_date"].drop_duplicates()}
        if cutoffs != {_as_date(spec.cutoff)}:
            raise AssertionError(
                f"fold_{index} cutoff mismatch: data={sorted(cutoffs)} spec={spec.cutoff}"
            )
        fold_values = {str(value) for value in frame["fold"].drop_duplicates()}
        accepted_fold_values = {str(index), f"fold_{index}", str(spec.name)}
        if len(fold_values) != 1 or not fold_values.issubset(accepted_fold_values):
            raise AssertionError(
                f"fold_{index} metadata mismatch: data={sorted(fold_values)} "
                f"expected={spec.name}"
            )
        if frame[cfg.ID_COL].duplicated().any():
            raise AssertionError(f"fold_{index} has duplicate {cfg.ID_COL} rows")
        target = frame[cfg.TARGET_COL].to_numpy(dtype=np.float64)
        target_log = frame[cfg.TARGET_LOG_COL].to_numpy(dtype=np.float64)
        if not np.isfinite(target).all() or (target < 0).any():
            raise AssertionError(f"fold_{index} has invalid targets")
        if not np.allclose(target_log, np.log1p(target), rtol=1e-5, atol=1e-5):
            raise AssertionError(f"fold_{index} target_log is inconsistent with target")

    for val_index in range(1, len(specs)):
        latest_training_target_end = max(
            _as_date(spec.target_end) for spec in specs[:val_index]
        )
        validation_cutoff = _as_date(specs[val_index].cutoff)
        if latest_training_target_end > validation_cutoff:
            raise AssertionError(
                "Temporal leakage: training target ends after validation cutoff: "
                f"{latest_training_target_end} > {validation_cutoff}"
            )


def _frame_fold_value(frame: pd.DataFrame, fallback: str) -> Any:
    values = frame["fold"].drop_duplicates().tolist()
    if len(values) != 1:
        raise AssertionError(f"Expected one fold value, got {values[:5]}")
    return values[0] if values else fallback


def _loader(torch, DataLoader, TensorDataset, x, y, batch_size, shuffle, seed):
    dataset = TensorDataset(
        torch.from_numpy(x),
        torch.from_numpy(y.astype(np.float32, copy=False)).reshape(-1, 1),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator if shuffle else None,
    )


def _predict_log(torch, model, x, batch_size, device) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start : start + batch_size]).to(device)
            output = model(batch).reshape(-1)
            predictions.append(output.detach().cpu().numpy())
    return np.concatenate(predictions).astype(np.float64, copy=False)


def _train_one_seed(
    torch,
    nn,
    DataLoader,
    TensorDataset,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    seed: int,
    device,
    hidden_dim: int,
    num_blocks: int,
    dropout: float,
    batch_size: int,
    max_epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
) -> tuple[Any, np.ndarray, int, float]:
    _set_seed(torch, seed)
    model = _make_model(nn, x_train.shape[1], hidden_dim, num_blocks, dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    criterion = nn.MSELoss()
    train_loader = _loader(
        torch,
        DataLoader,
        TensorDataset,
        x_train,
        y_train,
        batch_size,
        True,
        seed,
    )
    val_loader = _loader(
        torch,
        DataLoader,
        TensorDataset,
        x_val,
        y_val,
        batch_size,
        False,
        seed,
    )

    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    stale_epochs = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch_x, batch_y in train_loader:
            # BatchNorm cannot estimate variance from a one-row last batch.
            if len(batch_x) < 2:
                continue
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        model.eval()
        loss_sum = 0.0
        row_count = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                loss = criterion(model(batch_x), batch_y)
                loss_sum += float(loss.item()) * len(batch_x)
                row_count += len(batch_x)
        val_loss = loss_sum / max(row_count, 1)
        scheduler.step(val_loss)
        if val_loss < best_loss - 1e-8:
            best_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(
                {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
            )
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch == 1 or epoch % 5 == 0:
            print(
                f"    seed={seed} epoch={epoch:03d} "
                f"val_log_rmse={np.sqrt(val_loss):.6f}"
            )
        if stale_epochs >= patience:
            break

    if best_state is None:
        raise RuntimeError("NN training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    pred_log = _predict_log(torch, model, x_val, batch_size, device)
    return model, pred_log, best_epoch, best_loss


def _fit_final_seed(
    torch,
    nn,
    DataLoader,
    TensorDataset,
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
    epochs: int,
    device,
    hidden_dim: int,
    num_blocks: int,
    dropout: float,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
):
    _set_seed(torch, seed)
    model = _make_model(nn, x_train.shape[1], hidden_dim, num_blocks, dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = nn.MSELoss()
    loader = _loader(
        torch,
        DataLoader,
        TensorDataset,
        x_train,
        y_train,
        batch_size,
        True,
        seed,
    )
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_x, batch_y in loader:
            if len(batch_x) < 2:
                continue
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"    final seed={seed} epoch={epoch:03d}/{epochs:03d}")
    return model


def _array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).view(np.uint8))
    return digest.hexdigest()


def _json_hash(payload: Any) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def _save_standardizer_atomic(scaler: Standardizer, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    with temporary.open("wb") as stream:
        np.savez(stream, mean=scaler.mean, scale=scaler.scale)
    temporary.replace(path)


def _save_model_atomic(torch, payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    torch.save(payload, temporary)
    temporary.replace(path)


def _checkpoint_paths(oof_dir: Path, model_dir: Path, fold_name: str):
    return (
        oof_dir / f"oof_nn_{fold_name}.csv",
        oof_dir / f"oof_nn_{fold_name}.meta.json",
        model_dir / f"normalizer_{fold_name}.npz",
    )


def _make_oof_part(val_df: pd.DataFrame, val_spec, pred_log: np.ndarray, cfg):
    pred_log = np.clip(np.asarray(pred_log, dtype=np.float64), 0.0, None)
    return pd.DataFrame(
        {
            cfg.ID_COL: val_df[cfg.ID_COL].to_numpy(dtype=np.int64, copy=False),
            "fold": val_spec.name,
            "cutoff_date": val_spec.cutoff.isoformat(),
            "target": val_df[cfg.TARGET_COL].to_numpy(dtype=np.float64, copy=False),
            "pred": np.expm1(pred_log),
            "pred_log": pred_log,
        }
    )


def _validate_oof_part(part: pd.DataFrame, val_df: pd.DataFrame, val_spec, cfg, train):
    if part.columns.tolist() != CHECKPOINT_COLUMNS:
        raise AssertionError(
            f"{val_spec.name}: checkpoint columns={part.columns.tolist()}"
        )
    if len(part) != len(val_df):
        raise AssertionError(f"{val_spec.name}: checkpoint row count differs")
    if part.duplicated(KEY_COLUMNS).any():
        raise AssertionError(f"{val_spec.name}: duplicate checkpoint keys")
    if not part["fold"].astype(str).eq(val_spec.name).all():
        raise AssertionError(f"{val_spec.name}: checkpoint fold metadata differs")
    if not part["cutoff_date"].astype(str).eq(val_spec.cutoff.isoformat()).all():
        raise AssertionError(f"{val_spec.name}: checkpoint cutoff differs")
    if not np.array_equal(
        part[cfg.ID_COL].to_numpy(dtype=np.int64),
        val_df[cfg.ID_COL].to_numpy(dtype=np.int64),
    ):
        raise AssertionError(f"{val_spec.name}: checkpoint user_id/order differs")
    if not np.allclose(
        part["target"].to_numpy(dtype=np.float64),
        val_df[cfg.TARGET_COL].to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=1e-6,
    ):
        raise AssertionError(f"{val_spec.name}: checkpoint targets differ")
    numeric = part[["target", "pred", "pred_log"]].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all() or (numeric < 0).any():
        raise AssertionError(f"{val_spec.name}: checkpoint has invalid values")
    if not np.allclose(part["pred_log"], np.log1p(part["pred"]), atol=1e-10):
        raise AssertionError(f"{val_spec.name}: pred_log != log1p(pred)")
    return float(train.rmsle(part["target"], part["pred"]))


def _model_payload(
    model,
    *,
    seed: int,
    best_epoch: int,
    best_loss: float,
    feature_schema_sha256: str,
    normalizer_stats_sha256: str,
    nn_config_sha256: str,
    nn_config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "state_dict": {
            name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
        },
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "best_log_mse": float(best_loss),
        "feature_schema_sha256": feature_schema_sha256,
        "normalizer_stats_sha256": normalizer_stats_sha256,
        "nn_config_sha256": nn_config_sha256,
        "architecture": nn_config["architecture"],
    }


def _validate_model_payload(
    torch,
    model_path: Path,
    *,
    seed: int,
    feature_schema_sha256: str,
    normalizer_stats_sha256: str,
    nn_config_sha256: str,
) -> dict[str, Any]:
    try:
        payload = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(model_path, map_location="cpu")
    expected = {
        "seed": int(seed),
        "feature_schema_sha256": feature_schema_sha256,
        "normalizer_stats_sha256": normalizer_stats_sha256,
        "nn_config_sha256": nn_config_sha256,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise AssertionError(f"model payload {key} differs")
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict) or not state_dict:
        raise AssertionError("model state_dict is missing")
    if any(not bool(torch.isfinite(tensor).all()) for tensor in state_dict.values()):
        raise AssertionError("model state_dict contains non-finite values")
    if int(payload.get("best_epoch", 0)) < 1:
        raise AssertionError("model best_epoch is invalid")
    return payload


def _final_model_payload(
    model,
    *,
    seed: int,
    epochs: int,
    final_training_rows: int,
    dataset_fingerprint: str,
    feature_schema_sha256: str,
    nn_config_sha256: str,
    normalizer_stats_sha256: str,
    normalizer_file_sha256: str,
    nn_config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "artifact_kind": "final_refit_seed_model",
        "state_dict": {
            name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
        },
        "seed": int(seed),
        "epochs": int(epochs),
        "final_training_rows": int(final_training_rows),
        "dataset_fingerprint": dataset_fingerprint,
        "target_transform": "log1p",
        "feature_schema_sha256": feature_schema_sha256,
        "nn_configuration_sha256": nn_config_sha256,
        "normalizer_stats_sha256": normalizer_stats_sha256,
        "normalizer_file_sha256": normalizer_file_sha256,
        "architecture": nn_config["architecture"],
    }


def _load_valid_final_model(
    torch,
    path: Path,
    *,
    seed: int,
    epochs: int,
    final_training_rows: int,
    dataset_fingerprint: str,
    feature_schema_sha256: str,
    nn_config_sha256: str,
    normalizer_stats_sha256: str,
    normalizer_file_sha256: str,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        expected = {
            "artifact_kind": "final_refit_seed_model",
            "seed": int(seed),
            "epochs": int(epochs),
            "final_training_rows": int(final_training_rows),
            "dataset_fingerprint": dataset_fingerprint,
            "target_transform": "log1p",
            "feature_schema_sha256": feature_schema_sha256,
            "nn_configuration_sha256": nn_config_sha256,
            "normalizer_stats_sha256": normalizer_stats_sha256,
            "normalizer_file_sha256": normalizer_file_sha256,
        }
        for key, expected_value in expected.items():
            if payload.get(key) != expected_value:
                raise AssertionError(f"final model payload {key} differs")
        state_dict = payload.get("state_dict")
        if not isinstance(state_dict, dict) or not state_dict:
            raise AssertionError("final model state_dict is missing")
        if any(not bool(torch.isfinite(tensor).all()) for tensor in state_dict.values()):
            raise AssertionError("final model state_dict contains non-finite values")
        return payload
    except Exception as exc:
        print(
            f"final seed={seed}: existing model rejected "
            f"({type(exc).__name__}: {exc})"
        )
        return None


def _checkpoint_metadata(
    *,
    dataset_dir: Path,
    dataset_fingerprint: str,
    feature_cols: list[str],
    feature_dtypes: dict[str, str],
    feature_schema_sha256: str,
    nn_config: dict[str, Any],
    nn_config_sha256: str,
    val_idx: int,
    train_specs: Sequence[Any],
    val_spec,
    train_rows: int,
    validation_rows: int,
    scaler_path: Path,
    scaler: Standardizer,
    model_paths: dict[int, Path],
    seed_details: list[dict[str, Any]],
    checkpoint_path: Path,
    score: float,
    train,
    device: str,
    torch_version: str,
) -> dict[str, Any]:
    normalizer_stats_sha256 = _array_hash(scaler.mean, scaler.scale)
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pipeline": "temporal_walk_forward_tabular_resnet",
        "dataset_dir": str(dataset_dir.resolve()),
        "dataset_fingerprint": dataset_fingerprint,
        "feature_columns": feature_cols,
        "feature_dtypes": feature_dtypes,
        "feature_schema_sha256": feature_schema_sha256,
        "feature_selection": "fixed_common_numeric_no_target_importance",
        "nn_configuration": nn_config,
        "nn_configuration_sha256": nn_config_sha256,
        "seeds": nn_config["training"]["seeds"],
        "fold_index": int(val_idx),
        "fold": val_spec.name,
        "cutoff_date": val_spec.cutoff.isoformat(),
        "target_start": val_spec.target_start.isoformat(),
        "target_end": val_spec.target_end.isoformat(),
        "train_folds": [item.name for item in train_specs],
        "max_train_target_end": max(item.target_end for item in train_specs).isoformat(),
        "train_rows": int(train_rows),
        "validation_rows": int(validation_rows),
        "normalizer": {
            "identity": "numpy_standardizer_mean_std_ddof0_fit_on_train_only",
            "path": str(scaler_path.resolve()),
            "stats_sha256": normalizer_stats_sha256,
            "file_sha256": train._file_sha256(scaler_path),
        },
        "seed_results": seed_details,
        "best_epochs": {
            str(item["seed"]): int(item["best_epoch"]) for item in seed_details
        },
        "model_paths": {
            str(seed): str(path.resolve()) for seed, path in model_paths.items()
        },
        "model_sha256": {
            str(seed): train._file_sha256(path) for seed, path in model_paths.items()
        },
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": None,
        "rmsle": float(score),
        "runtime": {"device": device, "torch_version": torch_version},
    }


def _save_fold_checkpoint(
    part: pd.DataFrame,
    metadata: dict[str, Any],
    csv_path: Path,
    metadata_path: Path,
    train,
) -> dict[str, Any]:
    _atomic_write_csv(part, csv_path)
    complete = dict(metadata)
    complete["checkpoint_sha256"] = train._file_sha256(csv_path)
    _write_json(metadata_path, complete)
    return complete


def _load_valid_checkpoint(
    *,
    torch,
    csv_path: Path,
    metadata_path: Path,
    scaler_path: Path,
    model_paths: dict[int, Path],
    val_df: pd.DataFrame,
    val_spec,
    val_idx: int,
    dataset_dir: Path,
    dataset_fingerprint: str,
    feature_cols: list[str],
    feature_dtypes: dict[str, str],
    feature_schema_sha256: str,
    nn_config: dict[str, Any],
    nn_config_sha256: str,
    train_specs: Sequence[Any],
    train_rows: int,
    cfg,
    train,
) -> tuple[pd.DataFrame, dict[str, Any]] | None:
    if not csv_path.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "pipeline": "temporal_walk_forward_tabular_resnet",
            "dataset_dir": str(dataset_dir.resolve()),
            "dataset_fingerprint": dataset_fingerprint,
            "feature_columns": feature_cols,
            "feature_dtypes": feature_dtypes,
            "feature_schema_sha256": feature_schema_sha256,
            "feature_selection": "fixed_common_numeric_no_target_importance",
            "nn_configuration": nn_config,
            "nn_configuration_sha256": nn_config_sha256,
            "seeds": nn_config["training"]["seeds"],
            "fold_index": int(val_idx),
            "fold": val_spec.name,
            "cutoff_date": val_spec.cutoff.isoformat(),
            "target_start": val_spec.target_start.isoformat(),
            "target_end": val_spec.target_end.isoformat(),
            "train_folds": [item.name for item in train_specs],
            "max_train_target_end": max(
                item.target_end for item in train_specs
            ).isoformat(),
            "train_rows": int(train_rows),
            "validation_rows": int(len(val_df)),
            "checkpoint_path": str(csv_path.resolve()),
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise AssertionError(f"metadata {key} differs")
        if metadata.get("checkpoint_sha256") != train._file_sha256(csv_path):
            raise AssertionError("checkpoint SHA-256 differs")

        normalizer = metadata.get("normalizer", {})
        if normalizer.get("identity") != (
            "numpy_standardizer_mean_std_ddof0_fit_on_train_only"
        ):
            raise AssertionError("normalizer identity differs")
        if normalizer.get("path") != str(scaler_path.resolve()):
            raise AssertionError("normalizer path differs")
        if not scaler_path.is_file():
            raise AssertionError("normalizer file is missing")
        if normalizer.get("file_sha256") != train._file_sha256(scaler_path):
            raise AssertionError("normalizer file SHA-256 differs")
        with np.load(scaler_path) as saved:
            mean = np.asarray(saved["mean"], dtype=np.float32)
            scale = np.asarray(saved["scale"], dtype=np.float32)
        if mean.shape != (len(feature_cols),) or scale.shape != (len(feature_cols),):
            raise AssertionError("normalizer shape differs")
        normalizer_stats_sha256 = _array_hash(mean, scale)
        if normalizer.get("stats_sha256") != normalizer_stats_sha256:
            raise AssertionError("normalizer statistics hash differs")

        expected_model_paths = {
            str(seed): str(path.resolve()) for seed, path in model_paths.items()
        }
        if metadata.get("model_paths") != expected_model_paths:
            raise AssertionError("model paths differ")
        for seed, model_path in model_paths.items():
            if not model_path.is_file():
                raise AssertionError(f"model for seed={seed} is missing")
            if metadata.get("model_sha256", {}).get(str(seed)) != train._file_sha256(
                model_path
            ):
                raise AssertionError(f"model SHA-256 differs for seed={seed}")
            payload = _validate_model_payload(
                torch,
                model_path,
                seed=seed,
                feature_schema_sha256=feature_schema_sha256,
                normalizer_stats_sha256=normalizer_stats_sha256,
                nn_config_sha256=nn_config_sha256,
            )
            matching = [
                item for item in metadata.get("seed_results", []) if item.get("seed") == seed
            ]
            if len(matching) != 1:
                raise AssertionError(f"seed metadata differs for seed={seed}")
            if int(matching[0].get("best_epoch", 0)) != int(payload["best_epoch"]):
                raise AssertionError(f"best epoch differs for seed={seed}")

        part = pd.read_csv(csv_path)
        score = _validate_oof_part(part, val_df, val_spec, cfg, train)
        if not np.isclose(score, float(metadata["rmsle"]), rtol=0.0, atol=1e-12):
            raise AssertionError("checkpoint RMSLE differs")
        return part, metadata
    except Exception as exc:
        print(f"{val_spec.name}: checkpoint rejected ({type(exc).__name__}: {exc})")
        return None


def _metric_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "fold": metadata["fold"],
        "cutoff_date": metadata["cutoff_date"],
        "max_train_target_end": metadata["max_train_target_end"],
        "train_rows": int(metadata["train_rows"]),
        "validation_rows": int(metadata["validation_rows"]),
        "rmsle": float(metadata["rmsle"]),
        "best_epochs": metadata["best_epochs"],
        "checkpoint_source": "validated_resume",
    }


def _validate_and_save_oof(
    parts: list[pd.DataFrame],
    oof_path: Path,
    expected_rows_by_fold: dict[str, int],
    train,
) -> tuple[pd.DataFrame, float]:
    oof = pd.concat(parts, ignore_index=True)
    if oof.columns.tolist() != CHECKPOINT_COLUMNS:
        raise AssertionError("NN combined OOF column contract differs")
    if oof.duplicated(KEY_COLUMNS).any():
        raise AssertionError("NN OOF composite key is not unique")
    numeric = oof[["target", "pred", "pred_log"]].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all() or (numeric < 0).any():
        raise AssertionError("NN OOF contains invalid values")
    if not np.allclose(oof["pred_log"], np.log1p(oof["pred"]), atol=1e-10):
        raise AssertionError("NN combined OOF pred_log != log1p(pred)")
    actual_rows = oof.groupby("fold", sort=False).size().to_dict()
    if actual_rows != expected_rows_by_fold:
        raise AssertionError(f"NN OOF coverage differs: {actual_rows} != {expected_rows_by_fold}")
    score = float(train.rmsle(oof["target"], oof["pred"]))
    _atomic_write_csv(oof, oof_path)
    return oof, score


def run(args: argparse.Namespace) -> dict[str, Any]:
    cfg, time_split, train = _load_project_api()
    torch, nn, DataLoader, TensorDataset = _require_torch()

    if args.final_refit_only and args.skip_final_refit:
        raise ValueError("--final-refit-only cannot be combined with --skip-final-refit")
    if args.final_refit_only and not args.final_epochs:
        raise ValueError(
            "--final-refit-only requires the pre-committed --final-epochs contract"
        )

    dataset_dir = Path(args.dataset_dir or cfg.DATA_DIR)
    seeds = _parse_seeds(args.seeds)
    device = _preflight_device(torch, args.device)
    artifacts = train.resolve_artifact_layout(dataset_dir, None, args.run_name)
    model_dir = Path(args.model_dir or (artifacts.models_root / "nn"))
    oof_path = Path(args.oof_path or (artifacts.oof_dir / "oof_nn.csv"))
    if not artifacts.is_canonical:
        canonical_model_dir = (cfg.MODELS_DIR / "nn").resolve()
        canonical_oof_path = (cfg.OOF_DIR / "oof_nn.csv").resolve()
        if model_dir.resolve() == canonical_model_dir:
            raise ValueError(
                "An isolated/custom NN run cannot write the canonical NN model directory. "
                "Remove --model-dir or choose a non-canonical path."
            )
        if oof_path.resolve() == canonical_oof_path:
            raise ValueError(
                "An isolated/custom NN run cannot write canonical oof_nn.csv. "
                "Remove --oof-path or choose a non-canonical path."
            )
    print(
        f"Artifact mode={artifacts.artifact_mode} | run_name={artifacts.run_name} | "
        f"model_dir={model_dir} | oof={oof_path}"
    )
    specs = _build_fold_specs(time_split, cfg, cfg.N_FOLDS)
    first_fold, feature_cols, feature_dtypes = _load_checked_fold(
        dataset_dir, 0, specs[0], train, cfg
    )
    del first_fold
    gc.collect()
    row_counts = [
        len(pd.read_parquet(dataset_dir / f"{spec.name}.parquet", columns=[cfg.ID_COL]))
        for spec in specs
    ]
    dataset_fingerprint = train._dataset_fingerprint(dataset_dir, None)
    feature_schema = {"columns": feature_cols, "dtypes": feature_dtypes}
    feature_schema_sha256 = _json_hash(feature_schema)

    hidden_dim = args.hidden_dim or cfg.NN_HIDDEN_DIM
    num_blocks = args.num_blocks if args.num_blocks is not None else cfg.NN_NUM_BLOCKS
    dropout = args.dropout if args.dropout is not None else cfg.NN_DROPOUT
    batch_size = args.batch_size or cfg.NN_BATCH_SIZE
    max_epochs = args.max_epochs or cfg.NN_EPOCHS
    learning_rate = args.learning_rate or cfg.NN_LR
    weight_decay = (
        args.weight_decay if args.weight_decay is not None else cfg.NN_WEIGHT_DECAY
    )
    patience = args.patience or cfg.NN_PATIENCE
    nn_config = {
        "architecture": {
            "name": "TabularResNet",
            "input_dim": len(feature_cols),
            "hidden_dim": hidden_dim,
            "num_blocks": num_blocks,
            "dropout": dropout,
        },
        "training": {
            "seeds": list(seeds),
            "batch_size": batch_size,
            "max_cv_epochs": max_epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "patience": patience,
            "loss": "MSE_on_log1p_target",
            "scheduler": "ReduceLROnPlateau_factor0.5_patience2",
            "gradient_clip_norm": 5.0,
        },
        "prediction": "expm1(max(mean_seed_pred_log,0))",
    }
    nn_config_sha256 = _json_hash(nn_config)
    explicit_final_epochs = getattr(args, "final_epochs", None)
    parsed_final_epochs = (
        _parse_final_epochs(explicit_final_epochs, seeds)
        if explicit_final_epochs
        else None
    )

    if args.final_refit_only:
        selection_manifest_path = model_dir / "manifest.json"
        if not selection_manifest_path.is_file():
            raise FileNotFoundError(
                "Canonical NN selection manifest is required for --final-refit-only: "
                f"{selection_manifest_path}"
            )
        selection_manifest = json.loads(
            selection_manifest_path.read_text(encoding="utf-8")
        )
        expected_epochs = {"42": 15, "1337": 6, "2026": 8}
        exact_contract = {
            "feature_count": 436,
            "seeds": [42, 1337, 2026],
            "hidden_dim": 256,
            "num_blocks": 2,
            "dropout": 0.15,
            "batch_size": 4096,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "loss": "MSE_on_log1p_target",
        }
        observed_contract = {
            "feature_count": len(feature_cols),
            "seeds": list(seeds),
            "hidden_dim": hidden_dim,
            "num_blocks": num_blocks,
            "dropout": dropout,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "loss": nn_config["training"]["loss"],
        }
        if observed_contract != exact_contract:
            raise AssertionError(
                f"Final NN contract differs: {observed_contract} != {exact_contract}"
            )
        manifest_checks = {
            "dataset_fingerprint": dataset_fingerprint,
            "feature_count": len(feature_cols),
            "feature_schema_sha256": feature_schema_sha256,
            "nn_configuration_sha256": nn_config_sha256,
            "final_refit_rows": int(sum(row_counts)),
            "final_epochs_by_seed": expected_epochs,
        }
        for key, expected_value in manifest_checks.items():
            if selection_manifest.get(key) != expected_value:
                raise AssertionError(
                    f"Selection manifest {key} differs: "
                    f"{selection_manifest.get(key)!r} != {expected_value!r}"
                )
        if selection_manifest.get("nn_configuration") != nn_config:
            raise AssertionError("Selection manifest NN configuration differs")
        if {str(k): v for k, v in parsed_final_epochs.items()} != expected_epochs:
            raise AssertionError(
                "Final epochs must remain exactly 42:15,1337:6,2026:8"
            )
        print(
            "Validated canonical final NN contract | "
            f"manifest={selection_manifest_path} rows={sum(row_counts):,} "
            f"epochs={expected_epochs}"
        )

    print("=" * 78)
    print("TEMPORAL NN CV")
    print(f"device={device} folds={len(specs)} features={len(feature_cols)} seeds={seeds}")
    print("Feature selection: fixed common numeric columns; no global importance used")
    print(f"dataset_fingerprint={dataset_fingerprint}")
    print("=" * 78)

    oof_parts: list[pd.DataFrame] = []
    fold_results: list[dict[str, Any]] = []
    best_epochs_by_seed: dict[int, list[int]] = {seed: [] for seed in seeds}
    checkpoint_metadata_paths: list[str] = []
    cv_model_paths: list[Path] = []

    for val_index in range(1, len(specs)):
        spec = specs[val_index]
        train_specs = specs[:val_index]
        val_df, _, _ = _load_checked_fold(
            dataset_dir,
            val_index,
            spec,
            train,
            cfg,
            expected_features=feature_cols,
            expected_dtypes=feature_dtypes,
        )
        train_rows = int(sum(row_counts[:val_index]))
        csv_path, metadata_path, scaler_path = _checkpoint_paths(
            artifacts.oof_dir, model_dir, spec.name
        )
        model_paths = {
            seed: model_dir / f"model_{spec.name}_seed_{seed}.pt" for seed in seeds
        }
        print("-" * 78)
        print(
            f"validation={spec.name} | train_anchors="
            f"{[str(item.cutoff) for item in train_specs]} | "
            f"train_target_end={max(item.target_end for item in train_specs)} | "
            f"validation_cutoff={spec.cutoff} | "
            f"validation_target={spec.target_start}..{spec.target_end} | "
            f"rows={train_rows:,}/{len(val_df):,}"
        )

        checkpoint = _load_valid_checkpoint(
            torch=torch,
            csv_path=csv_path,
            metadata_path=metadata_path,
            scaler_path=scaler_path,
            model_paths=model_paths,
            val_df=val_df,
            val_spec=spec,
            val_idx=val_index,
            dataset_dir=dataset_dir,
            dataset_fingerprint=dataset_fingerprint,
            feature_cols=feature_cols,
            feature_dtypes=feature_dtypes,
            feature_schema_sha256=feature_schema_sha256,
            nn_config=nn_config,
            nn_config_sha256=nn_config_sha256,
            train_specs=train_specs,
            train_rows=train_rows,
            cfg=cfg,
            train=train,
        )
        if checkpoint is not None:
            part, metadata = checkpoint
            print(
                f"{spec.name}: reusing validated checkpoint | "
                f"RMSLE={metadata['rmsle']:.6f}, "
                f"best_epochs={metadata['best_epochs']}"
            )
            oof_parts.append(part)
            fold_results.append(_metric_from_metadata(metadata))
            checkpoint_metadata_paths.append(str(metadata_path.resolve()))
            for detail in metadata["seed_results"]:
                best_epochs_by_seed[int(detail["seed"])].append(
                    int(detail["best_epoch"])
                )
            cv_model_paths.extend(model_paths.values())
            del val_df, part
            gc.collect()
            continue

        if args.final_refit_only:
            raise RuntimeError(
                f"{spec.name}: canonical CV checkpoint did not validate; "
                "final-refit-only mode forbids CV retraining"
            )

        max_train_target_end = max(item.target_end for item in train_specs)
        if max_train_target_end > spec.cutoff:
            raise AssertionError("NN temporal leakage invariant failed")

        x_train, y_train = _load_training_arrays(
            dataset_dir,
            specs,
            val_index,
            row_counts,
            feature_cols,
            feature_dtypes,
            train,
            cfg,
        )
        x_val = _matrix(val_df, feature_cols)
        scaler = Standardizer.fit(x_train)
        scaler.transform(x_train)
        scaler.transform(x_val)
        y_val = val_df[cfg.TARGET_LOG_COL].to_numpy(dtype=np.float32)
        _save_standardizer_atomic(scaler, scaler_path)
        normalizer_stats_sha256 = _array_hash(scaler.mean, scaler.scale)

        seed_predictions: list[np.ndarray] = []
        seed_details: list[dict[str, Any]] = []
        for seed in seeds:
            model, pred_log_raw, best_epoch, best_loss = _train_one_seed(
                torch,
                nn,
                DataLoader,
                TensorDataset,
                x_train,
                y_train,
                x_val,
                y_val,
                seed=seed,
                device=device,
                hidden_dim=hidden_dim,
                num_blocks=num_blocks,
                dropout=dropout,
                batch_size=batch_size,
                max_epochs=max_epochs,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                patience=patience,
            )
            seed_predictions.append(pred_log_raw)
            best_epochs_by_seed[seed].append(best_epoch)
            detail = {
                "seed": seed,
                "best_epoch": best_epoch,
                "best_log_mse": float(best_loss),
            }
            seed_details.append(detail)
            _save_model_atomic(
                torch,
                _model_payload(
                    model,
                    seed=seed,
                    best_epoch=best_epoch,
                    best_loss=best_loss,
                    feature_schema_sha256=feature_schema_sha256,
                    normalizer_stats_sha256=normalizer_stats_sha256,
                    nn_config_sha256=nn_config_sha256,
                    nn_config=nn_config,
                ),
                model_paths[seed],
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        pred_log = np.clip(np.mean(seed_predictions, axis=0), 0.0, None)
        part = _make_oof_part(val_df, spec, pred_log, cfg)
        score = _validate_oof_part(part, val_df, spec, cfg, train)
        metadata = _checkpoint_metadata(
            dataset_dir=dataset_dir,
            dataset_fingerprint=dataset_fingerprint,
            feature_cols=feature_cols,
            feature_dtypes=feature_dtypes,
            feature_schema_sha256=feature_schema_sha256,
            nn_config=nn_config,
            nn_config_sha256=nn_config_sha256,
            val_idx=val_index,
            train_specs=train_specs,
            val_spec=spec,
            train_rows=train_rows,
            validation_rows=len(val_df),
            scaler_path=scaler_path,
            scaler=scaler,
            model_paths=model_paths,
            seed_details=seed_details,
            checkpoint_path=csv_path,
            score=score,
            train=train,
            device=str(device),
            torch_version=torch.__version__,
        )
        metadata = _save_fold_checkpoint(
            part, metadata, csv_path, metadata_path, train
        )
        print(
            f"{spec.name}: RMSLE={score:.6f}, "
            f"best_epochs={metadata['best_epochs']} | checkpoint={csv_path}"
        )
        oof_parts.append(part)
        metric = _metric_from_metadata(metadata)
        metric["checkpoint_source"] = "trained"
        fold_results.append(metric)
        checkpoint_metadata_paths.append(str(metadata_path.resolve()))
        cv_model_paths.extend(model_paths.values())

        del x_train, x_val, y_train, y_val, val_df, seed_predictions
        gc.collect()

    expected_rows_by_fold = {
        spec.name: row_counts[index]
        for index, spec in enumerate(specs)
        if index > 0
    }
    oof, pooled_score = _validate_and_save_oof(
        oof_parts, oof_path, expected_rows_by_fold, train
    )
    oof_rows = int(len(oof))
    print(f"Pooled NN OOF RMSLE={pooled_score:.6f}")
    print(f"OOF saved: {oof_path}")

    if parsed_final_epochs is not None:
        final_epochs_by_seed = parsed_final_epochs
        final_epoch_strategy = "explicit_fixed_post_cv"
    else:
        final_epochs_by_seed = {
            seed: max(1, int(round(float(np.median(epochs)))))
            for seed, epochs in best_epochs_by_seed.items()
        }
        final_epoch_strategy = "median_cv_best_epoch_by_seed"
    model_files: list[str] = []
    final_scaler_path = model_dir / "normalizer.npz"
    feature_path = model_dir / "features.json"
    model_dir.mkdir(parents=True, exist_ok=True)
    final_training_rows = int(sum(row_counts))
    final_matrix_bytes = final_training_rows * len(feature_cols) * np.dtype(np.float32).itemsize
    final_normalizer_stats_sha256: str | None = None
    final_normalizer_file_sha256: str | None = None
    final_feature_file_sha256: str | None = None
    final_matrix_path: Path | None = None
    final_matrix_removed = False
    final_matrix_storage = "in_memory_float32_array"

    # The combined OOF is already atomically persisted and validated. Releasing
    # it before allocating the final matrix keeps the peak below the proven
    # fold_5 CV peak (which held both train and validation matrices).
    del oof, oof_parts
    gc.collect()

    if not args.skip_final_refit:
        print("=" * 78)
        print("MEMORY-SAFE FINAL NN REFIT ON ALL LABELED SNAPSHOTS")
        print(
            f"rows={final_training_rows:,} features={len(feature_cols)} "
            f"float32_matrix_bytes={final_matrix_bytes:,} "
            f"epochs={final_epochs_by_seed}"
        )
        if args.final_array_dir is not None:
            final_matrix_path = Path(args.final_array_dir) / (
                f"nn_final_{dataset_fingerprint}_{final_training_rows}x"
                f"{len(feature_cols)}.float32.memmap"
            )
            final_matrix_storage = "disk_backed_float32_memmap"
            print(f"disk-backed training matrix={final_matrix_path}")
        x_all, y_all = _load_training_arrays(
            dataset_dir,
            specs,
            len(specs),
            row_counts,
            feature_cols,
            feature_dtypes,
            train,
            cfg,
            matrix_path=final_matrix_path,
        )
        final_scaler = Standardizer.fit(x_all)
        final_scaler.transform(x_all)
        if isinstance(x_all, np.memmap):
            x_all.flush()
        _save_standardizer_atomic(final_scaler, final_scaler_path)
        final_normalizer_stats_sha256 = _array_hash(
            final_scaler.mean, final_scaler.scale
        )
        final_normalizer_file_sha256 = train._file_sha256(final_scaler_path)
        _write_json(
            feature_path,
            {
                "dataset_fingerprint": dataset_fingerprint,
                "features": feature_cols,
                "dtypes": feature_dtypes,
                "feature_schema_sha256": feature_schema_sha256,
            },
        )
        final_feature_file_sha256 = train._file_sha256(feature_path)
        for seed in seeds:
            epochs = final_epochs_by_seed[seed]
            model_path = model_dir / f"model_seed_{seed}.pt"
            existing = _load_valid_final_model(
                torch,
                model_path,
                seed=seed,
                epochs=epochs,
                final_training_rows=final_training_rows,
                dataset_fingerprint=dataset_fingerprint,
                feature_schema_sha256=feature_schema_sha256,
                nn_config_sha256=nn_config_sha256,
                normalizer_stats_sha256=final_normalizer_stats_sha256,
                normalizer_file_sha256=final_normalizer_file_sha256,
            )
            if existing is not None:
                print(f"    final seed={seed}: reusing validated model ({epochs} epochs)")
                model_files.append(model_path.name)
                continue
            model = _fit_final_seed(
                torch,
                nn,
                DataLoader,
                TensorDataset,
                x_all,
                y_all,
                seed=seed,
                epochs=epochs,
                device=device,
                hidden_dim=hidden_dim,
                num_blocks=num_blocks,
                dropout=dropout,
                batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
            )
            _save_model_atomic(
                torch,
                _final_model_payload(
                    model,
                    seed=seed,
                    epochs=epochs,
                    final_training_rows=final_training_rows,
                    dataset_fingerprint=dataset_fingerprint,
                    feature_schema_sha256=feature_schema_sha256,
                    nn_config_sha256=nn_config_sha256,
                    normalizer_stats_sha256=final_normalizer_stats_sha256,
                    normalizer_file_sha256=final_normalizer_file_sha256,
                    nn_config=nn_config,
                ),
                model_path,
            )
            model_files.append(model_path.name)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if len(model_files) != len(seeds):
            raise AssertionError("Not all final NN seed models were persisted")
        if isinstance(x_all, np.memmap):
            x_all.flush()
        del x_all, y_all
        gc.collect()
        if final_matrix_path is not None and final_matrix_path.is_file():
            final_matrix_path.unlink()
            final_matrix_removed = True
            print(f"removed completed temporary matrix={final_matrix_path}")

    scores = [float(item["rmsle"]) for item in fold_results]
    latest_fold_score = scores[-1]
    mean_fold_score = float(np.mean(scores))
    std_fold_score = float(np.std(scores))
    cv_model_paths = list(dict.fromkeys(cv_model_paths))

    manifest = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pipeline": "temporal_walk_forward_tabular_resnet",
        "target_transform": "log1p",
        "effective_prediction": "expm1(max(pred_log, 0))",
        "dataset_dir": str(dataset_dir.resolve()),
        "dataset_fingerprint": dataset_fingerprint,
        "artifact_mode": artifacts.artifact_mode,
        "run_name": artifacts.run_name,
        "row_limit": None,
        "oof_path": str(oof_path.resolve()),
        "oof_rows": oof_rows,
        "pooled_oof_rmsle": pooled_score,
        "mean_fold_rmsle": mean_fold_score,
        "std_fold_rmsle": std_fold_score,
        "latest_fold_rmsle": latest_fold_score,
        "key_columns": KEY_COLUMNS,
        "feature_selection": "fixed_common_numeric_no_target_importance",
        "feature_count": len(feature_cols),
        "feature_columns": feature_cols,
        "feature_dtypes": feature_dtypes,
        "feature_schema_sha256": feature_schema_sha256,
        "feature_file": feature_path.name if feature_path.exists() else None,
        "feature_file_sha256": final_feature_file_sha256,
        "normalizer_file": final_scaler_path.name if final_scaler_path.exists() else None,
        "normalizer_file_sha256": final_normalizer_file_sha256,
        "normalizer_stats_sha256": final_normalizer_stats_sha256,
        "normalizer_identity": (
            "numpy_standardizer_mean_std_ddof0_fit_on_all_labeled_snapshots"
            if not args.skip_final_refit
            else None
        ),
        "model_files": model_files,
        "model_file_sha256": {
            filename: train._file_sha256(model_dir / filename)
            for filename in model_files
        },
        "final_refit_completed": (
            not args.skip_final_refit and len(model_files) == len(seeds)
        ),
        "final_refit_rows": final_training_rows,
        "final_epochs_by_seed": {str(k): v for k, v in final_epochs_by_seed.items()},
        "final_epoch_strategy": final_epoch_strategy,
        "final_matrix_storage": final_matrix_storage,
        "final_matrix_temporary_path": (
            str(final_matrix_path.resolve()) if final_matrix_path is not None else None
        ),
        "final_matrix_temporary_removed": final_matrix_removed,
        "final_matrix_bytes": final_matrix_bytes,
        "folds": fold_results,
        "nn_configuration": nn_config,
        "nn_configuration_sha256": nn_config_sha256,
        "cv_model_files": [str(path.resolve()) for path in cv_model_paths],
        "cv_model_sha256": {
            str(path.resolve()): train._file_sha256(path) for path in cv_model_paths
        },
        "fold_checkpoint_metadata": checkpoint_metadata_paths,
        "runtime": {"device": str(device), "torch_version": torch.__version__},
        "final_refit_note": (
            (
                "completed with sequential fold loading and a temporary "
                "disk-backed float32 memmap"
                if final_matrix_path is not None
                else "completed with sequential fold loading and one in-place float32 matrix"
            )
            if not args.skip_final_refit
            else "deferred until OOF evidence and ensemble value are established"
        ),
    }
    manifest_path = model_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    _atomic_write_csv(pd.DataFrame(fold_results), artifacts.reports_dir / "nn_fold_metrics.csv")
    print(f"NN manifest saved: {manifest_path}")
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the NN with the canonical temporal walk-forward folds."
    )
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--oof-path", type=Path, default=None)
    parser.add_argument(
        "--run-name",
        default=None,
        help=(
            "Isolated artifact namespace. A custom/non-full dataset is "
            "auto-isolated when this is omitted."
        ),
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--num-blocks", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument(
        "--final-epochs",
        default=None,
        help=(
            "Explicit fixed final epochs by seed, for example "
            "42:15,1337:6,2026:8. Omit to use median CV best epochs."
        ),
    )
    parser.add_argument(
        "--skip-final-refit",
        action="store_true",
        help="Development-only: produce OOF without the required final refit.",
    )
    parser.add_argument(
        "--final-refit-only",
        action="store_true",
        help=(
            "Validate and reuse every canonical CV checkpoint, then run only the "
            "fixed final refit. Any invalid CV checkpoint aborts instead of retraining."
        ),
    )
    parser.add_argument(
        "--final-array-dir",
        type=Path,
        default=None,
        help="Directory for the temporary disk-backed final float32 matrix.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
