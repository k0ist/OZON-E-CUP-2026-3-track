"""CatBoost on the same leakage-safe walk-forward folds as LightGBM."""

from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

import config as cfg
from data_loading import load_fold
from time_split import build_cv_folds, walk_forward_splits
from train import (
    _dataset_fingerprint,
    _file_sha256,
    _validate_loaded_fold,
    get_features,
    resolve_artifact_layout,
    rmsle,
)


def _load_checked_fold(
    dataset_dir: Path,
    fold_index: int,
    expected_spec,
    expected_features: list[str] | None = None,
    row_limit: int | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Load only one snapshot so CatBoost never retains future folds in RAM."""

    frame = load_fold(fold_index, data_dir=dataset_dir)
    if row_limit is not None and len(frame) > row_limit:
        frame = frame.sample(row_limit, random_state=cfg.RANDOM_STATE).sort_values(
            cfg.ID_COL
        ).reset_index(drop=True)
    _validate_loaded_fold(frame, expected_spec)
    features = get_features([frame])
    if expected_features is not None and features != expected_features:
        raise AssertionError(f"Feature schema mismatch in {expected_spec.name}")
    return frame, features


def _print_lazy_walk_forward_plan(
    dataset_dir: Path, registry: list, row_limit: int | None = None
) -> None:
    row_counts = [
        len(pd.read_parquet(dataset_dir / f"{spec.name}.parquet", columns=[cfg.ID_COL]))
        for spec in registry
    ]
    if row_limit is not None:
        row_counts = [min(count, row_limit) for count in row_counts]
    print("\nWALK-FORWARD CONTRACT (lazy CatBoost loading)")
    print("=" * 100)
    for val_idx, train_specs, val_spec in walk_forward_splits(registry):
        max_target_end = max(item.target_end for item in train_specs)
        print(
            f"validation={val_spec.name} | train anchors="
            f"{[item.cutoff.isoformat() for item in train_specs]} | "
            f"train target_end(max)={max_target_end} | "
            f"validation cutoff={val_spec.cutoff} | validation target="
            f"{val_spec.target_start}..{val_spec.target_end} | rows train="
            f"{sum(row_counts[:val_idx]):,}, validation={row_counts[val_idx]:,}"
        )


CHECKPOINT_COLUMNS = [
    cfg.ID_COL,
    "fold",
    "cutoff_date",
    "target",
    "pred",
    "pred_log",
]
CHECKPOINT_VERSION = 1


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _checkpoint_paths(oof_dir: Path, fold_name: str) -> tuple[Path, Path]:
    csv_path = oof_dir / f"oof_catboost_{fold_name}.csv"
    metadata_path = oof_dir / f"oof_catboost_{fold_name}.meta.json"
    return csv_path, metadata_path


def _model_matches_run(
    model: CatBoostRegressor,
    feature_cols: list[str],
    base_params: dict[str, Any],
) -> None:
    if list(model.feature_names_) != feature_cols:
        raise AssertionError("Persisted CatBoost model feature schema differs")
    if int(model.tree_count_) < 1 or int(model.tree_count_) > int(
        base_params["iterations"]
    ):
        raise AssertionError("Persisted CatBoost tree count is invalid")
    actual = model.get_all_params()
    exact = {
        "depth": int(base_params["depth"]),
        "random_seed": int(base_params["random_seed"]),
        "loss_function": str(base_params["loss_function"]),
        "task_type": str(base_params["task_type"]),
    }
    for key, expected in exact.items():
        if actual.get(key) != expected:
            raise AssertionError(
                f"Persisted CatBoost {key}={actual.get(key)!r}, expected {expected!r}"
            )
    if not np.isclose(
        float(actual.get("learning_rate")),
        float(base_params["learning_rate"]),
        rtol=1e-7,
        atol=1e-9,
    ):
        raise AssertionError("Persisted CatBoost learning_rate differs")


def _make_oof_part(val_df: pd.DataFrame, val_spec, pred_log: np.ndarray) -> pd.DataFrame:
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


def _validate_oof_part(
    part: pd.DataFrame,
    val_df: pd.DataFrame,
    val_spec,
) -> float:
    if part.columns.tolist() != CHECKPOINT_COLUMNS:
        raise AssertionError(
            f"{val_spec.name}: checkpoint columns={part.columns.tolist()}"
        )
    if len(part) != len(val_df):
        raise AssertionError(f"{val_spec.name}: checkpoint row count differs")
    if part.duplicated([cfg.ID_COL, "fold", "cutoff_date"]).any():
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
    if not np.allclose(part["pred_log"], np.log1p(part["pred"]), atol=1e-7):
        raise AssertionError(f"{val_spec.name}: pred_log != log1p(pred)")
    return rmsle(part["target"], part["pred"])


def _checkpoint_metadata(
    *,
    dataset_dir: Path,
    dataset_fingerprint: str,
    row_limit: int | None,
    base_params: dict[str, Any],
    feature_cols: list[str],
    feature_dtypes: dict[str, str],
    val_idx: int,
    train_specs: list,
    val_spec,
    train_rows: int,
    validation_rows: int,
    best_iteration: int,
    score: float,
    model_path: Path,
    checkpoint_path: Path,
    source: str,
) -> dict[str, Any]:
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "dataset_dir": str(dataset_dir.resolve()),
        "dataset_fingerprint": dataset_fingerprint,
        "row_limit": row_limit,
        "parameters": base_params,
        "feature_columns": feature_cols,
        "feature_dtypes": feature_dtypes,
        "fold_index": val_idx,
        "fold": val_spec.name,
        "cutoff_date": val_spec.cutoff.isoformat(),
        "target_start": val_spec.target_start.isoformat(),
        "target_end": val_spec.target_end.isoformat(),
        "train_folds": [item.name for item in train_specs],
        "max_train_target_end": max(item.target_end for item in train_specs).isoformat(),
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "best_iteration": int(best_iteration),
        "rmsle": float(score),
        "model_path": str(model_path.resolve()),
        "model_sha256": _file_sha256(model_path),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": None,
    }


def _save_fold_checkpoint(
    part: pd.DataFrame,
    metadata: dict[str, Any],
    csv_path: Path,
    metadata_path: Path,
) -> None:
    _atomic_write_csv(part, csv_path)
    metadata = dict(metadata)
    metadata["checkpoint_sha256"] = _file_sha256(csv_path)
    _atomic_write_json(metadata, metadata_path)


def _load_valid_checkpoint(
    *,
    csv_path: Path,
    metadata_path: Path,
    val_df: pd.DataFrame,
    val_spec,
    val_idx: int,
    dataset_dir: Path,
    dataset_fingerprint: str,
    row_limit: int | None,
    base_params: dict[str, Any],
    feature_cols: list[str],
    feature_dtypes: dict[str, str],
    train_specs: list,
    train_rows: int,
) -> tuple[pd.DataFrame, dict[str, Any]] | None:
    if not csv_path.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "dataset_dir": str(dataset_dir.resolve()),
            "dataset_fingerprint": dataset_fingerprint,
            "row_limit": row_limit,
            "parameters": base_params,
            "feature_columns": feature_cols,
            "feature_dtypes": feature_dtypes,
            "fold_index": val_idx,
            "fold": val_spec.name,
            "cutoff_date": val_spec.cutoff.isoformat(),
            "target_start": val_spec.target_start.isoformat(),
            "target_end": val_spec.target_end.isoformat(),
            "train_folds": [item.name for item in train_specs],
            "max_train_target_end": max(
                item.target_end for item in train_specs
            ).isoformat(),
            "train_rows": train_rows,
            "validation_rows": len(val_df),
            "checkpoint_path": str(csv_path.resolve()),
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise AssertionError(f"metadata {key} differs")
        if metadata.get("checkpoint_sha256") != _file_sha256(csv_path):
            raise AssertionError("checkpoint SHA-256 differs")
        model_path = Path(str(metadata.get("model_path", "")))
        if not model_path.is_file() or metadata.get("model_sha256") != _file_sha256(
            model_path
        ):
            raise AssertionError("model file/hash differs")
        part = pd.read_csv(csv_path)
        score = _validate_oof_part(part, val_df, val_spec)
        if not np.isclose(score, float(metadata["rmsle"]), rtol=0.0, atol=1e-12):
            raise AssertionError("checkpoint RMSLE differs")
        return part, metadata
    except Exception as exc:
        print(f"{val_spec.name}: checkpoint rejected ({type(exc).__name__}: {exc})")
        return None


def _save_model_atomic(model: CatBoostRegressor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    model.save_model(str(temporary), format="cbm")
    temporary.replace(path)


def _metric_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "fold": metadata["fold"],
        "cutoff_date": metadata["cutoff_date"],
        "max_train_target_end": metadata["max_train_target_end"],
        "train_rows": int(metadata["train_rows"]),
        "validation_rows": int(metadata["validation_rows"]),
        "rmsle": float(metadata["rmsle"]),
        "best_iteration": int(metadata["best_iteration"]),
        "checkpoint_source": metadata["source"],
    }


def _validate_and_save_oof(
    parts: list[pd.DataFrame],
    oof_path: Path,
    expected_rows_by_fold: dict[str, int],
) -> tuple[pd.DataFrame, float]:
    oof = pd.concat(parts, ignore_index=True)
    if oof.columns.tolist() != CHECKPOINT_COLUMNS:
        raise AssertionError("CatBoost combined OOF column contract differs")
    keys = [cfg.ID_COL, "fold", "cutoff_date"]
    if oof.duplicated(keys).any():
        raise AssertionError("CatBoost OOF composite key is not unique")
    numeric = oof[["target", "pred", "pred_log"]].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all() or (numeric < 0).any():
        raise AssertionError("CatBoost OOF contains invalid values")
    actual_rows = oof.groupby("fold", sort=False).size().to_dict()
    if actual_rows != expected_rows_by_fold:
        raise AssertionError(
            f"CatBoost OOF coverage differs: {actual_rows} != {expected_rows_by_fold}"
        )
    score = rmsle(oof["target"], oof["pred"])
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_csv(oof, oof_path)
    return oof, score


def train_catboost_cv(
    dataset_dir: str | Path | None = None,
    task_type: str = "CPU",
    iterations: int = 1500,
    learning_rate: float = 0.04,
    depth: int = 6,
    early_stopping_rounds: int = 100,
    final_refit: bool = True,
    row_limit: int | None = None,
    run_name: str | None = None,
    recover_existing_models: bool = False,
) -> dict:
    dataset_dir = Path(dataset_dir) if dataset_dir else cfg.DATA_DIR
    registry = build_cv_folds(cfg.N_FOLDS, cfg.STEP_DAYS)
    first_fold, feature_cols = _load_checked_fold(
        dataset_dir, 0, registry[0], row_limit=row_limit
    )
    feature_dtypes = {column: str(first_fold[column].dtype) for column in feature_cols}
    del first_fold
    gc.collect()
    _print_lazy_walk_forward_plan(dataset_dir, registry, row_limit=row_limit)

    artifacts = resolve_artifact_layout(dataset_dir, row_limit, run_name)
    model_dir = artifacts.models_root / "catboost"
    model_dir.mkdir(parents=True, exist_ok=True)
    artifacts.oof_dir.mkdir(parents=True, exist_ok=True)
    artifacts.reports_dir.mkdir(parents=True, exist_ok=True)
    oof_path = artifacts.oof_dir / "oof_catboost.csv"
    print(
        f"Artifact mode={artifacts.artifact_mode} | run_name={artifacts.run_name} | "
        f"model_dir={model_dir} | oof={oof_path} | "
        f"reports_dir={artifacts.reports_dir}"
    )

    base_params = {
        "iterations": iterations,
        "learning_rate": learning_rate,
        "depth": depth,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "random_seed": cfg.RANDOM_STATE,
        "verbose": 200,
        "task_type": task_type.upper(),
        "allow_writing_files": False,
        "thread_count": -1,
    }
    dataset_fingerprint = _dataset_fingerprint(dataset_dir, row_limit)
    row_counts = [
        len(pd.read_parquet(dataset_dir / f"{spec.name}.parquet", columns=[cfg.ID_COL]))
        for spec in registry
    ]
    if row_limit is not None:
        row_counts = [min(count, row_limit) for count in row_counts]

    oof_parts: list[pd.DataFrame] = []
    fold_metrics: list[dict] = []
    best_iterations: list[int] = []
    checkpoint_metadata_paths: list[str] = []

    # Deliberately starts at fold_1. Future folds are never present in train.
    for val_idx, train_specs, val_spec in walk_forward_splits(registry):
        val_df, _ = _load_checked_fold(
            dataset_dir,
            val_idx,
            val_spec,
            feature_cols,
            row_limit=row_limit,
        )
        train_rows = sum(row_counts[:val_idx])
        csv_path, metadata_path = _checkpoint_paths(artifacts.oof_dir, val_spec.name)
        model_path = model_dir / f"model_{val_spec.name}.cbm"
        checkpoint = _load_valid_checkpoint(
            csv_path=csv_path,
            metadata_path=metadata_path,
            val_df=val_df,
            val_spec=val_spec,
            val_idx=val_idx,
            dataset_dir=dataset_dir,
            dataset_fingerprint=dataset_fingerprint,
            row_limit=row_limit,
            base_params=base_params,
            feature_cols=feature_cols,
            feature_dtypes=feature_dtypes,
            train_specs=train_specs,
            train_rows=train_rows,
        )
        if checkpoint is not None:
            part, metadata = checkpoint
            print(
                f"{val_spec.name}: reusing validated checkpoint | "
                f"RMSLE={metadata['rmsle']:.6f}, "
                f"best_iteration={metadata['best_iteration']}"
            )
            oof_parts.append(part)
            fold_metrics.append(_metric_from_metadata(metadata))
            best_iterations.append(int(metadata["best_iteration"]))
            checkpoint_metadata_paths.append(str(metadata_path.resolve()))
            del val_df, part
            gc.collect()
            continue

        if recover_existing_models and model_path.is_file():
            recovered_model = CatBoostRegressor()
            recovered_model.load_model(str(model_path))
            _model_matches_run(recovered_model, feature_cols, base_params)
            recovered_pred_log = recovered_model.predict(val_df[feature_cols])
            part = _make_oof_part(val_df, val_spec, recovered_pred_log)
            score = _validate_oof_part(part, val_df, val_spec)
            best_iteration = int(recovered_model.tree_count_)
            metadata = _checkpoint_metadata(
                dataset_dir=dataset_dir,
                dataset_fingerprint=dataset_fingerprint,
                row_limit=row_limit,
                base_params=base_params,
                feature_cols=feature_cols,
                feature_dtypes=feature_dtypes,
                val_idx=val_idx,
                train_specs=train_specs,
                val_spec=val_spec,
                train_rows=train_rows,
                validation_rows=len(val_df),
                best_iteration=best_iteration,
                score=score,
                model_path=model_path,
                checkpoint_path=csv_path,
                source="recovered_existing_validated_model",
            )
            _save_fold_checkpoint(part, metadata, csv_path, metadata_path)
            metadata["checkpoint_sha256"] = _file_sha256(csv_path)
            print(
                f"{val_spec.name}: recovered checkpoint from exact persisted model | "
                f"RMSLE={score:.6f}, best_iteration={best_iteration}"
            )
            oof_parts.append(part)
            fold_metrics.append(_metric_from_metadata(metadata))
            best_iterations.append(best_iteration)
            checkpoint_metadata_paths.append(str(metadata_path.resolve()))
            del recovered_model, val_df, part
            gc.collect()
            continue

        train_parts = [
            _load_checked_fold(
                dataset_dir,
                index,
                registry[index],
                feature_cols,
                row_limit=row_limit,
            )[0]
            for index in range(val_idx)
        ]
        train_df = pd.concat(train_parts, ignore_index=True, copy=False)
        del train_parts
        gc.collect()
        max_train_target_end = max(item.target_end for item in train_specs)
        if max_train_target_end > val_spec.cutoff:
            raise AssertionError("CatBoost temporal leakage invariant failed")

        print(
            f"\nCATBOOST {val_spec.name}: train anchors="
            f"{[item.cutoff.isoformat() for item in train_specs]}, "
            f"train target_end(max)={max_train_target_end}, "
            f"validation cutoff={val_spec.cutoff}, target="
            f"{val_spec.target_start}..{val_spec.target_end}, "
            f"rows={len(train_df):,}/{len(val_df):,}"
        )
        model = CatBoostRegressor(**base_params)
        model.fit(
            train_df[feature_cols],
            train_df[cfg.TARGET_LOG_COL],
            eval_set=(val_df[feature_cols], val_df[cfg.TARGET_LOG_COL]),
            early_stopping_rounds=early_stopping_rounds,
            use_best_model=True,
        )
        best_iteration = int(model.get_best_iteration()) + 1
        if best_iteration <= 0:
            best_iteration = int(model.tree_count_)
        best_iterations.append(best_iteration)

        pred_log = model.predict(val_df[feature_cols])
        part = _make_oof_part(val_df, val_spec, pred_log)
        score = _validate_oof_part(part, val_df, val_spec)
        _save_model_atomic(model, model_path)
        metadata = _checkpoint_metadata(
            dataset_dir=dataset_dir,
            dataset_fingerprint=dataset_fingerprint,
            row_limit=row_limit,
            base_params=base_params,
            feature_cols=feature_cols,
            feature_dtypes=feature_dtypes,
            val_idx=val_idx,
            train_specs=train_specs,
            val_spec=val_spec,
            train_rows=len(train_df),
            validation_rows=len(val_df),
            best_iteration=best_iteration,
            score=score,
            model_path=model_path,
            checkpoint_path=csv_path,
            source="trained",
        )
        _save_fold_checkpoint(part, metadata, csv_path, metadata_path)
        metadata["checkpoint_sha256"] = _file_sha256(csv_path)
        print(
            f"{val_spec.name}: RMSLE={score:.6f}, best_iteration={best_iteration} | "
            f"checkpoint={csv_path}"
        )

        oof_parts.append(part)
        fold_metrics.append(_metric_from_metadata(metadata))
        checkpoint_metadata_paths.append(str(metadata_path.resolve()))
        del model, train_df, val_df, part
        gc.collect()

    expected_rows_by_fold = {
        spec.name: row_counts[index]
        for index, spec in enumerate(registry)
        if index > 0
    }
    combined_oof, pooled = _validate_and_save_oof(
        oof_parts, oof_path, expected_rows_by_fold
    )
    scores = [item["rmsle"] for item in fold_metrics]
    final_iterations = max(1, int(np.median(best_iterations)))
    final_path: Path | None = None
    if final_refit:
        final_parts = [
            _load_checked_fold(
                dataset_dir,
                index,
                spec,
                feature_cols,
                row_limit=row_limit,
            )[0]
            for index, spec in enumerate(registry)
        ]
        full_train = pd.concat(final_parts, ignore_index=True, copy=False)
        del final_parts
        gc.collect()
        final_params = dict(base_params)
        final_params["iterations"] = final_iterations
        final_params["verbose"] = 200
        final_model = CatBoostRegressor(**final_params)
        final_model.fit(
            full_train[feature_cols],
            full_train[cfg.TARGET_LOG_COL],
        )
        final_path = model_dir / "catboost_final.cbm"
        final_model.save_model(str(final_path))

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "catboost_log1p_gmv",
        "dataset_dir": str(dataset_dir.resolve()),
        "dataset_fingerprint": _dataset_fingerprint(dataset_dir, row_limit),
        "artifact_mode": artifacts.artifact_mode,
        "run_name": artifacts.run_name,
        "parameters": base_params,
        "feature_columns": feature_cols,
        "feature_dtypes": feature_dtypes,
        "folds": fold_metrics,
        "best_iterations": best_iterations,
        "final_iterations": final_iterations,
        "pooled_oof_rmsle": pooled,
        "mean_fold_rmsle": float(np.mean(scores)),
        "std_fold_rmsle": float(np.std(scores)),
        "latest_fold_rmsle": scores[-1],
        "final_model": str(final_path.resolve()) if final_path else None,
        "final_model_sha256": _file_sha256(final_path) if final_path else None,
        "oof_path": str(oof_path.resolve()),
        "oof_rows": int(len(combined_oof)),
        "fold_checkpoint_metadata": checkpoint_metadata_paths,
        "final_refit_completed": final_path is not None,
        "final_refit_note": (
            "completed on all labeled snapshots"
            if final_path is not None
            else "deferred to avoid the known all-fold RAM peak; OOF evidence is complete"
        ),
        "row_limit": row_limit,
    }
    (model_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(fold_metrics).to_csv(
        artifacts.reports_dir / "catboost_fold_metrics.csv", index=False
    )
    print(
        f"CATBOOST pooled={pooled:.6f}, mean={np.mean(scores):.6f}, "
        f"std={np.std(scores):.6f}, latest={scores[-1]:.6f}"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=cfg.DATA_DIR)
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="CPU")
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--no-final-refit", action="store_true")
    parser.add_argument("--row-limit", type=int, default=None)
    parser.add_argument(
        "--recover-existing-models",
        action="store_true",
        help=(
            "Bootstrap missing per-fold checkpoints from persisted models only "
            "after strict parameter and feature-schema validation."
        ),
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help=(
            "Isolated artifact namespace. Custom datasets and row-limited runs "
            "are auto-isolated when this is omitted."
        ),
    )
    args = parser.parse_args()
    train_catboost_cv(
        dataset_dir=args.dataset_dir,
        task_type=args.task_type,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        early_stopping_rounds=args.early_stopping_rounds,
        final_refit=not args.no_final_refit,
        row_limit=args.row_limit,
        run_name=args.run_name,
        recover_existing_models=args.recover_existing_models,
    )


if __name__ == "__main__":
    main()
