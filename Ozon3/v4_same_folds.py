"""Reproduce the frozen v4-stable recipe on the canonical temporal OOF rows.

This script deliberately does not rebuild snapshots or modify the original v4
artifacts.  It reuses the original raw component OOF predictions where they
already exist (current folds 2--5), trains only the missing historical
calibration fold and current fold 1, and writes an isolated, hash-bound run.

The hurdle probability calibrator is fitted in expanding time: a validation
fold may use only raw OOF predictions from earlier cutoffs.  This is stricter
than the old leave-one-fold-out analysis, which could use future OOF folds.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.isotonic import IsotonicRegression

import config as cfg
import experiments
import solution


RUN_NAME = "v4_stable_same_folds"
RUN_DIR = cfg.OOF_DIR / "runs" / RUN_NAME
RAW_DIR = RUN_DIR / "raw"
MODEL_DIR = cfg.MODELS_DIR / "runs" / RUN_NAME / "v4"
REPORT_PATH = cfg.PROJECT_DIR / "reports" / "v4_stable_nn_same_folds.json"
OOF_PATH = RUN_DIR / "oof_v4_stable.csv"
MANIFEST_PATH = RUN_DIR / "manifest.json"
NN_OOF_PATH = cfg.OOF_DIR / "oof_nn.csv"
SOURCE_OOF_DIR = cfg.DATA_DIR / "oof_v4"

CURRENT_FOLDS: tuple[tuple[str, dt.date], ...] = (
    ("fold_1", dt.date(2025, 9, 16)),
    ("fold_2", dt.date(2025, 10, 16)),
    ("fold_3", dt.date(2025, 11, 15)),
    ("fold_4", dt.date(2025, 12, 15)),
    ("fold_5", dt.date(2026, 1, 14)),
)
AUXILIARY_CALIBRATION_CUTOFF = dt.date(2025, 8, 17)

RAW_COLUMNS = (
    cfg.ID_COL,
    "cutoff_date",
    "target",
    "p_buy",
    "conditional_log",
    "deep95_log",
    "depth8_log",
)
KEY_COLUMNS = [cfg.ID_COL, "fold", "cutoff_date"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _rmsle(target: np.ndarray, prediction: np.ndarray) -> float:
    y = np.log1p(np.clip(np.asarray(target, dtype=np.float64), 0, None))
    z = np.log1p(np.clip(np.asarray(prediction, dtype=np.float64), 0, None))
    return float(np.sqrt(np.mean((y - z) ** 2)))


def _feature_contract() -> tuple[list[str], str]:
    feature_cols = experiments._feature_cols()
    schema = pq.ParquetFile(
        solution.SNAPSHOT_DIR / "snapshot_2026-01-14.parquet"
    ).schema_arrow
    fields = [
        {"name": name, "dtype": str(schema.field(name).type)} for name in feature_cols
    ]
    if len(feature_cols) != 400:
        raise AssertionError(f"Frozen v4 feature count changed: {len(feature_cols)}")
    return feature_cols, _json_sha256(fields)


def _snapshot_path(cutoff: dt.date) -> Path:
    return solution.SNAPSHOT_DIR / f"snapshot_{cutoff.isoformat()}.parquet"


def _canonical_fold(fold: str) -> pd.DataFrame:
    frame = pd.read_parquet(cfg.DATA_DIR / f"{fold}.parquet", columns=[cfg.ID_COL, "target"])
    frame[cfg.ID_COL] = frame[cfg.ID_COL].astype("int64")
    if frame[cfg.ID_COL].duplicated().any():
        raise AssertionError(f"{fold}: duplicate canonical user_id")
    return frame


def _save_model(model: lgb.Booster, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        model.model_to_string(num_iteration=model.current_iteration()), encoding="utf-8"
    )
    temporary.replace(path)
    return _sha256(path)


def _train_component(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_cols: list[str],
    train_label: np.ndarray,
    valid_label: np.ndarray,
    weights: np.ndarray,
    params: dict[str, Any],
    rounds: int,
    train_mask: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
) -> tuple[lgb.Booster, int]:
    if train_mask is None:
        train_mask = np.ones(len(train), dtype=bool)
    if valid_mask is None:
        valid_mask = np.ones(len(valid), dtype=bool)
    dtrain = lgb.Dataset(
        train.loc[train_mask, feature_cols],
        label=np.asarray(train_label)[train_mask],
        weight=np.asarray(weights)[train_mask],
        free_raw_data=True,
    )
    dvalid = lgb.Dataset(
        valid.loc[valid_mask, feature_cols],
        label=np.asarray(valid_label)[valid_mask],
        reference=dtrain,
        free_raw_data=True,
    )
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=rounds,
        valid_sets=[dvalid],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(100)],
    )
    return model, int(model.best_iteration)


def _raw_checkpoint_paths(cutoff: dt.date) -> tuple[Path, Path]:
    stem = f"raw_{cutoff.isoformat()}"
    return RAW_DIR / f"{stem}.parquet", RAW_DIR / f"{stem}.meta.json"


def _raw_contract(
    cutoff: dt.date,
    train_anchors: list[dt.date],
    feature_schema_sha256: str,
    include_direct: bool,
) -> dict[str, Any]:
    source_paths = [_snapshot_path(value) for value in [*train_anchors, cutoff]]
    return {
        "run": RUN_NAME,
        "cutoff": cutoff.isoformat(),
        "train_anchors": [value.isoformat() for value in train_anchors],
        "target_end_invariant": all(
            anchor + dt.timedelta(days=30) <= cutoff for anchor in train_anchors
        ),
        "feature_count": 400,
        "feature_schema_sha256": feature_schema_sha256,
        "include_direct": include_direct,
        "snapshot_sha256": {str(path): _sha256(path) for path in source_paths},
        "deep95_params": (
            {**experiments.BASE_PARAMS, **experiments.TUNING_CONFIGS["deep95"],
             "objective": "regression", "metric": "rmse", "seed": 810}
            if include_direct else None
        ),
        "depth8_params": (
            {**experiments.BASE_PARAMS, **experiments.TUNING_CONFIGS["depth8"],
             "objective": "regression", "metric": "rmse", "seed": 810}
            if include_direct else None
        ),
        "hurdle_classifier_params": {
            **experiments.BASE_PARAMS,
            "objective": "binary",
            "metric": "binary_logloss",
            "seed": 303,
        },
        "hurdle_positive_params": {
            **experiments.BASE_PARAMS,
            "objective": "regression",
            "metric": "rmse",
            "seed": 304,
        },
        "direct_weighting": "uniform",
        "hurdle_weighting": "exp82",
        "early_stopping_rounds": 80,
    }


def _validate_raw_checkpoint(
    data_path: Path,
    meta_path: Path,
    expected_contract_sha256: str,
    include_direct: bool,
) -> pd.DataFrame | None:
    if not data_path.exists() or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("contract_sha256") != expected_contract_sha256:
            return None
        if meta.get("oof_sha256") != _sha256(data_path):
            return None
        frame = pd.read_parquet(data_path)
        required = {cfg.ID_COL, "cutoff_date", "target", "p_buy", "conditional_log"}
        if include_direct:
            required.update({"deep95_log", "depth8_log"})
        if not required.issubset(frame.columns):
            return None
        numeric = frame[list(required - {"cutoff_date", cfg.ID_COL})].to_numpy(dtype=np.float64)
        if not np.isfinite(numeric).all() or frame[cfg.ID_COL].duplicated().any():
            return None
        return frame
    except Exception:
        return None


def _train_missing_raw_fold(
    cutoff: dt.date,
    feature_cols: list[str],
    feature_schema_sha256: str,
    include_direct: bool,
) -> pd.DataFrame:
    train_anchors = experiments._train_anchors(cutoff)
    if not train_anchors:
        raise AssertionError(f"No historical training anchors for {cutoff}")
    contract = _raw_contract(
        cutoff, train_anchors, feature_schema_sha256, include_direct
    )
    if not contract["target_end_invariant"]:
        raise AssertionError(f"{cutoff}: training target overlaps validation features")
    contract_sha256 = _json_sha256(contract)
    data_path, meta_path = _raw_checkpoint_paths(cutoff)
    reusable = _validate_raw_checkpoint(
        data_path, meta_path, contract_sha256, include_direct
    )
    if reusable is not None:
        print(f"[v4] reuse validated raw checkpoint {data_path.name}")
        return reusable

    needed_cols = feature_cols + [cfg.ID_COL, "target"]
    print(f"[v4] train raw cutoff={cutoff}, train={train_anchors}")
    parts = [pd.read_parquet(_snapshot_path(anchor), columns=needed_cols) for anchor in train_anchors]
    valid = pd.read_parquet(_snapshot_path(cutoff), columns=needed_cols)
    train = pd.concat(parts, ignore_index=True)
    y_train = np.clip(train["target"].to_numpy(dtype=np.float64), 0, None)
    y_valid = np.clip(valid["target"].to_numpy(dtype=np.float64), 0, None)
    z_train = np.log1p(y_train)
    z_valid = np.log1p(y_valid)
    result = valid[[cfg.ID_COL, "target"]].copy()
    result["cutoff_date"] = cutoff.isoformat()
    result["deep95_log"] = np.nan
    result["depth8_log"] = np.nan
    best_iterations: dict[str, int] = {}
    model_hashes: dict[str, str] = {}

    if include_direct:
        uniform = np.ones(len(train), dtype=np.float32)
        for name in ("deep95", "depth8"):
            params = dict(experiments.BASE_PARAMS)
            params.update(experiments.TUNING_CONFIGS[name])
            params.update({"objective": "regression", "metric": "rmse", "seed": 810})
            model, best = _train_component(
                train, valid, feature_cols, z_train, z_valid, uniform, params, 1800
            )
            result[f"{name}_log"] = np.clip(
                model.predict(valid[feature_cols]), 0, None
            )
            best_iterations[name] = best
            model_hashes[name] = _save_model(
                model, MODEL_DIR / f"{name}_{cutoff.isoformat()}.txt"
            )
            del model
            gc.collect()
        del uniform

    hurdle_weights = experiments._weights_for_anchors(
        parts, train_anchors, cutoff, "exp82"
    )
    binary_train = (y_train > 0).astype(np.float32)
    binary_valid = (y_valid > 0).astype(np.float32)
    classifier_params = dict(experiments.BASE_PARAMS)
    classifier_params.update(
        {"objective": "binary", "metric": "binary_logloss", "seed": 303}
    )
    classifier, best_classifier = _train_component(
        train,
        valid,
        feature_cols,
        binary_train,
        binary_valid,
        hurdle_weights,
        classifier_params,
        1400,
    )
    result["p_buy"] = np.clip(classifier.predict(valid[feature_cols]), 0, 1)
    best_iterations["hurdle_classifier"] = best_classifier
    model_hashes["hurdle_classifier"] = _save_model(
        classifier, MODEL_DIR / f"hurdle_classifier_{cutoff.isoformat()}.txt"
    )
    del classifier
    gc.collect()

    positive_train = y_train > 0
    positive_valid = y_valid > 0
    positive_params = dict(experiments.BASE_PARAMS)
    positive_params.update(
        {"objective": "regression", "metric": "rmse", "seed": 304}
    )
    positive, best_positive = _train_component(
        train,
        valid,
        feature_cols,
        z_train,
        z_valid,
        hurdle_weights,
        positive_params,
        1400,
        positive_train,
        positive_valid,
    )
    result["conditional_log"] = np.clip(
        positive.predict(valid[feature_cols]), 0, None
    )
    best_iterations["hurdle_positive"] = best_positive
    model_hashes["hurdle_positive"] = _save_model(
        positive, MODEL_DIR / f"hurdle_positive_{cutoff.isoformat()}.txt"
    )
    del positive, train, valid, parts, hurdle_weights, y_train, y_valid, z_train, z_valid
    gc.collect()

    _atomic_parquet(data_path, result)
    meta = {
        "contract": contract,
        "contract_sha256": contract_sha256,
        "best_iterations": best_iterations,
        "model_sha256": model_hashes,
        "rows": int(len(result)),
        "oof_sha256": _sha256(data_path),
    }
    _atomic_json(meta_path, meta)
    print(f"[v4] checkpointed {data_path.name}, rows={len(result)}")
    return result


def _normalize_existing_raw(
    fold: str,
    cutoff: dt.date,
    feature_schema_sha256: str,
) -> pd.DataFrame:
    source_path = SOURCE_OOF_DIR / f"oof_{cutoff.isoformat()}.parquet"
    source_meta_path = SOURCE_OOF_DIR / f"metadata_{cutoff.isoformat()}.json"
    if not source_path.exists() or not source_meta_path.exists():
        raise FileNotFoundError(f"Missing original v4 OOF source for {cutoff}")
    source_meta = json.loads(source_meta_path.read_text(encoding="utf-8"))
    if source_meta.get("n_features") != 400:
        raise AssertionError(f"{cutoff}: original v4 feature count is not 400")
    required = [
        cfg.ID_COL,
        "target",
        "p_buy",
        "pred_hurdle_log",
        "pred_hurdle_amount",
        "pred_log_tune_deep95",
        "pred_log_tune_depth8",
    ]
    source = pd.read_parquet(source_path, columns=required)
    p_buy = np.clip(source["p_buy"].to_numpy(dtype=np.float64), 0, 1)
    hurdle_amount = np.clip(
        source["pred_hurdle_amount"].to_numpy(dtype=np.float64), 0, None
    )
    conditional_amount = np.divide(
        hurdle_amount,
        p_buy,
        out=np.zeros_like(hurdle_amount),
        where=p_buy > 0,
    )
    conditional_log = np.log1p(conditional_amount)
    reconstructed_hurdle = np.expm1(p_buy * conditional_log)
    mismatch = float(
        np.max(
            np.abs(
                reconstructed_hurdle
                - source["pred_hurdle_log"].to_numpy(dtype=np.float64)
            )
        )
    )
    if mismatch > 1e-8:
        raise AssertionError(
            f"{cutoff}: cannot reconstruct conditional hurdle log; max diff={mismatch}"
        )
    normalized = pd.DataFrame(
        {
            cfg.ID_COL: source[cfg.ID_COL].astype("int64"),
            "cutoff_date": cutoff.isoformat(),
            "target": source["target"].to_numpy(dtype=np.float64),
            "p_buy": p_buy,
            "conditional_log": conditional_log,
            "deep95_log": np.log1p(
                np.clip(source["pred_log_tune_deep95"].to_numpy(dtype=np.float64), 0, None)
            ),
            "depth8_log": np.log1p(
                np.clip(source["pred_log_tune_depth8"].to_numpy(dtype=np.float64), 0, None)
            ),
        }
    )
    data_path, meta_path = _raw_checkpoint_paths(cutoff)
    source_binding = {
        "run": RUN_NAME,
        "fold": fold,
        "cutoff": cutoff.isoformat(),
        "feature_count": 400,
        "feature_schema_sha256": feature_schema_sha256,
        "source_oof": str(source_path),
        "source_oof_sha256": _sha256(source_path),
        "source_metadata": str(source_meta_path),
        "source_metadata_sha256": _sha256(source_meta_path),
        "source_best_iterations": {
            key: source_meta["best_iterations"][key]
            for key in (
                "log_tune_deep95",
                "log_tune_depth8",
                "hurdle_classifier",
                "hurdle_positive",
            )
        },
        "hurdle_reconstruction_max_abs": mismatch,
    }
    contract_sha256 = _json_sha256(source_binding)
    reusable = _validate_raw_checkpoint(
        data_path, meta_path, contract_sha256, include_direct=True
    )
    if reusable is not None:
        print(f"[v4] reuse normalized source checkpoint {data_path.name}")
        return reusable
    _atomic_parquet(data_path, normalized)
    _atomic_json(
        meta_path,
        {
            "contract": source_binding,
            "contract_sha256": contract_sha256,
            "rows": int(len(normalized)),
            "oof_sha256": _sha256(data_path),
        },
    )
    print(f"[v4] normalized original source {source_path.name}")
    return normalized


def _attach_canonical_target(
    raw: pd.DataFrame, fold: str, cutoff: dt.date
) -> tuple[pd.DataFrame, dict[str, Any]]:
    canonical = _canonical_fold(fold)
    merged = canonical.merge(
        raw,
        on=cfg.ID_COL,
        how="outer",
        suffixes=("_canonical", "_v4"),
        indicator=True,
        validate="one_to_one",
    )
    coverage = merged["_merge"].value_counts().to_dict()
    if coverage.get("both", 0) != len(canonical) or len(merged) != len(canonical):
        raise AssertionError(f"{fold}: v4/current coverage mismatch: {coverage}")
    target_diff = np.abs(
        merged["target_canonical"].to_numpy(dtype=np.float64)
        - merged["target_v4"].to_numpy(dtype=np.float64)
    )
    zero_equal = np.array_equal(
        merged["target_canonical"].to_numpy() > 0,
        merged["target_v4"].to_numpy() > 0,
    )
    if not zero_equal:
        raise AssertionError(f"{fold}: target zero indicator differs")
    result = merged[
        [cfg.ID_COL, "target_canonical", "p_buy", "conditional_log", "deep95_log", "depth8_log"]
    ].rename(columns={"target_canonical": "target"})
    # Preserve the exact float32 value as a full-precision decimal in CSV so
    # re-reading the artifact matches the canonical NN OOF target exactly.
    result["target"] = result["target"].astype(np.float64)
    result["fold"] = fold
    result["cutoff_date"] = cutoff.isoformat()
    diagnostics = {
        "coverage": {str(key): int(value) for key, value in coverage.items()},
        "legacy_target_max_abs_delta": float(target_diff.max()),
        "legacy_target_rows_different": int((target_diff != 0).sum()),
        "target_zero_indicator_exact": zero_equal,
    }
    return result, diagnostics


def _fit_expanding_stable(
    auxiliary: pd.DataFrame, folds: list[pd.DataFrame]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    history = auxiliary[["p_buy", "target", "cutoff_date"]].copy()
    outputs: list[pd.DataFrame] = []
    calibration_steps: list[dict[str, Any]] = []
    for frame in folds:
        fold = str(frame["fold"].iloc[0])
        cutoff = str(frame["cutoff_date"].iloc[0])
        isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        isotonic.fit(
            np.clip(history["p_buy"].to_numpy(dtype=np.float64), 0, 1),
            (history["target"].to_numpy(dtype=np.float64) > 0).astype(np.float64),
        )
        calibrated_p = np.clip(
            isotonic.predict(np.clip(frame["p_buy"].to_numpy(dtype=np.float64), 0, 1)),
            0,
            1,
        )
        hurdle_log = calibrated_p * np.clip(
            frame["conditional_log"].to_numpy(dtype=np.float64), 0, None
        )
        pred_log = np.mean(
            np.column_stack(
                [
                    np.clip(frame["deep95_log"].to_numpy(dtype=np.float64), 0, None),
                    np.clip(frame["depth8_log"].to_numpy(dtype=np.float64), 0, None),
                    np.clip(hurdle_log, 0, None),
                ]
            ),
            axis=1,
        )
        out = frame[[cfg.ID_COL, "fold", "cutoff_date", "target"]].copy()
        out["pred_log"] = np.clip(pred_log, 0, None)
        out["pred"] = np.expm1(out["pred_log"].to_numpy(dtype=np.float64))
        outputs.append(out[[cfg.ID_COL, "fold", "cutoff_date", "target", "pred", "pred_log"]])
        calibration_steps.append(
            {
                "fold": fold,
                "cutoff": cutoff,
                "fit_rows": int(len(history)),
                "fit_cutoffs": sorted(history.get("cutoff_date", pd.Series(dtype=str)).astype(str).unique().tolist()),
                "isotonic_thresholds": int(len(isotonic.X_thresholds_)),
            }
        )
        append = frame[["p_buy", "target", "cutoff_date"]].copy()
        history = pd.concat([history, append], ignore_index=True)
    return pd.concat(outputs, ignore_index=True), calibration_steps


def _validate_oof(oof: pd.DataFrame, nn: pd.DataFrame) -> dict[str, Any]:
    if list(oof.columns) != [cfg.ID_COL, "fold", "cutoff_date", "target", "pred", "pred_log"]:
        raise AssertionError(f"Unexpected OOF columns: {list(oof.columns)}")
    if oof.duplicated(KEY_COLUMNS).any():
        raise AssertionError("Duplicate v4 OOF key")
    numeric = oof[["target", "pred", "pred_log"]].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all() or (numeric < 0).any():
        raise AssertionError("Invalid v4 OOF numeric values")
    pred_contract_error = float(
        np.max(np.abs(oof["pred_log"].to_numpy() - np.log1p(oof["pred"].to_numpy())))
    )
    if pred_contract_error > 1e-12:
        raise AssertionError(f"pred_log contract error={pred_contract_error}")
    aligned = oof.merge(
        nn,
        on=KEY_COLUMNS,
        how="outer",
        suffixes=("_v4", "_nn"),
        indicator=True,
        validate="one_to_one",
    )
    coverage = aligned["_merge"].value_counts().to_dict()
    if coverage.get("both", 0) != len(oof) or len(oof) != len(nn):
        raise AssertionError(f"NN/v4 key coverage mismatch: {coverage}")
    target_delta = np.abs(
        aligned["target_v4"].to_numpy(dtype=np.float64)
        - aligned["target_nn"].to_numpy(dtype=np.float64)
    )
    if target_delta.max(initial=0.0) > 1e-12:
        raise AssertionError(f"NN/v4 targets differ: {target_delta.max()}")
    return {
        "rows": int(len(oof)),
        "duplicate_keys": 0,
        "pred_log_max_abs_error": pred_contract_error,
        "aligned_rows": int(coverage.get("both", 0)),
        "rows_lost_v4": int(coverage.get("right_only", 0)),
        "rows_lost_nn": int(coverage.get("left_only", 0)),
        "target_max_abs_delta": float(target_delta.max(initial=0.0)),
    }


def _optimal_v4_weight(v4_log: np.ndarray, nn_log: np.ndarray, y_log: np.ndarray) -> float:
    delta = v4_log - nn_log
    denominator = float(np.dot(delta, delta))
    if denominator <= 0:
        return 0.5
    return float(np.clip(np.dot(delta, y_log - nn_log) / denominator, 0, 1))


def _analysis(oof: pd.DataFrame, nn: pd.DataFrame) -> dict[str, Any]:
    aligned = oof.merge(
        nn,
        on=KEY_COLUMNS,
        how="inner",
        suffixes=("_v4", "_nn"),
        validate="one_to_one",
    ).sort_values(["fold", cfg.ID_COL], kind="stable")
    y_log = np.log1p(aligned["target_v4"].to_numpy(dtype=np.float64))
    v4_log = aligned["pred_log_v4"].to_numpy(dtype=np.float64)
    nn_log = aligned["pred_log_nn"].to_numpy(dtype=np.float64)
    fold_values = aligned["fold"].astype(str).to_numpy()

    standalone: dict[str, Any] = {}
    for name, values in (("v4", v4_log), ("nn", nn_log)):
        standalone[name] = {
            "pooled_rmsle": float(np.sqrt(np.mean((y_log - values) ** 2))),
            "fold_rmsle": {
                fold: float(np.sqrt(np.mean((y_log[fold_values == fold] - values[fold_values == fold]) ** 2)))
                for fold, _ in CURRENT_FOLDS
            },
        }

    correlations = {
        "pooled_prediction_log": float(np.corrcoef(v4_log, nn_log)[0, 1]),
        "pooled_residual_log": float(
            np.corrcoef(v4_log - y_log, nn_log - y_log)[0, 1]
        ),
        "per_fold": {},
    }
    for fold, _ in CURRENT_FOLDS:
        mask = fold_values == fold
        correlations["per_fold"][fold] = {
            "prediction_log": float(np.corrcoef(v4_log[mask], nn_log[mask])[0, 1]),
            "residual_log": float(
                np.corrcoef(v4_log[mask] - y_log[mask], nn_log[mask] - y_log[mask])[0, 1]
            ),
        }

    fixed: dict[str, Any] = {}
    for v4_weight in (0.95, 0.9, 0.8, 0.7, 0.6, 0.5):
        blended = v4_weight * v4_log + (1 - v4_weight) * nn_log
        key = f"v4_{int(round(100 * v4_weight))}_nn_{int(round(100 * (1 - v4_weight)))}"
        fixed[key] = {
            "v4_weight": v4_weight,
            "nn_weight": 1 - v4_weight,
            "pooled_rmsle": float(np.sqrt(np.mean((y_log - blended) ** 2))),
            "fold_rmsle": {
                fold: float(
                    np.sqrt(
                        np.mean(
                            (y_log[fold_values == fold] - blended[fold_values == fold]) ** 2
                        )
                    )
                )
                for fold, _ in CURRENT_FOLDS
            },
        }

    meta_pred = np.full(len(aligned), np.nan, dtype=np.float64)
    steps: list[dict[str, Any]] = []
    for index in range(1, len(CURRENT_FOLDS)):
        train_folds = [name for name, _ in CURRENT_FOLDS[:index]]
        valid_fold = CURRENT_FOLDS[index][0]
        fit = np.isin(fold_values, train_folds)
        held = fold_values == valid_fold
        weight = _optimal_v4_weight(v4_log[fit], nn_log[fit], y_log[fit])
        meta_pred[held] = weight * v4_log[held] + (1 - weight) * nn_log[held]
        steps.append(
            {
                "validation_fold": valid_fold,
                "fit_folds": train_folds,
                "v4_weight": weight,
                "nn_weight": 1 - weight,
                "rows": int(held.sum()),
                "rmsle": float(np.sqrt(np.mean((y_log[held] - meta_pred[held]) ** 2))),
            }
        )
    evaluable = np.isfinite(meta_pred)
    meta_score = float(np.sqrt(np.mean((y_log[evaluable] - meta_pred[evaluable]) ** 2)))
    v4_same = float(np.sqrt(np.mean((y_log[evaluable] - v4_log[evaluable]) ** 2)))
    nn_same = float(np.sqrt(np.mean((y_log[evaluable] - nn_log[evaluable]) ** 2)))
    best_same = min(v4_same, nn_same)

    all_weight = _optimal_v4_weight(v4_log, nn_log, y_log)
    all_blend = all_weight * v4_log + (1 - all_weight) * nn_log
    return {
        "aligned_rows": int(len(aligned)),
        "standalone": standalone,
        "correlations": correlations,
        "fixed_log_blends": fixed,
        "expanding_temporal_meta_cv": {
            "steps": steps,
            "evaluable_rows": int(evaluable.sum()),
            "pooled_rmsle": meta_score,
            "v4_standalone_same_rows": v4_same,
            "nn_standalone_same_rows": nn_same,
            "best_standalone_same_rows": best_same,
            "delta_vs_best_standalone": meta_score - best_same,
            "gate_passed": bool(meta_score < best_same),
        },
        "all_oof_simplex_in_sample_diagnostic": {
            "v4_weight": all_weight,
            "nn_weight": 1 - all_weight,
            "rmsle": float(np.sqrt(np.mean((y_log - all_blend) ** 2))),
        },
    }


def run() -> dict[str, Any]:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    feature_cols, feature_schema_sha256 = _feature_contract()

    auxiliary = _train_missing_raw_fold(
        AUXILIARY_CALIBRATION_CUTOFF,
        feature_cols,
        feature_schema_sha256,
        include_direct=False,
    )
    auxiliary = auxiliary[[cfg.ID_COL, "target", "p_buy", "cutoff_date"]].copy()

    raw_folds: list[pd.DataFrame] = []
    legacy_target_diagnostics: dict[str, Any] = {}
    for fold, cutoff in CURRENT_FOLDS:
        if fold == "fold_1":
            raw = _train_missing_raw_fold(
                cutoff, feature_cols, feature_schema_sha256, include_direct=True
            )
        else:
            raw = _normalize_existing_raw(fold, cutoff, feature_schema_sha256)
        canonical, diagnostics = _attach_canonical_target(raw, fold, cutoff)
        raw_folds.append(canonical)
        legacy_target_diagnostics[fold] = diagnostics

    oof, calibration_steps = _fit_expanding_stable(auxiliary, raw_folds)
    nn = pd.read_csv(NN_OOF_PATH)
    nn["cutoff_date"] = nn["cutoff_date"].astype(str)
    validation = _validate_oof(oof, nn)
    _atomic_csv(OOF_PATH, oof)
    analysis = _analysis(oof, nn)

    source_hashes = {
        str(path): _sha256(path)
        for path in sorted(SOURCE_OOF_DIR.glob("oof_*.parquet"))
        if path.name != "oof_all_analyzed.parquet"
    }
    report = {
        "run": RUN_NAME,
        "recipe": {
            "features": 400,
            "feature_schema_sha256": feature_schema_sha256,
            "direct_components": ["deep95", "depth8"],
            "direct_target": "log1p(target)",
            "direct_weighting": "uniform",
            "hurdle": "isotonic(P(target>0)) * conditional_log1p_positive_GMV",
            "hurdle_weighting": "exp82",
            "blend": "equal arithmetic mean of three component pred_log values",
            "final_prediction": "expm1(max(pred_log, 0))",
            "frozen_source": "finalize_v4.py and experiments.py",
        },
        "temporal_protocol": {
            "validation_folds": [fold for fold, _ in CURRENT_FOLDS],
            "validation_cutoffs": [cutoff.isoformat() for _, cutoff in CURRENT_FOLDS],
            "component_training": "all v4 snapshots with target_end <= validation cutoff",
            "hurdle_calibration": "expanding earlier raw OOF only",
            "auxiliary_calibration_cutoff": AUXILIARY_CALIBRATION_CUTOFF.isoformat(),
            "calibration_steps": calibration_steps,
        },
        "validation": validation,
        "legacy_target_serialization": legacy_target_diagnostics,
        "analysis": analysis,
        "selection_evidence_warning": (
            "The frozen v4 recipe was historically selected using cutoffs "
            "2025-10-16 through 2026-01-14, which are the current folds 2-5. "
            "Predictions are row-aligned and temporally trained, but scores on "
            "those folds are not independent of historical recipe selection."
        ),
        "sources": {
            "nn_oof": str(NN_OOF_PATH),
            "nn_oof_sha256": _sha256(NN_OOF_PATH),
            "original_v4_oof_sha256": source_hashes,
            "original_v4_submission": str(
                cfg.SUB_DIR / "submission_v4_stable_logblend.csv"
            ),
            "original_v4_submission_sha256": _sha256(
                cfg.SUB_DIR / "submission_v4_stable_logblend.csv"
            ),
        },
        "artifacts": {
            "oof": str(OOF_PATH),
            "oof_sha256": _sha256(OOF_PATH),
            "raw_dir": str(RAW_DIR),
            "model_dir": str(MODEL_DIR),
        },
    }
    _atomic_json(REPORT_PATH, report)
    manifest = {
        "run": RUN_NAME,
        "complete": True,
        "report": str(REPORT_PATH),
        "report_sha256": _sha256(REPORT_PATH),
        "oof": str(OOF_PATH),
        "oof_sha256": _sha256(OOF_PATH),
        "feature_schema_sha256": feature_schema_sha256,
        "rows": int(len(oof)),
        "folds": [fold for fold, _ in CURRENT_FOLDS],
    }
    _atomic_json(MANIFEST_PATH, manifest)
    print(json.dumps(analysis, indent=2))
    print(f"[v4] OOF: {OOF_PATH}")
    print(f"[v4] report: {REPORT_PATH}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
