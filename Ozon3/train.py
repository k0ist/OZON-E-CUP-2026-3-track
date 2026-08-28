"""Leakage-safe LightGBM walk-forward CV and final refit.

The project OOF contract is:
    user_id, fold, cutoff_date, target, pred, pred_log

`fold_0` is the first labelled training snapshot. Validation starts at
`fold_1`, so every validation model has strictly historical labelled data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd

import config as cfg
from features import get_feature_groups
from time_split import build_cv_folds, validate_fold_contract, walk_forward_splits


META_COLUMNS = {
    cfg.ID_COL,
    cfg.TARGET_COL,
    cfg.TARGET_LOG_COL,
    "fold",
    "cutoff_date",
    "target_start_date",
    "target_end_date",
    "first_order_dt",
    "last_order_dt",
}


BASELINE_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.02,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.80,
    "bagging_fraction": 0.80,
    "bagging_freq": 1,
    "lambda_l1": 0.05,
    "lambda_l2": 1.0,
    "max_bin": 255,
    "verbosity": -1,
    "num_threads": -1,
    "seed": cfg.RANDOM_STATE,
    "feature_fraction_seed": cfg.RANDOM_STATE,
    "bagging_seed": cfg.RANDOM_STATE,
    "data_random_seed": cfg.RANDOM_STATE,
    "deterministic": True,
    "force_row_wise": True,
}


@dataclass(frozen=True)
class ArtifactLayout:
    """Resolved output namespace for one reproducible experiment run."""

    run_name: str | None
    artifact_mode: str
    models_root: Path
    oof_dir: Path
    reports_dir: Path
    tuning_dir: Path

    @property
    def is_canonical(self) -> bool:
        return self.artifact_mode == "canonical"


def _normalize_run_name(run_name: str) -> str:
    value = str(run_name).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", value):
        raise ValueError(
            "run_name must be 1..80 characters and contain only letters, "
            "digits, '.', '_' or '-' (starting with a letter or digit)."
        )
    return value


def _dataset_fingerprint(dataset_dir: Path, row_limit: int | None) -> str:
    digest = hashlib.sha256()
    digest.update(str(dataset_dir.resolve()).encode("utf-8"))
    digest.update(f"|row_limit={row_limit}".encode("ascii"))
    candidates = [*sorted(dataset_dir.glob("fold_*.parquet"))]
    test_path = dataset_dir / "test_features.parquet"
    if test_path.exists():
        candidates.append(test_path)
    for path in candidates:
        stat = path.stat()
        digest.update(
            f"|{path.name}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")
        )
    return digest.hexdigest()[:10]


def _file_sha256(path: str | Path) -> str:
    """Hash a persisted model/report without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_dataset_status(
    dataset_dir: Path,
    row_limit: int | None,
) -> tuple[bool, str]:
    """Prove that a run is the full canonical dataset before canonical writes."""

    if row_limit is not None:
        return False, f"row_limit={row_limit}"
    if dataset_dir.resolve() != cfg.DATA_DIR.resolve():
        return False, f"custom dataset_dir={dataset_dir.resolve()}"
    if Path(cfg.DATA_PATH).resolve() == Path(cfg.SYNTHETIC_DATA_PATH).resolve():
        return False, "ECUP_USE_SYNTHETIC/configured synthetic input"

    test_path = dataset_dir / "test_features.parquet"
    if not test_path.is_file():
        return False, f"missing {test_path.name} full-dataset canary"
    try:
        test_meta = pd.read_parquet(
            test_path,
            columns=[cfg.ID_COL, "fold", "cutoff_date"],
        )
    except Exception as exc:
        return False, f"invalid {test_path.name}: {type(exc).__name__}: {exc}"

    if len(test_meta) != cfg.EXPECTED_SUBMISSION_ROWS:
        return False, (
            f"{test_path.name} rows={len(test_meta)}, "
            f"expected={cfg.EXPECTED_SUBMISSION_ROWS}"
        )
    if test_meta[cfg.ID_COL].isna().any() or test_meta[cfg.ID_COL].duplicated().any():
        return False, f"{test_path.name} has null/duplicate user_id"
    if not test_meta["fold"].astype(str).eq("test").all():
        return False, f"{test_path.name} fold metadata is not 'test'"
    cutoffs = pd.to_datetime(test_meta["cutoff_date"], errors="coerce").dt.date
    if cutoffs.isna().any() or not cutoffs.eq(cfg.HIST_END).all():
        return False, f"{test_path.name} cutoff metadata is not {cfg.HIST_END}"
    return True, "full canonical dataset"


def resolve_artifact_layout(
    dataset_dir: str | Path,
    row_limit: int | None,
    run_name: str | None,
) -> ArtifactLayout:
    """Keep smoke/custom/limited runs physically separate from production.

    Omitting ``run_name`` writes canonical artifacts only when the dataset is
    proven to be the complete canonical build and no row limit is active.
    Otherwise a deterministic isolated run name is assigned automatically.
    """

    dataset_dir = Path(dataset_dir).resolve()
    if row_limit is not None and row_limit < 1:
        raise ValueError(f"row_limit must be positive, got {row_limit}")

    canonical_ready, reason = _canonical_dataset_status(dataset_dir, row_limit)
    explicit_run_name = _normalize_run_name(run_name) if run_name is not None else None
    if explicit_run_name is None and canonical_ready:
        return ArtifactLayout(
            run_name=None,
            artifact_mode="canonical",
            models_root=cfg.MODELS_DIR,
            oof_dir=cfg.OOF_DIR,
            reports_dir=cfg.REPORTS_DIR,
            tuning_dir=cfg.DATA_DIR,
        )

    effective_run_name = explicit_run_name
    if effective_run_name is None:
        dataset_slug = re.sub(r"[^A-Za-z0-9_-]+", "_", dataset_dir.name).strip("_")
        dataset_slug = (dataset_slug or "dataset")[:40]
        limit_slug = f"rows{row_limit}" if row_limit is not None else "full"
        fingerprint = _dataset_fingerprint(dataset_dir, row_limit)
        effective_run_name = _normalize_run_name(
            f"auto_{dataset_slug}_{limit_slug}_{fingerprint}"
        )
        print(
            "Artifact safety: canonical outputs are disabled "
            f"({reason}); auto-isolating run as {effective_run_name!r}."
        )
    else:
        print(f"Artifact isolation: using explicit run_name={effective_run_name!r}.")

    return ArtifactLayout(
        run_name=effective_run_name,
        artifact_mode="isolated",
        models_root=cfg.MODELS_DIR / "runs" / effective_run_name,
        oof_dir=cfg.OOF_DIR / "runs" / effective_run_name,
        reports_dir=cfg.REPORTS_DIR / "runs" / effective_run_name,
        tuning_dir=cfg.DATA_DIR / "runs" / effective_run_name,
    )


def rmsle(y_true, y_pred) -> float:
    y_true = np.clip(np.asarray(y_true, dtype=np.float64), 0.0, None)
    y_pred = np.clip(np.asarray(y_pred, dtype=np.float64), 0.0, None)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def _single_value(df: pd.DataFrame, column: str):
    values = pd.Series(df[column]).drop_duplicates()
    if len(values) != 1:
        raise AssertionError(
            f"Expected one {column!r} value, found {values.astype(str).tolist()[:5]}"
        )
    return values.iloc[0]


def _validate_loaded_fold(df: pd.DataFrame, expected_fold) -> None:
    required = {
        cfg.ID_COL,
        cfg.TARGET_COL,
        cfg.TARGET_LOG_COL,
        "fold",
        "cutoff_date",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"{expected_fold.name} is missing {missing}. Rebuild with build_dataset.py."
        )

    fold_name = str(_single_value(df, "fold"))
    cutoff = pd.Timestamp(_single_value(df, "cutoff_date")).date()
    if fold_name != expected_fold.name or cutoff != expected_fold.cutoff:
        raise AssertionError(
            f"Fold metadata mismatch: file={expected_fold.name}, "
            f"metadata=({fold_name}, {cutoff}), expected cutoff={expected_fold.cutoff}"
        )
    if df.duplicated([cfg.ID_COL, "fold", "cutoff_date"]).any():
        raise AssertionError(f"Duplicate OOF key candidates in {expected_fold.name}")
    if df[cfg.ID_COL].isna().any():
        raise AssertionError(f"Null user_id in {expected_fold.name}")

    target = df[cfg.TARGET_COL].to_numpy(dtype=np.float64, copy=False)
    target_log = df[cfg.TARGET_LOG_COL].to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(target).all() or not np.isfinite(target_log).all():
        raise AssertionError(f"Non-finite target in {expected_fold.name}")
    if (target < 0).any():
        raise AssertionError(f"Negative target in {expected_fold.name}")
    if not np.allclose(target_log, np.log1p(target), rtol=1e-5, atol=1e-5):
        raise AssertionError(f"target_log != log1p(target) in {expected_fold.name}")


def load_folds(
    dataset_dir: str | Path | None = None,
    row_limit: int | None = None,
) -> list[pd.DataFrame]:
    """Load labelled snapshots in registry order, never filename order alone."""
    dataset_dir = Path(dataset_dir) if dataset_dir else cfg.DATA_DIR
    expected_folds = build_cv_folds(
        n_folds=cfg.N_FOLDS,
        step_days=cfg.STEP_DAYS,
    )
    validate_fold_contract(expected_folds)

    folds: list[pd.DataFrame] = []
    for expected in expected_folds:
        path = dataset_dir / f"{expected.name}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run build_dataset.py before model training."
            )
        df = pd.read_parquet(path)
        if row_limit is not None and len(df) > row_limit:
            df = df.sample(row_limit, random_state=cfg.RANDOM_STATE).sort_values(
                cfg.ID_COL
            ).reset_index(drop=True)
        _validate_loaded_fold(df, expected)
        print(
            f"{path.name}: rows={len(df):,}, users={df[cfg.ID_COL].nunique():,}, "
            f"cutoff={expected.cutoff}"
        )
        folds.append(df)
    return folds


def get_features(
    folds: list[pd.DataFrame],
    drop_feature_groups: Iterable[str] | None = None,
) -> list[str]:
    if not folds:
        raise ValueError("No folds supplied")

    first = folds[0]
    feature_cols = [
        column
        for column in first.columns
        if column not in META_COLUMNS
        and pd.api.types.is_numeric_dtype(first[column])
    ]
    if not feature_cols:
        raise ValueError("No numeric feature columns found")

    expected_columns = feature_cols
    expected_dtypes = {column: str(first[column].dtype) for column in feature_cols}
    for idx, fold in enumerate(folds):
        current = [
            column
            for column in fold.columns
            if column not in META_COLUMNS
            and pd.api.types.is_numeric_dtype(fold[column])
        ]
        if current != expected_columns:
            missing = sorted(set(expected_columns) - set(current))
            extra = sorted(set(current) - set(expected_columns))
            raise AssertionError(
                f"Feature schema mismatch in fold_{idx}: missing={missing}, extra={extra}"
            )
        dtype_mismatch = {
            column: (expected_dtypes[column], str(fold[column].dtype))
            for column in feature_cols
            if str(fold[column].dtype) != expected_dtypes[column]
        }
        if dtype_mismatch:
            raise AssertionError(f"Feature dtype mismatch in fold_{idx}: {dtype_mismatch}")
        bad = [
            column
            for column in feature_cols
            if not np.isfinite(fold[column].to_numpy(copy=False)).all()
        ]
        if bad:
            raise AssertionError(f"Non-finite features in fold_{idx}: {bad[:10]}")

    drop_feature_groups = set(drop_feature_groups or [])
    if drop_feature_groups:
        groups = get_feature_groups(feature_cols)
        unknown = drop_feature_groups - set(groups)
        if unknown:
            raise ValueError(
                f"Unknown feature groups {sorted(unknown)}; available={sorted(groups)}"
            )
        dropped = {
            column
            for group in drop_feature_groups
            for column in groups[group]
        }
        feature_cols = [column for column in feature_cols if column not in dropped]
        print(
            f"Dropped feature groups={sorted(drop_feature_groups)}; "
            f"removed={len(dropped)}, retained={len(feature_cols)}"
        )
    return feature_cols


def print_walk_forward_plan(folds: list[pd.DataFrame]) -> None:
    registry = build_cv_folds(cfg.N_FOLDS, cfg.STEP_DAYS)
    validate_fold_contract(registry)
    print("\nWALK-FORWARD CONTRACT")
    print("=" * 100)
    for val_idx, train_specs, val_spec in walk_forward_splits(registry):
        train_anchors = [item.cutoff.isoformat() for item in train_specs]
        train_target_ends = [item.target_end.isoformat() for item in train_specs]
        train_rows = sum(len(folds[i]) for i in range(val_idx))
        val_rows = len(folds[val_idx])
        max_target_end = max(item.target_end for item in train_specs)
        if max_target_end > val_spec.cutoff:
            raise AssertionError(
                f"Leakage: train target ends {max_target_end} after {val_spec.cutoff}"
            )
        print(
            f"validation={val_spec.name} | train anchors={train_anchors} | "
            f"train target_end={train_target_ends} (max={max_target_end}) | "
            f"validation cutoff={val_spec.cutoff} | validation target="
            f"{val_spec.target_start}..{val_spec.target_end} | "
            f"rows train={train_rows:,}, validation={val_rows:,}"
        )


def resolve_lgbm_params(
    params_source: str,
    params_path: str | Path | None,
    device: str,
) -> tuple[dict, str]:
    params = dict(BASELINE_PARAMS)
    source_description = "baseline"
    if params_source == "best":
        path = cfg.DATA_DIR / "best_params.json"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run bounded tune.py or use --params-source baseline."
            )
        params.update(json.loads(path.read_text(encoding="utf-8")))
        source_description = str(path)
    elif params_source == "path":
        if params_path is None:
            raise ValueError("--params-path is required when --params-source path")
        path = Path(params_path)
        params.update(json.loads(path.read_text(encoding="utf-8")))
        source_description = str(path.resolve())
    elif params_source != "baseline":
        raise ValueError(f"Unknown params source: {params_source}")

    params["device_type"] = "cuda" if device in {"gpu", "cuda"} else "cpu"
    return params, source_description


def train_single_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    params: dict,
    num_boost_round: int = 5000,
    early_stopping_rounds: int = 100,
) -> lgb.Booster:
    train_data = lgb.Dataset(
        train_df[feature_cols],
        label=train_df[cfg.TARGET_LOG_COL],
        feature_name=feature_cols,
        free_raw_data=False,
    )
    val_data = lgb.Dataset(
        val_df[feature_cols],
        label=val_df[cfg.TARGET_LOG_COL],
        feature_name=feature_cols,
        reference=train_data,
        free_raw_data=False,
    )
    return lgb.train(
        params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=[val_data],
        valid_names=["valid"],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=200),
        ],
    )


def _save_lgbm_model(model: lgb.Booster, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # save_model can fail on some Cyrillic Windows/OneDrive paths.
    path.write_text(model.model_to_string(), encoding="utf-8")


def _predict_log(model: lgb.Booster, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    pred_log = model.predict(
        frame[features],
        num_iteration=model.best_iteration or model.current_iteration(),
    )
    return np.clip(np.asarray(pred_log, dtype=np.float64), 0.0, None)


def _validate_oof(oof: pd.DataFrame) -> None:
    required = {cfg.ID_COL, "fold", "cutoff_date", "target", "pred", "pred_log"}
    missing = required - set(oof.columns)
    if missing:
        raise AssertionError(f"OOF missing columns: {sorted(missing)}")
    keys = [cfg.ID_COL, "fold", "cutoff_date"]
    if oof.duplicated(keys).any():
        raise AssertionError("OOF composite key is not unique")
    values = oof[["target", "pred", "pred_log"]].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise AssertionError("OOF contains NaN or infinity")
    if (oof[["target", "pred", "pred_log"]].to_numpy() < 0).any():
        raise AssertionError("OOF contains negative values")
    if not np.allclose(oof["pred_log"], np.log1p(oof["pred"]), atol=1e-7):
        raise AssertionError("OOF pred_log is not log1p(pred)")


def _robust_iterations(best_iterations: list[int]) -> int:
    valid = [int(value) for value in best_iterations if int(value) > 0]
    if not valid:
        raise ValueError("No positive best_iteration values")
    return max(1, int(np.median(valid)))


def _json_ready(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON serialize {type(value)!r}")


def train_temporal_cv(
    device: str = "cpu",
    dataset_dir: str | Path | None = None,
    params_source: str = "baseline",
    params_path: str | Path | None = None,
    drop_feature_groups: Iterable[str] | None = None,
    num_boost_round: int = 5000,
    early_stopping_rounds: int = 100,
    final_refit: bool = True,
    row_limit: int | None = None,
    run_name: str | None = None,
) -> dict:
    dataset_dir = Path(dataset_dir) if dataset_dir else cfg.DATA_DIR
    folds = load_folds(dataset_dir=dataset_dir, row_limit=row_limit)
    feature_cols = get_features(folds, drop_feature_groups=drop_feature_groups)
    print_walk_forward_plan(folds)
    print(f"\nTEMPORAL LIGHTGBM CV: features={len(feature_cols)}")

    params, params_description = resolve_lgbm_params(
        params_source=params_source,
        params_path=params_path,
        device=device,
    )
    artifacts = resolve_artifact_layout(dataset_dir, row_limit, run_name)
    model_dir = artifacts.models_root / "lgbm"
    model_dir.mkdir(parents=True, exist_ok=True)
    artifacts.oof_dir.mkdir(parents=True, exist_ok=True)
    artifacts.reports_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Artifact mode={artifacts.artifact_mode} | run_name={artifacts.run_name} | "
        f"model_dir={model_dir} | oof_dir={artifacts.oof_dir} | "
        f"reports_dir={artifacts.reports_dir}"
    )

    registry = build_cv_folds(cfg.N_FOLDS, cfg.STEP_DAYS)
    prefix_models: list[lgb.Booster] = []
    oof_parts: list[pd.DataFrame] = []
    fold_metrics: list[dict] = []
    importance_parts: list[pd.DataFrame] = []
    best_iterations: list[int] = []

    for val_idx, train_specs, val_spec in walk_forward_splits(registry):
        train_df = pd.concat(folds[:val_idx], ignore_index=True)
        val_df = folds[val_idx]
        max_train_target_end = max(item.target_end for item in train_specs)
        if max_train_target_end > val_spec.cutoff:
            raise AssertionError("Temporal leakage invariant failed before fit")

        print("\n" + "-" * 100)
        print(
            f"FIT {val_spec.name}: train={len(train_df):,}, val={len(val_df):,}, "
            f"max_train_target_end={max_train_target_end}, val_cutoff={val_spec.cutoff}"
        )
        try:
            model = train_single_model(
                train_df,
                val_df,
                feature_cols,
                params,
                num_boost_round=num_boost_round,
                early_stopping_rounds=early_stopping_rounds,
            )
        except Exception:
            if params.get("device_type") == "cpu":
                raise
            print("GPU LightGBM failed; retrying this fold on CPU.")
            params["device_type"] = "cpu"
            model = train_single_model(
                train_df,
                val_df,
                feature_cols,
                params,
                num_boost_round=num_boost_round,
                early_stopping_rounds=early_stopping_rounds,
            )

        prefix_models.append(model)
        best_iteration = int(model.best_iteration or model.current_iteration())
        best_iterations.append(best_iteration)
        pred_log = _predict_log(model, val_df, feature_cols)
        pred = np.expm1(pred_log)
        prefix_predictions = [
            _predict_log(prefix_model, val_df, feature_cols)
            for prefix_model in prefix_models
        ]
        temporal_ensemble_log = np.mean(prefix_predictions, axis=0)
        temporal_ensemble_pred = np.expm1(temporal_ensemble_log)
        fold_score = rmsle(val_df[cfg.TARGET_COL], pred)
        ensemble_score = rmsle(val_df[cfg.TARGET_COL], temporal_ensemble_pred)

        model_path = model_dir / f"model_{val_spec.name}.txt"
        _save_lgbm_model(model, model_path)
        print(
            f"{val_spec.name}: RMSLE={fold_score:.6f}, "
            f"prefix-log-ensemble={ensemble_score:.6f}, best_iteration={best_iteration}"
        )

        oof_parts.append(
            pd.DataFrame(
                {
                    cfg.ID_COL: val_df[cfg.ID_COL].to_numpy(),
                    "fold": val_spec.name,
                    "cutoff_date": val_spec.cutoff.isoformat(),
                    "target": val_df[cfg.TARGET_COL].to_numpy(dtype=np.float64),
                    "pred": pred,
                    "pred_log": pred_log,
                    "pred_temporal_ensemble": temporal_ensemble_pred,
                    "pred_log_temporal_ensemble": temporal_ensemble_log,
                }
            )
        )
        fold_metrics.append(
            {
                "fold": val_spec.name,
                "cutoff_date": val_spec.cutoff.isoformat(),
                "train_anchors": [item.cutoff.isoformat() for item in train_specs],
                "max_train_target_end": max_train_target_end.isoformat(),
                "target_start": val_spec.target_start.isoformat(),
                "target_end": val_spec.target_end.isoformat(),
                "train_rows": len(train_df),
                "validation_rows": len(val_df),
                "rmsle": fold_score,
                "temporal_ensemble_rmsle": ensemble_score,
                "best_iteration": best_iteration,
            }
        )
        importance_parts.append(
            pd.DataFrame(
                {
                    "fold": val_spec.name,
                    "cutoff_date": val_spec.cutoff.isoformat(),
                    "feature": feature_cols,
                    "importance": model.feature_importance(importance_type="gain"),
                }
            )
        )

    oof = pd.concat(oof_parts, ignore_index=True)
    _validate_oof(oof)
    pooled = rmsle(oof["target"], oof["pred"])
    pooled_temporal_ensemble = rmsle(oof["target"], oof["pred_temporal_ensemble"])
    fold_scores = [row["rmsle"] for row in fold_metrics]
    print("\n" + "=" * 100)
    print(
        f"POOLED OOF RMSLE={pooled:.6f}; mean={np.mean(fold_scores):.6f}; "
        f"std={np.std(fold_scores):.6f}; latest={fold_scores[-1]:.6f}; "
        f"prefix-log-ensemble pooled={pooled_temporal_ensemble:.6f}"
    )

    oof_path = artifacts.oof_dir / "oof_lgbm.csv"
    oof.to_csv(oof_path, index=False)
    pd.DataFrame(fold_metrics).to_csv(
        artifacts.reports_dir / "lgbm_fold_metrics.csv", index=False
    )
    fold_importance = pd.concat(importance_parts, ignore_index=True)
    fold_importance.to_csv(
        artifacts.reports_dir / "lgbm_feature_importances_by_fold.csv", index=False
    )
    aggregate_importance = (
        fold_importance.groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
    )
    aggregate_importance.to_csv(
        artifacts.reports_dir / "lgbm_feature_importances.csv", index=False
    )

    final_iterations = _robust_iterations(best_iterations)
    final_model_path: Path | None = None
    if final_refit:
        if registry[-1].target_end > cfg.HIST_END:
            raise AssertionError("Final labelled target extends beyond known history")
        full_train = pd.concat(folds, ignore_index=True)
        final_dataset = lgb.Dataset(
            full_train[feature_cols],
            label=full_train[cfg.TARGET_LOG_COL],
            feature_name=feature_cols,
            free_raw_data=False,
        )
        final_params = dict(params)
        final_params["metric"] = "None"
        final_model = lgb.train(
            final_params,
            final_dataset,
            num_boost_round=final_iterations,
            callbacks=[lgb.log_evaluation(0)],
        )
        final_model_path = model_dir / "lgbm_final.txt"
        _save_lgbm_model(final_model, final_model_path)
        print(
            f"Final refit: rows={len(full_train):,}, snapshots={len(folds)}, "
            f"iterations=median{best_iterations}={final_iterations}"
        )

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "lightgbm_log1p_gmv",
        "dataset_dir": str(dataset_dir.resolve()),
        "dataset_fingerprint": _dataset_fingerprint(dataset_dir, row_limit),
        "artifact_mode": artifacts.artifact_mode,
        "run_name": artifacts.run_name,
        "params_source": params_description,
        "params": params,
        "feature_columns": feature_cols,
        "feature_dtypes": {column: str(folds[0][column].dtype) for column in feature_cols},
        "drop_feature_groups": sorted(drop_feature_groups or []),
        "folds": fold_metrics,
        "best_iterations": best_iterations,
        "final_iterations": final_iterations,
        "pooled_oof_rmsle": pooled,
        "mean_fold_rmsle": float(np.mean(fold_scores)),
        "std_fold_rmsle": float(np.std(fold_scores)),
        "latest_fold_rmsle": fold_scores[-1],
        "pooled_prefix_temporal_log_ensemble_rmsle": pooled_temporal_ensemble,
        "primary_inference": "final_refit",
        "final_model": str(final_model_path.resolve()) if final_model_path else None,
        "final_model_sha256": (
            _file_sha256(final_model_path) if final_model_path is not None else None
        ),
        "oof_path": str(oof_path.resolve()),
        "btyd_backend": getattr(cfg, "BTYD_BACKEND", "fallback"),
        "row_limit": row_limit,
    }
    manifest_path = model_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_ready),
        encoding="utf-8",
    )
    (artifacts.reports_dir / "lgbm_cv_summary.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_ready),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=cfg.DATA_DIR)
    parser.add_argument(
        "--params-source", choices=["baseline", "best", "path"], default="baseline"
    )
    parser.add_argument("--params-path", type=Path)
    parser.add_argument("--device", choices=["cpu", "gpu", "cuda"], default="cpu")
    parser.add_argument("--drop-feature-group", action="append", default=[])
    parser.add_argument("--num-boost-round", type=int, default=5000)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--no-final-refit", action="store_true")
    parser.add_argument(
        "--row-limit",
        type=int,
        default=None,
        help="Smoke-test only; never use limited rows for reported CV.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help=(
            "Isolated artifact namespace. If omitted, canonical outputs are used "
            "only for the full canonical dataset without --row-limit; all other "
            "runs are auto-isolated."
        ),
    )
    args = parser.parse_args()
    train_temporal_cv(
        device=args.device,
        dataset_dir=args.dataset_dir,
        params_source=args.params_source,
        params_path=args.params_path,
        drop_feature_groups=args.drop_feature_group,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
        final_refit=not args.no_final_refit,
        row_limit=args.row_limit,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
