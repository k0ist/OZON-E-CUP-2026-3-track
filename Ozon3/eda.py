"""Reproducible temporal EDA and aligned-OOF diagnostics.

This module reads fold parquet files and standardized OOF CSV files. It only
writes diagnostics: it never trains a model or creates a submission.

Standard OOF columns:
    user_id, fold, cutoff_date, target, pred[, pred_log]
"""

from __future__ import annotations

import argparse
import math
import re
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

import config as cfg


OOF_REQUIRED = {"user_id", "fold", "cutoff_date", "target", "pred"}
KEY = ["_user_key", "fold", "cutoff_date"]
TARGET_QUANTILES = (0.0, 0.5, 0.75, 0.9, 0.95, 0.99, 0.995, 0.999, 1.0)

# Ordered aliases document which semantics were selected. The exact resolution
# for every fold is also written to column_resolution.csv.
ALIASES = {
    "recency_purchase": (
        "recency_purchase_days",
        "recency_purchase",
        "purchase_recency_days",
    ),
    "purchase_frequency": (
        "purchase_frequency",
        "n_purchase_days",
        "purchase_days_lifetime",
        "n_purchase_days_total",
        "to_ord_sum_365d",
    ),
    "historical_gmv": (
        "total_purchase_gmv",
        "gmv_sum_365d",
        "gmv_sum_180d",
        "gmv_sum_90d",
        "gmv_sum_30d",
    ),
    "history_orders": (
        "to_ord_sum_30d",
        "orders_sum_30d",
        "to_ord_sum_90d",
        "orders_sum_90d",
        "to_ord_sum_365d",
    ),
    "history_searches": (
        "searches_sum_30d",
        "search_sum_30d",
        "searches_sum_90d",
        "searches_sum_365d",
    ),
    "target_orders": ("target_orders", "orders_target"),
}
ACTIVE_COUNT = (
    "n_active_days_30d",
    "active_days_30d",
    "n_active_days_14d",
    "active_days_14d",
)
ACTIVE_RECENCY = ("recency_active_days", "recency_days")
DIRECT_FEATURES = (
    "gmv_sum_30d",
    "gmv_sum_90d",
    "gmv_sum_180d",
    "gmv_sum_365d",
    "to_ord_sum_30d",
    "to_ord_sum_90d",
    "searches_sum_30d",
    "searches_sum_90d",
    "purchase_days_lifetime",
    "tenure_days",
)
OUTPUT_NAMES = (
    "diagnostic_status.csv",
    "input_files.csv",
    "column_resolution.csv",
    "target_distribution.csv",
    "target_extremes.csv",
    "temporal_drift.csv",
    "key_feature_drift.csv",
    "latest_fold_drift.csv",
    "oof_input_summary.csv",
    "model_scores.csv",
    "model_fold_scores.csv",
    "cohort_errors.csv",
    "cohort_thresholds.csv",
    "largest_log_errors.csv",
    "false_positives.csv",
    "false_negatives.csv",
    "prediction_calibration.csv",
    "oof_coverage.csv",
    "pairwise_model_diagnostics.csv",
    "pairwise_log_blends.csv",
    "pairwise_log_blends_by_fold.csv",
    "prediction_correlation.csv",
    "prediction_log_correlation.csv",
    "residual_correlation.csv",
    "eda_report.md",
)


def rmsle(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    true = np.clip(np.asarray(y_true, dtype=np.float64), 0.0, None)
    pred = np.clip(np.asarray(y_pred, dtype=np.float64), 0.0, None)
    if not len(true):
        return float("nan")
    return float(np.sqrt(np.mean(np.square(np.log1p(pred) - np.log1p(true)))))


def _natural_key(path: Path) -> list[object]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def _user_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def _user_series(series: pd.Series) -> pd.Series:
    return series.map(_user_value).astype("string")


def _fold_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return re.sub(
        r"^fold[\s_-]*", "", str(value).strip(), flags=re.IGNORECASE
    )


def _fold_series(series: pd.Series) -> pd.Series:
    return series.map(_fold_value).astype("string")


def _cutoff_series(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    normalized = parsed.dt.strftime("%Y-%m-%d")
    return normalized.fillna(series.astype("string").str.strip()).fillna("")


def _numeric(series: pd.Series) -> pd.Series:
    result = pd.to_numeric(series, errors="coerce").astype("float64")
    return result.where(np.isfinite(result))


def _first(columns: Iterable[str], aliases: Iterable[str]) -> str | None:
    available = set(columns)
    return next((column for column in aliases if column in available), None)


def _status(
    records: list[dict[str, object]],
    component: str,
    state: str,
    detail: str,
) -> None:
    records.append({"component": component, "status": state, "detail": detail})
    print(f"[{state.upper()}] {component}: {detail}")


def _schema(path: Path) -> list[str]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required to read selected parquet columns."
        ) from exc
    return list(pq.ParquetFile(path).schema_arrow.names)


def _clean_output(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Remove only this script's known outputs, avoiding stale results when a
    # model is absent on a later run.
    for name in OUTPUT_NAMES:
        path = output_dir / name
        if path.is_file():
            path.unlink()


def _read_fold(
    path: Path,
    statuses: list[dict[str, object]],
    resolutions: list[dict[str, object]],
) -> pd.DataFrame | None:
    available = _schema(path)
    if not {"user_id", "target"}.issubset(available):
        _status(
            statuses,
            f"fold:{path.name}",
            "skipped",
            "Required user_id and target columns are not both present.",
        )
        return None

    wanted = {
        "user_id",
        "target",
        "target_log",
        "fold",
        "cutoff_date",
        "anchor_date",
        *DIRECT_FEATURES,
        *ACTIVE_COUNT,
        *ACTIVE_RECENCY,
    }
    for candidates in ALIASES.values():
        wanted.update(candidates)
    usecols = [column for column in available if column in wanted]
    frame = pd.read_parquet(path, columns=usecols)
    frame["target"] = _numeric(frame["target"])
    bad_target = int(frame["target"].isna().sum())
    negative_target = int((frame["target"] < 0).sum())
    if bad_target or negative_target:
        _status(
            statuses,
            f"fold:{path.name}",
            "skipped",
            f"Invalid target rows: non-finite={bad_target}, negative={negative_target}.",
        )
        return None

    frame["_user_key"] = _user_series(frame["user_id"])
    if (frame["_user_key"] == "").any():
        _status(statuses, f"fold:{path.name}", "skipped", "Missing user_id.")
        return None

    if "fold" in frame:
        frame["fold"] = _fold_series(frame["fold"])
        fold_source = "fold"
    else:
        frame["fold"] = _fold_value(path.stem)
        fold_source = "filename"
    cutoff_source = _first(frame.columns, ("cutoff_date", "anchor_date"))
    if cutoff_source:
        frame["cutoff_date"] = _cutoff_series(frame[cutoff_source])
    else:
        frame["cutoff_date"] = ""
    resolutions.extend(
        [
            {
                "source_file": path.name,
                "canonical_column": "fold",
                "source_column": fold_source,
            },
            {
                "source_file": path.name,
                "canonical_column": "cutoff_date",
                "source_column": cutoff_source or "unavailable",
            },
        ]
    )

    for canonical, candidates in ALIASES.items():
        source = _first(frame.columns, candidates)
        frame[canonical] = _numeric(frame[source]) if source else np.nan
        resolutions.append(
            {
                "source_file": path.name,
                "canonical_column": canonical,
                "source_column": source or "unavailable",
            }
        )

    count_source = _first(frame.columns, ACTIVE_COUNT)
    recency_source = _first(frame.columns, ACTIVE_RECENCY)
    if count_source:
        active_values = _numeric(frame[count_source])
        frame["is_active"] = (active_values > 0).where(active_values.notna())
        active_source = f"{count_source} > 0"
    elif recency_source:
        active_values = _numeric(frame[recency_source])
        frame["is_active"] = (active_values <= 30).where(active_values.notna())
        active_source = f"{recency_source} <= 30"
    else:
        frame["is_active"] = np.nan
        active_source = "unavailable"
    resolutions.append(
        {
            "source_file": path.name,
            "canonical_column": "is_active",
            "source_column": active_source,
        }
    )

    for column in DIRECT_FEATURES:
        if column in frame:
            frame[column] = _numeric(frame[column])
    frame["target_log_eda"] = np.log1p(frame["target"].clip(lower=0))
    frame["_source_file"] = path.name
    cutoff = str(frame["cutoff_date"].iloc[0]) or "unavailable"
    _status(
        statuses,
        f"fold:{path.name}",
        "loaded",
        f"{len(frame):,} rows; fold={frame['fold'].iloc[0]}; cutoff={cutoff}.",
    )
    return frame


def load_folds(
    dataset_dir: Path,
    pattern: str,
    statuses: list[dict[str, object]],
) -> tuple[list[pd.DataFrame], pd.DataFrame, list[Path]]:
    files = sorted(dataset_dir.glob(pattern), key=_natural_key)
    if not files:
        _status(
            statuses,
            "fold_dataset",
            "unavailable",
            f"No files matched {dataset_dir / pattern}; target, drift, and "
            "feature-cohort diagnostics will be skipped.",
        )
        return [], pd.DataFrame(), []
    frames: list[pd.DataFrame] = []
    resolutions: list[dict[str, object]] = []
    for path in files:
        try:
            frame = _read_fold(path, statuses, resolutions)
        except Exception as exc:
            _status(
                statuses,
                f"fold:{path.name}",
                "skipped",
                f"{type(exc).__name__}: {exc}",
            )
            continue
        if frame is not None:
            frames.append(frame)
    # Filenames are expected to follow fold_0, fold_1, ... but cutoff metadata
    # is the temporal source of truth.  Sort by it whenever every loaded fold
    # supplies a cutoff so that "latest fold" diagnostics cannot silently use
    # a lexicographically last, temporally earlier file.
    if frames and all(str(frame["cutoff_date"].iloc[0]) for frame in frames):
        frames.sort(key=lambda frame: str(frame["cutoff_date"].iloc[0]))
    if not frames:
        _status(
            statuses,
            "fold_dataset",
            "unavailable",
            "Fold files exist, but none passed validation.",
        )
    return frames, pd.DataFrame(resolutions), files


def target_diagnostics(
    folds: list[pd.DataFrame], top_n: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, object]] = []
    extremes: list[pd.DataFrame] = []
    for frame in folds:
        target = frame["target"]
        log_target = frame["target_log_eda"]
        row: dict[str, object] = {
            "fold": frame["fold"].iloc[0],
            "cutoff_date": frame["cutoff_date"].iloc[0],
            "rows": len(frame),
            "users": frame["_user_key"].nunique(),
            "zero_target_share": float((target == 0).mean()),
            "target_sum": float(target.sum()),
            "target_mean": float(target.mean()),
            "target_std": float(target.std(ddof=0)),
            "log_target_mean": float(log_target.mean()),
            "log_target_std": float(log_target.std(ddof=0)),
        }
        for quantile, value in target.quantile(TARGET_QUANTILES).items():
            row[f"target_q{quantile:g}"] = float(value)
        for quantile, value in log_target.quantile(TARGET_QUANTILES).items():
            row[f"log_target_q{quantile:g}"] = float(value)
        summaries.append(row)
        columns = [
            column
            for column in (
                "user_id",
                "fold",
                "cutoff_date",
                "target",
                "target_log_eda",
                "recency_purchase",
                "purchase_frequency",
                "historical_gmv",
                "is_active",
            )
            if column in frame
        ]
        top = frame.nlargest(min(top_n, len(frame)), "target")[columns].copy()
        top.insert(0, "rank_within_fold", np.arange(1, len(top) + 1))
        extremes.append(top)
    return pd.DataFrame(summaries), pd.concat(extremes, ignore_index=True)


def _distribution(
    values: pd.Series, fold: str, cutoff: str, feature: str
) -> dict[str, object]:
    values = _numeric(values)
    valid = values.dropna()
    row: dict[str, object] = {
        "fold": fold,
        "cutoff_date": cutoff,
        "feature": feature,
        "rows": len(values),
        "missing_share": float(values.isna().mean()),
        "mean": float(valid.mean()) if len(valid) else np.nan,
        "std": float(valid.std(ddof=0)) if len(valid) else np.nan,
    }
    for quantile in (0.0, 0.1, 0.5, 0.9, 0.99, 1.0):
        row[f"q{quantile:g}"] = (
            float(valid.quantile(quantile)) if len(valid) else np.nan
        )
    return row


def drift_diagnostics(
    folds: list[pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    canonical = (
        "target",
        "target_log_eda",
        "recency_purchase",
        "purchase_frequency",
        "historical_gmv",
        "history_orders",
        "history_searches",
        "is_active",
        *DIRECT_FEATURES,
    )
    temporal_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    for frame in folds:
        fold = str(frame["fold"].iloc[0])
        cutoff = str(frame["cutoff_date"].iloc[0])
        temporal_rows.append(
            {
                "fold": fold,
                "cutoff_date": cutoff,
                "rows": len(frame),
                "users": frame["_user_key"].nunique(),
                "active_user_share": (
                    float(frame["is_active"].mean())
                    if frame["is_active"].notna().any()
                    else np.nan
                ),
                "zero_target_share": float((frame["target"] == 0).mean()),
                "target_gmv_sum": float(frame["target"].sum()),
                "target_gmv_mean": float(frame["target"].mean()),
                "target_orders_sum": (
                    float(frame["target_orders"].sum())
                    if "target_orders" in frame
                    and frame["target_orders"].notna().any()
                    else np.nan
                ),
                "history_gmv_sum": (
                    float(frame["historical_gmv"].sum())
                    if frame["historical_gmv"].notna().any()
                    else np.nan
                ),
                "history_orders_sum": (
                    float(frame["history_orders"].sum())
                    if frame["history_orders"].notna().any()
                    else np.nan
                ),
                "history_searches_sum": (
                    float(frame["history_searches"].sum())
                    if frame["history_searches"].notna().any()
                    else np.nan
                ),
            }
        )
        for feature in dict.fromkeys(canonical):
            if feature in frame and frame[feature].notna().any():
                feature_rows.append(
                    _distribution(frame[feature], fold, cutoff, feature)
                )

    latest_rows: list[dict[str, object]] = []
    if len(folds) > 1:
        earlier = pd.concat(folds[:-1], ignore_index=True)
        latest = folds[-1]
        for feature in dict.fromkeys(canonical):
            if feature not in earlier or feature not in latest:
                continue
            reference = _numeric(earlier[feature]).dropna()
            current = _numeric(latest[feature]).dropna()
            if not len(reference) or not len(current):
                continue
            ref_mean = float(reference.mean())
            cur_mean = float(current.mean())
            ref_std = float(reference.std(ddof=0))
            latest_rows.append(
                {
                    "feature": feature,
                    "reference_folds": ",".join(
                        str(value) for value in pd.unique(earlier["fold"])
                    ),
                    "latest_fold": str(latest["fold"].iloc[0]),
                    "reference_rows": len(reference),
                    "latest_rows": len(current),
                    "reference_mean": ref_mean,
                    "latest_mean": cur_mean,
                    "relative_mean_change": (cur_mean - ref_mean)
                    / (abs(ref_mean) + 1e-12),
                    "standardized_mean_difference": (cur_mean - ref_mean)
                    / (ref_std + 1e-12),
                    "reference_q50": float(reference.quantile(0.5)),
                    "latest_q50": float(current.quantile(0.5)),
                    "reference_q90": float(reference.quantile(0.9)),
                    "latest_q90": float(current.quantile(0.9)),
                }
            )
    return (
        pd.DataFrame(temporal_rows),
        pd.DataFrame(feature_rows),
        pd.DataFrame(latest_rows),
    )


def _model_name(path: Path) -> str:
    return path.stem[4:] if path.stem.lower().startswith("oof_") else path.stem


def load_oof_models(
    oof_dir: Path,
    pattern: str,
    statuses: list[dict[str, object]],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, list[Path]]:
    files = sorted(oof_dir.glob(pattern), key=_natural_key)
    if not files:
        _status(
            statuses,
            "oof_models",
            "unavailable",
            f"No standardized OOF files matched {oof_dir / pattern}; model, "
            "cohort-error, and diversity diagnostics will be skipped.",
        )
        return {}, pd.DataFrame(), []

    models: dict[str, pd.DataFrame] = {}
    summaries: list[dict[str, object]] = []
    for path in files:
        model = _model_name(path)
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            _status(statuses, f"oof:{model}", "skipped", f"Read failed: {exc}")
            continue
        missing = sorted(OOF_REQUIRED.difference(frame.columns))
        if missing:
            _status(
                statuses,
                f"oof:{model}",
                "skipped",
                f"Missing standardized columns: {missing}.",
            )
            continue
        columns = ["user_id", "fold", "cutoff_date", "target", "pred"]
        if "pred_log" in frame:
            columns.append("pred_log")
        frame = frame[columns].copy()
        frame["_user_key"] = _user_series(frame["user_id"])
        frame["fold"] = _fold_series(frame["fold"])
        frame["cutoff_date"] = _cutoff_series(frame["cutoff_date"])
        frame["target"] = _numeric(frame["target"])
        frame["pred"] = _numeric(frame["pred"])
        if "pred_log" in frame:
            frame["pred_log"] = _numeric(frame["pred_log"])

        blank_key = int(
            (
                (frame["_user_key"] == "")
                | (frame["fold"] == "")
                | (frame["cutoff_date"] == "")
            ).sum()
        )
        nonfinite = int(frame[["target", "pred"]].isna().any(axis=1).sum())
        negative_target = int((frame["target"] < 0).sum())
        duplicate_key = int(frame.duplicated(KEY).sum())
        if blank_key or nonfinite or negative_target or duplicate_key:
            _status(
                statuses,
                f"oof:{model}",
                "skipped",
                "Invalid standardized OOF: "
                f"blank_key={blank_key}, non_finite={nonfinite}, "
                f"negative_target={negative_target}, duplicate_key={duplicate_key}.",
            )
            continue

        negative_prediction = int((frame["pred"] < 0).sum())
        frame["model"] = model
        models[model] = frame
        score = rmsle(frame["target"], frame["pred"])
        summaries.append(
            {
                "model": model,
                "source_file": str(path.resolve()),
                "rows": len(frame),
                "folds": frame["fold"].nunique(),
                "cutoffs": frame["cutoff_date"].nunique(),
                "cutoff_min": frame["cutoff_date"].min(),
                "cutoff_max": frame["cutoff_date"].max(),
                "negative_prediction_rows": negative_prediction,
                "pooled_rmsle": score,
            }
        )
        _status(
            statuses,
            f"oof:{model}",
            "loaded" if not negative_prediction else "warning",
            f"{len(frame):,} rows; pooled RMSLE={score:.6f}; "
            f"negative predictions={negative_prediction}.",
        )

    if not models:
        _status(
            statuses,
            "oof_models",
            "unavailable",
            "OOF files exist, but none passed standardized schema checks.",
        )
    return models, pd.DataFrame(summaries), files


def _feature_lookup(folds: list[pd.DataFrame]) -> pd.DataFrame:
    if not folds:
        return pd.DataFrame()
    possible = (
        "recency_purchase",
        "purchase_frequency",
        "historical_gmv",
        "is_active",
        "history_orders",
        "history_searches",
        *DIRECT_FEATURES,
    )
    feature_columns = [
        column for column in possible if any(column in frame for frame in folds)
    ]
    pieces = []
    for frame in folds:
        keep = [
            "user_id",
            "_user_key",
            "fold",
            "cutoff_date",
            *[column for column in feature_columns if column in frame],
        ]
        piece = frame[keep].copy()
        for column in feature_columns:
            if column not in piece:
                piece[column] = np.nan
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def attach_fold_features(
    oof: pd.DataFrame,
    lookup: pd.DataFrame,
    model: str,
    statuses: list[dict[str, object]],
) -> pd.DataFrame:
    if lookup.empty:
        _status(
            statuses,
            f"feature_join:{model}",
            "skipped",
            "Fold datasets are unavailable; history-based cohorts were not "
            "attached.",
        )
        result = oof.copy()
        result["_dataset_match"] = False
        return result
    feature_columns = [
        column
        for column in lookup.columns
        if column not in {"user_id", "_user_key", "fold", "cutoff_date"}
    ]
    candidate = lookup[KEY + feature_columns].copy()
    if candidate.duplicated(KEY).any():
        _status(
            statuses,
            f"feature_join:{model}",
            "skipped",
            "Fold feature keys (user_id + fold + cutoff_date) are not unique; "
            "feature cohorts were not attached.",
        )
        result = oof.copy()
        result["_dataset_match"] = False
        return result
    candidate["_dataset_match"] = True
    best = oof.merge(candidate, on=KEY, how="left", validate="one_to_one")
    matched = best["_dataset_match"].eq(True)
    best_rate = float(matched.mean())
    best["_dataset_match"] = matched
    _status(
        statuses,
        f"feature_join:{model}",
        "loaded" if best_rate >= 0.95 else "warning",
        f"Matched {best_rate:.2%} by exact temporal key {KEY}.",
    )
    return best


def _metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    if not len(frame):
        return {
            "rows": 0,
            "rmsle": np.nan,
            "mean_target": np.nan,
            "mean_prediction": np.nan,
            "raw_bias": np.nan,
            "mean_log_residual": np.nan,
        }
    true = frame["target"].to_numpy(dtype=np.float64)
    pred = np.clip(frame["pred"].to_numpy(dtype=np.float64), 0.0, None)
    residual = np.log1p(pred) - np.log1p(np.clip(true, 0.0, None))
    return {
        "rows": len(frame),
        "rmsle": float(np.sqrt(np.mean(np.square(residual)))),
        "mean_target": float(np.mean(true)),
        "mean_prediction": float(np.mean(pred)),
        "raw_bias": float(np.mean(pred - true)),
        "mean_log_residual": float(np.mean(residual)),
    }


def model_scores(
    models: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pooled_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    for model, frame in models.items():
        work = frame.copy()
        work["_squared_log_error"] = np.square(
            np.log1p(work["pred"].clip(lower=0))
            - np.log1p(work["target"].clip(lower=0))
        )
        total_error = float(work["_squared_log_error"].sum())
        row: dict[str, object] = {"model": model, **_metrics(work)}
        row.update(
            {
                "folds": work["fold"].nunique(),
                "negative_prediction_rows": int((work["pred"] < 0).sum()),
                "zero_target_error_share": (
                    float(
                        work.loc[work["target"] == 0, "_squared_log_error"].sum()
                        / total_error
                    )
                    if total_error > 0
                    else np.nan
                ),
                "top_1pct_target_error_share": (
                    float(
                        work.loc[
                            work["target"] >= work["target"].quantile(0.99),
                            "_squared_log_error",
                        ].sum()
                        / total_error
                    )
                    if total_error > 0
                    else np.nan
                ),
            }
        )
        pooled_rows.append(row)
        for (fold, cutoff), group in work.groupby(
            ["fold", "cutoff_date"], sort=True, dropna=False
        ):
            fold_rows.append(
                {
                    "model": model,
                    "fold": fold,
                    "cutoff_date": cutoff,
                    **_metrics(group),
                }
            )
    return pd.DataFrame(pooled_rows), pd.DataFrame(fold_rows)


def _quantile_band(values: pd.Series, prefix: str) -> pd.Series:
    numeric = _numeric(values)
    result = pd.Series("missing", index=values.index, dtype="string")
    result.loc[numeric == 0] = "zero"
    positive = numeric > 0
    if positive.any():
        rank = numeric.loc[positive].rank(method="average", pct=True)
        result.loc[positive] = pd.cut(
            rank,
            [0.0, 0.25, 0.5, 0.75, 1.0],
            labels=[
                f"{prefix}_q1",
                f"{prefix}_q2",
                f"{prefix}_q3",
                f"{prefix}_q4",
            ],
            include_lowest=True,
        ).astype("string")
    return result


def _high_value_band(values: pd.Series) -> tuple[pd.Series, float, float]:
    numeric = _numeric(values)
    result = pd.Series("missing", index=values.index, dtype="string")
    result.loc[numeric == 0] = "zero"
    positive = numeric[numeric > 0]
    if not len(positive):
        return result, np.nan, np.nan
    p90 = float(positive.quantile(0.90))
    p99 = float(positive.quantile(0.99))
    result.loc[(numeric > 0) & (numeric < p90)] = "below_positive_p90"
    result.loc[(numeric >= p90) & (numeric < p99)] = "positive_p90_p99"
    result.loc[numeric >= p99] = "positive_top_1pct"
    return result, p90, p99


def add_cohorts(
    frame: pd.DataFrame, model: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = frame.copy()
    thresholds: list[dict[str, object]] = []
    work["future_target_status"] = np.where(
        work["target"] == 0, "target_zero", "target_nonzero"
    )
    target_band, p90, p99 = _high_value_band(work["target"])
    work["future_value_band"] = target_band
    thresholds.extend(
        [
            {"model": model, "cohort": "future_value_band", "threshold": "p90", "value": p90},
            {"model": model, "cohort": "future_value_band", "threshold": "p99", "value": p99},
        ]
    )
    if "is_active" in work and work["is_active"].notna().any():
        activity = pd.Series(
            np.where(work["is_active"] > 0, "active", "inactive"),
            index=work.index,
            dtype="string",
        )
        work["activity_status"] = activity.where(
            work["is_active"].notna(), "missing"
        )
    if "recency_purchase" in work and work["recency_purchase"].notna().any():
        work["recency_band"] = (
            pd.cut(
                _numeric(work["recency_purchase"]),
                [-np.inf, 7, 30, 90, 180, np.inf],
                labels=["0_7", "8_30", "31_90", "91_180", "181_plus"],
            )
            .astype("string")
            .fillna("missing")
        )
    if "purchase_frequency" in work and work["purchase_frequency"].notna().any():
        work["frequency_band"] = _quantile_band(
            work["purchase_frequency"], "positive"
        )
    if "historical_gmv" in work and work["historical_gmv"].notna().any():
        work["historical_gmv_band"] = _quantile_band(
            work["historical_gmv"], "positive"
        )
        history_band, p90, p99 = _high_value_band(work["historical_gmv"])
        work["historical_value_band"] = history_band
        thresholds.extend(
            [
                {
                    "model": model,
                    "cohort": "historical_value_band",
                    "threshold": "p90",
                    "value": p90,
                },
                {
                    "model": model,
                    "cohort": "historical_value_band",
                    "threshold": "p99",
                    "value": p99,
                },
            ]
        )
    return work, pd.DataFrame(thresholds)


def cohort_errors(frame: pd.DataFrame, model: str) -> pd.DataFrame:
    cohort_columns = [
        column
        for column in (
            "future_target_status",
            "future_value_band",
            "activity_status",
            "recency_band",
            "frequency_band",
            "historical_gmv_band",
            "historical_value_band",
        )
        if column in frame
    ]
    scopes: list[tuple[str, str, pd.DataFrame]] = [("pooled", "", frame)]
    scopes.extend(
        (f"fold={fold};cutoff={cutoff}", str(fold), group)
        for (fold, cutoff), group in frame.groupby(
            ["fold", "cutoff_date"], sort=True, dropna=False
        )
    )
    rows: list[dict[str, object]] = []
    for scope, fold, scoped in scopes:
        total_error = float(
            np.square(
                np.log1p(scoped["pred"].clip(lower=0))
                - np.log1p(scoped["target"].clip(lower=0))
            ).sum()
        )
        for cohort_type in cohort_columns:
            for cohort, group in scoped.groupby(
                cohort_type, sort=True, dropna=False
            ):
                group_error = float(
                    np.square(
                        np.log1p(group["pred"].clip(lower=0))
                        - np.log1p(group["target"].clip(lower=0))
                    ).sum()
                )
                rows.append(
                    {
                        "model": model,
                        "scope": scope,
                        "fold": fold,
                        "cohort_type": cohort_type,
                        "cohort": str(cohort),
                        "row_share": len(group) / len(scoped),
                        "squared_log_error_share": (
                            group_error / total_error if total_error > 0 else np.nan
                        ),
                        **_metrics(group),
                    }
                )
    return pd.DataFrame(rows)


def error_diagnostics(
    frame: pd.DataFrame,
    model: str,
    top_n: int,
    calibration_bins: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = frame.copy()
    work["pred_clipped"] = work["pred"].clip(lower=0)
    work["log_residual"] = np.log1p(work["pred_clipped"]) - np.log1p(
        work["target"].clip(lower=0)
    )
    work["squared_log_error"] = np.square(work["log_residual"])
    columns = [
        column
        for column in (
            "user_id",
            "fold",
            "cutoff_date",
            "target",
            "pred",
            "pred_clipped",
            "log_residual",
            "squared_log_error",
            "recency_purchase",
            "purchase_frequency",
            "historical_gmv",
            "is_active",
        )
        if column in work
    ]
    largest = work.nlargest(min(top_n, len(work)), "squared_log_error")[
        columns
    ].copy()
    largest.insert(0, "model", model)
    zero_pool = work.loc[work["target"] == 0]
    false_positive = zero_pool.nlargest(
        min(top_n, len(zero_pool)), "pred_clipped"
    )[columns].copy()
    false_positive.insert(0, "model", model)
    positive = work.loc[work["target"] > 0]
    if len(positive):
        threshold = float(positive["target"].quantile(0.90))
        pool = positive.loc[
            (positive["target"] >= threshold) & (positive["log_residual"] < 0)
        ]
        false_negative = pool.nsmallest(
            min(top_n, len(pool)), "log_residual"
        )[columns].copy()
    else:
        false_negative = pd.DataFrame(columns=columns)
    false_negative.insert(0, "model", model)

    calibration_rows: list[dict[str, object]] = []
    scopes: list[tuple[str, str, pd.DataFrame]] = [("pooled", "", work)]
    scopes.extend(
        (f"fold={fold};cutoff={cutoff}", str(fold), group)
        for (fold, cutoff), group in work.groupby(
            ["fold", "cutoff_date"], sort=True, dropna=False
        )
    )
    for scope, fold, scoped in scopes:
        if not len(scoped):
            continue
        bins = max(1, min(calibration_bins, len(scoped)))
        bin_id = pd.qcut(
            scoped["pred_clipped"].rank(method="first"),
            q=bins,
            labels=False,
            duplicates="drop",
        )
        binned = scoped.assign(prediction_quantile=bin_id.astype("int64") + 1)
        for quantile, group in binned.groupby("prediction_quantile", sort=True):
            calibration_rows.append(
                {
                    "model": model,
                    "scope": scope,
                    "fold": fold,
                    "prediction_quantile": int(quantile),
                    "prediction_min": float(group["pred_clipped"].min()),
                    "prediction_max": float(group["pred_clipped"].max()),
                    **_metrics(group),
                }
            )
    return (
        largest,
        false_positive,
        false_negative,
        pd.DataFrame(calibration_rows),
    )


def diversity_diagnostics(
    models: dict[str, pd.DataFrame],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame | None,
    pd.DataFrame | None,
    pd.DataFrame | None,
]:
    coverage_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    for model_a, model_b in combinations(sorted(models), 2):
        left = models[model_a][KEY + ["target", "pred"]].rename(
            columns={"target": "target_a", "pred": "pred_a"}
        )
        right = models[model_b][KEY + ["target", "pred"]].rename(
            columns={"target": "target_b", "pred": "pred_b"}
        )
        outer = left.merge(
            right, on=KEY, how="outer", indicator=True, validate="one_to_one"
        )
        aligned = outer.loc[outer["_merge"] == "both"].copy()
        mismatch = (
            int(
                (
                    ~np.isclose(
                        aligned["target_a"],
                        aligned["target_b"],
                        rtol=1e-6,
                        atol=1e-6,
                    )
                ).sum()
            )
            if len(aligned)
            else 0
        )
        coverage_rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "rows_a": len(left),
                "rows_b": len(right),
                "aligned_rows": len(aligned),
                "only_a_rows": int((outer["_merge"] == "left_only").sum()),
                "only_b_rows": int((outer["_merge"] == "right_only").sum()),
                "target_mismatch_rows": mismatch,
                "coverage_a": len(aligned) / len(left) if len(left) else np.nan,
                "coverage_b": len(aligned) / len(right) if len(right) else np.nan,
            }
        )
        if not len(aligned) or mismatch:
            continue
        target = aligned["target_a"].to_numpy(dtype=np.float64)
        pred_a = np.clip(aligned["pred_a"].to_numpy(dtype=np.float64), 0, None)
        pred_b = np.clip(aligned["pred_b"].to_numpy(dtype=np.float64), 0, None)
        z_true = np.log1p(target)
        z_a, z_b = np.log1p(pred_a), np.log1p(pred_b)
        blend = np.expm1(0.5 * z_a + 0.5 * z_b)
        score_a, score_b = rmsle(target, pred_a), rmsle(target, pred_b)
        blend_score = rmsle(target, blend)
        pair_rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "aligned_rows": len(aligned),
                "prediction_correlation": float(np.corrcoef(pred_a, pred_b)[0, 1]),
                "prediction_log_correlation": float(np.corrcoef(z_a, z_b)[0, 1]),
                "residual_correlation": float(
                    np.corrcoef(z_a - z_true, z_b - z_true)[0, 1]
                ),
                "rmsle_a_on_intersection": score_a,
                "rmsle_b_on_intersection": score_b,
                "equal_log_blend_rmsle": blend_score,
                "blend_gain_vs_best": min(score_a, score_b) - blend_score,
            }
        )
        aligned["pred_a_clipped"] = pred_a
        aligned["pred_b_clipped"] = pred_b
        aligned["blend_pred"] = blend
        for (fold, cutoff), group in aligned.groupby(
            ["fold", "cutoff_date"], sort=True, dropna=False
        ):
            fold_a = rmsle(group["target_a"], group["pred_a_clipped"])
            fold_b = rmsle(group["target_a"], group["pred_b_clipped"])
            fold_blend = rmsle(group["target_a"], group["blend_pred"])
            fold_rows.append(
                {
                    "model_a": model_a,
                    "model_b": model_b,
                    "fold": fold,
                    "cutoff_date": cutoff,
                    "rows": len(group),
                    "rmsle_a": fold_a,
                    "rmsle_b": fold_b,
                    "equal_log_blend_rmsle": fold_blend,
                    "blend_gain_vs_best": min(fold_a, fold_b) - fold_blend,
                }
            )

    raw_corr = log_corr = residual_corr = None
    if len(models) >= 2:
        names = sorted(models)
        common = models[names[0]][KEY + ["target", "pred"]].rename(
            columns={"target": "_target", "pred": names[0]}
        )
        accepted = [names[0]]
        for model in names[1:]:
            candidate = models[model][KEY + ["target", "pred"]].rename(
                columns={"target": "_candidate_target", "pred": model}
            )
            merged = common.merge(
                candidate, on=KEY, how="inner", validate="one_to_one"
            )
            if len(merged) and np.allclose(
                merged["_target"],
                merged["_candidate_target"],
                rtol=1e-6,
                atol=1e-6,
            ):
                common = merged.drop(columns="_candidate_target")
                accepted.append(model)
        if len(accepted) >= 2 and len(common):
            raw = common[accepted].clip(lower=0)
            log_values = np.log1p(raw)
            residuals = log_values.sub(np.log1p(common["_target"]), axis=0)
            raw_corr = raw.corr()
            log_corr = log_values.corr()
            residual_corr = residuals.corr()
            for matrix in (raw_corr, log_corr, residual_corr):
                matrix.index.name = f"all_common_rows={len(common)}"
    return (
        pd.DataFrame(coverage_rows),
        pd.DataFrame(pair_rows),
        pd.DataFrame(fold_rows),
        raw_corr,
        log_corr,
        residual_corr,
    )


def _md_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return ""
        return f"{float(value):.6g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def markdown_table(
    frame: pd.DataFrame,
    columns: Sequence[str] | None = None,
    max_rows: int = 12,
) -> str:
    if frame.empty:
        return "_Unavailable._"
    selected = list(columns) if columns is not None else list(frame.columns)
    selected = [column for column in selected if column in frame]
    view = frame[selected].head(max_rows)
    header = "| " + " | ".join(selected) + " |"
    divider = "| " + " | ".join("---" for _ in selected) + " |"
    rows = [
        "| " + " | ".join(_md_value(value) for value in row) + " |"
        for row in view.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def next_experiments(
    scores: pd.DataFrame,
    cohorts: pd.DataFrame,
    calibration: pd.DataFrame,
    latest_drift: pd.DataFrame,
    pairwise: pd.DataFrame,
) -> list[str]:
    suggestions: list[str] = []
    if scores.empty:
        return [
            "First generate at least one standardized, leakage-free temporal OOF "
            "file. Without it, cohort errors and ensemble gains cannot be measured."
        ]

    best = scores.sort_values("rmsle").iloc[0]
    zero_share = float(best.get("zero_target_error_share", 0.0))
    if zero_share >= 0.35:
        suggestions.append(
            "Test a fold-safe purchase-probability or hurdle calibration on "
            f"{best['model']}: future-zero rows contribute {zero_share:.1%} of "
            "its squared log error."
        )

    if not cohorts.empty and {"scope", "cohort_type"}.issubset(cohorts):
        recency = cohorts.loc[
            (cohorts["scope"] == "pooled")
            & (cohorts["cohort_type"] == "recency_band")
        ]
        if not recency.empty:
            eligible = recency.loc[
                recency["rows"] >= max(100, int(0.005 * recency["rows"].sum()))
            ]
            if not eligible.empty:
                worst = eligible.sort_values("rmsle", ascending=False).iloc[0]
                suggestions.append(
                    "Target the weakest recency cohort with one matched-fold "
                    f"ablation: {worst['cohort']} has RMSLE {worst['rmsle']:.4f} "
                    f"on {int(worst['rows']):,} rows."
                )

    if not pairwise.empty:
        best_pair = pairwise.sort_values(
            "blend_gain_vs_best", ascending=False
        ).iloc[0]
        if best_pair["blend_gain_vs_best"] > 0.001:
            suggestions.append(
                "Fit non-negative log-space OOF weights for "
                f"{best_pair['model_a']} and {best_pair['model_b']}; their equal "
                f"blend gains {best_pair['blend_gain_vs_best']:.5f} RMSLE on "
                "aligned rows."
            )
        elif best_pair["residual_correlation"] > 0.99:
            suggestions.append(
                "Do not add another near-duplicate model only for ensembling: "
                f"the best available pair has residual correlation "
                f"{best_pair['residual_correlation']:.4f} without material gain."
            )

    if not latest_drift.empty:
        shifted = latest_drift.assign(
            abs_smd=latest_drift["standardized_mean_difference"].abs()
        ).sort_values("abs_smd", ascending=False)
        if len(shifted) and shifted.iloc[0]["abs_smd"] >= 0.25:
            row = shifted.iloc[0]
            suggestions.append(
                "Before aggressive recency weighting, run a controlled temporal "
                f"test for drift in {row['feature']} (latest-vs-earlier SMD "
                f"{row['standardized_mean_difference']:.3f})."
            )

    if not calibration.empty and {"scope", "mean_log_residual"}.issubset(
        calibration
    ):
        pooled = calibration.loc[calibration["scope"] == "pooled"]
        if not pooled.empty:
            worst = pooled.loc[pooled["mean_log_residual"].abs().idxmax()]
            if abs(worst["mean_log_residual"]) >= 0.10:
                suggestions.append(
                    "Evaluate fold-safe monotone calibration because prediction "
                    f"quantile {int(worst['prediction_quantile'])} of "
                    f"{worst['model']} has mean log residual "
                    f"{worst['mean_log_residual']:.3f}."
                )

    if not suggestions:
        suggestions.append(
            "Run one matched-seed ablation for the feature family implicated by "
            "the largest stable cohort error; these diagnostics do not justify "
            "a broader search."
        )
    return suggestions[:4]


def write_report(
    output_dir: Path,
    statuses: pd.DataFrame,
    target_summary: pd.DataFrame,
    temporal_drift: pd.DataFrame,
    latest_drift: pd.DataFrame,
    scores: pd.DataFrame,
    fold_scores: pd.DataFrame,
    cohorts: pd.DataFrame,
    calibration: pd.DataFrame,
    coverage: pd.DataFrame,
    pairwise: pd.DataFrame,
) -> Path:
    if not latest_drift.empty:
        largest_drift = latest_drift.assign(
            abs_smd=latest_drift["standardized_mean_difference"].abs()
        ).sort_values("abs_smd", ascending=False)
    else:
        largest_drift = latest_drift
    if not cohorts.empty:
        cohort_view = cohorts.loc[
            (cohorts["scope"] == "pooled") & (cohorts["rows"] >= 100)
        ].sort_values(["model", "rmsle"], ascending=[True, False])
    else:
        cohort_view = cohorts
    if not calibration.empty:
        calibration_view = calibration.loc[calibration["scope"] == "pooled"]
    else:
        calibration_view = calibration

    lines = [
        "# Temporal EDA and OOF diagnostics",
        "",
        "This report uses only the fold and OOF artifacts found during this run. "
        "Missing or invalid inputs are marked unavailable; no result is inferred "
        "from filenames or leaderboard history.",
        "",
        "## Input and validation status",
        "",
        markdown_table(statuses, max_rows=50),
        "",
        "## Target by temporal fold",
        "",
        markdown_table(
            target_summary,
            (
                "fold",
                "cutoff_date",
                "rows",
                "zero_target_share",
                "target_mean",
                "target_q0.5",
                "target_q0.9",
                "target_q0.99",
                "target_q1",
                "log_target_mean",
            ),
            20,
        ),
        "",
        "## Temporal drift",
        "",
        markdown_table(temporal_drift, max_rows=20),
        "",
        "Largest latest-fold standardized mean differences:",
        "",
        markdown_table(
            largest_drift,
            (
                "feature",
                "reference_mean",
                "latest_mean",
                "standardized_mean_difference",
            ),
            12,
        ),
        "",
        "## Standalone OOF performance",
        "",
        markdown_table(
            scores.sort_values("rmsle") if not scores.empty else scores,
            max_rows=20,
        ),
        "",
        "Fold-by-fold scores:",
        "",
        markdown_table(fold_scores, max_rows=40),
        "",
        "## Cohort and calibration findings",
        "",
        markdown_table(
            cohort_view,
            (
                "model",
                "cohort_type",
                "cohort",
                "rows",
                "rmsle",
                "mean_target",
                "mean_prediction",
                "mean_log_residual",
                "squared_log_error_share",
            ),
            30,
        ),
        "",
        "Prediction-quantile calibration:",
        "",
        markdown_table(calibration_view, max_rows=30),
        "",
        "## Model diversity and equal log-space blends",
        "",
        "Coverage is explicit because user_id alone is not a valid temporal OOF key.",
        "",
        markdown_table(coverage, max_rows=30),
        "",
        markdown_table(
            pairwise.sort_values("blend_gain_vs_best", ascending=False)
            if not pairwise.empty
            else pairwise,
            max_rows=20,
        ),
        "",
        "## What should we try next and why",
        "",
    ]
    lines.extend(
        f"{number}. {text}"
        for number, text in enumerate(
            next_experiments(
                scores, cohorts, calibration, latest_drift, pairwise
            ),
            start=1,
        )
    )
    path = output_dir / "eda_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _save(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        return
    columns = [
        column for column in frame.columns if not str(column).startswith("_")
    ]
    frame[columns].to_csv(path, index=False)


def run(
    dataset_dir: Path,
    oof_dir: Path,
    output_dir: Path,
    *,
    fold_pattern: str = "fold_*.parquet",
    oof_pattern: str = "oof_*.csv",
    top_n: int = 100,
    calibration_bins: int = 10,
    require_folds: bool = False,
    require_oof: bool = False,
) -> Path:
    dataset_dir = dataset_dir.resolve()
    oof_dir = oof_dir.resolve()
    output_dir = output_dir.resolve()
    _clean_output(output_dir)
    status_records: list[dict[str, object]] = []

    folds, resolutions, fold_files = load_folds(
        dataset_dir, fold_pattern, status_records
    )
    models, oof_summary, oof_files = load_oof_models(
        oof_dir, oof_pattern, status_records
    )
    if require_folds and not folds:
        raise FileNotFoundError(
            f"No valid fold datasets found at {dataset_dir / fold_pattern}."
        )
    if require_oof and not models:
        raise FileNotFoundError(
            f"No valid standardized OOF found at {oof_dir / oof_pattern}."
        )
    if not folds and not models:
        raise FileNotFoundError(
            "Neither valid fold datasets nor standardized OOF files are "
            f"available. Expected {dataset_dir / fold_pattern} and "
            f"{oof_dir / oof_pattern}."
        )

    inputs = pd.DataFrame(
        [
            {"kind": "fold", "path": str(path.resolve()), "bytes": path.stat().st_size}
            for path in fold_files
        ]
        + [
            {"kind": "oof", "path": str(path.resolve()), "bytes": path.stat().st_size}
            for path in oof_files
        ]
    )
    if folds:
        target_summary, target_extremes = target_diagnostics(folds, top_n)
        temporal_drift, feature_drift, latest_drift = drift_diagnostics(folds)
    else:
        target_summary = target_extremes = pd.DataFrame()
        temporal_drift = feature_drift = latest_drift = pd.DataFrame()

    score_table, fold_score_table = model_scores(models)
    lookup = _feature_lookup(folds)
    cohort_parts: list[pd.DataFrame] = []
    threshold_parts: list[pd.DataFrame] = []
    largest_parts: list[pd.DataFrame] = []
    false_positive_parts: list[pd.DataFrame] = []
    false_negative_parts: list[pd.DataFrame] = []
    calibration_parts: list[pd.DataFrame] = []
    for model, oof in models.items():
        enriched = attach_fold_features(oof, lookup, model, status_records)
        enriched, thresholds = add_cohorts(enriched, model)
        threshold_parts.append(thresholds)
        cohort_parts.append(cohort_errors(enriched, model))
        largest, false_positive, false_negative, calibration = error_diagnostics(
            enriched, model, top_n, calibration_bins
        )
        largest_parts.append(largest)
        false_positive_parts.append(false_positive)
        false_negative_parts.append(false_negative)
        calibration_parts.append(calibration)

    def combine(parts: list[pd.DataFrame]) -> pd.DataFrame:
        valid = [part for part in parts if not part.empty]
        return pd.concat(valid, ignore_index=True) if valid else pd.DataFrame()

    cohort_table = combine(cohort_parts)
    threshold_table = combine(threshold_parts)
    largest_table = combine(largest_parts)
    false_positive_table = combine(false_positive_parts)
    false_negative_table = combine(false_negative_parts)
    calibration_table = combine(calibration_parts)
    (
        coverage,
        pairwise,
        pairwise_by_fold,
        prediction_corr,
        prediction_log_corr,
        residual_corr,
    ) = diversity_diagnostics(models)
    statuses = pd.DataFrame(status_records)

    pairwise_blends = (
        pairwise[
            [
                "model_a",
                "model_b",
                "aligned_rows",
                "rmsle_a_on_intersection",
                "rmsle_b_on_intersection",
                "equal_log_blend_rmsle",
                "blend_gain_vs_best",
            ]
        ]
        if not pairwise.empty
        else pairwise
    )
    outputs = {
        "diagnostic_status.csv": statuses,
        "input_files.csv": inputs,
        "column_resolution.csv": resolutions,
        "target_distribution.csv": target_summary,
        "target_extremes.csv": target_extremes,
        "temporal_drift.csv": temporal_drift,
        "key_feature_drift.csv": feature_drift,
        "latest_fold_drift.csv": latest_drift,
        "oof_input_summary.csv": oof_summary,
        "model_scores.csv": score_table,
        "model_fold_scores.csv": fold_score_table,
        "cohort_errors.csv": cohort_table,
        "cohort_thresholds.csv": threshold_table,
        "largest_log_errors.csv": largest_table,
        "false_positives.csv": false_positive_table,
        "false_negatives.csv": false_negative_table,
        "prediction_calibration.csv": calibration_table,
        "oof_coverage.csv": coverage,
        "pairwise_model_diagnostics.csv": pairwise,
        "pairwise_log_blends.csv": pairwise_blends,
        "pairwise_log_blends_by_fold.csv": pairwise_by_fold,
    }
    for filename, frame in outputs.items():
        _save(frame, output_dir / filename)
    if prediction_corr is not None:
        prediction_corr.to_csv(output_dir / "prediction_correlation.csv")
    if prediction_log_corr is not None:
        prediction_log_corr.to_csv(output_dir / "prediction_log_correlation.csv")
    if residual_corr is not None:
        residual_corr.to_csv(output_dir / "residual_correlation.csv")

    report = write_report(
        output_dir,
        statuses,
        target_summary,
        temporal_drift,
        latest_drift,
        score_table,
        fold_score_table,
        cohort_table,
        calibration_table,
        coverage,
        pairwise,
    )
    print(f"EDA report: {report}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Temporal EDA, cohort errors, and aligned OOF diversity."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=cfg.DATA_DIR,
        help=f"Fold dataset directory (default: {cfg.DATA_DIR}).",
    )
    parser.add_argument(
        "--oof-dir",
        type=Path,
        default=cfg.OOF_DIR,
        help=f"Standardized OOF directory (default: {cfg.OOF_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=cfg.REPORTS_DIR / "eda",
        help=(
            "Diagnostic output directory "
            f"(default: {cfg.REPORTS_DIR / 'eda'})."
        ),
    )
    parser.add_argument("--fold-pattern", default="fold_*.parquet")
    parser.add_argument("--oof-pattern", default="oof_*.csv")
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--require-folds", action="store_true")
    parser.add_argument("--require-oof", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_n < 1:
        raise ValueError("--top-n must be positive.")
    if args.calibration_bins < 2:
        raise ValueError("--calibration-bins must be at least 2.")
    run(
        args.dataset_dir,
        args.oof_dir,
        args.output_dir,
        fold_pattern=args.fold_pattern,
        oof_pattern=args.oof_pattern,
        top_n=args.top_n,
        calibration_bins=args.calibration_bins,
        require_folds=args.require_folds,
        require_oof=args.require_oof,
    )


if __name__ == "__main__":
    main()
