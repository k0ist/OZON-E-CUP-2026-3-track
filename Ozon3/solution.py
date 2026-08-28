"""Competition-grade pipeline for E-CUP 2026, track 3.

The public metric is RMSLE, therefore the model predicts ``log1p(GMV)``
directly.  Features are built as user snapshots at several historical cutoff
dates.  The final prediction is an ensemble of two LightGBM models and simple
time-series forecasts, calibrated on the latest fully observable 30-day
window.  A recent cross-sectional ensemble adds another independently
validated correction layer.

Typical usage (from the Ozon3 directory)::

    python solution.py build
    python solution.py train
    python solution.py meta

The last two commands write ready-to-upload files to ``submissions/``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

import config as cfg


SNAPSHOT_DIR = cfg.DATA_DIR / "snapshots_v2"
MODEL_DIR = cfg.DATA_DIR / "models_v2"
REPORT_PATH = cfg.DATA_DIR / "validation_report.json"

# Sums over these windows are the backbone of the model.  The 365-day window
# also allows the tree model to estimate long-term customer value and churn.
LOOKBACKS = (7, 14, 30, 60, 90, 180, 365)
BLOCK_DAYS = 30
N_BLOCKS = 6

SUM_COLS = tuple(cfg.ALL_NUMERIC_COLS)
KEY_COLS = ("gmv", "gmv_search", "gmv_cat", "to_ord", "to_cart", "searches")
BLOCK_COLS = ("gmv", "to_ord", "to_cart", "searches")
RATIO_WINDOWS = (30, 90, 365)
DECAY_HALF_LIVES = (7, 14, 30, 45, 60, 90, 120, 180, 365)

NON_FEATURE_COLS = {
    cfg.ID_COL,
    "target",
    "target_orders",
    "target_purchase_days",
    "anchor_date",
}


@dataclass(frozen=True)
class Snapshot:
    anchor: dt.date
    target_start: dt.date
    target_end: dt.date
    path: Path


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Exact competition metric, including clipping negative predictions."""
    z_true = np.log1p(np.clip(np.asarray(y_true), 0, None))
    z_pred = np.log1p(np.clip(np.asarray(y_pred), 0, None))
    return float(np.sqrt(mean_squared_error(z_true, z_pred)))


def _snapshot_path(anchor: dt.date) -> Path:
    return SNAPSHOT_DIR / f"snapshot_{anchor.isoformat()}.parquet"


def make_train_anchors(n_snapshots: int = 8, step_days: int = 30) -> list[dt.date]:
    """Return labelled anchors ending at the latest observable target window."""
    latest = cfg.HIST_END - dt.timedelta(days=cfg.TARGET_LEN_DAYS)
    anchors = [latest - dt.timedelta(days=step_days * i) for i in range(n_snapshots)]
    anchors = sorted(anchors)
    if anchors[0] - dt.timedelta(days=max(LOOKBACKS) - 1) < cfg.HIST_START:
        # Partial long windows are valid and get an explicit coverage feature.
        print("[anchors] earliest snapshots have partial 365-day history")
    return anchors


def snapshot_spec(anchor: dt.date) -> Snapshot:
    return Snapshot(
        anchor=anchor,
        target_start=anchor + dt.timedelta(days=1),
        target_end=anchor + dt.timedelta(days=cfg.TARGET_LEN_DAYS),
        path=_snapshot_path(anchor),
    )


def _days_available(start: dt.date, end: dt.date) -> int:
    lo = max(start, cfg.HIST_START)
    hi = min(end, cfg.HIST_END)
    return max(0, (hi - lo).days + 1)


def _conditional_sum(col: str, mask: pl.Expr, alias: str) -> pl.Expr:
    return (
        pl.col(col)
        .filter(mask)
        .sum()
        .fill_null(0)
        .cast(pl.Float32)
        .alias(alias)
    )


def _recency_expr(anchor: dt.date, condition: pl.Expr, alias: str) -> pl.Expr:
    return (
        (pl.lit(anchor) - pl.col(cfg.DATE_COL).filter(condition).max())
        .dt.total_days()
        .fill_null(10_000)
        .cast(pl.Float32)
        .alias(alias)
    )


def _aggregation_expressions(anchor: dt.date) -> list[pl.Expr]:
    date_col = pl.col(cfg.DATE_COL)
    expressions: list[pl.Expr] = [
        ((pl.lit(anchor) - date_col.min()).dt.total_days() + 1)
        .cast(pl.Float32)
        .alias("tenure_days"),
        (pl.lit(anchor) - date_col.max())
        .dt.total_days()
        .cast(pl.Float32)
        .alias("recency_active_days"),
        pl.len().cast(pl.Float32).alias("active_days_lifetime"),
        (pl.col("gmv") > 0).sum().cast(pl.Float32).alias("purchase_days_lifetime"),
        (pl.col("to_cart") > 0).sum().cast(pl.Float32).alias("cart_days_lifetime"),
        (pl.col("search") > 0).sum().cast(pl.Float32).alias("search_days_lifetime"),
        (pl.col("cat") > 0).sum().cast(pl.Float32).alias("catalog_days_lifetime"),
        _recency_expr(anchor, pl.col("gmv") > 0, "recency_purchase_days"),
        _recency_expr(anchor, pl.col("to_cart") > 0, "recency_cart_days"),
        _recency_expr(anchor, pl.col("search") > 0, "recency_search_days"),
        _recency_expr(anchor, pl.col("cat") > 0, "recency_catalog_days"),
        pl.col(cfg.DATE_COL)
        .filter(pl.col("gmv") > 0)
        .sort()
        .diff()
        .dt.total_days()
        .mean()
        .fill_null(-1)
        .cast(pl.Float32)
        .alias("purchase_gap_mean"),
        pl.col(cfg.DATE_COL)
        .filter(pl.col("gmv") > 0)
        .sort()
        .diff()
        .dt.total_days()
        .std()
        .fill_null(-1)
        .cast(pl.Float32)
        .alias("purchase_gap_std"),
    ]

    # Smooth alternatives to hard lookback boundaries.
    age_days = (pl.lit(anchor) - date_col).dt.total_days().cast(pl.Float64)
    for half_life in DECAY_HALF_LIVES:
        decay = (-age_days / half_life).exp()
        for col in KEY_COLS:
            expressions.append(
                (pl.col(col).cast(pl.Float64) * decay)
                .sum()
                .cast(pl.Float32)
                .alias(f"{col}_ewm_hl{half_life}")
            )

    for days in LOOKBACKS:
        start = anchor - dt.timedelta(days=days - 1)
        mask = date_col.is_between(start, anchor)
        for col in SUM_COLS:
            expressions.append(_conditional_sum(col, mask, f"{col}_sum_{days}d"))

        expressions.extend(
            [
                mask.sum().cast(pl.Float32).alias(f"active_days_{days}d"),
                (mask & (pl.col("gmv") > 0))
                .sum()
                .cast(pl.Float32)
                .alias(f"purchase_days_{days}d"),
                pl.col("gmv")
                .filter(mask)
                .mean()
                .fill_null(0)
                .cast(pl.Float32)
                .alias(f"gmv_active_mean_{days}d"),
                pl.col("gmv")
                .filter(mask)
                .max()
                .fill_null(0)
                .cast(pl.Float32)
                .alias(f"gmv_daily_max_{days}d"),
                pl.col("gmv")
                .filter(mask)
                .std()
                .fill_null(0)
                .cast(pl.Float32)
                .alias(f"gmv_daily_std_{days}d"),
            ]
        )

    # Disjoint monthly blocks preserve the sequence of recent behaviour.
    for block in range(N_BLOCKS):
        end = anchor - dt.timedelta(days=BLOCK_DAYS * block)
        start = end - dt.timedelta(days=BLOCK_DAYS - 1)
        mask = date_col.is_between(start, end)
        for col in BLOCK_COLS:
            expressions.append(_conditional_sum(col, mask, f"{col}_block_{block}"))

    # Two seasonal proxies: exact calendar alignment and exact weekday
    # alignment.  365 is calendar-like; 364 is exactly 52 weeks.
    for lag in (365, 364):
        start = anchor + dt.timedelta(days=1 - lag)
        end = anchor + dt.timedelta(days=cfg.TARGET_LEN_DAYS - lag)
        mask = date_col.is_between(start, end)
        for col in KEY_COLS:
            expressions.append(
                _conditional_sum(col, mask, f"{col}_sum_lag{lag}_target")
            )

    return expressions


def _add_derived_features(frame: pl.DataFrame, anchor: dt.date) -> pl.DataFrame:
    eps = 1.0
    expressions: list[pl.Expr] = []

    # Coverage makes partial windows explicit instead of silently treating
    # missing calendar history as genuine zero activity.
    for days in LOOKBACKS:
        start = anchor - dt.timedelta(days=days - 1)
        coverage = _days_available(start, anchor)
        expressions.append(pl.lit(coverage / days).cast(pl.Float32).alias(f"coverage_{days}d"))

    for lag in (365, 364):
        start = anchor + dt.timedelta(days=1 - lag)
        end = anchor + dt.timedelta(days=cfg.TARGET_LEN_DAYS - lag)
        coverage = _days_available(start, end)
        expressions.append(
            pl.lit(coverage / cfg.TARGET_LEN_DAYS)
            .cast(pl.Float32)
            .alias(f"coverage_lag{lag}_target")
        )

    for days in RATIO_WINDOWS:
        gmv = pl.col(f"gmv_sum_{days}d")
        orders = pl.col(f"to_ord_sum_{days}d")
        carts = pl.col(f"to_cart_sum_{days}d")
        searches = pl.col(f"searches_sum_{days}d")
        expressions.extend(
            [
                (gmv / (orders + eps)).cast(pl.Float32).alias(f"aov_{days}d"),
                (orders / (carts + eps)).cast(pl.Float32).alias(f"cart_to_order_{days}d"),
                (carts / (searches + eps)).cast(pl.Float32).alias(f"search_to_cart_{days}d"),
                (pl.col(f"gmv_search_sum_{days}d") / (gmv + eps))
                .cast(pl.Float32)
                .alias(f"search_gmv_share_{days}d"),
                (pl.col(f"active_days_{days}d") / days)
                .cast(pl.Float32)
                .alias(f"activity_density_{days}d"),
            ]
        )

    for half_life in DECAY_HALF_LIVES:
        gmv = pl.col(f"gmv_ewm_hl{half_life}")
        orders = pl.col(f"to_ord_ewm_hl{half_life}")
        carts = pl.col(f"to_cart_ewm_hl{half_life}")
        searches = pl.col(f"searches_ewm_hl{half_life}")
        expressions.extend(
            [
                (gmv / (orders + eps)).cast(pl.Float32).alias(f"aov_ewm_hl{half_life}"),
                (orders / (carts + eps))
                .cast(pl.Float32)
                .alias(f"cart_to_order_ewm_hl{half_life}"),
                (carts / (searches + eps))
                .cast(pl.Float32)
                .alias(f"search_to_cart_ewm_hl{half_life}"),
            ]
        )

    # Recent-vs-previous trends.  Log ratios are much more stable for GMV.
    for col in ("gmv", "to_ord", "to_cart", "searches"):
        recent = pl.col(f"{col}_sum_30d")
        previous = (pl.col(f"{col}_sum_60d") - recent).clip(lower_bound=0)
        earlier_monthly = (
            (pl.col(f"{col}_sum_90d") - recent).clip(lower_bound=0) / 2.0
        )
        expressions.extend(
            [
                (recent - previous).cast(pl.Float32).alias(f"{col}_momentum_30v30"),
                ((recent + eps).log() - (previous + eps).log())
                .cast(pl.Float32)
                .alias(f"{col}_logratio_30v30"),
                ((recent + eps).log() - (earlier_monthly + eps).log())
                .cast(pl.Float32)
                .alias(f"{col}_logratio_30v90"),
            ]
        )

    # Log transforms align the most skewed predictors with the target space.
    for col in ("gmv", "to_ord", "to_cart", "searches"):
        for days in LOOKBACKS:
            expressions.append(
                pl.col(f"{col}_sum_{days}d")
                .log1p()
                .cast(pl.Float32)
                .alias(f"log1p_{col}_sum_{days}d")
            )

    for lag in (365, 364):
        for col in KEY_COLS:
            expressions.append(
                pl.col(f"{col}_sum_lag{lag}_target")
                .log1p()
                .cast(pl.Float32)
                .alias(f"log1p_{col}_sum_lag{lag}_target")
            )

    for col in BLOCK_COLS:
        block_names = [f"{col}_block_{block}" for block in range(N_BLOCKS)]
        block_exprs = [pl.col(name) for name in block_names]
        block_mean = pl.mean_horizontal(block_exprs)
        expressions.extend(
            [
                block_mean.cast(pl.Float32).alias(f"{col}_block_mean"),
                pl.max_horizontal(block_exprs).cast(pl.Float32).alias(f"{col}_block_max"),
                pl.sum_horizontal([(expr > 0).cast(pl.Float32) for expr in block_exprs])
                .alias(f"{col}_nonzero_blocks"),
                (
                    pl.sum_horizontal([(expr - block_mean) ** 2 for expr in block_exprs])
                    / N_BLOCKS
                )
                .sqrt()
                .cast(pl.Float32)
                .alias(f"{col}_block_std"),
                ((block_exprs[0] + eps).log() - (pl.mean_horizontal(block_exprs[1:4]) + eps).log())
                .cast(pl.Float32)
                .alias(f"{col}_block_recent_logratio"),
            ]
        )
        for block, name in enumerate(block_names):
            expressions.append(
                pl.col(name).log1p().cast(pl.Float32).alias(f"log1p_{col}_block_{block}")
            )

    target_midpoint = anchor + dt.timedelta(days=1 + cfg.TARGET_LEN_DAYS // 2)
    angle = 2.0 * math.pi * target_midpoint.timetuple().tm_yday / 365.25
    expressions.extend(
        [
            pl.lit(math.sin(angle)).cast(pl.Float32).alias("target_doy_sin"),
            pl.lit(math.cos(angle)).cast(pl.Float32).alias("target_doy_cos"),
            pl.lit(target_midpoint.weekday()).cast(pl.Float32).alias("target_mid_weekday"),
        ]
    )

    frame = frame.with_columns(expressions)
    numeric_cols = [name for name, dtype in frame.schema.items() if dtype.is_numeric()]
    return frame.with_columns(
        [
            pl.col(name).fill_nan(0).fill_null(0)
            for name in numeric_cols
            if name != cfg.ID_COL
        ]
    )


def build_snapshot(anchor: dt.date, include_target: bool) -> Path:
    """Build one user-level snapshot using a streaming Polars aggregation."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _snapshot_path(anchor)
    t0 = time.time()
    print(f"[build] anchor={anchor} -> {out_path.name}")

    history = (
        pl.scan_parquet(cfg.DATA_PATH)
        .filter(pl.col(cfg.DATE_COL) <= pl.lit(anchor))
        .group_by(cfg.ID_COL)
        .agg(_aggregation_expressions(anchor))
    )
    features = history.collect(engine="streaming")
    features = _add_derived_features(features, anchor)
    features = features.with_columns(pl.lit(anchor).alias("anchor_date"))

    if include_target:
        target_start = anchor + dt.timedelta(days=1)
        target_end = anchor + dt.timedelta(days=cfg.TARGET_LEN_DAYS)
        target = (
            pl.scan_parquet(cfg.DATA_PATH)
            .filter(pl.col(cfg.DATE_COL).is_between(target_start, target_end))
            .group_by(cfg.ID_COL)
            .agg(
                [
                    pl.col("gmv").sum().cast(pl.Float32).alias("target"),
                    pl.col("to_ord").sum().cast(pl.Float32).alias("target_orders"),
                    (pl.col("gmv") > 0)
                    .sum()
                    .cast(pl.Float32)
                    .alias("target_purchase_days"),
                ]
            )
            .collect(engine="streaming")
        )
        features = features.join(target, on=cfg.ID_COL, how="left")
        features = features.with_columns(
            [
                pl.col("target").fill_null(0),
                pl.col("target_orders").fill_null(0),
                pl.col("target_purchase_days").fill_null(0),
            ]
        )

    features.write_parquet(out_path, compression="zstd", statistics=True)
    elapsed = time.time() - t0
    print(f"[build] {features.shape}, {elapsed:.1f}s, {out_path.stat().st_size / 2**20:.1f} MiB")
    return out_path


def build_all(n_snapshots: int = 8, step_days: int = 30, force: bool = False) -> list[Path]:
    anchors = make_train_anchors(n_snapshots=n_snapshots, step_days=step_days)
    outputs: list[Path] = []
    for anchor in anchors:
        path = _snapshot_path(anchor)
        if force or not path.exists():
            build_snapshot(anchor, include_target=True)
        else:
            print(f"[build] reuse {path.name}")
        outputs.append(path)

    test_path = _snapshot_path(cfg.HIST_END)
    if force or not test_path.exists():
        build_snapshot(cfg.HIST_END, include_target=False)
    else:
        print(f"[build] reuse {test_path.name}")
    outputs.append(test_path)

    manifest = {
        "train_anchors": [anchor.isoformat() for anchor in anchors],
        "test_anchor": cfg.HIST_END.isoformat(),
        "step_days": step_days,
        "n_snapshots": n_snapshots,
    }
    (SNAPSHOT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return outputs


def _load_manifest() -> dict:
    path = SNAPSHOT_DIR / "manifest.json"
    if not path.exists():
        raise FileNotFoundError("No snapshot manifest. Run `python solution.py build` first.")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_snapshot(anchor: dt.date) -> pd.DataFrame:
    path = _snapshot_path(anchor)
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run the build command first")
    return pd.read_parquet(path)


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    return [col for col in frame.columns if col not in NON_FEATURE_COLS]


MODEL_CONFIGS = (
    {
        "num_leaves": 31,
        "min_data_in_leaf": 250,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "lambda_l1": 0.2,
        "lambda_l2": 3.0,
        "seed": 42,
    },
    {
        "num_leaves": 63,
        "min_data_in_leaf": 500,
        "feature_fraction": 0.72,
        "bagging_fraction": 0.90,
        "lambda_l1": 0.5,
        "lambda_l2": 8.0,
        "seed": 314159,
    },
    {
        "num_leaves": 127,
        "min_data_in_leaf": 180,
        "feature_fraction": 0.68,
        "bagging_fraction": 0.80,
        "lambda_l1": 1.0,
        "lambda_l2": 12.0,
        "seed": 2026,
    },
)


def _lgb_params(config: dict) -> dict:
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.035,
        "max_depth": -1,
        "max_bin": 127,
        "bagging_freq": 1,
        "verbosity": -1,
        "num_threads": max(1, (os.cpu_count() or 4) - 1),
        "deterministic": True,
        "force_col_wise": True,
        "feature_pre_filter": False,
    }
    params.update(config)
    return params


def _concat_training(
    anchors: list[dt.date], latest_reference: dt.date
) -> tuple[pd.DataFrame, np.ndarray]:
    parts: list[pd.DataFrame] = []
    weights: list[np.ndarray] = []
    for anchor in anchors:
        part = _read_snapshot(anchor)
        age_days = (latest_reference - anchor).days
        # Recent snapshots better match the test distribution without throwing
        # away the older examples needed for robust rare-user estimates.
        weight = float(0.82 ** (age_days / 30.0))
        parts.append(part)
        weights.append(np.full(len(part), weight, dtype=np.float32))
        print(f"[train] {anchor}: rows={len(part):,}, weight={weight:.3f}")
    return pd.concat(parts, ignore_index=True), np.concatenate(weights)


def _train_models(
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame | None,
    weights: np.ndarray,
    feature_cols: list[str],
    rounds: list[int] | None = None,
) -> tuple[list[lgb.Booster], list[int]]:
    y_train_log = np.log1p(np.clip(train_frame["target"].to_numpy(), 0, None))
    dtrain = lgb.Dataset(
        train_frame[feature_cols],
        label=y_train_log,
        weight=weights,
        free_raw_data=False,
    )
    models: list[lgb.Booster] = []
    best_rounds: list[int] = []

    if valid_frame is not None:
        y_valid_log = np.log1p(np.clip(valid_frame["target"].to_numpy(), 0, None))
        dvalid = lgb.Dataset(
            valid_frame[feature_cols], label=y_valid_log, reference=dtrain, free_raw_data=False
        )
    else:
        dvalid = None

    for index, config in enumerate(MODEL_CONFIGS):
        print(f"[train] LightGBM member {index + 1}/{len(MODEL_CONFIGS)}")
        if dvalid is not None:
            model = lgb.train(
                _lgb_params(config),
                dtrain,
                num_boost_round=2500,
                valid_sets=[dvalid],
                valid_names=["valid"],
                callbacks=[lgb.early_stopping(120), lgb.log_evaluation(100)],
            )
            best = int(model.best_iteration)
        else:
            if rounds is None:
                raise ValueError("Fixed training rounds are required without a validation set")
            best = int(rounds[index])
            model = lgb.train(
                _lgb_params(config),
                dtrain,
                num_boost_round=best,
                callbacks=[lgb.log_evaluation(100)],
            )
        models.append(model)
        best_rounds.append(best)
    return models, best_rounds


def _predict_log(models: list[lgb.Booster], frame: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    predictions = [
        model.predict(frame[feature_cols], num_iteration=model.best_iteration)
        for model in models
    ]
    return np.clip(np.mean(predictions, axis=0), 0, None)


def _stack_components(frame: pd.DataFrame, model_log_prediction: np.ndarray) -> pd.DataFrame:
    components = pd.DataFrame(index=frame.index)
    components["model"] = np.clip(model_log_prediction, 0, None)
    for days in (30, 60, 90, 180):
        monthly_rate = np.clip(frame[f"gmv_sum_{days}d"].to_numpy() * 30.0 / days, 0, None)
        components[f"persistence_{days}d"] = np.log1p(monthly_rate)
    for lag in (365, 364):
        value = np.clip(frame[f"gmv_sum_lag{lag}_target"].to_numpy(), 0, None)
        components[f"seasonal_lag_{lag}"] = np.log1p(value)

    recent = frame["gmv_sum_30d"].to_numpy()
    previous = np.clip(frame["gmv_sum_60d"].to_numpy() - recent, 0, None)
    trend = np.clip(recent + 0.5 * (recent - previous), 0, None)
    components["trend_30v30"] = np.log1p(trend)
    return components.astype(np.float32)


def _fit_stacker(
    components: pd.DataFrame, target: np.ndarray, user_ids: np.ndarray
) -> tuple[Ridge, dict]:
    target_log = np.log1p(np.clip(target, 0, None))
    user_values = np.asarray(user_ids, dtype=np.int64)
    meta_train = ((user_values * 37 + 17) % 100) < 50
    alpha_grid = (0.0, 10.0, 100.0, 1000.0, 10_000.0)
    trials: list[dict] = []
    best_alpha = alpha_grid[0]
    best_score = float("inf")

    for alpha in alpha_grid:
        model = Ridge(alpha=alpha, positive=True)
        model.fit(components.loc[meta_train], target_log[meta_train])
        pred = np.clip(model.predict(components.loc[~meta_train]), 0, None)
        score = float(np.sqrt(mean_squared_error(target_log[~meta_train], pred)))
        trials.append({"alpha": alpha, "rmsle": score})
        if score < best_score:
            best_score, best_alpha = score, alpha

    stacker = Ridge(alpha=best_alpha, positive=True)
    stacker.fit(components, target_log)
    report = {
        "meta_holdout_rmsle": best_score,
        "alpha_trials": trials,
        "selected_alpha": best_alpha,
        "intercept": float(stacker.intercept_),
        "coefficients": {
            name: float(value)
            for name, value in zip(components.columns, stacker.coef_)
        },
    }
    return stacker, report


def _save_stacker(stacker: Ridge, columns: list[str], path: Path) -> None:
    payload = {
        "columns": columns,
        "coef": [float(value) for value in stacker.coef_],
        "intercept": float(stacker.intercept_),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _save_booster(model: lgb.Booster, path: Path) -> None:
    """Save through Python because LightGBM cannot open Cyrillic Windows paths."""
    path.write_text(model.model_to_string(num_iteration=model.best_iteration), encoding="utf-8")


def _apply_stacker(components: pd.DataFrame, payload: dict) -> np.ndarray:
    matrix = components[payload["columns"]].to_numpy(dtype=np.float64)
    coef = np.asarray(payload["coef"], dtype=np.float64)
    return np.clip(matrix @ coef + float(payload["intercept"]), 0, None)


def _write_final_predictions(
    final_models: list[lgb.Booster], feature_cols: list[str]
) -> None:
    """Run test inference, validate the schema, and write submission variants."""
    for index, model in enumerate(final_models):
        _save_booster(model, MODEL_DIR / f"final_model_{index}.txt")

    test = _read_snapshot(cfg.HIST_END)
    test_model_log = _predict_log(final_models, test, feature_cols)
    test_components = _stack_components(test, test_model_log)
    stack_payload = json.loads((MODEL_DIR / "stacker.json").read_text(encoding="utf-8"))
    test_stacked_log = _apply_stacker(test_components, stack_payload)

    # A conservative second candidate hedges against time drift in the stacker.
    test_conservative_log = 0.70 * test_stacked_log + 0.30 * test_model_log
    predictions = {
        "submission_v2_stacked.csv": np.expm1(test_stacked_log),
        "submission_v2_conservative.csv": np.expm1(test_conservative_log),
        "submission_v2_model_only.csv": np.expm1(test_model_log),
    }

    cfg.SUB_DIR.mkdir(parents=True, exist_ok=True)
    sample = pd.read_csv(cfg.SAMPLE_SUBMISSION_PATH, usecols=[cfg.ID_COL])
    prediction_map_index = pd.Series(np.arange(len(test)), index=test[cfg.ID_COL].to_numpy())
    positions = prediction_map_index.reindex(sample[cfg.ID_COL]).to_numpy()
    if np.isnan(positions).any():
        missing = int(np.isnan(positions).sum())
        raise ValueError(f"Test features are missing {missing} sample user IDs")
    positions = positions.astype(np.int64)

    for filename, raw_prediction in predictions.items():
        submission = sample.copy()
        submission["predict"] = np.clip(raw_prediction[positions], 0, None)
        if submission.shape != (250_000, 2) or submission["predict"].isna().any():
            raise ValueError(f"Invalid submission shape or values for {filename}")
        output_path = cfg.SUB_DIR / filename
        submission.to_csv(output_path, index=False)
        print(
            f"[submit] {output_path} | mean={submission['predict'].mean():.3f}, "
            f"median={submission['predict'].median():.3f}, max={submission['predict'].max():.3f}"
        )


def train_and_predict() -> dict:
    """Run last-window validation, refit on all labels, and create submissions."""
    manifest = _load_manifest()
    anchors = [dt.date.fromisoformat(value) for value in manifest["train_anchors"]]
    if len(anchors) < 3:
        raise ValueError("At least three snapshots are required")

    valid_anchor = anchors[-1]
    train_anchors = anchors[:-1]
    valid = _read_snapshot(valid_anchor)
    train, train_weights = _concat_training(train_anchors, latest_reference=valid_anchor)
    feature_cols = _feature_columns(train)
    print(f"[train] features={len(feature_cols)}, train={train.shape}, valid={valid.shape}")

    models, best_rounds = _train_models(
        train, valid, train_weights, feature_cols, rounds=None
    )
    valid_model_log = _predict_log(models, valid, feature_cols)
    valid_model_pred = np.expm1(valid_model_log)
    y_valid = valid["target"].to_numpy()

    components = _stack_components(valid, valid_model_log)
    stacker, stack_report = _fit_stacker(
        components, y_valid, valid[cfg.ID_COL].to_numpy()
    )
    stacked_valid_log = np.clip(stacker.predict(components), 0, None)

    metrics = {
        "valid_anchor": valid_anchor.isoformat(),
        "target_start": (valid_anchor + dt.timedelta(days=1)).isoformat(),
        "target_end": (valid_anchor + dt.timedelta(days=cfg.TARGET_LEN_DAYS)).isoformat(),
        "rows": len(valid),
        "target_nonzero_share": float((y_valid > 0).mean()),
        "model_rmsle": rmsle(y_valid, valid_model_pred),
        "stacked_full_fit_rmsle_optimistic": rmsle(y_valid, np.expm1(stacked_valid_log)),
        "persistence_30d_rmsle": rmsle(y_valid, valid["gmv_sum_30d"].to_numpy()),
        "persistence_60d_rmsle": rmsle(y_valid, valid["gmv_sum_60d"].to_numpy() / 2.0),
        "seasonal_365d_rmsle": rmsle(y_valid, valid["gmv_sum_lag365_target"].to_numpy()),
        "best_rounds": best_rounds,
        "stacker": stack_report,
    }
    print("\n=== Honest latest-window validation ===")
    for name, value in metrics.items():
        if name.endswith("rmsle"):
            print(f"{name}: {value:.6f}")
    print(f"stacker meta-holdout RMSLE: {stack_report['meta_holdout_rmsle']:.6f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for index, model in enumerate(models):
        _save_booster(model, MODEL_DIR / f"cv_model_{index}.txt")
    _save_stacker(stacker, list(components.columns), MODEL_DIR / "stacker.json")

    # Refit using the most recent labelled snapshot as well.  Fixed rounds from
    # the honest holdout avoid reusing the test-like window for early stopping.
    del train, valid, models
    gc.collect()
    full_train, full_weights = _concat_training(anchors, latest_reference=cfg.HIST_END)
    final_models, _ = _train_models(
        full_train, None, full_weights, feature_cols, rounds=best_rounds
    )
    _write_final_predictions(final_models, feature_cols)
    REPORT_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def finalize_from_saved_cv() -> None:
    """Resume after validation: refit using saved CV rounds and create CSVs."""
    manifest = _load_manifest()
    anchors = [dt.date.fromisoformat(value) for value in manifest["train_anchors"]]
    reference = _read_snapshot(anchors[-1])
    feature_cols = _feature_columns(reference)

    cv_models: list[lgb.Booster] = []
    for index in range(len(MODEL_CONFIGS)):
        path = MODEL_DIR / f"cv_model_{index}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run `python solution.py train` first")
        cv_models.append(lgb.Booster(model_str=path.read_text(encoding="utf-8")))
    rounds = [model.current_iteration() for model in cv_models]
    print(f"[finalize] reuse CV rounds: {rounds}")

    full_train, full_weights = _concat_training(anchors, latest_reference=cfg.HIST_END)
    final_models, _ = _train_models(
        full_train, None, full_weights, feature_cols, rounds=rounds
    )
    _write_final_predictions(final_models, feature_cols)


def _load_saved_boosters(prefix: str) -> list[lgb.Booster]:
    models: list[lgb.Booster] = []
    for index in range(len(MODEL_CONFIGS)):
        path = MODEL_DIR / f"{prefix}_{index}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        models.append(lgb.Booster(model_str=path.read_text(encoding="utf-8")))
    return models


def _write_log_submission(filename: str, test: pd.DataFrame, prediction_log: np.ndarray) -> Path:
    sample = pd.read_csv(cfg.SAMPLE_SUBMISSION_PATH, usecols=[cfg.ID_COL])
    prediction_by_user = pd.Series(
        np.expm1(np.clip(prediction_log, 0, None)), index=test[cfg.ID_COL].to_numpy()
    )
    submission = sample.copy()
    submission["predict"] = prediction_by_user.reindex(sample[cfg.ID_COL]).to_numpy()
    if submission.shape != (250_000, 2):
        raise ValueError(f"Unexpected submission shape: {submission.shape}")
    if submission["predict"].isna().any() or (submission["predict"] < 0).any():
        raise ValueError("Submission contains missing or negative predictions")
    path = cfg.SUB_DIR / filename
    submission.to_csv(path, index=False)
    print(
        f"[submit] {path} | mean={submission['predict'].mean():.3f}, "
        f"median={submission['predict'].median():.3f}, max={submission['predict'].max():.3f}"
    )
    return path


def train_recent_meta() -> dict:
    """Fit a recent cross-sectional model and validate on unseen users.

    The base model is time-validated.  This second layer learns the latest
    cross-sectional calibration on half of the users and is evaluated on the
    other half before being refit on all users.  It is particularly useful for
    conditional use of the annual seasonal proxy.
    """
    manifest = _load_manifest()
    anchors = [dt.date.fromisoformat(value) for value in manifest["train_anchors"]]
    valid = _read_snapshot(anchors[-1])
    test = _read_snapshot(cfg.HIST_END)

    cv_models = _load_saved_boosters("cv_model")
    final_models = _load_saved_boosters("final_model")
    core_feature_cols = cv_models[0].feature_name()
    feature_cols = _feature_columns(valid)
    valid_core_log = _predict_log(cv_models, valid, core_feature_cols)
    test_core_log = _predict_log(final_models, test, core_feature_cols)

    valid_meta = valid[feature_cols].copy()
    test_meta = test[feature_cols].copy()
    valid_meta["core_log_prediction"] = valid_core_log.astype(np.float32)
    test_meta["core_log_prediction"] = test_core_log.astype(np.float32)
    meta_features = feature_cols + ["core_log_prediction"]
    y_log = np.log1p(np.clip(valid["target"].to_numpy(), 0, None))

    # Stable hash-like split is independent of row order and target values.
    # Polars group-by output order is not guaranteed, so an RNG-by-row split
    # would make validation change after an otherwise identical feature build.
    user_values = valid[cfg.ID_COL].to_numpy(dtype=np.int64)
    meta_train_mask = ((user_values * 37 + 17) % 100) < 50
    recent_params = _lgb_params(
        {
            "num_leaves": 31,
            "min_data_in_leaf": 120,
            "feature_fraction": 0.82,
            "bagging_fraction": 0.85,
            "lambda_l1": 0.5,
            "lambda_l2": 6.0,
            "seed": 777,
        }
    )
    recent_params["learning_rate"] = 0.025
    dtrain = lgb.Dataset(
        valid_meta.loc[meta_train_mask, meta_features],
        label=y_log[meta_train_mask],
        free_raw_data=False,
    )
    dvalid = lgb.Dataset(
        valid_meta.loc[~meta_train_mask, meta_features],
        label=y_log[~meta_train_mask],
        reference=dtrain,
        free_raw_data=False,
    )
    recent_cv = lgb.train(
        recent_params,
        dtrain,
        num_boost_round=1800,
        valid_sets=[dvalid],
        valid_names=["unseen_users"],
        callbacks=[lgb.early_stopping(120), lgb.log_evaluation(100)],
    )
    recent_valid_log = np.clip(
        recent_cv.predict(
            valid_meta.loc[~meta_train_mask, meta_features],
            num_iteration=recent_cv.best_iteration,
        ),
        0,
        None,
    )
    recent_score = float(
        np.sqrt(mean_squared_error(y_log[~meta_train_mask], recent_valid_log))
    )

    recent_params_2 = _lgb_params(
        {
            "num_leaves": 63,
            "min_data_in_leaf": 220,
            "feature_fraction": 0.70,
            "bagging_fraction": 0.90,
            "lambda_l1": 1.0,
            "lambda_l2": 12.0,
            "seed": 202603,
        }
    )
    recent_params_2["learning_rate"] = 0.025
    recent_cv_2 = lgb.train(
        recent_params_2,
        dtrain,
        num_boost_round=1800,
        valid_sets=[dvalid],
        valid_names=["unseen_users_2"],
        callbacks=[lgb.early_stopping(120), lgb.log_evaluation(100)],
    )
    recent_valid_log_2 = np.clip(
        recent_cv_2.predict(
            valid_meta.loc[~meta_train_mask, meta_features],
            num_iteration=recent_cv_2.best_iteration,
        ),
        0,
        None,
    )
    recent_score_2 = float(
        np.sqrt(mean_squared_error(y_log[~meta_train_mask], recent_valid_log_2))
    )
    best_member_weight = 0.5
    best_recent_score = float("inf")
    best_recent_valid_log = recent_valid_log
    for weight in np.linspace(0.0, 1.0, 51):
        member_blend = weight * recent_valid_log + (1.0 - weight) * recent_valid_log_2
        score = float(np.sqrt(mean_squared_error(y_log[~meta_train_mask], member_blend)))
        if score < best_recent_score:
            best_member_weight = float(weight)
            best_recent_score = score
            best_recent_valid_log = member_blend

    # Fit a leakage-free affine/base calibration on the same meta-train users.
    base_components = _stack_components(valid, valid_core_log)
    base_calibrator = Ridge(alpha=1000.0, positive=True)
    base_calibrator.fit(base_components.loc[meta_train_mask], y_log[meta_train_mask])
    calibrated_base_valid = np.clip(
        base_calibrator.predict(base_components.loc[~meta_train_mask]), 0, None
    )
    base_score = float(
        np.sqrt(mean_squared_error(y_log[~meta_train_mask], calibrated_base_valid))
    )

    best_weight = 0.0
    best_blend_score = base_score
    for weight in np.linspace(0.0, 1.0, 51):
        blend = weight * best_recent_valid_log + (1.0 - weight) * calibrated_base_valid
        score = float(np.sqrt(mean_squared_error(y_log[~meta_train_mask], blend)))
        if score < best_blend_score:
            best_weight, best_blend_score = float(weight), score

    print("\n=== Recent-model unseen-user validation ===")
    print(f"calibrated base RMSLE: {base_score:.6f}")
    print(f"recent meta 1 RMSLE:   {recent_score:.6f}")
    print(f"recent meta 2 RMSLE:   {recent_score_2:.6f}")
    print(
        f"recent ensemble RMSLE: {best_recent_score:.6f} "
        f"(member-1 weight={best_member_weight:.2f})"
    )
    print(f"best blend RMSLE:      {best_blend_score:.6f} (recent weight={best_weight:.2f})")

    full_recent_data = lgb.Dataset(valid_meta[meta_features], label=y_log)
    recent_final = lgb.train(
        recent_params,
        full_recent_data,
        num_boost_round=int(recent_cv.best_iteration),
        callbacks=[lgb.log_evaluation(100)],
    )
    _save_booster(recent_final, MODEL_DIR / "recent_meta.txt")
    recent_final_2 = lgb.train(
        recent_params_2,
        full_recent_data,
        num_boost_round=int(recent_cv_2.best_iteration),
        callbacks=[lgb.log_evaluation(100)],
    )
    _save_booster(recent_final_2, MODEL_DIR / "recent_meta_2.txt")
    recent_test_log_1 = np.clip(recent_final.predict(test_meta[meta_features]), 0, None)
    recent_test_log_2 = np.clip(recent_final_2.predict(test_meta[meta_features]), 0, None)
    recent_test_log = (
        best_member_weight * recent_test_log_1
        + (1.0 - best_member_weight) * recent_test_log_2
    )

    stack_payload = json.loads((MODEL_DIR / "stacker.json").read_text(encoding="utf-8"))
    stacked_test_log = _apply_stacker(_stack_components(test, test_core_log), stack_payload)
    blend_test_log = best_weight * recent_test_log + (1.0 - best_weight) * stacked_test_log

    _write_log_submission("submission_v3_recent_meta.csv", test, recent_test_log)
    _write_log_submission("submission_v3_blend.csv", test, blend_test_log)

    report = {
        "valid_anchor": anchors[-1].isoformat(),
        "unseen_user_rows": int((~meta_train_mask).sum()),
        "recent_model_best_iteration": int(recent_cv.best_iteration),
        "recent_model_2_best_iteration": int(recent_cv_2.best_iteration),
        "calibrated_base_rmsle": base_score,
        "recent_meta_1_rmsle": recent_score,
        "recent_meta_2_rmsle": recent_score_2,
        "recent_ensemble_rmsle": best_recent_score,
        "recent_member_1_weight": best_member_weight,
        "blend_rmsle": best_blend_score,
        "recent_blend_weight": best_weight,
    }
    (cfg.DATA_DIR / "recent_meta_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build train/test snapshots")
    build_parser.add_argument("--n-snapshots", type=int, default=8)
    build_parser.add_argument("--step-days", type=int, default=30)
    build_parser.add_argument("--force", action="store_true")

    subparsers.add_parser("train", help="Validate, fit, and create submissions")
    subparsers.add_parser("finalize", help="Resume final fit from saved CV models")
    subparsers.add_parser("meta", help="Fit a recent unseen-user meta model")
    all_parser = subparsers.add_parser("all", help="Build everything, then train")
    all_parser.add_argument("--n-snapshots", type=int, default=8)
    all_parser.add_argument("--step-days", type=int, default=30)
    all_parser.add_argument("--force", action="store_true")

    args = parser.parse_args()
    if args.command in {"build", "all"}:
        build_all(
            n_snapshots=args.n_snapshots,
            step_days=args.step_days,
            force=args.force,
        )
    if args.command in {"train", "all"}:
        train_and_predict()
    elif args.command == "finalize":
        finalize_from_saved_cv()
    elif args.command == "meta":
        train_recent_meta()


if __name__ == "__main__":
    main()
