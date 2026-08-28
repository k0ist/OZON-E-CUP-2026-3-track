"""Reproducible temporal OOF experiments for E-CUP 2026 track 3.

This module intentionally keeps model selection separate from final-test
inference.  Every calibration and blend is fitted on other temporal folds and
applied to the held-out fold.  The output is the common evidence base requested
for comparing the historical autoregression, log-target LightGBM, Tweedie,
hurdle, and frequency x monetary formulations.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import os
import re
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression

import config as cfg
import solution


OOF_DIR = cfg.DATA_DIR / "oof_v4"
EXPERIMENT_DIR = cfg.PROJECT_DIR / "experiments"
EXPERIMENT_LOG = EXPERIMENT_DIR / "experiment_log.csv"
RESULTS_PATH = EXPERIMENT_DIR / "temporal_cv_results.csv"
CORRELATION_PATH = EXPERIMENT_DIR / "oof_residual_correlations.csv"
ZERO_PATH = EXPERIMENT_DIR / "zero_target_analysis.csv"
ZERO_SEGMENT_PATH = EXPERIMENT_DIR / "zero_target_segment_analysis.csv"
CALIBRATION_PATH = EXPERIMENT_DIR / "calibration_deltas.csv"
SEGMENT_PATH = EXPERIMENT_DIR / "segment_performance.csv"
LARGEST_ERRORS_PATH = EXPERIMENT_DIR / "largest_oof_errors.csv"
BLEND_PATH = EXPERIMENT_DIR / "blend_weights.json"
ABLATION_PATH = EXPERIMENT_DIR / "feature_ablation.csv"
TUNING_PATH = EXPERIMENT_DIR / "lgb_tuning.csv"
TUNING_SELECTION_PATH = EXPERIMENT_DIR / "lgb_tuning_selection.json"

VALID_ANCHORS = (
    dt.date(2025, 10, 16),
    dt.date(2025, 11, 15),
    dt.date(2025, 12, 15),
    dt.date(2026, 1, 14),
)

META_COLS = {
    cfg.ID_COL,
    "target",
    "target_orders",
    "target_purchase_days",
    "anchor_date",
}

SEGMENT_COLS = [
    "recency_active_days",
    "recency_purchase_days",
    "tenure_days",
    "purchase_days_30d",
    "purchase_days_90d",
    "purchase_days_180d",
    "gmv_sum_30d",
    "gmv_sum_90d",
    "gmv_sum_180d",
    "to_ord_sum_30d",
    "to_ord_ewm_hl90",
]

BASE_PARAMS = {
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 250,
    "feature_fraction": 0.80,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l1": 0.5,
    "lambda_l2": 6.0,
    "max_bin": 127,
    "verbosity": -1,
    "num_threads": max(1, (os.cpu_count() or 4) - 1),
    "feature_pre_filter": False,
    "force_col_wise": True,
    "seed": cfg.RANDOM_STATE,
}

TUNING_CONFIGS = {
    "selected_base": {},
    "compact31": {
        "learning_rate": 0.04,
        "num_leaves": 31,
        "min_data_in_leaf": 350,
        "feature_fraction": 0.90,
        "bagging_fraction": 0.90,
        "lambda_l1": 0.5,
        "lambda_l2": 8.0,
        "max_bin": 127,
    },
    "regularized63": {
        "learning_rate": 0.035,
        "num_leaves": 63,
        "min_data_in_leaf": 500,
        "feature_fraction": 0.90,
        "bagging_fraction": 0.90,
        "lambda_l1": 2.0,
        "lambda_l2": 15.0,
        "max_bin": 255,
    },
    "deep95": {
        "learning_rate": 0.035,
        "num_leaves": 95,
        "min_data_in_leaf": 350,
        "feature_fraction": 0.80,
        "bagging_fraction": 0.85,
        "lambda_l1": 1.0,
        "lambda_l2": 10.0,
        "max_bin": 127,
    },
    "wide127": {
        "learning_rate": 0.03,
        "num_leaves": 127,
        "min_data_in_leaf": 450,
        "feature_fraction": 0.75,
        "bagging_fraction": 0.85,
        "lambda_l1": 2.0,
        "lambda_l2": 18.0,
        "max_bin": 255,
    },
    "depth8": {
        "learning_rate": 0.04,
        "num_leaves": 63,
        "max_depth": 8,
        "min_data_in_leaf": 300,
        "feature_fraction": 1.0,
        "bagging_fraction": 0.90,
        "lambda_l1": 0.5,
        "lambda_l2": 8.0,
        "max_bin": 255,
    },
}


def _semantic_feature_groups(feature_cols: list[str]) -> dict[str, list[str]]:
    """Return auditable, intentionally overlapping semantic feature groups."""

    def select(*patterns: str) -> list[str]:
        return [
            col
            for col in feature_cols
            if any(re.search(pattern, col) is not None for pattern in patterns)
        ]

    return {
        "recency": select(r"^recency_"),
        "frequency": select(r"to_ord", r"purchase_days", r"active_days"),
        "monetary": select(r"gmv", r"aov"),
        "funnel": select(r"search", r"cat", r"cart"),
        "momentum": select(r"momentum", r"logratio"),
        "user_trend": select(
            r"momentum", r"logratio", r"_block_[0-5]$", r"ewm_hl(7|14|30|45|60|90)"
        ),
        "purchase_cycles": select(r"purchase_gap", r"_block_", r"block_(mean|max|std)"),
        "seasonality": select(r"_lag36[45]_target", r"^target_doy_", r"target_mid_weekday"),
        "long_term_history": select(r"lifetime", r"180d", r"365d", r"hl180", r"hl365"),
        "inactivity_reactivation": select(
            r"^recency_active", r"^recency_purchase", r"tenure", r"nonzero_blocks",
            r"activity_density", r"active_days_(90|180|365)d",
        ),
        "value_concentration": select(
            r"daily_(max|std)", r"active_mean", r"block_(max|std|nonzero)",
        ),
        "ratios": select(r"ratio", r"_share", r"_density", r"^aov_", r"cart_to_order"),
    }


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_log = np.log1p(np.clip(np.asarray(y_true), 0, None))
    pred_log = np.log1p(np.clip(np.asarray(y_pred), 0, None))
    return float(np.sqrt(np.mean((y_log - pred_log) ** 2)))


def _manifest_anchors() -> list[dt.date]:
    manifest = json.loads(
        (solution.SNAPSHOT_DIR / "manifest.json").read_text(encoding="utf-8")
    )
    return [dt.date.fromisoformat(value) for value in manifest["train_anchors"]]


def _path(anchor: dt.date) -> Path:
    return solution.SNAPSHOT_DIR / f"snapshot_{anchor.isoformat()}.parquet"


def _feature_cols() -> list[str]:
    schema = pq.ParquetFile(_path(VALID_ANCHORS[-1])).schema.names
    return [col for col in schema if col not in META_COLS]


def _load_fold(anchor: dt.date, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(_path(anchor), columns=columns)


def _train_anchors(valid_anchor: dt.date) -> list[dt.date]:
    # With a 30-day stride, every earlier target ends no later than the valid
    # anchor.  Therefore its target never overlaps the held-out target.
    return [anchor for anchor in _manifest_anchors() if anchor < valid_anchor]


def _weights_for_anchors(
    train_parts: list[pd.DataFrame],
    train_anchors: list[dt.date],
    valid_anchor: dt.date,
    scheme: str,
) -> np.ndarray:
    values: list[np.ndarray] = []
    for part, anchor in zip(train_parts, train_anchors):
        age_steps = (valid_anchor - anchor).days / 30.0
        if scheme == "uniform":
            weight = 1.0
        elif scheme == "exp82":
            weight = 0.82**age_steps
        elif scheme == "exp65":
            weight = 0.65**age_steps
        elif scheme == "exp40":
            weight = 0.40**age_steps
        elif scheme == "exp25":
            weight = 0.25**age_steps
        elif scheme == "linear":
            weight = 1.0 / (1.0 + age_steps)
        else:
            raise ValueError(scheme)
        values.append(np.full(len(part), weight, dtype=np.float32))
    return np.concatenate(values)


def _train_lgb(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_cols: list[str],
    label_train: np.ndarray,
    label_valid: np.ndarray,
    weights: np.ndarray,
    params_override: dict,
    positive_train_mask: np.ndarray | None = None,
    positive_valid_mask: np.ndarray | None = None,
    rounds: int = 1400,
) -> tuple[lgb.Booster, int]:
    params = dict(BASE_PARAMS)
    params.update(params_override)
    train_mask = (
        np.ones(len(train), dtype=bool)
        if positive_train_mask is None
        else positive_train_mask
    )
    valid_mask = (
        np.ones(len(valid), dtype=bool)
        if positive_valid_mask is None
        else positive_valid_mask
    )
    dtrain = lgb.Dataset(
        train.loc[train_mask, feature_cols],
        label=np.asarray(label_train)[train_mask],
        weight=weights[train_mask],
        free_raw_data=True,
    )
    dvalid = lgb.Dataset(
        valid.loc[valid_mask, feature_cols],
        label=np.asarray(label_valid)[valid_mask],
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


def _fit_one_temporal_fold(valid_anchor: dt.date, force: bool = False) -> Path:
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OOF_DIR / f"oof_{valid_anchor.isoformat()}.parquet"
    if out_path.exists() and not force:
        print(f"[oof] reuse {out_path.name}")
        return out_path

    feature_cols = _feature_cols()
    needed_cols = list(dict.fromkeys(feature_cols + list(META_COLS)))
    train_anchors = _train_anchors(valid_anchor)
    print(f"\n[oof] valid={valid_anchor}, train={train_anchors}")
    t0 = time.time()
    parts = [_load_fold(anchor, columns=needed_cols) for anchor in train_anchors]
    valid = _load_fold(valid_anchor, columns=needed_cols)
    train = pd.concat(parts, ignore_index=True)

    y_train = np.clip(train["target"].to_numpy(dtype=np.float64), 0, None)
    y_valid = np.clip(valid["target"].to_numpy(dtype=np.float64), 0, None)
    z_train = np.log1p(y_train)
    z_valid = np.log1p(y_valid)
    orders_train = np.clip(train["target_orders"].to_numpy(dtype=np.float64), 0, None)
    orders_valid = np.clip(valid["target_orders"].to_numpy(dtype=np.float64), 0, None)

    oof = valid[[cfg.ID_COL, "target", "target_orders", "target_purchase_days"] + SEGMENT_COLS].copy()
    oof["anchor_date"] = valid_anchor.isoformat()
    oof["pred_ar30"] = np.clip(valid["gmv_sum_30d"].to_numpy(), 0, None)
    oof["pred_ar60"] = np.clip(valid["gmv_sum_60d"].to_numpy() / 2.0, 0, None)
    metadata: dict[str, object] = {
        "valid_anchor": valid_anchor.isoformat(),
        "train_anchors": [anchor.isoformat() for anchor in train_anchors],
        "n_features": len(feature_cols),
        "best_iterations": {},
    }

    # A. Direct regression on the exact competition target space.  Three
    # weighting schemes are compared on identical folds.
    for scheme in ("uniform", "exp82", "exp65", "linear"):
        weights = _weights_for_anchors(parts, train_anchors, valid_anchor, scheme)
        model, best = _train_lgb(
            train,
            valid,
            feature_cols,
            z_train,
            z_valid,
            weights,
            {"objective": "regression", "metric": "rmse", "seed": 100 + len(scheme)},
        )
        pred_log = np.clip(model.predict(valid[feature_cols]), 0, None)
        oof[f"pred_log_{scheme}"] = np.expm1(pred_log)
        metadata["best_iterations"][f"log_{scheme}"] = best
        del model, weights
        gc.collect()

    # B. Tweedie objective on raw GMV.  Early stopping still monitors the
    # competition metric via a custom evaluator.
    def tweedie_rmsle(pred: np.ndarray, dataset: lgb.Dataset):
        return "rmsle", rmsle(dataset.get_label(), pred), False

    weights = _weights_for_anchors(parts, train_anchors, valid_anchor, "exp82")
    dtrain_tw = lgb.Dataset(train[feature_cols], label=y_train, weight=weights)
    dvalid_tw = lgb.Dataset(valid[feature_cols], label=y_valid, reference=dtrain_tw)
    tw_params = dict(BASE_PARAMS)
    tw_params.update(
        {
            "objective": "tweedie",
            "tweedie_variance_power": 1.3,
            "metric": "None",
            "seed": 202,
        }
    )
    tweedie = lgb.train(
        tw_params,
        dtrain_tw,
        # The first common fold was still far behind the log-target model at
        # 500 rounds (and improved only marginally through round 1400).  Keep
        # the objective in the benchmark, but cap this dominated branch so the
        # remaining temporal folds finish in practical time.
        num_boost_round=500,
        valid_sets=[dvalid_tw],
        valid_names=["valid"],
        feval=tweedie_rmsle,
        callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(100)],
    )
    oof["pred_tweedie"] = np.clip(tweedie.predict(valid[feature_cols]), 0, None)
    metadata["best_iterations"]["tweedie"] = int(tweedie.best_iteration)
    del tweedie, dtrain_tw, dvalid_tw
    gc.collect()

    # C. Hurdle model: classifier plus positive conditional log-GMV model.
    binary_train = (y_train > 0).astype(np.float32)
    binary_valid = (y_valid > 0).astype(np.float32)
    classifier, best_clf = _train_lgb(
        train,
        valid,
        feature_cols,
        binary_train,
        binary_valid,
        weights,
        {"objective": "binary", "metric": "binary_logloss", "seed": 303},
    )
    pos_train = y_train > 0
    pos_valid = y_valid > 0
    positive_reg, best_pos = _train_lgb(
        train,
        valid,
        feature_cols,
        z_train,
        z_valid,
        weights,
        {"objective": "regression", "metric": "rmse", "seed": 304},
        positive_train_mask=pos_train,
        positive_valid_mask=pos_valid,
    )
    p_buy = np.clip(classifier.predict(valid[feature_cols]), 0, 1)
    conditional_log = np.clip(positive_reg.predict(valid[feature_cols]), 0, None)
    conditional_amount = np.expm1(conditional_log)
    oof["p_buy"] = p_buy
    oof["pred_hurdle_log"] = np.expm1(p_buy * conditional_log)
    oof["pred_hurdle_amount"] = p_buy * conditional_amount
    metadata["best_iterations"]["hurdle_classifier"] = best_clf
    metadata["best_iterations"]["hurdle_positive"] = best_pos
    del classifier, positive_reg
    gc.collect()

    # D. Frequency x monetary decomposition.
    order_log_train = np.log1p(orders_train)
    order_log_valid = np.log1p(orders_valid)
    order_model, best_orders = _train_lgb(
        train,
        valid,
        feature_cols,
        order_log_train,
        order_log_valid,
        weights,
        {"objective": "regression", "metric": "rmse", "seed": 404},
    )
    monetary_train_mask = (orders_train > 0) & (y_train > 0)
    monetary_valid_mask = (orders_valid > 0) & (y_valid > 0)
    aov_train = np.zeros_like(y_train)
    aov_valid = np.zeros_like(y_valid)
    aov_train[monetary_train_mask] = np.log1p(
        y_train[monetary_train_mask] / orders_train[monetary_train_mask]
    )
    aov_valid[monetary_valid_mask] = np.log1p(
        y_valid[monetary_valid_mask] / orders_valid[monetary_valid_mask]
    )
    aov_model, best_aov = _train_lgb(
        train,
        valid,
        feature_cols,
        aov_train,
        aov_valid,
        weights,
        {"objective": "regression", "metric": "rmse", "seed": 405},
        positive_train_mask=monetary_train_mask,
        positive_valid_mask=monetary_valid_mask,
    )
    pred_orders = np.expm1(np.clip(order_model.predict(valid[feature_cols]), 0, None))
    pred_aov = np.expm1(np.clip(aov_model.predict(valid[feature_cols]), 0, None))
    oof["pred_frequency_monetary"] = pred_orders * pred_aov
    metadata["best_iterations"]["frequency"] = best_orders
    metadata["best_iterations"]["monetary"] = best_aov
    del order_model, aov_model, train, valid, parts, weights
    gc.collect()

    oof.to_parquet(out_path, index=False)
    (OOF_DIR / f"metadata_{valid_anchor.isoformat()}.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"[oof] saved {out_path.name}, {time.time() - t0:.1f}s")
    return out_path


def run_oof(force: bool = False) -> None:
    for anchor in VALID_ANCHORS:
        _fit_one_temporal_fold(anchor, force=force)


def run_feature_ablation(weight_scheme: str = "exp65") -> None:
    """Run resumable leave-one-semantic-group-out tests on the common folds."""
    feature_cols = _feature_cols()
    groups = _semantic_feature_groups(feature_cols)
    empty = [name for name, cols in groups.items() if not cols]
    if empty:
        raise ValueError(f"Empty feature groups: {empty}")
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    if ABLATION_PATH.exists():
        records = pd.read_csv(ABLATION_PATH).to_dict("records")
    else:
        records = []
    completed = {
        (str(row["valid_anchor"]), str(row["removed_group"])) for row in records
    }

    for valid_anchor in VALID_ANCHORS:
        pending = [
            (name, cols)
            for name, cols in groups.items()
            if (valid_anchor.isoformat(), name) not in completed
        ]
        if not pending:
            print(f"[ablation] reuse all groups for {valid_anchor}")
            continue
        oof_path = OOF_DIR / f"oof_{valid_anchor.isoformat()}.parquet"
        if not oof_path.exists():
            raise FileNotFoundError(f"Missing {oof_path}; run OOF first")
        base_oof = pd.read_parquet(oof_path, columns=["target", f"pred_log_{weight_scheme}"])
        base_score = rmsle(base_oof["target"], base_oof[f"pred_log_{weight_scheme}"])
        train_anchors = _train_anchors(valid_anchor)
        needed_cols = list(dict.fromkeys(feature_cols + ["target"]))
        parts = [_load_fold(anchor, columns=needed_cols) for anchor in train_anchors]
        valid = _load_fold(valid_anchor, columns=needed_cols)
        train = pd.concat(parts, ignore_index=True)
        weights = _weights_for_anchors(parts, train_anchors, valid_anchor, weight_scheme)
        z_train = np.log1p(np.clip(train["target"].to_numpy(dtype=np.float64), 0, None))
        z_valid = np.log1p(np.clip(valid["target"].to_numpy(dtype=np.float64), 0, None))

        for group_name, removed_cols in pending:
            kept_cols = [col for col in feature_cols if col not in set(removed_cols)]
            print(
                f"[ablation] valid={valid_anchor} remove={group_name} "
                f"({len(removed_cols)} cols; keep={len(kept_cols)})"
            )
            model, best_iteration = _train_lgb(
                train,
                valid,
                kept_cols,
                z_train,
                z_valid,
                weights,
                {"objective": "regression", "metric": "rmse", "seed": 700},
                rounds=900,
            )
            prediction = np.expm1(
                np.clip(model.predict(valid[kept_cols]), 0, None)
            )
            score = rmsle(valid["target"].to_numpy(), prediction)
            records.append(
                {
                    "valid_anchor": valid_anchor.isoformat(),
                    "removed_group": group_name,
                    "removed_feature_count": len(removed_cols),
                    "kept_feature_count": len(kept_cols),
                    "weight_scheme": weight_scheme,
                    "best_iteration": best_iteration,
                    "full_model_rmsle": base_score,
                    "ablated_rmsle": score,
                    "delta_ablated_minus_full": score - base_score,
                }
            )
            pd.DataFrame(records).sort_values(
                ["removed_group", "valid_anchor"]
            ).to_csv(ABLATION_PATH, index=False)
            del model, prediction
            gc.collect()
        del train, valid, parts, weights, base_oof
        gc.collect()

    result = pd.read_csv(ABLATION_PATH)
    summary = (
        result.groupby("removed_group", as_index=False)
        .agg(
            folds=("valid_anchor", "nunique"),
            removed_feature_count=("removed_feature_count", "first"),
            mean_delta=("delta_ablated_minus_full", "mean"),
            std_delta=("delta_ablated_minus_full", "std"),
            worst_fold_delta=("delta_ablated_minus_full", "min"),
        )
        .sort_values("mean_delta", ascending=False)
    )
    print("\n=== Leave-one-group-out deltas (positive means the group helps) ===")
    print(summary.to_string(index=False))


def run_ablation_seed_control(weight_scheme: str = "uniform") -> None:
    """Correct leave-group-out deltas with a same-seed full-feature control."""
    if not ABLATION_PATH.exists():
        raise FileNotFoundError("Run feature ablation first")
    feature_cols = _feature_cols()
    control_col = "pred_log_ablation_control_seed700"
    for valid_anchor in VALID_ANCHORS:
        out_path = OOF_DIR / f"oof_{valid_anchor.isoformat()}.parquet"
        oof = pd.read_parquet(out_path)
        if control_col in oof:
            print(f"[ablation-control] reuse {valid_anchor}")
            continue
        train_anchors = _train_anchors(valid_anchor)
        needed_cols = list(dict.fromkeys(feature_cols + ["target"]))
        parts = [_load_fold(anchor, columns=needed_cols) for anchor in train_anchors]
        valid = _load_fold(valid_anchor, columns=needed_cols)
        train = pd.concat(parts, ignore_index=True)
        weights = _weights_for_anchors(parts, train_anchors, valid_anchor, weight_scheme)
        z_train = np.log1p(np.clip(train["target"].to_numpy(dtype=np.float64), 0, None))
        z_valid = np.log1p(np.clip(valid["target"].to_numpy(dtype=np.float64), 0, None))
        print(f"[ablation-control] valid={valid_anchor}, full={len(feature_cols)}, seed=700")
        model, best_iteration = _train_lgb(
            train,
            valid,
            feature_cols,
            z_train,
            z_valid,
            weights,
            {"objective": "regression", "metric": "rmse", "seed": 700},
            rounds=900,
        )
        oof[control_col] = np.expm1(
            np.clip(model.predict(valid[feature_cols]), 0, None)
        )
        temporary = out_path.with_suffix(".tmp.parquet")
        oof.to_parquet(temporary, index=False)
        temporary.replace(out_path)
        metadata_path = OOF_DIR / f"metadata_{valid_anchor.isoformat()}.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.setdefault("best_iterations", {})["ablation_control_seed700"] = best_iteration
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        del train, valid, parts, weights, model, oof
        gc.collect()

    ablation = pd.read_csv(ABLATION_PATH)
    controls = {}
    for anchor in VALID_ANCHORS:
        oof = pd.read_parquet(
            OOF_DIR / f"oof_{anchor.isoformat()}.parquet",
            columns=["target", control_col],
        )
        controls[anchor.isoformat()] = rmsle(oof["target"], oof[control_col])
    ablation["original_different_seed_full_rmsle"] = ablation["full_model_rmsle"]
    ablation["full_model_rmsle"] = ablation["valid_anchor"].map(controls)
    ablation["delta_ablated_minus_full"] = (
        ablation["ablated_rmsle"] - ablation["full_model_rmsle"]
    )
    ablation.to_csv(ABLATION_PATH, index=False)
    summary = (
        ablation.groupby("removed_group", as_index=False)
        .agg(
            folds=("valid_anchor", "nunique"),
            mean_delta=("delta_ablated_minus_full", "mean"),
            std_delta=("delta_ablated_minus_full", "std"),
            nonpositive_folds=("delta_ablated_minus_full", lambda values: int((values <= 0).sum())),
        )
        .sort_values("mean_delta", ascending=False)
    )
    print("\n=== Seed-corrected leave-one-group-out deltas ===")
    print(summary.to_string(index=False))


def run_strong_recency_weighting() -> None:
    """Add strong-decay and latest-snapshot direct models to the common OOF."""
    feature_cols = _feature_cols()
    specifications = {
        "exp40": (None, "exp40"),
        "exp25": (None, "exp25"),
        "latest1": (1, "uniform"),
        "latest2": (2, "uniform"),
    }
    for valid_anchor in VALID_ANCHORS:
        out_path = OOF_DIR / f"oof_{valid_anchor.isoformat()}.parquet"
        if not out_path.exists():
            raise FileNotFoundError(f"Missing {out_path}; run OOF first")
        oof = pd.read_parquet(out_path)
        pending = [name for name in specifications if f"pred_log_{name}" not in oof]
        if not pending:
            print(f"[strong-recency] reuse all variants for {valid_anchor}")
            continue
        all_train_anchors = _train_anchors(valid_anchor)
        needed_cols = list(dict.fromkeys(feature_cols + ["target"]))
        valid = _load_fold(valid_anchor, columns=needed_cols)
        z_valid = np.log1p(np.clip(valid["target"].to_numpy(dtype=np.float64), 0, None))
        for name in pending:
            keep_latest, scheme = specifications[name]
            train_anchors = (
                all_train_anchors
                if keep_latest is None
                else all_train_anchors[-keep_latest:]
            )
            parts = [_load_fold(anchor, columns=needed_cols) for anchor in train_anchors]
            train = pd.concat(parts, ignore_index=True)
            weights = _weights_for_anchors(parts, train_anchors, valid_anchor, scheme)
            z_train = np.log1p(
                np.clip(train["target"].to_numpy(dtype=np.float64), 0, None)
            )
            print(
                f"[strong-recency] valid={valid_anchor}, model={name}, "
                f"train_anchors={train_anchors}"
            )
            model, best_iteration = _train_lgb(
                train,
                valid,
                feature_cols,
                z_train,
                z_valid,
                weights,
                {"objective": "regression", "metric": "rmse", "seed": 620},
                rounds=1200,
            )
            oof[f"pred_log_{name}"] = np.expm1(
                np.clip(model.predict(valid[feature_cols]), 0, None)
            )
            metadata_path = OOF_DIR / f"metadata_{valid_anchor.isoformat()}.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata.setdefault("best_iterations", {})[f"log_{name}"] = best_iteration
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            temporary = out_path.with_suffix(".tmp.parquet")
            oof.to_parquet(temporary, index=False)
            temporary.replace(out_path)
            del model, train, parts, weights, z_train
            gc.collect()
        del valid, oof
        gc.collect()


def _resolve_tuning_features(drop_groups: str) -> tuple[list[str], list[str]]:
    feature_cols = _feature_cols()
    groups = _semantic_feature_groups(feature_cols)
    if drop_groups == "none":
        selected_groups: list[str] = []
    elif drop_groups == "auto":
        if not ABLATION_PATH.exists():
            raise FileNotFoundError("Run `experiments.py ablate` before auto tuning")
        ablation = pd.read_csv(ABLATION_PATH)
        summary = (
            ablation.groupby("removed_group")
            .agg(
                mean_delta=("delta_ablated_minus_full", "mean"),
                nonpositive_folds=("delta_ablated_minus_full", lambda values: int((values <= 0).sum())),
                folds=("valid_anchor", "nunique"),
            )
            .reset_index()
        )
        candidates = summary[
            (summary["mean_delta"] < -0.0002)
            & (summary["nonpositive_folds"] >= 3)
            & (summary["folds"] == len(VALID_ANCHORS))
        ].sort_values("mean_delta")
        # Semantic groups overlap; drop only the single most consistently
        # harmful group to avoid silently removing useful shared features.
        selected_groups = [] if candidates.empty else [str(candidates.iloc[0]["removed_group"])]
    else:
        selected_groups = [value.strip() for value in drop_groups.split(",") if value.strip()]
        unknown = [name for name in selected_groups if name not in groups]
        if unknown:
            raise ValueError(f"Unknown feature groups: {unknown}")
    removed = set().union(*(set(groups[name]) for name in selected_groups)) if selected_groups else set()
    return [col for col in feature_cols if col not in removed], selected_groups


def run_lgb_tuning(weight_scheme: str = "exp82", drop_groups: str = "auto") -> None:
    """Run a compact, resumable LightGBM search on stabilized features."""
    selected_cols, selected_groups = _resolve_tuning_features(drop_groups)
    TUNING_SELECTION_PATH.write_text(
        json.dumps(
            {
                "weight_scheme": weight_scheme,
                "drop_groups": selected_groups,
                "selected_feature_count": len(selected_cols),
                "configs": TUNING_CONFIGS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for valid_anchor in VALID_ANCHORS:
        out_path = OOF_DIR / f"oof_{valid_anchor.isoformat()}.parquet"
        if not out_path.exists():
            raise FileNotFoundError(f"Missing {out_path}; run OOF first")
        oof = pd.read_parquet(out_path)
        pending = [name for name in TUNING_CONFIGS if f"pred_log_tune_{name}" not in oof]
        if not pending:
            print(f"[tune] reuse all variants for {valid_anchor}")
            continue
        train_anchors = _train_anchors(valid_anchor)
        needed_cols = list(dict.fromkeys(selected_cols + ["target"]))
        parts = [_load_fold(anchor, columns=needed_cols) for anchor in train_anchors]
        valid = _load_fold(valid_anchor, columns=needed_cols)
        train = pd.concat(parts, ignore_index=True)
        weights = _weights_for_anchors(parts, train_anchors, valid_anchor, weight_scheme)
        z_train = np.log1p(np.clip(train["target"].to_numpy(dtype=np.float64), 0, None))
        z_valid = np.log1p(np.clip(valid["target"].to_numpy(dtype=np.float64), 0, None))
        for name in pending:
            params = dict(TUNING_CONFIGS[name])
            params.update({"objective": "regression", "metric": "rmse", "seed": 810})
            print(
                f"[tune] valid={valid_anchor}, config={name}, "
                f"features={len(selected_cols)}, drop={selected_groups}"
            )
            model, best_iteration = _train_lgb(
                train,
                valid,
                selected_cols,
                z_train,
                z_valid,
                weights,
                params,
                rounds=1800,
            )
            oof[f"pred_log_tune_{name}"] = np.expm1(
                np.clip(model.predict(valid[selected_cols]), 0, None)
            )
            metadata_path = OOF_DIR / f"metadata_{valid_anchor.isoformat()}.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata.setdefault("best_iterations", {})[f"log_tune_{name}"] = best_iteration
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            temporary = out_path.with_suffix(".tmp.parquet")
            oof.to_parquet(temporary, index=False)
            temporary.replace(out_path)
            del model
            gc.collect()
        del train, valid, parts, weights, oof
        gc.collect()

    all_oof = pd.concat(
        [pd.read_parquet(OOF_DIR / f"oof_{anchor.isoformat()}.parquet") for anchor in VALID_ANCHORS],
        ignore_index=True,
    )
    rows = []
    for name, params in TUNING_CONFIGS.items():
        col = f"pred_log_tune_{name}"
        scores = _fold_scores(all_oof, col)
        values = np.asarray(list(scores.values()))
        rows.append(
            {
                "config": name,
                "feature_count": len(selected_cols),
                "drop_groups": ",".join(selected_groups) or "none",
                "weight_scheme": weight_scheme,
                "hyperparameters": json.dumps(params),
                "fold_rmsle": json.dumps(scores),
                "mean_rmsle": float(values.mean()),
                "std_rmsle": float(values.std(ddof=1)),
                "oof_rmsle": rmsle(all_oof["target"], all_oof[col]),
            }
        )
    pd.DataFrame(rows).sort_values("oof_rmsle").to_csv(TUNING_PATH, index=False)
    print(pd.DataFrame(rows).sort_values("oof_rmsle").to_string(index=False))


def run_selected_hurdle(
    weight_scheme: str = "uniform",
    drop_groups: str = "inactivity_reactivation",
) -> None:
    """Recheck hurdle after feature selection instead of assuming transfer."""
    selected_cols, selected_groups = _resolve_tuning_features(drop_groups)
    for valid_anchor in VALID_ANCHORS:
        out_path = OOF_DIR / f"oof_{valid_anchor.isoformat()}.parquet"
        if not out_path.exists():
            raise FileNotFoundError(f"Missing {out_path}; run OOF first")
        oof = pd.read_parquet(out_path)
        if "pred_hurdle_selected_log" in oof:
            print(f"[selected-hurdle] reuse {valid_anchor}")
            continue
        train_anchors = _train_anchors(valid_anchor)
        needed_cols = list(dict.fromkeys(selected_cols + ["target"]))
        parts = [_load_fold(anchor, columns=needed_cols) for anchor in train_anchors]
        valid = _load_fold(valid_anchor, columns=needed_cols)
        train = pd.concat(parts, ignore_index=True)
        weights = _weights_for_anchors(parts, train_anchors, valid_anchor, weight_scheme)
        y_train = np.clip(train["target"].to_numpy(dtype=np.float64), 0, None)
        y_valid = np.clip(valid["target"].to_numpy(dtype=np.float64), 0, None)
        z_train = np.log1p(y_train)
        z_valid = np.log1p(y_valid)
        print(
            f"[selected-hurdle] valid={valid_anchor}, features={len(selected_cols)}, "
            f"drop={selected_groups}"
        )
        classifier, best_clf = _train_lgb(
            train,
            valid,
            selected_cols,
            (y_train > 0).astype(np.float32),
            (y_valid > 0).astype(np.float32),
            weights,
            {"objective": "binary", "metric": "binary_logloss", "seed": 930},
        )
        positive_train = y_train > 0
        positive_valid = y_valid > 0
        regressor, best_reg = _train_lgb(
            train,
            valid,
            selected_cols,
            z_train,
            z_valid,
            weights,
            {"objective": "regression", "metric": "rmse", "seed": 931},
            positive_train_mask=positive_train,
            positive_valid_mask=positive_valid,
        )
        p_buy = np.clip(classifier.predict(valid[selected_cols]), 0, 1)
        conditional_log = np.clip(regressor.predict(valid[selected_cols]), 0, None)
        conditional_amount = np.expm1(conditional_log)
        oof["p_buy_selected"] = p_buy
        oof["pred_hurdle_selected_log"] = np.expm1(p_buy * conditional_log)
        oof["pred_hurdle_selected_amount"] = p_buy * conditional_amount
        metadata_path = OOF_DIR / f"metadata_{valid_anchor.isoformat()}.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.setdefault("best_iterations", {})["hurdle_selected_classifier"] = best_clf
        metadata.setdefault("best_iterations", {})["hurdle_selected_positive"] = best_reg
        metadata["hurdle_selected_drop_groups"] = selected_groups
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        temporary = out_path.with_suffix(".tmp.parquet")
        oof.to_parquet(temporary, index=False)
        temporary.replace(out_path)
        del train, valid, parts, weights, classifier, regressor, oof
        gc.collect()


def _fold_scores(frame: pd.DataFrame, pred_col: str) -> dict[str, float]:
    return {
        str(anchor): rmsle(part["target"].to_numpy(), part[pred_col].to_numpy())
        for anchor, part in frame.groupby("anchor_date", sort=True)
    }


def _segment(frame: pd.DataFrame) -> np.ndarray:
    result = np.full(len(frame), "occasional", dtype=object)
    result[frame["purchase_days_180d"].to_numpy() == 0] = "no_purchase_180d"
    result[frame["recency_active_days"].to_numpy() > 30] = "inactive_30d"
    result[frame["purchase_days_90d"].to_numpy() >= 3] = "regular"
    return result


def _loo_affine(frame: pd.DataFrame, pred_col: str, segmented: bool = False) -> np.ndarray:
    result = np.zeros(len(frame), dtype=np.float64)
    folds = frame["anchor_date"].unique()
    segments = _segment(frame) if segmented else np.full(len(frame), "all", dtype=object)
    x_all = np.log1p(np.clip(frame[pred_col].to_numpy(), 0, None))
    y_all = np.log1p(np.clip(frame["target"].to_numpy(), 0, None))
    for fold in folds:
        held = frame["anchor_date"].to_numpy() == fold
        train_mask = ~held
        for segment in np.unique(segments):
            fit_mask = train_mask & (segments == segment)
            apply_mask = held & (segments == segment)
            if fit_mask.sum() < 100 or apply_mask.sum() == 0:
                continue
            design = np.column_stack([x_all[fit_mask], np.ones(fit_mask.sum())])
            coef, *_ = np.linalg.lstsq(design, y_all[fit_mask], rcond=None)
            result[apply_mask] = np.clip(
                coef[0] * x_all[apply_mask] + coef[1], 0, None
            )
    return np.expm1(result)


def _rolling_affine(frame: pd.DataFrame, pred_col: str) -> np.ndarray:
    """Fit on the immediately preceding temporal fold and apply forward."""
    x = np.log1p(np.clip(frame[pred_col].to_numpy(), 0, None))
    y = np.log1p(np.clip(frame["target"].to_numpy(), 0, None))
    fold_values = frame["anchor_date"].to_numpy()
    folds = sorted(frame["anchor_date"].unique())
    result = x.copy()
    for previous, current in zip(folds[:-1], folds[1:]):
        fit = fold_values == previous
        apply = fold_values == current
        design = np.column_stack([x[fit], np.ones(fit.sum())])
        coef, *_ = np.linalg.lstsq(design, y[fit], rcond=None)
        result[apply] = np.clip(coef[0] * x[apply] + coef[1], 0, None)
    return np.expm1(result)


def _user_crossfit_affine(
    frame: pd.DataFrame, pred_col: str, segmented: bool = False
) -> np.ndarray:
    """Cross-fit same-period calibration by anonymized user hash halves."""
    x = np.log1p(np.clip(frame[pred_col].to_numpy(), 0, None))
    y = np.log1p(np.clip(frame["target"].to_numpy(), 0, None))
    users = frame[cfg.ID_COL].to_numpy(dtype=np.int64)
    halves = ((users * 37 + 17) % 100) < 50
    folds = frame["anchor_date"].to_numpy()
    segments = _segment(frame) if segmented else np.full(len(frame), "all", dtype=object)
    result = np.zeros(len(frame), dtype=np.float64)
    for fold in frame["anchor_date"].unique():
        in_fold = folds == fold
        for held_half in (False, True):
            for segment in np.unique(segments[in_fold]):
                fit = in_fold & (halves != held_half) & (segments == segment)
                apply = in_fold & (halves == held_half) & (segments == segment)
                if fit.sum() < 100 or not np.any(apply):
                    result[apply] = x[apply]
                    continue
                design = np.column_stack([x[fit], np.ones(fit.sum())])
                coef, *_ = np.linalg.lstsq(design, y[fit], rcond=None)
                result[apply] = np.clip(coef[0] * x[apply] + coef[1], 0, None)
    return np.expm1(result)


def _loo_isotonic(frame: pd.DataFrame, pred_col: str) -> np.ndarray:
    result = np.zeros(len(frame), dtype=np.float64)
    x_all = np.log1p(np.clip(frame[pred_col].to_numpy(), 0, None))
    y_all = np.log1p(np.clip(frame["target"].to_numpy(), 0, None))
    for fold in frame["anchor_date"].unique():
        held = frame["anchor_date"].to_numpy() == fold
        model = IsotonicRegression(out_of_bounds="clip", y_min=0)
        model.fit(x_all[~held], y_all[~held])
        result[held] = model.predict(x_all[held])
    return np.expm1(np.clip(result, 0, None))


def _loo_multiplicative_scale(frame: pd.DataFrame, pred_col: str) -> np.ndarray:
    result = np.zeros(len(frame), dtype=np.float64)
    y = frame["target"].to_numpy(dtype=np.float64)
    pred = np.clip(frame[pred_col].to_numpy(dtype=np.float64), 0, None)
    fold_values = frame["anchor_date"].to_numpy()
    grid = np.linspace(0.55, 1.45, 91)
    for fold in frame["anchor_date"].unique():
        held = fold_values == fold
        scores = [rmsle(y[~held], pred[~held] * scale) for scale in grid]
        result[held] = pred[held] * grid[int(np.argmin(scores))]
    return result


def _loo_upper_clip(frame: pd.DataFrame, pred_col: str) -> np.ndarray:
    result_log = np.zeros(len(frame), dtype=np.float64)
    x = np.log1p(np.clip(frame[pred_col].to_numpy(dtype=np.float64), 0, None))
    y = np.log1p(np.clip(frame["target"].to_numpy(dtype=np.float64), 0, None))
    fold_values = frame["anchor_date"].to_numpy()
    quantiles = (0.95, 0.975, 0.99, 0.995, 0.999, 1.0)
    for fold in frame["anchor_date"].unique():
        held = fold_values == fold
        caps = [float(np.quantile(x[~held], quantile)) for quantile in quantiles]
        scores = [np.mean((y[~held] - np.minimum(x[~held], cap)) ** 2) for cap in caps]
        result_log[held] = np.minimum(x[held], caps[int(np.argmin(scores))])
    return np.expm1(result_log)


def _loo_decile_calibration(frame: pd.DataFrame, pred_col: str) -> np.ndarray:
    result_log = np.zeros(len(frame), dtype=np.float64)
    x = np.log1p(np.clip(frame[pred_col].to_numpy(dtype=np.float64), 0, None))
    y = np.log1p(np.clip(frame["target"].to_numpy(dtype=np.float64), 0, None))
    fold_values = frame["anchor_date"].to_numpy()
    for fold in frame["anchor_date"].unique():
        held = fold_values == fold
        edges = np.unique(np.quantile(x[~held], np.linspace(0, 1, 11)))
        if len(edges) < 3:
            result_log[held] = x[held]
            continue
        train_bin = np.clip(np.digitize(x[~held], edges[1:-1]), 0, len(edges) - 2)
        held_bin = np.clip(np.digitize(x[held], edges[1:-1]), 0, len(edges) - 2)
        priors = np.array(
            [
                y[~held][train_bin == index].mean()
                if np.any(train_bin == index)
                else y[~held].mean()
                for index in range(len(edges) - 1)
            ]
        )
        result_log[held] = priors[held_bin]
    return np.expm1(np.clip(result_log, 0, None))


def _loo_cohort_shrink(frame: pd.DataFrame, pred_col: str) -> np.ndarray:
    result_log = np.zeros(len(frame), dtype=np.float64)
    x = np.log1p(np.clip(frame[pred_col].to_numpy(dtype=np.float64), 0, None))
    y = np.log1p(np.clip(frame["target"].to_numpy(dtype=np.float64), 0, None))
    segments = _segment(frame)
    fold_values = frame["anchor_date"].to_numpy()
    alpha_grid = np.linspace(0.0, 1.0, 21)
    for fold in frame["anchor_date"].unique():
        held = fold_values == fold
        for segment in np.unique(segments):
            fit = (~held) & (segments == segment)
            apply = held & (segments == segment)
            if fit.sum() < 100 or not np.any(apply):
                result_log[apply] = x[apply]
                continue
            prior = float(y[fit].mean())
            scores = [np.mean((y[fit] - (alpha * x[fit] + (1 - alpha) * prior)) ** 2) for alpha in alpha_grid]
            alpha = alpha_grid[int(np.argmin(scores))]
            result_log[apply] = alpha * x[apply] + (1 - alpha) * prior
    return np.expm1(np.clip(result_log, 0, None))


def _loo_hurdle_calibrations(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    p = np.clip(frame["p_buy"].to_numpy(dtype=np.float64), 1e-6, 1.0)
    amount = np.divide(
        np.clip(frame["pred_hurdle_amount"].to_numpy(dtype=np.float64), 0, None),
        p,
    )
    conditional_log = np.log1p(amount)
    y = np.log1p(np.clip(frame["target"].to_numpy(dtype=np.float64), 0, None))
    binary = (frame["target"].to_numpy() > 0).astype(np.float64)
    fold_values = frame["anchor_date"].to_numpy()
    segments = _segment(frame)
    iso_log = np.zeros(len(frame), dtype=np.float64)
    exponent_log = np.zeros(len(frame), dtype=np.float64)
    gated_log = np.zeros(len(frame), dtype=np.float64)
    segment_iso_log = np.zeros(len(frame), dtype=np.float64)
    segment_exponent_log = np.zeros(len(frame), dtype=np.float64)
    gamma_grid = np.linspace(0.45, 2.0, 32)
    threshold_grid = np.linspace(0.0, 0.60, 31)
    for fold in frame["anchor_date"].unique():
        held = fold_values == fold
        isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        isotonic.fit(p[~held], binary[~held])
        calibrated_p = isotonic.predict(p[held])
        iso_log[held] = calibrated_p * conditional_log[held]
        gamma_scores = [
            np.mean((y[~held] - p[~held] ** gamma * conditional_log[~held]) ** 2)
            for gamma in gamma_grid
        ]
        gamma = gamma_grid[int(np.argmin(gamma_scores))]
        exponent_log[held] = p[held] ** gamma * conditional_log[held]
        threshold_scores = [
            np.mean(
                (
                    y[~held]
                    - np.where(
                        p[~held] >= threshold,
                        p[~held] * conditional_log[~held],
                        0.0,
                    )
                )
                ** 2
            )
            for threshold in threshold_grid
        ]
        threshold = threshold_grid[int(np.argmin(threshold_scores))]
        gated_log[held] = np.where(
            p[held] >= threshold, p[held] * conditional_log[held], 0.0
        )
        for segment in np.unique(segments):
            fit = (~held) & (segments == segment)
            apply = held & (segments == segment)
            if fit.sum() < 100 or not np.any(apply):
                segment_iso_log[apply] = p[apply] * conditional_log[apply]
                segment_exponent_log[apply] = p[apply] * conditional_log[apply]
                continue
            segment_isotonic = IsotonicRegression(
                out_of_bounds="clip", y_min=0, y_max=1
            )
            segment_isotonic.fit(p[fit], binary[fit])
            segment_iso_log[apply] = (
                segment_isotonic.predict(p[apply]) * conditional_log[apply]
            )
            segment_gamma_scores = [
                np.mean((y[fit] - p[fit] ** gamma * conditional_log[fit]) ** 2)
                for gamma in gamma_grid
            ]
            segment_gamma = gamma_grid[int(np.argmin(segment_gamma_scores))]
            segment_exponent_log[apply] = (
                p[apply] ** segment_gamma * conditional_log[apply]
            )
    return {
        "pred_hurdle_probability_isotonic": np.expm1(np.clip(iso_log, 0, None)),
        "pred_hurdle_probability_exponent": np.expm1(np.clip(exponent_log, 0, None)),
        "pred_hurdle_probability_gate": np.expm1(np.clip(gated_log, 0, None)),
        "pred_hurdle_segment_probability_isotonic": np.expm1(
            np.clip(segment_iso_log, 0, None)
        ),
        "pred_hurdle_segment_probability_exponent": np.expm1(
            np.clip(segment_exponent_log, 0, None)
        ),
    }


def _loo_decomposition_probability(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Convert frequency x monetary amount to expected log-GMV using P(buy)."""
    p = np.clip(frame["p_buy"].to_numpy(dtype=np.float64), 1e-6, 1.0)
    amount_log = np.log1p(
        np.clip(frame["pred_frequency_monetary"].to_numpy(dtype=np.float64), 0, None)
    )
    y = np.log1p(np.clip(frame["target"].to_numpy(dtype=np.float64), 0, None))
    binary = (frame["target"].to_numpy() > 0).astype(np.float64)
    fold_values = frame["anchor_date"].to_numpy()
    exponent_log = np.zeros(len(frame), dtype=np.float64)
    isotonic_log = np.zeros(len(frame), dtype=np.float64)
    gamma_grid = np.linspace(0.30, 1.30, 41)
    for fold in frame["anchor_date"].unique():
        held = fold_values == fold
        scores = [
            np.mean((y[~held] - p[~held] ** gamma * amount_log[~held]) ** 2)
            for gamma in gamma_grid
        ]
        gamma = gamma_grid[int(np.argmin(scores))]
        exponent_log[held] = p[held] ** gamma * amount_log[held]
        isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        isotonic.fit(p[~held], binary[~held])
        isotonic_log[held] = isotonic.predict(p[held]) * amount_log[held]
    return {
        "pred_frequency_probability_exponent": np.expm1(
            np.clip(exponent_log, 0, None)
        ),
        "pred_frequency_probability_isotonic": np.expm1(
            np.clip(isotonic_log, 0, None)
        ),
    }


def _fit_constrained_log_blend(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    n_models = x.shape[1]

    def objective(params: np.ndarray) -> float:
        weights = params[:n_models]
        intercept = params[-1]
        residual = y - (x @ weights + intercept)
        return float(np.mean(residual**2))

    initial = np.r_[np.full(n_models, 1.0 / n_models), 0.0]
    bounds = [(0.0, 1.5)] * n_models + [(-1.0, 1.0)]
    result = minimize(objective, initial, method="L-BFGS-B", bounds=bounds)
    return result.x[:n_models], float(result.x[-1])


def _loo_blend(frame: pd.DataFrame, pred_cols: list[str]) -> tuple[np.ndarray, dict]:
    result = np.zeros(len(frame), dtype=np.float64)
    x_all = np.column_stack(
        [np.log1p(np.clip(frame[col].to_numpy(), 0, None)) for col in pred_cols]
    )
    y_all = np.log1p(np.clip(frame["target"].to_numpy(), 0, None))
    fold_weights: dict[str, object] = {}
    for fold in frame["anchor_date"].unique():
        held = frame["anchor_date"].to_numpy() == fold
        weights, intercept = _fit_constrained_log_blend(x_all[~held], y_all[~held])
        result[held] = np.clip(x_all[held] @ weights + intercept, 0, None)
        fold_weights[str(fold)] = {
            "weights": dict(zip(pred_cols, map(float, weights))),
            "intercept": intercept,
        }
    final_weights, final_intercept = _fit_constrained_log_blend(x_all, y_all)
    payload = {
        "prediction_columns": pred_cols,
        "fold_weights": fold_weights,
        "final_weights": dict(zip(pred_cols, map(float, final_weights))),
        "final_intercept": final_intercept,
    }
    return np.expm1(result), payload


def _fit_constrained_raw_blend(x: np.ndarray, y_log: np.ndarray) -> np.ndarray:
    n_models = x.shape[1]

    def objective(weights: np.ndarray) -> float:
        prediction_log = np.log1p(np.clip(x @ weights, 0, None))
        return float(np.mean((y_log - prediction_log) ** 2))

    result = minimize(
        objective,
        np.full(n_models, 1.0 / n_models),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_models,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"maxiter": 100, "ftol": 1e-10},
    )
    return np.clip(result.x, 0, 1)


def _loo_raw_blend(frame: pd.DataFrame, pred_cols: list[str]) -> tuple[np.ndarray, dict]:
    result = np.zeros(len(frame), dtype=np.float64)
    x = np.column_stack(
        [np.clip(frame[col].to_numpy(dtype=np.float64), 0, None) for col in pred_cols]
    )
    y_log = np.log1p(np.clip(frame["target"].to_numpy(dtype=np.float64), 0, None))
    fold_values = frame["anchor_date"].to_numpy()
    fold_weights: dict[str, object] = {}
    for fold in frame["anchor_date"].unique():
        held = fold_values == fold
        weights = _fit_constrained_raw_blend(x[~held], y_log[~held])
        result[held] = np.clip(x[held] @ weights, 0, None)
        fold_weights[str(fold)] = dict(zip(pred_cols, map(float, weights)))
    final_weights = _fit_constrained_raw_blend(x, y_log)
    return result, {
        "prediction_columns": pred_cols,
        "fold_weights": fold_weights,
        "final_weights": dict(zip(pred_cols, map(float, final_weights))),
    }


def _append_known_feedback(rows: list[dict]) -> list[dict]:
    rows.extend(
        [
            {
                "experiment_id": "public_v3_blend_20260820",
                "model": "v3_blend",
                "feature_groups": "400 features + recent user-holdout meta",
                "target_transform": "log1p",
                "training_snapshots": "8 monthly",
                "sample_weighting": "exp82",
                "hyperparameters": "see solution.py",
                "fold_rmsle": "not available for historical run",
                "mean_rmsle": np.nan,
                "std_rmsle": np.nan,
                "oof_rmsle": 1.668124,
                "public_lb": 1.6769052187,
                "notes": "REGRESSION vs team best; local value is user-holdout, not temporal OOF",
            },
            {
                "experiment_id": "team_best_known_20260820",
                "model": "team_best_unknown_artifact",
                "feature_groups": "unknown",
                "target_transform": "unknown",
                "training_snapshots": "unknown",
                "sample_weighting": "unknown",
                "hyperparameters": "artifact absent from supplied workspace",
                "fold_rmsle": "unknown",
                "mean_rmsle": np.nan,
                "std_rmsle": np.nan,
                "oof_rmsle": np.nan,
                "public_lb": 1.65,
                "notes": "Primary benchmark. Must not be conflated with official sample without confirmation.",
            },
        ]
    )
    return rows


def _append_auxiliary_experiments(rows: list[dict]) -> list[dict]:
    if ABLATION_PATH.exists():
        ablation = pd.read_csv(ABLATION_PATH)
        for group, part in ablation.groupby("removed_group", sort=True):
            fold_scores = dict(zip(part["valid_anchor"], part["ablated_rmsle"]))
            values = part["ablated_rmsle"].to_numpy(dtype=float)
            rows.append(
                {
                    "experiment_id": f"ablation_remove_{group}",
                    "model": "log_uniform_group_ablation",
                    "feature_groups": f"full_400 minus {group}",
                    "target_transform": "log1p",
                    "training_snapshots": "rolling monthly; 4 held-out temporal folds",
                    "sample_weighting": str(part["weight_scheme"].iloc[0]),
                    "hyperparameters": "BASE_PARAMS; see experiments.py",
                    "fold_rmsle": json.dumps(fold_scores),
                    "mean_rmsle": float(values.mean()),
                    "std_rmsle": float(values.std(ddof=1)),
                    "oof_rmsle": float(np.sqrt(np.mean(values**2))),
                    "public_lb": np.nan,
                    "notes": (
                        f"leave-one-semantic-group-out; mean delta="
                        f"{part['delta_ablated_minus_full'].mean():+.6f}; semantic groups overlap"
                    ),
                }
            )
    if TUNING_PATH.exists():
        tuning = pd.read_csv(TUNING_PATH)
        for _, row in tuning.iterrows():
            rows.append(
                {
                    "experiment_id": f"lgb_tune_{row['config']}",
                    "model": f"log_tune_{row['config']}",
                    "feature_groups": f"selected_{int(row['feature_count'])}; drop={row['drop_groups']}",
                    "target_transform": "log1p",
                    "training_snapshots": "rolling monthly; 4 held-out temporal folds",
                    "sample_weighting": row["weight_scheme"],
                    "hyperparameters": row["hyperparameters"],
                    "fold_rmsle": row["fold_rmsle"],
                    "mean_rmsle": row["mean_rmsle"],
                    "std_rmsle": row["std_rmsle"],
                    "oof_rmsle": row["oof_rmsle"],
                    "public_lb": np.nan,
                    "notes": "compact post-ablation LightGBM search",
                }
            )
    return rows


def analyze_oof() -> None:
    paths = [OOF_DIR / f"oof_{anchor.isoformat()}.parquet" for anchor in VALID_ANCHORS]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing OOF files: {missing}. Run `experiments.py run`.")
    frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    pred_cols = [col for col in frame if col.startswith("pred_")]

    # Honest leave-one-temporal-fold-out calibration variants.  Select the
    # direct model using pooled OOF only; every calibrator itself is then fitted
    # without the fold to which it is applied.
    direct_cols = [col for col in frame if col.startswith("pred_log_")]
    direct_cols = [col for col in direct_cols if not any(tag in col for tag in ("affine", "isotonic", "clip", "decile", "shrink"))]
    direct_ranking = sorted(
        direct_cols,
        key=lambda col: rmsle(frame["target"].to_numpy(), frame[col].to_numpy()),
    )
    best_direct = direct_ranking[0]
    calibration_prefix = best_direct.removeprefix("pred_")
    frame[f"pred_{calibration_prefix}_affine"] = _loo_affine(frame, best_direct)
    frame[f"pred_{calibration_prefix}_segment_affine"] = _loo_affine(
        frame, best_direct, segmented=True
    )
    frame[f"pred_{calibration_prefix}_rolling_affine"] = _rolling_affine(
        frame, best_direct
    )
    frame[f"pred_{calibration_prefix}_user_xfit_affine"] = _user_crossfit_affine(
        frame, best_direct
    )
    frame[f"pred_{calibration_prefix}_user_xfit_segment_affine"] = _user_crossfit_affine(
        frame, best_direct, segmented=True
    )
    frame[f"pred_{calibration_prefix}_isotonic"] = _loo_isotonic(frame, best_direct)
    frame[f"pred_{calibration_prefix}_multiplicative"] = _loo_multiplicative_scale(
        frame, best_direct
    )
    frame[f"pred_{calibration_prefix}_upper_clip"] = _loo_upper_clip(frame, best_direct)
    frame[f"pred_{calibration_prefix}_decile"] = _loo_decile_calibration(frame, best_direct)
    frame[f"pred_{calibration_prefix}_cohort_shrink"] = _loo_cohort_shrink(frame, best_direct)
    for col, values in _loo_hurdle_calibrations(frame).items():
        frame[col] = values
    if {"p_buy_selected", "pred_hurdle_selected_amount"}.issubset(frame.columns):
        selected_view = frame[
            [cfg.ID_COL, "anchor_date", "target"] + SEGMENT_COLS
        ].copy()
        selected_view["p_buy"] = frame["p_buy_selected"].to_numpy()
        selected_view["pred_hurdle_amount"] = frame[
            "pred_hurdle_selected_amount"
        ].to_numpy()
        for col, values in _loo_hurdle_calibrations(selected_view).items():
            renamed = col.replace("pred_hurdle_", "pred_hurdle_selected_", 1)
            frame[renamed] = values
        del selected_view
    for col, values in _loo_decomposition_probability(frame).items():
        frame[col] = values

    hurdle_candidates = [
        col
        for col in frame
        if col == "pred_hurdle_log"
        or col == "pred_hurdle_selected_log"
        or col.startswith("pred_hurdle_probability_")
        or col.startswith("pred_hurdle_segment_probability_")
        or col.startswith("pred_hurdle_selected_probability_")
        or col.startswith("pred_hurdle_selected_segment_probability_")
        or col.startswith("pred_hurdle_segment_probability_")
        or col.startswith("pred_hurdle_selected_probability_")
        or col.startswith("pred_hurdle_selected_segment_probability_")
    ]
    best_hurdle = min(
        hurdle_candidates,
        key=lambda col: rmsle(frame["target"].to_numpy(), frame[col].to_numpy()),
    )

    blend_candidates = list(
        dict.fromkeys(
            direct_ranking[:2]
            + [
                best_hurdle,
                "pred_tweedie",
                "pred_frequency_probability_exponent",
                "pred_ar60",
            ]
        )
    )
    frame["pred_simple_log_blend"] = np.expm1(
        np.mean(
            np.column_stack(
                [np.log1p(np.clip(frame[col].to_numpy(), 0, None)) for col in blend_candidates[:3]]
            ),
            axis=1,
        )
    )
    frame["pred_simple_raw_blend"] = np.mean(
        np.column_stack(
            [np.clip(frame[col].to_numpy(), 0, None) for col in blend_candidates[:3]]
        ),
        axis=1,
    )
    frame["pred_oof_blend"], blend_payload = _loo_blend(frame, blend_candidates)
    frame["pred_oof_raw_blend"], raw_blend_payload = _loo_raw_blend(
        frame, blend_candidates
    )
    blend_payload["selected_best_direct"] = best_direct
    blend_payload["selected_best_hurdle"] = best_hurdle
    blend_payload["raw_space_blend"] = raw_blend_payload
    BLEND_PATH.parent.mkdir(parents=True, exist_ok=True)
    BLEND_PATH.write_text(json.dumps(blend_payload, indent=2), encoding="utf-8")
    pred_cols = [col for col in frame if col.startswith("pred_")]

    result_rows: list[dict] = []
    log_rows: list[dict] = []
    for pred_col in pred_cols:
        fold_scores = _fold_scores(frame, pred_col)
        values = np.asarray(list(fold_scores.values()))
        pooled = rmsle(frame["target"].to_numpy(), frame[pred_col].to_numpy())
        result_rows.append(
            {
                "model": pred_col.removeprefix("pred_"),
                **fold_scores,
                "latest_temporal_rmsle": fold_scores[str(VALID_ANCHORS[-1])],
                "mean_rmsle": float(values.mean()),
                "std_rmsle": float(values.std(ddof=1)),
                "oof_rmsle": pooled,
            }
        )
        log_rows.append(
            {
                "experiment_id": f"temporal_v4_{pred_col.removeprefix('pred_')}",
                "model": pred_col.removeprefix("pred_"),
                "feature_groups": "full_400" if "ar" not in pred_col else "autoregressive",
                "target_transform": (
                    "log1p" if "log" in pred_col else "raw/decomposed"
                ),
                "training_snapshots": "rolling monthly; 4 held-out temporal folds",
                "sample_weighting": next(
                    (name for name in ("uniform", "exp82", "exp65", "linear") if name in pred_col),
                    "exp82 or n/a",
                ),
                "hyperparameters": "see experiments.py",
                "fold_rmsle": json.dumps(fold_scores),
                "mean_rmsle": float(values.mean()),
                "std_rmsle": float(values.std(ddof=1)),
                "oof_rmsle": pooled,
                "public_lb": np.nan,
                "notes": (
                    "secondary same-period user-crossfit calibration; not a primary temporal selection score"
                    if "user_xfit" in pred_col
                    else "common temporal OOF"
                ),
            }
        )

    results = pd.DataFrame(result_rows).sort_values("oof_rmsle")
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)
    log_rows = _append_auxiliary_experiments(log_rows)
    pd.DataFrame(_append_known_feedback(log_rows)).to_csv(EXPERIMENT_LOG, index=False)

    base_score = rmsle(frame["target"].to_numpy(), frame[best_direct].to_numpy())
    calibration_cols = [
        col
        for col in pred_cols
        if col.startswith(f"pred_{calibration_prefix}_")
        or col.startswith("pred_hurdle_probability_")
        or col.startswith("pred_frequency_probability_")
    ]
    calibration_rows = []
    for col in calibration_cols:
        fold_scores = _fold_scores(frame, col)
        score = rmsle(frame["target"].to_numpy(), frame[col].to_numpy())
        calibration_rows.append(
            {
                "base_model": best_direct.removeprefix("pred_"),
                "method": col.removeprefix("pred_"),
                "fold_rmsle": json.dumps(fold_scores),
                "oof_rmsle": score,
                "delta_vs_best_direct": score - base_score,
            }
        )
    pd.DataFrame(calibration_rows).sort_values("oof_rmsle").to_csv(
        CALIBRATION_PATH, index=False
    )

    # Error diversity in the exact metric space.
    y_log = np.log1p(frame["target"].to_numpy())
    residuals = pd.DataFrame(
        {
            col.removeprefix("pred_"): y_log
            - np.log1p(np.clip(frame[col].to_numpy(), 0, None))
            for col in pred_cols
        }
    )
    residuals.corr().to_csv(CORRELATION_PATH)

    # Zero-target contribution and harmful false-positive tail.
    zero = frame["target"].to_numpy() == 0
    zero_rows: list[dict] = []
    for col in pred_cols:
        log_pred = np.log1p(np.clip(frame[col].to_numpy(), 0, None))
        squared_error = (y_log - log_pred) ** 2
        zero_rows.append(
            {
                "model": col.removeprefix("pred_"),
                "zero_share": float(zero.mean()),
                "zero_rows_mean_squared_log_error": float(squared_error[zero].mean()),
                "zero_rows_share_total_squared_error": float(
                    squared_error[zero].sum() / squared_error.sum()
                ),
                "zero_pred_gt_10_share": float((frame.loc[zero, col] > 10).mean()),
                "zero_pred_gt_100_share": float((frame.loc[zero, col] > 100).mean()),
                "nonzero_rows_rmsle": float(np.sqrt(squared_error[~zero].mean())),
                "top_1pct_rows_share_total_squared_error": float(
                    np.sort(squared_error)[-max(1, len(squared_error) // 100) :].sum()
                    / squared_error.sum()
                ),
            }
        )
    pd.DataFrame(zero_rows).sort_values(
        "zero_rows_mean_squared_log_error"
    ).to_csv(ZERO_PATH, index=False)

    segment_values = _segment(frame)
    zero_segment_rows: list[dict] = []
    for col in pred_cols:
        log_pred = np.log1p(np.clip(frame[col].to_numpy(), 0, None))
        for segment in np.unique(segment_values):
            mask = zero & (segment_values == segment)
            if not np.any(mask):
                continue
            zero_segment_rows.append(
                {
                    "model": col.removeprefix("pred_"),
                    "segment": segment,
                    "zero_rows": int(mask.sum()),
                    "mean_false_positive_prediction": float(frame.loc[mask, col].mean()),
                    "median_false_positive_prediction": float(frame.loc[mask, col].median()),
                    "zero_rows_mean_squared_log_error": float(np.mean(log_pred[mask] ** 2)),
                    "zero_pred_gt_10_share": float((frame.loc[mask, col] > 10).mean()),
                    "zero_pred_gt_100_share": float((frame.loc[mask, col] > 100).mean()),
                }
            )
    pd.DataFrame(zero_segment_rows).sort_values(
        ["model", "zero_rows_mean_squared_log_error"], ascending=[True, False]
    ).to_csv(ZERO_SEGMENT_PATH, index=False)

    segment_rows: list[dict] = []
    for col in pred_cols:
        for segment in np.unique(segment_values):
            mask = segment_values == segment
            segment_rows.append(
                {
                    "model": col.removeprefix("pred_"),
                    "segment": segment,
                    "rows": int(mask.sum()),
                    "target_zero_share": float(zero[mask].mean()),
                    "rmsle": rmsle(frame.loc[mask, "target"], frame.loc[mask, col]),
                    "mean_prediction": float(frame.loc[mask, col].mean()),
                    "mean_target": float(frame.loc[mask, "target"].mean()),
                }
            )
    pd.DataFrame(segment_rows).sort_values(["model", "rmsle"], ascending=[True, False]).to_csv(
        SEGMENT_PATH, index=False
    )

    # Save the exact rows driving the metric for manual diagnosis.  Restrict to
    # the best direct, hurdle and selected blend so the artifact stays compact.
    diagnostic_models = list(
        dict.fromkeys([best_direct, "pred_hurdle_log", "pred_oof_blend"])
    )
    largest_parts = []
    for col in diagnostic_models:
        log_error = np.abs(
            np.log1p(np.clip(frame["target"].to_numpy(), 0, None))
            - np.log1p(np.clip(frame[col].to_numpy(), 0, None))
        )
        take = np.argpartition(log_error, -1000)[-1000:]
        part = frame.iloc[take][
            [cfg.ID_COL, "anchor_date", "target", "p_buy"] + SEGMENT_COLS
        ].copy()
        part["model"] = col.removeprefix("pred_")
        part["prediction"] = frame[col].to_numpy()[take]
        part["absolute_log_error"] = log_error[take]
        part["segment"] = segment_values[take]
        largest_parts.append(part)
    pd.concat(largest_parts, ignore_index=True).sort_values(
        "absolute_log_error", ascending=False
    ).to_csv(LARGEST_ERRORS_PATH, index=False)

    frame.to_parquet(OOF_DIR / "oof_all_analyzed.parquet", index=False)
    print("\n=== Temporal CV results ===")
    print(results.to_string(index=False))
    print(f"\nSaved: {RESULTS_PATH}, {EXPERIMENT_LOG}, {ZERO_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--force", action="store_true")
    sub.add_parser("analyze")
    sub.add_parser("strong-recency")
    ablation_control_parser = sub.add_parser("ablation-control")
    ablation_control_parser.add_argument(
        "--weight-scheme",
        choices=("uniform", "exp82", "exp65", "exp40", "exp25", "linear"),
        default="uniform",
    )
    ablation_parser = sub.add_parser("ablate")
    ablation_parser.add_argument(
        "--weight-scheme",
        choices=("uniform", "exp82", "exp65", "exp40", "exp25", "linear"),
        default="exp65",
    )
    tuning_parser = sub.add_parser("tune")
    tuning_parser.add_argument(
        "--weight-scheme",
        choices=("uniform", "exp82", "exp65", "exp40", "exp25", "linear"),
        default="exp82",
    )
    tuning_parser.add_argument(
        "--drop-groups",
        default="auto",
        help="auto, none, or a comma-separated list of semantic groups",
    )
    selected_hurdle_parser = sub.add_parser("selected-hurdle")
    selected_hurdle_parser.add_argument(
        "--weight-scheme",
        choices=("uniform", "exp82", "exp65", "exp40", "exp25", "linear"),
        default="uniform",
    )
    selected_hurdle_parser.add_argument(
        "--drop-groups", default="inactivity_reactivation"
    )
    args = parser.parse_args()
    if args.command == "run":
        run_oof(force=args.force)
    elif args.command == "analyze":
        analyze_oof()
    elif args.command == "ablate":
        run_feature_ablation(weight_scheme=args.weight_scheme)
    elif args.command == "strong-recency":
        run_strong_recency_weighting()
    elif args.command == "ablation-control":
        run_ablation_seed_control(weight_scheme=args.weight_scheme)
    elif args.command == "tune":
        run_lgb_tuning(
            weight_scheme=args.weight_scheme, drop_groups=args.drop_groups
        )
    elif args.command == "selected-hurdle":
        run_selected_hurdle(
            weight_scheme=args.weight_scheme, drop_groups=args.drop_groups
        )


if __name__ == "__main__":
    main()
