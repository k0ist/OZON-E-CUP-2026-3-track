"""Train the selected v4 models and write at most three final submissions."""

from __future__ import annotations

import gc
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

import config as cfg
import experiments
import solution


MODEL_DIR = cfg.DATA_DIR / "models_v4"
REPORT_PATH = cfg.PROJECT_DIR / "experiments" / "final_candidates.json"


def _save_model(model: lgb.Booster, name: str) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / name).write_text(
        model.model_to_string(num_iteration=model.current_iteration()),
        encoding="utf-8",
    )


def _train_fixed(
    train: pd.DataFrame,
    feature_cols: list[str],
    label: np.ndarray,
    weights: np.ndarray,
    params: dict,
    rounds: int,
    mask: np.ndarray | None = None,
) -> lgb.Booster:
    if mask is None:
        mask = np.ones(len(train), dtype=bool)
    dataset = lgb.Dataset(
        train.loc[mask, feature_cols],
        label=np.asarray(label)[mask],
        weight=np.asarray(weights)[mask],
        free_raw_data=True,
    )
    return lgb.train(
        params,
        dataset,
        num_boost_round=int(rounds),
        callbacks=[lgb.log_evaluation(100)],
    )


def _fit_oof_calibrators() -> tuple[IsotonicRegression, dict[str, dict[str, float]]]:
    frames = []
    columns = [
        cfg.ID_COL,
        "anchor_date",
        "target",
        "p_buy",
        "pred_log_tune_deep95",
    ] + experiments.SEGMENT_COLS
    for anchor in experiments.VALID_ANCHORS:
        frames.append(
            pd.read_parquet(
                experiments.OOF_DIR / f"oof_{anchor.isoformat()}.parquet",
                columns=columns,
            )
        )
    oof = pd.concat(frames, ignore_index=True)
    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    isotonic.fit(
        np.clip(oof["p_buy"].to_numpy(dtype=np.float64), 0, 1),
        (oof["target"].to_numpy() > 0).astype(np.float64),
    )

    # The shift-hedge deliberately mirrors deployment: fit calibration on the
    # latest fully labelled month, then apply it to the next month.
    latest = oof[oof["anchor_date"] == experiments.VALID_ANCHORS[-1].isoformat()].copy()
    segments = experiments._segment(latest)
    x = np.log1p(
        np.clip(latest["pred_log_tune_deep95"].to_numpy(dtype=np.float64), 0, None)
    )
    y = np.log1p(np.clip(latest["target"].to_numpy(dtype=np.float64), 0, None))
    coefficients: dict[str, dict[str, float]] = {}
    for segment in np.unique(segments):
        mask = segments == segment
        design = np.column_stack([x[mask], np.ones(mask.sum())])
        coef, *_ = np.linalg.lstsq(design, y[mask], rcond=None)
        coefficients[str(segment)] = {
            "slope": float(coef[0]),
            "intercept": float(coef[1]),
            "rows": int(mask.sum()),
        }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "calibration.json").write_text(
        json.dumps(
            {
                "isotonic_x_thresholds": isotonic.X_thresholds_.tolist(),
                "isotonic_y_thresholds": isotonic.y_thresholds_.tolist(),
                "latest_segment_affine": coefficients,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    del oof, latest, frames
    gc.collect()
    return isotonic, coefficients


def _apply_segment_affine(
    test: pd.DataFrame,
    prediction_log: np.ndarray,
    coefficients: dict[str, dict[str, float]],
) -> np.ndarray:
    result = np.asarray(prediction_log, dtype=np.float64).copy()
    segments = experiments._segment(test)
    for segment, params in coefficients.items():
        mask = segments == segment
        result[mask] = (
            float(params["slope"]) * prediction_log[mask]
            + float(params["intercept"])
        )
    return np.clip(result, 0, None)


def _write_submission(
    name: str,
    test: pd.DataFrame,
    prediction_log: np.ndarray,
) -> dict[str, float | int | str]:
    sample = pd.read_csv(cfg.SAMPLE_SUBMISSION_PATH, usecols=[cfg.ID_COL])
    values = pd.Series(
        np.expm1(np.clip(prediction_log, 0, None)),
        index=test[cfg.ID_COL].to_numpy(),
    ).reindex(sample[cfg.ID_COL])
    submission = sample.copy()
    submission["predict"] = values.to_numpy(dtype=np.float64)
    if submission.shape != (250_000, 2):
        raise ValueError(f"{name}: invalid shape {submission.shape}")
    prediction = submission["predict"].to_numpy()
    if not np.isfinite(prediction).all():
        raise ValueError(f"{name}: NaN or inf detected")
    if (prediction < 0).any():
        raise ValueError(f"{name}: negative predictions detected")
    if not submission[cfg.ID_COL].equals(sample[cfg.ID_COL]):
        raise ValueError(f"{name}: user order changed")
    cfg.SUB_DIR.mkdir(parents=True, exist_ok=True)
    path = cfg.SUB_DIR / name
    submission.to_csv(path, index=False)
    summary: dict[str, float | int | str] = {
        "path": str(path),
        "rows": int(len(submission)),
        "mean": float(prediction.mean()),
        "median": float(np.median(prediction)),
        "max": float(prediction.max()),
        "nan": int(np.isnan(prediction).sum()),
        "inf": int(np.isinf(prediction).sum()),
        "negative": int((prediction < 0).sum()),
    }
    print(f"[submit] {name}: {summary}")
    return summary


def train_final() -> None:
    isotonic, affine_coefficients = _fit_oof_calibrators()
    manifest = json.loads(
        (solution.SNAPSHOT_DIR / "manifest.json").read_text(encoding="utf-8")
    )
    anchors = [pd.Timestamp(value).date() for value in manifest["train_anchors"]]
    feature_cols = experiments._feature_cols()
    needed_cols = feature_cols + [cfg.ID_COL, "target"]
    print(f"[final] load {len(anchors)} train snapshots, features={len(feature_cols)}")
    parts = [pd.read_parquet(experiments._path(anchor), columns=needed_cols) for anchor in anchors]
    train = pd.concat(parts, ignore_index=True)
    test = pd.read_parquet(experiments._path(cfg.HIST_END), columns=feature_cols + [cfg.ID_COL])
    y = np.clip(train["target"].to_numpy(dtype=np.float64), 0, None)
    y_log = np.log1p(y)
    uniform_weights = np.ones(len(train), dtype=np.float32)
    hurdle_weights = experiments._weights_for_anchors(
        parts, anchors, cfg.HIST_END, "exp82"
    )
    latest_metadata = json.loads(
        (
            experiments.OOF_DIR
            / f"metadata_{experiments.VALID_ANCHORS[-1].isoformat()}.json"
        ).read_text(encoding="utf-8")
    )
    rounds = latest_metadata["best_iterations"]

    deep_params = dict(experiments.BASE_PARAMS)
    deep_params.update(experiments.TUNING_CONFIGS["deep95"])
    deep_params.update({"objective": "regression", "metric": "rmse", "seed": 810})
    deep = _train_fixed(
        train,
        feature_cols,
        y_log,
        uniform_weights,
        deep_params,
        rounds["log_tune_deep95"],
    )
    deep_log = np.clip(deep.predict(test[feature_cols]), 0, None)
    _save_model(deep, "deep95.txt")
    del deep
    gc.collect()

    depth_params = dict(experiments.BASE_PARAMS)
    depth_params.update(experiments.TUNING_CONFIGS["depth8"])
    depth_params.update({"objective": "regression", "metric": "rmse", "seed": 810})
    depth = _train_fixed(
        train,
        feature_cols,
        y_log,
        uniform_weights,
        depth_params,
        rounds["log_tune_depth8"],
    )
    depth_log = np.clip(depth.predict(test[feature_cols]), 0, None)
    _save_model(depth, "depth8.txt")
    del depth
    gc.collect()

    classifier_params = dict(experiments.BASE_PARAMS)
    classifier_params.update(
        {"objective": "binary", "metric": "binary_logloss", "seed": 303}
    )
    classifier = _train_fixed(
        train,
        feature_cols,
        (y > 0).astype(np.float32),
        hurdle_weights,
        classifier_params,
        rounds["hurdle_classifier"],
    )
    p_buy = np.clip(classifier.predict(test[feature_cols]), 0, 1)
    _save_model(classifier, "hurdle_classifier.txt")
    del classifier
    gc.collect()

    positive_params = dict(experiments.BASE_PARAMS)
    positive_params.update({"objective": "regression", "metric": "rmse", "seed": 304})
    positive = y > 0
    positive_model = _train_fixed(
        train,
        feature_cols,
        y_log,
        hurdle_weights,
        positive_params,
        rounds["hurdle_positive"],
        mask=positive,
    )
    conditional_log = np.clip(positive_model.predict(test[feature_cols]), 0, None)
    _save_model(positive_model, "hurdle_positive.txt")
    del positive_model, train, parts, hurdle_weights, uniform_weights, y, y_log
    gc.collect()

    calibrated_p = np.clip(isotonic.predict(p_buy), 0, 1)
    hurdle_log = np.clip(calibrated_p * conditional_log, 0, None)
    stable_log = np.mean(np.column_stack([deep_log, depth_log, hurdle_log]), axis=1)
    shift_hedge_log = _apply_segment_affine(test, deep_log, affine_coefficients)

    candidate_reports = {
        "strongest": {
            "file": "submission_v4_strongest_hurdle_iso.csv",
            "primary_oof_rmsle": 1.716553253,
            "mean_temporal_rmsle": 1.716450,
            "std_temporal_rmsle": 0.026951,
            "role": "best simple primary temporal model",
            "schema": _write_submission(
                "submission_v4_strongest_hurdle_iso.csv", test, hurdle_log
            ),
        },
        "stable": {
            "file": "submission_v4_stable_logblend.csv",
            "primary_oof_rmsle": 1.716745,
            "mean_temporal_rmsle": 1.716652,
            "std_temporal_rmsle": 0.024909,
            "role": "lower fold variance; equal OOF-justified log blend",
            "schema": _write_submission(
                "submission_v4_stable_logblend.csv", test, stable_log
            ),
        },
        "shift_hedge": {
            "file": "submission_v4_shift_hedge.csv",
            "secondary_user_xfit_oof_rmsle": 1.713029,
            "rolling_temporal_proxy_oof_rmsle": 1.717533,
            "role": "latest-period calibration hedge; highest temporal-shift risk",
            "schema": _write_submission(
                "submission_v4_shift_hedge.csv", test, shift_hedge_log
            ),
        },
    }
    REPORT_PATH.write_text(
        json.dumps(
            {
                "feature_count": len(feature_cols),
                "rounds": {
                    "deep95": rounds["log_tune_deep95"],
                    "depth8": rounds["log_tune_depth8"],
                    "hurdle_classifier": rounds["hurdle_classifier"],
                    "hurdle_positive": rounds["hurdle_positive"],
                },
                "candidates": candidate_reports,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[final] report: {REPORT_PATH}")


if __name__ == "__main__":
    train_final()
