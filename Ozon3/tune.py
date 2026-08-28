"""Bounded LightGBM search with development folds and one held-out latest fold.

This intentionally replaces the old 40-trial Optuna search. Candidate selection
uses walk-forward validation folds 1..N-2; only the selected candidate is then
evaluated on the latest fold. The holdout score is reported separately and is
never folded into candidate selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg
from time_split import build_cv_folds
from train import (
    BASELINE_PARAMS,
    _predict_log,
    get_features,
    load_folds,
    resolve_artifact_layout,
    rmsle,
    train_single_model,
)


CANDIDATES = {
    "baseline": {},
    "compact_31": {
        "learning_rate": 0.025,
        "num_leaves": 31,
        "max_depth": 7,
        "min_data_in_leaf": 150,
        "feature_fraction": 0.90,
        "bagging_fraction": 0.85,
        "lambda_l2": 1.0,
    },
    "depth8_95": {
        "learning_rate": 0.025,
        "num_leaves": 95,
        "max_depth": 8,
        "min_data_in_leaf": 120,
        "feature_fraction": 0.80,
        "bagging_fraction": 0.80,
        "lambda_l2": 2.0,
    },
    "regularized_63": {
        "learning_rate": 0.02,
        "num_leaves": 63,
        "max_depth": 8,
        "min_data_in_leaf": 180,
        "feature_fraction": 0.75,
        "bagging_fraction": 0.85,
        "lambda_l1": 0.10,
        "lambda_l2": 5.0,
    },
    "wide_127": {
        "learning_rate": 0.02,
        "num_leaves": 127,
        "max_depth": 9,
        "min_data_in_leaf": 150,
        "feature_fraction": 0.70,
        "bagging_fraction": 0.80,
        "lambda_l2": 3.0,
    },
}


def _score_candidate(
    name: str,
    overrides: dict,
    folds: list[pd.DataFrame],
    feature_cols: list[str],
    val_indices: list[int],
    num_boost_round: int,
    early_stopping_rounds: int,
) -> dict:
    registry = build_cv_folds(cfg.N_FOLDS, cfg.STEP_DAYS)
    params = dict(BASELINE_PARAMS)
    params.update(overrides)
    params["device_type"] = "cpu"
    scores: list[float] = []
    iterations: list[int] = []
    fold_scores: dict[str, float] = {}

    for val_idx in val_indices:
        spec = registry[val_idx]
        train_specs = registry[:val_idx]
        max_target_end = max(item.target_end for item in train_specs)
        if max_target_end > spec.cutoff:
            raise AssertionError("Tuning temporal leakage invariant failed")
        train_df = pd.concat(folds[:val_idx], ignore_index=True)
        val_df = folds[val_idx]
        model = train_single_model(
            train_df,
            val_df,
            feature_cols,
            params,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
        )
        pred = np.expm1(_predict_log(model, val_df, feature_cols))
        score = rmsle(val_df[cfg.TARGET_COL], pred)
        scores.append(score)
        iterations.append(int(model.best_iteration or model.current_iteration()))
        fold_scores[spec.name] = score
        print(f"{name} {spec.name}: {score:.6f}")

    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))
    latest_dev = float(scores[-1])
    # Stability-aware selection; every term is visible in the report.
    selection_score = mean_score + 0.10 * std_score + 0.05 * abs(
        latest_dev - mean_score
    )
    return {
        "candidate": name,
        "overrides": overrides,
        "fold_scores": fold_scores,
        "mean_rmsle": mean_score,
        "std_rmsle": std_score,
        "latest_development_rmsle": latest_dev,
        "selection_score": selection_score,
        "best_iterations": iterations,
    }


def bounded_search(
    dataset_dir: str | Path | None = None,
    num_boost_round: int = 2500,
    early_stopping_rounds: int = 75,
    row_limit: int | None = None,
    run_name: str | None = None,
) -> dict:
    dataset_dir = Path(dataset_dir) if dataset_dir else cfg.DATA_DIR
    folds = load_folds(dataset_dir=dataset_dir, row_limit=row_limit)
    feature_cols = get_features(folds)
    if len(folds) < 4:
        raise ValueError("Need at least four snapshots for bounded tuning")

    holdout_idx = len(folds) - 1
    development_indices = list(range(1, holdout_idx))
    artifacts = resolve_artifact_layout(dataset_dir, row_limit, run_name)
    artifacts.tuning_dir.mkdir(parents=True, exist_ok=True)
    artifacts.reports_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Artifact mode={artifacts.artifact_mode} | run_name={artifacts.run_name} | "
        f"tuning_dir={artifacts.tuning_dir} | reports_dir={artifacts.reports_dir}"
    )
    results = [
        _score_candidate(
            name,
            overrides,
            folds,
            feature_cols,
            development_indices,
            num_boost_round,
            early_stopping_rounds,
        )
        for name, overrides in CANDIDATES.items()
    ]
    selected = min(results, key=lambda item: item["selection_score"])

    holdout_result = _score_candidate(
        selected["candidate"],
        selected["overrides"],
        folds,
        feature_cols,
        [holdout_idx],
        num_boost_round,
        early_stopping_rounds,
    )
    holdout_spec = build_cv_folds(cfg.N_FOLDS, cfg.STEP_DAYS)[holdout_idx]
    holdout_score = holdout_result["fold_scores"][holdout_spec.name]

    # Only parameter overrides are persisted. train.py overlays them on the
    # deterministic baseline and records the source in its manifest.
    best_params_path = artifacts.tuning_dir / "best_params.json"
    best_params_path.write_text(
        json.dumps(selected["overrides"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    flat_rows = []
    for result in results:
        row = {
            "candidate": result["candidate"],
            "mean_rmsle": result["mean_rmsle"],
            "std_rmsle": result["std_rmsle"],
            "latest_development_rmsle": result["latest_development_rmsle"],
            "selection_score": result["selection_score"],
            "selected": result["candidate"] == selected["candidate"],
        }
        row.update(result["fold_scores"])
        flat_rows.append(row)
    pd.DataFrame(flat_rows).to_csv(
        artifacts.reports_dir / "lgbm_bounded_tuning.csv", index=False
    )
    report = {
        "dataset_dir": str(dataset_dir.resolve()),
        "artifact_mode": artifacts.artifact_mode,
        "run_name": artifacts.run_name,
        "row_limit": row_limit,
        "best_params_path": str(best_params_path.resolve()),
        "selection_folds": [
            build_cv_folds(cfg.N_FOLDS, cfg.STEP_DAYS)[idx].name
            for idx in development_indices
        ],
        "held_out_fold": holdout_spec.name,
        "candidates": results,
        "selected_candidate": selected["candidate"],
        "selected_overrides": selected["overrides"],
        "held_out_latest_rmsle": holdout_score,
        "warning": (
            "The held-out score is unbiased only relative to this bounded run. "
            "Historical project experiments may already have inspected the same date."
        ),
    }
    (artifacts.reports_dir / "lgbm_bounded_tuning.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"Selected={selected['candidate']} on development folds; "
        f"held-out {holdout_spec.name} RMSLE={holdout_score:.6f}"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=cfg.DATA_DIR)
    parser.add_argument("--num-boost-round", type=int, default=2500)
    parser.add_argument("--early-stopping-rounds", type=int, default=75)
    parser.add_argument("--row-limit", type=int, default=None)
    parser.add_argument(
        "--run-name",
        default=None,
        help=(
            "Isolated artifact namespace. Custom datasets and row-limited runs "
            "are auto-isolated when this is omitted."
        ),
    )
    args = parser.parse_args()
    bounded_search(
        dataset_dir=args.dataset_dir,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
        row_limit=args.row_limit,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
