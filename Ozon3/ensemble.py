"""Aligned temporal-OOF diagnostics and log-space ensemble optimization.

This script never creates a submission.  Its only inputs are standardized OOF
files and its outputs are auditable reports plus a manifest containing the
weights that may later be used by deterministic final inference. Ensemble
validation uses expanding temporal meta-CV; pooled all-OOF optimization is
reported only as an in-sample diagnostic and deployment fit.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


KEY_COLUMNS = ["user_id", "fold", "cutoff_date"]
REQUIRED_COLUMNS = KEY_COLUMNS + ["target", "pred"]
DEFAULT_OOF_FILES = {
    "lgbm": "oof_lgbm.csv",
    "catboost": "oof_catboost.csv",
    "nn": "oof_nn.csv",
}
WEIGHT_PRUNE_THRESHOLD = 1e-8


def rmsle(y_true, y_pred) -> float:
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise ValueError("RMSLE received NaN or infinity")
    if (true < 0).any() or (pred < 0).any():
        raise ValueError("RMSLE requires non-negative targets and predictions")
    return float(np.sqrt(np.mean(np.square(np.log1p(pred) - np.log1p(true)))))


def _require_scipy_minimize():
    try:
        from scipy.optimize import minimize
    except ModuleNotFoundError as exc:
        if exc.name == "scipy":
            raise RuntimeError(
                "SciPy is required for constrained OOF weight optimization. "
                "Install it with your project environment before running ensemble.py."
            ) from exc
        raise
    return minimize


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_record(path: Path) -> dict[str, Any]:
    """Return a stable content binding for an input artifact."""

    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "bytes": int(stat.st_size),
    }


def _parse_source(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("OOF source must have the form model=path.csv")
    name, raw_path = value.split("=", 1)
    name = name.strip().lower()
    if not re.fullmatch(r"[a-z0-9_\-]+", name):
        raise argparse.ArgumentTypeError(f"Invalid model name: {name!r}")
    path = Path(raw_path.strip())
    if not raw_path.strip():
        raise argparse.ArgumentTypeError("OOF path is empty")
    return name, path


def _prune_weights(
    names: Sequence[str],
    weights: np.ndarray,
    threshold: float = WEIGHT_PRUNE_THRESHOLD,
) -> tuple[np.ndarray, dict[str, float], list[str]]:
    """Zero numerical dust, renormalize, and omit inactive models from JSON."""

    cleaned = np.asarray(weights, dtype=np.float64).copy()
    if len(cleaned) != len(names):
        raise ValueError("Weight vector and model names have different lengths")
    cleaned[cleaned < threshold] = 0.0
    if not np.any(cleaned > 0):
        cleaned[int(np.argmax(weights))] = 1.0
    cleaned /= float(cleaned.sum())
    active = {
        str(name): float(weight)
        for name, weight in zip(names, cleaned)
        if weight > 0.0
    }
    dropped = [str(name) for name, weight in zip(names, cleaned) if weight == 0.0]
    return cleaned, active, dropped


def _trainer_manifest_record(
    manifest_path: Path,
    source_path: Path,
) -> dict[str, Any]:
    """Hash a trainer manifest and record any checkable OOF-path claim."""

    record = _artifact_record(manifest_path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid trainer manifest {manifest_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Trainer manifest must contain a JSON object: {manifest_path}")

    declared_oof = payload.get("oof_path")
    declared_resolved: Path | None = None
    if declared_oof:
        declared_resolved = Path(str(declared_oof)).expanduser().resolve()
    record.update(
        {
            "model": payload.get("model") or payload.get("pipeline"),
            "created_utc": payload.get("created_at_utc") or payload.get("created_utc"),
            "row_limit": payload.get("row_limit"),
            "declared_oof_path": (
                str(declared_resolved) if declared_resolved is not None else None
            ),
            "declared_oof_matches_source": (
                str(declared_resolved).casefold() == str(source_path.resolve()).casefold()
                if declared_resolved is not None
                else None
            ),
        }
    )
    return record


def _normalize_oof(name: str, path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{name} OOF file not found: {path}")
    frame = pd.read_csv(path)
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(
            f"{name} OOF is legacy/incomplete; missing {missing}. "
            "Rerun the corresponding trainer to produce standardized OOF."
        )
    frame = frame.loc[:, REQUIRED_COLUMNS].copy()
    user_ids = pd.to_numeric(frame["user_id"], errors="raise").to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(user_ids).all() or not np.equal(user_ids, np.floor(user_ids)).all():
        raise ValueError(f"{name} OOF contains invalid user_id values")
    frame["user_id"] = user_ids.astype(np.int64)
    frame["cutoff_date"] = pd.to_datetime(
        frame["cutoff_date"], errors="raise"
    ).dt.strftime("%Y-%m-%d")
    frame["fold"] = frame["fold"].astype(str)
    frame["target"] = pd.to_numeric(frame["target"], errors="raise")
    frame["pred"] = pd.to_numeric(frame["pred"], errors="raise")
    values = frame[["target", "pred"]].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} OOF contains NaN or infinity")
    if (values < 0).any():
        raise ValueError(f"{name} OOF contains negative target/prediction values")
    duplicated = frame.duplicated(KEY_COLUMNS, keep=False)
    if duplicated.any():
        example = frame.loc[duplicated, KEY_COLUMNS].head().to_dict("records")
        raise ValueError(f"{name} OOF has duplicate composite keys: {example}")
    return frame.sort_values(KEY_COLUMNS, kind="stable").reset_index(drop=True)


def _align_exact(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Inner-align after proving every model has exactly the same key coverage."""

    names = list(frames)
    if len(names) < 2:
        raise ValueError("At least two OOF models are required for an ensemble")

    reference_name = names[0]
    reference = frames[reference_name]
    aligned = reference.rename(
        columns={"target": "target", "pred": f"pred__{reference_name}"}
    )
    for name in names[1:]:
        current = frames[name]
        coverage = reference[KEY_COLUMNS].merge(
            current[KEY_COLUMNS],
            on=KEY_COLUMNS,
            how="outer",
            indicator=True,
            validate="one_to_one",
        )
        mismatch = coverage["_merge"] != "both"
        if mismatch.any():
            counts = coverage.loc[mismatch, "_merge"].value_counts().to_dict()
            sample = coverage.loc[mismatch, KEY_COLUMNS + ["_merge"]].head().to_dict(
                "records"
            )
            raise ValueError(
                f"OOF coverage mismatch: {reference_name} vs {name}; "
                f"counts={counts}, sample={sample}"
            )

        aligned = aligned.merge(
            current.rename(
                columns={"target": f"target__{name}", "pred": f"pred__{name}"}
            ),
            on=KEY_COLUMNS,
            how="inner",
            validate="one_to_one",
        )
        other_target = aligned.pop(f"target__{name}").to_numpy(dtype=np.float64)
        reference_target = aligned["target"].to_numpy(dtype=np.float64)
        if not np.allclose(reference_target, other_target, rtol=1e-7, atol=1e-6):
            max_delta = float(np.max(np.abs(reference_target - other_target)))
            raise ValueError(
                f"Target mismatch after alignment for {name}; max_abs_delta={max_delta}"
            )

    if len(aligned) != len(reference):
        raise AssertionError("Inner alignment unexpectedly changed row coverage")
    return aligned.sort_values(KEY_COLUMNS, kind="stable").reset_index(drop=True)


def _optimize_weights(pred_log: np.ndarray, target_log: np.ndarray) -> np.ndarray:
    minimize = _require_scipy_minimize()
    model_count = pred_log.shape[1]
    initial = np.full(model_count, 1.0 / model_count, dtype=np.float64)

    def objective(weights: np.ndarray) -> float:
        residual = pred_log @ weights - target_log
        return float(np.mean(np.square(residual)))

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * model_count,
        constraints=[{"type": "eq", "fun": lambda weights: weights.sum() - 1.0}],
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"OOF weight optimization failed: {result.message}")
    weights = np.clip(np.asarray(result.x, dtype=np.float64), 0.0, 1.0)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0:
        raise RuntimeError("OOF optimizer returned invalid weights")
    return weights / total


def _fold_records(
    aligned: pd.DataFrame,
    predictions: np.ndarray,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    work = aligned[KEY_COLUMNS + ["target"]].copy()
    work["_prediction"] = predictions
    for (cutoff, fold), part in work.groupby(
        ["cutoff_date", "fold"], sort=True, observed=True
    ):
        records.append(
            {
                "fold": str(fold),
                "cutoff_date": str(cutoff),
                "rows": int(len(part)),
                "rmsle": rmsle(part["target"], part["_prediction"]),
            }
        )
    return records


def _correlation_dict(frame: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    correlation = frame.corr(method="pearson")
    result: dict[str, dict[str, float | None]] = {}
    for row in correlation.index:
        result[str(row)] = {}
        for column in correlation.columns:
            value = float(correlation.loc[row, column])
            result[str(row)][str(column)] = value if np.isfinite(value) else None
    return result


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = _json_safe(payload)
    path.write_text(
        json.dumps(safe, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def analyze(
    sources: dict[str, Path],
    trainer_manifests: dict[str, Path] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    frames = {name: _normalize_oof(name, path) for name, path in sources.items()}
    aligned = _align_exact(frames)
    names = list(frames)
    target = aligned["target"].to_numpy(dtype=np.float64)
    target_log = np.log1p(target)
    pred_raw = np.column_stack(
        [aligned[f"pred__{name}"].to_numpy(dtype=np.float64) for name in names]
    )
    pred_log = np.log1p(pred_raw)
    residual_log = pred_log - target_log[:, None]

    standalone: dict[str, Any] = {}
    for index, name in enumerate(names):
        standalone[name] = {
            "pooled_rmsle": rmsle(target, pred_raw[:, index]),
            "folds": _fold_records(aligned, pred_raw[:, index]),
        }

    pairwise: list[dict[str, Any]] = []
    for left, right in combinations(range(len(names)), 2):
        pair_matrix = pred_log[:, [left, right]]
        equal_weights = np.array([0.5, 0.5], dtype=np.float64)
        optimal_weights = _optimize_weights(pair_matrix, target_log)
        optimal_weights, optimal_weight_map, dropped = _prune_weights(
            [names[left], names[right]], optimal_weights
        )
        equal_log = pair_matrix @ equal_weights
        optimal_log = pair_matrix @ optimal_weights
        equal_pred = np.expm1(np.clip(equal_log, 0.0, None))
        pairwise.append(
            {
                "models": [names[left], names[right]],
                "equal_weights": equal_weights.tolist(),
                "equal_log_blend_rmsle": float(
                    np.sqrt(np.mean(np.square(equal_log - target_log)))
                ),
                "equal_log_blend_folds": _fold_records(aligned, equal_pred),
                "optimal_all_oof_weights": optimal_weight_map,
                "dropped_tiny_weights": dropped,
                "weight_prune_threshold": WEIGHT_PRUNE_THRESHOLD,
                "optimal_all_oof_in_sample_rmsle": float(
                    np.sqrt(np.mean(np.square(optimal_log - target_log)))
                ),
                "interpretation": (
                    "The optimized pairwise weights and score use the same pooled "
                    "OOF rows; this is an in-sample diagnostic, not validation evidence."
                ),
            }
        )

    equal_all_weights = np.full(len(names), 1.0 / len(names), dtype=np.float64)
    equal_all_log = pred_log @ equal_all_weights
    equal_all_pred = np.expm1(np.clip(equal_all_log, 0.0, None))
    equal_all = {
        "weights": {
            name: float(equal_all_weights[index]) for index, name in enumerate(names)
        },
        "rmsle": rmsle(target, equal_all_pred),
        "folds": _fold_records(aligned, equal_all_pred),
    }

    all_weights = _optimize_weights(pred_log, target_log)
    all_weights, all_weight_map, all_dropped = _prune_weights(names, all_weights)
    all_blend_log = pred_log @ all_weights
    all_blend_pred = np.expm1(np.clip(all_blend_log, 0.0, None))
    all_oof_score = rmsle(target, all_blend_pred)

    # Expanding temporal meta-CV: weights for fold i see only OOF folds before i.
    # The first validation fold has no earlier OOF fold and is intentionally not scored.
    fold_table = (
        aligned[["cutoff_date", "fold"]]
        .drop_duplicates()
        .sort_values(["cutoff_date", "fold"], kind="stable")
        .reset_index(drop=True)
    )
    if len(fold_table) < 2:
        raise ValueError(
            "Expanding temporal ensemble validation requires at least two OOF folds"
        )
    cutoff_values = pd.to_datetime(fold_table["cutoff_date"], errors="raise")
    if cutoff_values.duplicated().any() or not cutoff_values.is_monotonic_increasing:
        raise ValueError(
            "Temporal ensemble requires one strictly increasing cutoff_date per OOF fold"
        )
    fold_order = {
        (str(row.cutoff_date), str(row.fold)): index
        for index, row in enumerate(fold_table.itertuples(index=False))
    }
    row_order = np.fromiter(
        (
            fold_order[(str(cutoff), str(fold))]
            for cutoff, fold in aligned[["cutoff_date", "fold"]].itertuples(
                index=False, name=None
            )
        ),
        dtype=np.int64,
        count=len(aligned),
    )
    temporal_log = np.full(len(aligned), np.nan, dtype=np.float64)
    temporal_folds: list[dict[str, Any]] = []
    first_row = fold_table.iloc[0]
    excluded_folds = [
        {
            "fold": str(first_row["fold"]),
            "cutoff_date": str(first_row["cutoff_date"]),
            "rows": int((row_order == 0).sum()),
            "reason": "no earlier OOF fold is available to fit ensemble weights",
        }
    ]
    for fold_index in range(1, len(fold_table)):
        row = fold_table.iloc[fold_index]
        holdout = row_order == fold_index
        train_mask = row_order < fold_index
        if not holdout.any() or not train_mask.any():
            raise AssertionError("Cannot construct expanding temporal ensemble split")
        weights = _optimize_weights(pred_log[train_mask], target_log[train_mask])
        weights, weight_map, dropped = _prune_weights(names, weights)
        temporal_log[holdout] = pred_log[holdout] @ weights
        fold_prediction = np.expm1(np.clip(temporal_log[holdout], 0.0, None))
        temporal_folds.append(
            {
                "fold": str(row["fold"]),
                "cutoff_date": str(row["cutoff_date"]),
                "rows": int(holdout.sum()),
                "weight_training_folds": int(fold_index),
                "weight_training_rows": int(train_mask.sum()),
                "weight_training_max_cutoff": str(
                    fold_table.iloc[fold_index - 1]["cutoff_date"]
                ),
                "weights": weight_map,
                "dropped_tiny_weights": dropped,
                "rmsle": rmsle(target[holdout], fold_prediction),
            }
        )
    evaluable_mask = np.isfinite(temporal_log)
    if evaluable_mask.sum() != int((row_order > 0).sum()):
        raise AssertionError("Temporal meta-CV coverage does not match future folds")
    temporal_pred = np.full(len(aligned), np.nan, dtype=np.float64)
    temporal_pred[evaluable_mask] = np.expm1(
        np.clip(temporal_log[evaluable_mask], 0.0, None)
    )
    temporal_score = rmsle(target[evaluable_mask], temporal_pred[evaluable_mask])
    standalone_evaluable = {
        name: rmsle(target[evaluable_mask], pred_raw[evaluable_mask, index])
        for index, name in enumerate(names)
    }

    prediction_raw_frame = pd.DataFrame(pred_raw, columns=names)
    prediction_log_frame = pd.DataFrame(pred_log, columns=names)
    residual_frame = pd.DataFrame(residual_log, columns=names)

    artifact_sources: dict[str, dict[str, Any]] = {}
    for name, path in sources.items():
        oof_record = _artifact_record(path)
        oof_record["rows"] = int(len(frames[name]))
        artifact_sources[name] = {
            "oof": oof_record,
            "trainer_manifest": (
                _trainer_manifest_record((trainer_manifests or {})[name], path)
                if name in (trainer_manifests or {})
                else None
            ),
        }

    report = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "metric": "RMSLE",
        "blend_space": "log1p(pred)",
        "constraints": "weights >= 0; sum(weights) = 1",
        "key_columns": KEY_COLUMNS,
        "models": names,
        "rows": int(len(aligned)),
        "fold_count": int(len(fold_table)),
        "coverage_exact": True,
        "standalone": standalone,
        "prediction_correlation_raw": _correlation_dict(prediction_raw_frame),
        "prediction_correlation_log": _correlation_dict(prediction_log_frame),
        "residual_correlation_log": _correlation_dict(residual_frame),
        "pairwise_blends": pairwise,
        "equal_all_models": equal_all,
        "optimal_all_oof": {
            "weights": all_weight_map,
            "dropped_tiny_weights": all_dropped,
            "weight_prune_threshold": WEIGHT_PRUNE_THRESHOLD,
            "rmsle": all_oof_score,
            "folds": _fold_records(aligned, all_blend_pred),
            "interpretation": (
                "Deployment-fit candidate and in-sample diagnostic only: weights and "
                "score use the same pooled OOF rows. Do not treat this RMSLE as "
                "validation evidence; use expanding_temporal_meta_cv below."
            ),
        },
        "expanding_temporal_meta_cv": {
            "protocol": (
                "For validation fold i, fit weights on OOF folds with temporal "
                "order < i and evaluate only fold i."
            ),
            "pooled_rmsle_evaluable_rows": temporal_score,
            "evaluable_rows": int(evaluable_mask.sum()),
            "excluded_rows": int((~evaluable_mask).sum()),
            "excluded_folds": excluded_folds,
            "standalone_pooled_rmsle_evaluable_rows": standalone_evaluable,
            "folds": temporal_folds,
        },
        # Backwards-compatible compact source listing. The richer binding below
        # additionally records trainer manifests.
        "sources": {
            name: dict(binding["oof"])
            for name, binding in artifact_sources.items()
        },
        "artifact_binding": {
            "hash_algorithm": "sha256",
            "interpretation": (
                "Weights are bound to the exact OOF bytes and, where found, the "
                "trainer manifest bytes listed below."
            ),
            "sources": artifact_sources,
        },
    }
    return report, aligned


def _print_report(report: dict[str, Any]) -> None:
    print("=" * 78)
    print(f"ALIGNED OOF: rows={report['rows']:,} folds={report['fold_count']}")
    print("=" * 78)
    standalone_rows = [
        {"model": name, "pooled_rmsle": values["pooled_rmsle"]}
        for name, values in report["standalone"].items()
    ]
    print(pd.DataFrame(standalone_rows).to_string(index=False))
    print("\nResidual correlations in log space:")
    print(pd.DataFrame(report["residual_correlation_log"]).to_string())
    print("\nOptimal all-OOF weights:")
    print(json.dumps(report["optimal_all_oof"]["weights"], indent=2))
    print(f"all-OOF blend RMSLE={report['optimal_all_oof']['rmsle']:.6f}")
    print(
        "expanding temporal meta-CV blend RMSLE (evaluable rows)="
        f"{report['expanding_temporal_meta_cv']['pooled_rmsle_evaluable_rows']:.6f}"
    )
    print(
        "temporal meta-CV coverage="
        f"{report['expanding_temporal_meta_cv']['evaluable_rows']:,}/"
        f"{report['rows']:,} rows; earliest fold intentionally excluded"
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    import config as cfg

    oof_dir = Path(args.oof_dir or cfg.OOF_DIR)
    if args.oof:
        parsed = [_parse_source(value) for value in args.oof]
        sources = dict(parsed)
        if len(sources) != len(parsed):
            raise ValueError("Duplicate model names in --oof arguments")
    else:
        candidates = {
            name: oof_dir / filename for name, filename in DEFAULT_OOF_FILES.items()
        }
        sources = {name: path for name, path in candidates.items() if path.is_file()}
        if len(sources) < 2:
            missing = [str(path) for path in candidates.values() if not path.is_file()]
            raise FileNotFoundError(
                "Need at least two standardized OOF files. Missing candidates: "
                + ", ".join(missing)
            )

    manifest_candidates = {
        name: Path(cfg.MODELS_DIR) / name / "manifest.json" for name in sources
    }
    trainer_manifests = {
        name: path for name, path in manifest_candidates.items() if path.is_file()
    }
    if args.trainer_manifest:
        parsed_manifests = [_parse_source(value) for value in args.trainer_manifest]
        explicit_manifests = dict(parsed_manifests)
        if len(explicit_manifests) != len(parsed_manifests):
            raise ValueError("Duplicate model names in --trainer-manifest arguments")
        unknown = sorted(set(explicit_manifests) - set(sources))
        if unknown:
            raise ValueError(
                "Trainer manifests without matching OOF sources: " + ", ".join(unknown)
            )
        trainer_manifests.update(explicit_manifests)

    missing_manifests = [
        f"{name}={path}" for name, path in trainer_manifests.items() if not path.is_file()
    ]
    if missing_manifests:
        raise FileNotFoundError(
            "Trainer manifest files not found: " + ", ".join(missing_manifests)
        )

    report, _ = analyze(sources, trainer_manifests=trainer_manifests)
    model_dir = Path(args.model_dir or (cfg.MODELS_DIR / "ensemble"))
    report_path = Path(args.report_path or (cfg.REPORTS_DIR / "ensemble_report.json"))
    manifest_path = model_dir / "manifest.json"
    _write_json(report_path, report)
    _write_json(manifest_path, report)
    _print_report(report)
    print(f"Report saved: {report_path}")
    print(f"Ensemble manifest saved: {manifest_path}")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Align temporal OOF predictions and optimize log-space weights."
    )
    parser.add_argument("--oof-dir", type=Path, default=None)
    parser.add_argument(
        "--oof",
        action="append",
        default=None,
        metavar="MODEL=PATH",
        help="Explicit OOF source; repeat for each model.",
    )
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument(
        "--trainer-manifest",
        action="append",
        default=None,
        metavar="MODEL=PATH",
        help=(
            "Optional explicit trainer manifest to hash-bind; repeat per model. "
            "By default canonical data/models/<model>/manifest.json files are used "
            "when present."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
