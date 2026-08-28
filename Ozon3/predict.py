"""Deterministic final inference and strict one-file submission validation."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import config as cfg
from data_loading import load_sample_submission
from train import _dataset_fingerprint, get_features


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_project_path(path: Path) -> str:
    """Record project-relative paths so Cyrillic parent paths stay portable."""

    path = Path(path)
    try:
        return path.resolve().relative_to(cfg.PROJECT_DIR.resolve()).as_posix()
    except ValueError:
        return str(path)


def _require_bound_file(record: dict, *, label: str) -> Path:
    path = Path(str(record.get("path", "")))
    if not path.is_file():
        raise RuntimeError(f"{label} is missing: {path}")
    expected_size = record.get("size_bytes")
    if expected_size is not None and path.stat().st_size != int(expected_size):
        raise RuntimeError(f"{label} size changed after ensemble fitting: {path}")
    expected_hash = record.get("sha256")
    if not expected_hash or _sha256(path) != str(expected_hash):
        raise RuntimeError(f"{label} SHA-256 changed after ensemble fitting: {path}")
    return path


def _read_manifest(model_name: str) -> dict:
    path = cfg.MODELS_DIR / model_name / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run the corresponding CV + final refit first."
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("row_limit") is not None:
        raise RuntimeError(
            f"Refusing submission from smoke-test manifest with row_limit={manifest['row_limit']}"
        )
    if manifest.get("artifact_mode") != "canonical" or manifest.get("run_name") is not None:
        raise RuntimeError(
            f"Refusing non-canonical {model_name} manifest: "
            f"artifact_mode={manifest.get('artifact_mode')!r}, "
            f"run_name={manifest.get('run_name')!r}"
        )
    dataset_dir = Path(str(manifest.get("dataset_dir", ""))).resolve()
    if dataset_dir != cfg.DATA_DIR.resolve():
        raise RuntimeError(
            f"{model_name} was trained from {dataset_dir}, expected {cfg.DATA_DIR.resolve()}"
        )
    expected_dataset_fingerprint = manifest.get("dataset_fingerprint")
    actual_dataset_fingerprint = _dataset_fingerprint(cfg.DATA_DIR, None)
    if expected_dataset_fingerprint != actual_dataset_fingerprint:
        raise RuntimeError(
            f"{model_name} dataset fingerprint changed after training: "
            f"manifest={expected_dataset_fingerprint}, current={actual_dataset_fingerprint}"
        )
    return manifest


def _validate_final_model_artifact(manifest: dict, model_name: str) -> Path:
    model_path = Path(str(manifest.get("final_model", "")))
    expected_dir = (cfg.MODELS_DIR / model_name).resolve()
    if not model_path.is_file() or not model_path.resolve().is_relative_to(expected_dir):
        raise RuntimeError(
            f"Invalid canonical final model path for {model_name}: {model_path}"
        )
    expected_hash = manifest.get("final_model_sha256")
    if not expected_hash or _sha256(model_path) != str(expected_hash):
        raise RuntimeError(f"Final {model_name} model hash does not match its manifest")
    return model_path


def _validate_test_features(
    test_df: pd.DataFrame,
    feature_cols: list[str],
    feature_dtypes: dict[str, str] | None = None,
) -> None:
    if cfg.ID_COL not in test_df:
        raise ValueError(f"test_features is missing {cfg.ID_COL}")
    if test_df[cfg.ID_COL].duplicated().any():
        raise ValueError("Duplicate user_id in test_features")
    missing = [column for column in feature_cols if column not in test_df]
    if missing:
        raise ValueError(f"test_features is missing model features: {missing[:20]}")
    bad = [
        column
        for column in feature_cols
        if not pd.api.types.is_numeric_dtype(test_df[column])
        or not np.isfinite(test_df[column].to_numpy(copy=False)).all()
    ]
    if bad:
        raise ValueError(f"Invalid test feature columns: {bad[:20]}")
    actual_features = list(get_features([test_df]))
    if actual_features != feature_cols:
        missing = sorted(set(feature_cols) - set(actual_features))
        extra = sorted(set(actual_features) - set(feature_cols))
        raise ValueError(
            f"Exact test feature schema differs: missing={missing[:20]}, extra={extra[:20]}"
        )
    if feature_dtypes is not None:
        actual_dtypes = {column: str(test_df[column].dtype) for column in feature_cols}
        if actual_dtypes != feature_dtypes:
            mismatch = {
                column: (feature_dtypes.get(column), actual_dtypes.get(column))
                for column in feature_cols
                if feature_dtypes.get(column) != actual_dtypes.get(column)
            }
            raise ValueError(f"Test feature dtypes differ: {dict(list(mismatch.items())[:20])}")
    if "cutoff_date" in test_df:
        cutoffs = pd.to_datetime(test_df["cutoff_date"]).dt.date.unique().tolist()
        if cutoffs != [cfg.HIST_END]:
            raise AssertionError(
                f"test cutoff must be {cfg.HIST_END}, found {cutoffs}"
            )


def _validate_test_sample_contract(
    test_df: pd.DataFrame, sample: pd.DataFrame
) -> None:
    expected_rows = getattr(cfg, "EXPECTED_SUBMISSION_ROWS", 250_000)
    if len(test_df) != expected_rows:
        raise AssertionError(
            f"test_features rows={len(test_df):,}, expected={expected_rows:,}"
        )
    if len(sample) != expected_rows:
        raise AssertionError(
            f"official sample rows={len(sample):,}, expected={expected_rows:,}"
        )
    if sample.columns.tolist() != [cfg.ID_COL, "predict"]:
        raise AssertionError("Official sample schema differs")
    test_ids = test_df[cfg.ID_COL].to_numpy(dtype=np.int64, copy=False)
    sample_ids = sample[cfg.ID_COL].to_numpy(dtype=np.int64, copy=False)
    if not np.array_equal(np.sort(test_ids), np.sort(sample_ids)):
        raise AssertionError("test_features user_id universe differs from official sample")
    if not np.array_equal(test_ids, sample_ids):
        raise AssertionError("test_features user_id order differs from official sample")


def _distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise AssertionError("Prediction distribution contains invalid values")
    quantiles = np.quantile(values, [0.5, 0.9, 0.95, 0.99, 0.999])
    return {
        "min": float(values.min()),
        "median": float(quantiles[0]),
        "mean": float(values.mean()),
        "p90": float(quantiles[1]),
        "p95": float(quantiles[2]),
        "p99": float(quantiles[3]),
        "p99.9": float(quantiles[4]),
        "max": float(values.max()),
    }


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def predict_lgbm(test_df: pd.DataFrame) -> np.ndarray:
    manifest = _read_manifest("lgbm")
    features = manifest["feature_columns"]
    _validate_test_features(test_df, features)
    model_path = _validate_final_model_artifact(manifest, "lgbm")
    # model_str is robust to Cyrillic Windows paths.
    model = lgb.Booster(model_str=model_path.read_text(encoding="utf-8"))
    pred_log = np.clip(model.predict(test_df[features]), 0.0, None)
    return np.expm1(pred_log)


def predict_catboost(test_df: pd.DataFrame) -> np.ndarray:
    try:
        from catboost import CatBoostRegressor
    except ImportError as exc:
        raise RuntimeError("CatBoost is required for CatBoost inference") from exc
    manifest = _read_manifest("catboost")
    features = manifest["feature_columns"]
    _validate_test_features(test_df, features)
    model_path = _validate_final_model_artifact(manifest, "catboost")
    model = CatBoostRegressor()
    model.load_model(str(model_path))
    pred_log = np.clip(model.predict(test_df[features]), 0.0, None)
    return np.expm1(pred_log)


def predict_nn(
    test_df: pd.DataFrame,
    *,
    return_seed_predictions: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[int, np.ndarray]]:
    from train_nn import Standardizer, _make_model, _matrix, _predict_log, _require_torch

    manifest = _read_manifest("nn")
    if not manifest.get("final_refit_completed"):
        raise RuntimeError("NN final refit is absent; rerun train_nn.py without --skip-final-refit")
    model_dir = cfg.MODELS_DIR / "nn"
    feature_path = model_dir / manifest["feature_file"]
    normalizer_path = model_dir / manifest["normalizer_file"]
    if _sha256(feature_path) != manifest.get("feature_file_sha256"):
        raise RuntimeError("NN feature schema file hash mismatch")
    feature_payload = json.loads(feature_path.read_text(encoding="utf-8"))
    features = list(feature_payload["features"])
    feature_dtypes = dict(feature_payload["dtypes"])
    expected_feature_contract = {
        "dataset_fingerprint": manifest["dataset_fingerprint"],
        "features": manifest["feature_columns"],
        "dtypes": manifest["feature_dtypes"],
        "feature_schema_sha256": manifest["feature_schema_sha256"],
    }
    if feature_payload != expected_feature_contract:
        raise RuntimeError("NN feature schema file differs from manifest")
    _validate_test_features(test_df, features, feature_dtypes)

    if _sha256(normalizer_path) != manifest.get("normalizer_file_sha256"):
        raise RuntimeError("NN normalizer file hash mismatch")

    torch, nn, _, _ = _require_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats = np.load(normalizer_path)
    scaler = Standardizer(
        mean=stats["mean"].astype(np.float32),
        scale=stats["scale"].astype(np.float32),
    )
    if scaler.mean.shape != (len(features),) or scaler.scale.shape != (len(features),):
        raise RuntimeError("NN normalizer shape differs from feature contract")
    if not np.isfinite(scaler.mean).all() or not np.isfinite(scaler.scale).all():
        raise RuntimeError("NN normalizer contains non-finite statistics")
    if (scaler.scale <= 0).any():
        raise RuntimeError("NN normalizer contains non-positive scales")
    from train_nn import _array_hash

    if _array_hash(scaler.mean, scaler.scale) != manifest.get(
        "normalizer_stats_sha256"
    ):
        raise RuntimeError("NN normalizer statistics hash mismatch")
    x_test = scaler.transform(_matrix(test_df, features))
    predictions: dict[int, np.ndarray] = {}
    nn_configuration = manifest["nn_configuration"]
    architecture = nn_configuration["architecture"]
    training = nn_configuration["training"]
    batch_size = int(training["batch_size"])
    expected_seeds = [int(seed) for seed in training["seeds"]]
    final_epochs = {
        int(seed): int(epochs)
        for seed, epochs in manifest["final_epochs_by_seed"].items()
    }
    expected_filenames = [f"model_seed_{seed}.pt" for seed in expected_seeds]
    if manifest["model_files"] != expected_filenames:
        raise RuntimeError(
            f"NN final model list differs: {manifest['model_files']} != {expected_filenames}"
        )
    for seed, filename in zip(expected_seeds, expected_filenames):
        expected_hash = manifest.get("model_file_sha256", {}).get(filename)
        if not expected_hash or _sha256(model_dir / filename) != expected_hash:
            raise RuntimeError(f"NN model hash mismatch: {model_dir / filename}")
        try:
            checkpoint = torch.load(
                model_dir / filename, map_location=device, weights_only=True
            )
        except TypeError:
            checkpoint = torch.load(model_dir / filename, map_location=device)
        expected_checkpoint = {
            "artifact_kind": "final_refit_seed_model",
            "seed": seed,
            "epochs": final_epochs[seed],
            "final_training_rows": int(manifest["final_refit_rows"]),
            "dataset_fingerprint": manifest["dataset_fingerprint"],
            "target_transform": "log1p",
            "feature_schema_sha256": manifest["feature_schema_sha256"],
            "nn_configuration_sha256": manifest["nn_configuration_sha256"],
            "normalizer_stats_sha256": manifest["normalizer_stats_sha256"],
            "normalizer_file_sha256": manifest["normalizer_file_sha256"],
            "architecture": architecture,
        }
        for key, expected_value in expected_checkpoint.items():
            if checkpoint.get(key) != expected_value:
                raise RuntimeError(f"NN {filename} payload {key} differs")
        model = _make_model(
            nn,
            input_dim=len(features),
            hidden_dim=int(architecture["hidden_dim"]),
            num_blocks=int(architecture["num_blocks"]),
            dropout=float(architecture["dropout"]),
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        predictions[seed] = _predict_log(torch, model, x_test, batch_size, device)
        del model, checkpoint
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if not predictions:
        raise RuntimeError("NN manifest contains no final model files")
    pred_log = np.clip(np.mean(list(predictions.values()), axis=0), 0.0, None)
    prediction = np.expm1(pred_log)
    if not np.isfinite(prediction).all() or (prediction < 0).any():
        raise RuntimeError("NN inference produced NaN, infinity, or negative predictions")
    if return_seed_predictions:
        return prediction, predictions
    return prediction


def _nn_inference_diagnostics(
    test_df: pd.DataFrame,
    predictions: np.ndarray,
    seed_pred_log: dict[int, np.ndarray],
) -> dict:
    oof_path = cfg.OOF_DIR / "oof_nn.csv"
    if not oof_path.is_file():
        raise FileNotFoundError(f"Missing canonical NN OOF for diagnostics: {oof_path}")
    oof_prediction = pd.read_csv(oof_path, usecols=["pred"])["pred"].to_numpy(
        dtype=np.float64, copy=False
    )
    test_stats = _distribution(predictions)
    oof_stats = _distribution(oof_prediction)
    seed_matrix = np.column_stack(
        [seed_pred_log[seed] for seed in sorted(seed_pred_log)]
    ).astype(np.float64, copy=False)
    if not np.isfinite(seed_matrix).all():
        raise AssertionError("Per-seed NN log predictions contain invalid values")
    seed_log_std = seed_matrix.std(axis=1)
    top_indices = np.argsort(predictions)[-20:][::-1]
    top_predictions = []
    for index in top_indices:
        top_predictions.append(
            {
                "user_id": int(test_df[cfg.ID_COL].iloc[index]),
                "predict": float(predictions[index]),
                "pred_log": float(np.log1p(predictions[index])),
                "seed_pred_log": {
                    str(seed): float(seed_pred_log[seed][index])
                    for seed in sorted(seed_pred_log)
                },
                "seed_log_std": float(seed_log_std[index]),
            }
        )

    def ratio(numerator: float, denominator: float) -> float | None:
        return float(numerator / denominator) if denominator > 0 else None

    comparison = {
        "test_to_oof_mean_ratio": ratio(test_stats["mean"], oof_stats["mean"]),
        "test_to_oof_p99_ratio": ratio(test_stats["p99"], oof_stats["p99"]),
        "test_to_oof_p99.9_ratio": ratio(test_stats["p99.9"], oof_stats["p99.9"]),
        "test_to_oof_max_ratio": ratio(test_stats["max"], oof_stats["max"]),
        "test_rows_above_oof_max": int((predictions > oof_stats["max"]).sum()),
        "test_rows_above_oof_p99.9": int((predictions > oof_stats["p99.9"]).sum()),
        "seed_log_std_p99.9": float(np.quantile(seed_log_std, 0.999)),
        "seed_log_std_max": float(seed_log_std.max()),
    }
    warnings: list[str] = []
    if comparison["test_to_oof_p99.9_ratio"] is not None and comparison[
        "test_to_oof_p99.9_ratio"
    ] > 3.0:
        warnings.append("test p99.9 exceeds OOF p99.9 by more than 3x")
    if comparison["test_to_oof_max_ratio"] is not None and comparison[
        "test_to_oof_max_ratio"
    ] > 10.0:
        warnings.append("test maximum exceeds OOF maximum by more than 10x")
    return {
        "test_prediction_distribution": test_stats,
        "nn_oof_prediction_distribution": oof_stats,
        "test_vs_oof": comparison,
        "pathology_warnings": warnings,
        "top_20_test_predictions": top_predictions,
        "interpretation": (
            "No clipping or calibration was applied. Warnings compare test extremes "
            "only with pre-existing NN OOF evidence and do not alter predictions."
        ),
    }


def _load_external_prediction(
    path: Path,
    test_df: pd.DataFrame,
    prediction_column: str = "pred",
) -> np.ndarray:
    frame = pd.read_csv(path)
    if prediction_column not in frame and "predict" in frame:
        prediction_column = "predict"
    required = {cfg.ID_COL, prediction_column}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path} must contain {sorted(required)}")
    if frame[cfg.ID_COL].duplicated().any():
        raise ValueError(f"Duplicate user_id in {path}")
    aligned = test_df[[cfg.ID_COL]].merge(
        frame[[cfg.ID_COL, prediction_column]],
        on=cfg.ID_COL,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    values = aligned[prediction_column].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"Invalid predictions in {path}")
    return values


def predict_ensemble(test_df: pd.DataFrame) -> np.ndarray:
    manifest_path = cfg.MODELS_DIR / "ensemble" / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Run ensemble.py on aligned OOF first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    weights = {
        name: float(value)
        for name, value in manifest.get("optimal_all_oof", {}).get("weights", {}).items()
        if float(value) > 1e-8
    }
    if not weights or any(float(value) < 0 for value in weights.values()):
        raise ValueError("Invalid ensemble weights")
    total = float(sum(weights.values()))
    if not np.isclose(total, 1.0, atol=1e-7):
        raise ValueError(f"Ensemble weights sum to {total}, expected 1")

    bindings = manifest.get("artifact_binding", {}).get("sources", {})
    for model_name in weights:
        binding = bindings.get(model_name)
        if not isinstance(binding, dict):
            raise RuntimeError(f"Ensemble lacks artifact binding for {model_name}")
        _require_bound_file(binding.get("oof") or {}, label=f"{model_name} OOF")
        trainer_binding = binding.get("trainer_manifest")
        if not isinstance(trainer_binding, dict):
            raise RuntimeError(f"Ensemble lacks trainer-manifest binding for {model_name}")
        manifest_bound_path = _require_bound_file(
            trainer_binding, label=f"{model_name} trainer manifest"
        )
        canonical_manifest = (cfg.MODELS_DIR / model_name / "manifest.json").resolve()
        if manifest_bound_path.resolve() != canonical_manifest:
            raise RuntimeError(
                f"Ensemble {model_name} weights are bound to {manifest_bound_path}, "
                f"not canonical {canonical_manifest}"
            )
        if trainer_binding.get("declared_oof_matches_source") is not True:
            raise RuntimeError(
                f"{model_name} trainer manifest does not declare the bound OOF source"
            )

    predictions: dict[str, np.ndarray] = {}
    for model_name in weights:
        if model_name == "lgbm":
            predictions[model_name] = predict_lgbm(test_df)
        elif model_name == "catboost":
            predictions[model_name] = predict_catboost(test_df)
        elif model_name == "nn":
            predictions[model_name] = predict_nn(test_df)
        else:
            raise ValueError(f"Unsupported ensemble model: {model_name}")

    blend_log = np.zeros(len(test_df), dtype=np.float64)
    for model_name, weight in weights.items():
        blend_log += float(weight) * np.log1p(predictions[model_name])
    return np.expm1(blend_log)


def build_submission(
    test_df: pd.DataFrame,
    predictions: np.ndarray,
    output_path: Path,
) -> pd.DataFrame:
    sample = load_sample_submission()
    expected_rows = getattr(cfg, "EXPECTED_SUBMISSION_ROWS", 250_000)
    if len(sample) != expected_rows:
        raise AssertionError(
            f"Official sample must have {expected_rows:,} rows, found {len(sample):,}"
        )
    if sample.columns.tolist() != [cfg.ID_COL, "predict"]:
        raise AssertionError(
            f"Official sample schema must be [{cfg.ID_COL}, predict], "
            f"found {sample.columns.tolist()}"
        )

    raw = pd.DataFrame(
        {cfg.ID_COL: test_df[cfg.ID_COL].to_numpy(), "predict": predictions}
    )
    if raw[cfg.ID_COL].duplicated().any():
        raise AssertionError("Duplicate test user_id before submission alignment")
    submission = sample[[cfg.ID_COL]].merge(
        raw,
        on=cfg.ID_COL,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if len(submission) != expected_rows:
        raise AssertionError("Submission row count changed during alignment")
    if not np.array_equal(submission[cfg.ID_COL], sample[cfg.ID_COL]):
        raise AssertionError("Submission user order differs from official sample")
    values = submission["predict"].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise AssertionError("Submission contains NaN, infinity, or negative prediction")
    if submission.columns.tolist() != [cfg.ID_COL, "predict"]:
        raise AssertionError("Unexpected submission columns")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    submission.to_csv(temporary, index=False)
    temporary.replace(output_path)
    persisted = pd.read_csv(output_path)
    if persisted.columns.tolist() != [cfg.ID_COL, "predict"] or len(
        persisted
    ) != expected_rows:
        raise AssertionError("Persisted submission schema/row count differs")
    if not np.array_equal(
        persisted[cfg.ID_COL].to_numpy(dtype=np.int64),
        sample[cfg.ID_COL].to_numpy(dtype=np.int64),
    ):
        raise AssertionError("Persisted submission user_id order differs")
    persisted_values = persisted["predict"].to_numpy(dtype=np.float64)
    if not np.isfinite(persisted_values).all() or (persisted_values < 0).any():
        raise AssertionError("Persisted submission contains invalid predictions")
    print(
        f"Submission validated and saved: {output_path} | rows={len(submission):,} | "
        f"mean={values.mean():.6f} | median={np.median(values):.6f}"
    )
    return submission


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=cfg.DATA_DIR)
    parser.add_argument(
        "--model", choices=["lgbm", "catboost", "nn", "ensemble"], default="nn"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=cfg.SUB_DIR / "submission_nn_primary.csv",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=cfg.REPORTS_DIR / "nn_final_inference.json",
    )
    args = parser.parse_args()
    test_path = args.dataset_dir / "test_features.parquet"
    if not test_path.exists():
        raise FileNotFoundError(f"Missing {test_path}; run build_dataset.py")
    test_df = pd.read_parquet(test_path)
    sample = load_sample_submission()
    _validate_test_sample_contract(test_df, sample)
    if args.model == "lgbm":
        predictions = predict_lgbm(test_df)
    elif args.model == "catboost":
        predictions = predict_catboost(test_df)
    elif args.model == "nn":
        predictions, seed_predictions = predict_nn(
            test_df, return_seed_predictions=True
        )
    else:
        predictions = predict_ensemble(test_df)

    if args.model == "nn":
        diagnostics = _nn_inference_diagnostics(
            test_df, predictions, seed_predictions
        )
        manifest_path = cfg.MODELS_DIR / "nn" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        diagnostics.update(
            {
                "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "selected_model": "standalone_temporal_nn",
                "dataset_fingerprint": manifest["dataset_fingerprint"],
                "final_training_rows": int(manifest["final_refit_rows"]),
                "final_epochs_by_seed": manifest["final_epochs_by_seed"],
                "model_manifest_path": _portable_project_path(manifest_path),
                "model_manifest_sha256": _sha256(manifest_path),
                "model_artifacts": {
                    filename: {
                        "path": _portable_project_path(
                            cfg.MODELS_DIR / "nn" / filename
                        ),
                        "sha256": manifest["model_file_sha256"][filename],
                    }
                    for filename in manifest["model_files"]
                },
                "normalizer_path": _portable_project_path(
                    cfg.MODELS_DIR / "nn" / manifest["normalizer_file"]
                ),
                "normalizer_sha256": manifest["normalizer_file_sha256"],
                "submission_created": False,
                "prediction_formula": "expm1(max(mean seed pred_log, 0))",
            }
        )
        print("NN test prediction distribution:")
        print(json.dumps(diagnostics["test_prediction_distribution"], indent=2))
        print("NN OOF prediction distribution:")
        print(json.dumps(diagnostics["nn_oof_prediction_distribution"], indent=2))
        print("NN extreme comparison:")
        print(json.dumps(diagnostics["test_vs_oof"], indent=2))
        if diagnostics["pathology_warnings"]:
            diagnostics["inference_status"] = "blocked_before_submission"
            _write_json_atomic(args.report_path, diagnostics)
            raise RuntimeError(
                "Suspicious NN test extremes require diagnosis before submission: "
                + "; ".join(diagnostics["pathology_warnings"])
            )

        submission = build_submission(test_df, predictions, args.output)
        diagnostics.update(
            {
                "inference_status": "validated_submission_created",
                "submission_created": True,
                "submission_path": _portable_project_path(args.output),
                "submission_sha256": _sha256(args.output),
                "submission_rows": int(len(submission)),
                "submission_columns": submission.columns.tolist(),
                "submission_validated": True,
            }
        )
        _write_json_atomic(args.report_path, diagnostics)
        print(f"Inference report saved: {args.report_path}")
    else:
        build_submission(test_df, predictions, args.output)


if __name__ == "__main__":
    main()
