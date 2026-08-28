import argparse
import time
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

import config as cfg
from data_loading import load_raw, load_sample_submission
from features import build_features_for_cutoff, build_target, optimize_dtypes
from time_split import build_cv_folds, build_test_fold, validate_fold_contract


UserPolicy = Literal["seen_only", "all_users"]
METADATA_COLUMNS = ("fold", "cutoff_date")
TARGET_COLUMNS = (cfg.TARGET_COL, cfg.TARGET_LOG_COL)


def _ordered_int64_ids(values: pd.Series) -> pd.Series:
    ids = pd.Series(values).drop_duplicates().astype("int64").reset_index(drop=True)
    if ids.isna().any():
        raise ValueError("User universe contains missing user_id values.")
    return ids


def _fill_unseen_feature_rows(
    observed_features: pd.DataFrame,
    desired_user_ids: pd.Series,
    existing_user_ids: pd.Series,
    cutoff,
) -> pd.DataFrame:
    """Reindex features without asking the current feature builder for unseen IDs."""

    base = pd.DataFrame({cfg.ID_COL: _ordered_int64_ids(desired_user_ids)})
    features = base.merge(
        observed_features,
        on=cfg.ID_COL,
        how="left",
        sort=False,
        validate="one_to_one",
    )
    features["cutoff_date"] = pd.Timestamp(cutoff)
    existing_set = set(existing_user_ids.astype("int64").tolist())
    features["seen_before_anchor"] = (
        features[cfg.ID_COL].isin(existing_set).astype("int8")
    )

    for column in features.columns:
        if column in (cfg.ID_COL, "cutoff_date"):
            continue
        if pd.api.types.is_numeric_dtype(features[column]):
            features[column] = (
                features[column]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0)
            )
    return features


def _validate_prepared_frame(
    frame: pd.DataFrame,
    *,
    fold_name: str,
    cutoff,
    is_test: bool,
    expected_test_ids: pd.Series | None = None,
) -> None:
    required = {cfg.ID_COL, *METADATA_COLUMNS}
    missing = required.difference(frame.columns)
    if missing:
        raise AssertionError(f"{fold_name}: missing required columns: {sorted(missing)}")
    if frame.empty:
        raise AssertionError(f"{fold_name}: prepared frame is empty.")
    if frame[cfg.ID_COL].dtype != np.dtype("int64"):
        raise AssertionError(
            f"{fold_name}: user_id dtype={frame[cfg.ID_COL].dtype}, expected int64."
        )
    if frame[cfg.ID_COL].isna().any() or frame[cfg.ID_COL].duplicated().any():
        raise AssertionError(f"{fold_name}: user_id must be non-null and unique.")
    if not frame["fold"].eq(fold_name).all():
        raise AssertionError(f"{fold_name}: inconsistent fold metadata.")
    cutoff_values = pd.to_datetime(frame["cutoff_date"], errors="raise")
    if cutoff_values.isna().any() or not cutoff_values.eq(pd.Timestamp(cutoff)).all():
        raise AssertionError(f"{fold_name}: inconsistent cutoff_date metadata.")

    target_presence = [column in frame.columns for column in TARGET_COLUMNS]
    if is_test and any(target_presence):
        raise AssertionError("test_features must not contain target columns.")
    if not is_test and not all(target_presence):
        raise AssertionError(f"{fold_name}: labeled fold lacks target columns.")

    for column in frame.select_dtypes(include=[np.number]).columns:
        if not np.isfinite(frame[column].to_numpy()).all():
            raise AssertionError(f"{fold_name}: {column} contains NaN or infinity.")
    if not is_test and (frame[cfg.TARGET_COL] < 0).any():
        raise AssertionError(f"{fold_name}: target contains negative values.")

    if expected_test_ids is not None:
        expected = _ordered_int64_ids(expected_test_ids).to_numpy()
        actual = frame[cfg.ID_COL].to_numpy()
        if len(actual) != len(expected) or not np.array_equal(actual, expected):
            raise AssertionError(
                "test user_id values/order do not exactly match official sample submission."
            )


def _feature_schema(frame: pd.DataFrame) -> tuple[tuple[str, str], ...]:
    return tuple(
        (column, str(frame[column].dtype))
        for column in frame.columns
        if column not in TARGET_COLUMNS
    )


def _canonicalize_prepared_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    """Use one deterministic dtype contract for every fold and test."""

    frame = optimize_dtypes(frame).copy()
    frame[cfg.ID_COL] = frame[cfg.ID_COL].astype("int64")
    frame["cutoff_date"] = pd.to_datetime(frame["cutoff_date"], errors="raise")
    frame["fold"] = frame["fold"].astype("string")

    protected = {cfg.ID_COL, *METADATA_COLUMNS, *TARGET_COLUMNS}
    for column in frame.columns:
        if column in protected:
            continue
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise TypeError(
                f"Feature {column!r} must be numeric, got {frame[column].dtype}."
            )
        frame[column] = frame[column].astype("float32")

    for column in TARGET_COLUMNS:
        if column in frame.columns:
            frame[column] = frame[column].astype("float32")
    return frame


def _align_to_feature_contract(
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Add data-dependent missing features as zeros and enforce column order."""

    current_feature_columns = [
        column for column in frame.columns if column not in TARGET_COLUMNS
    ]
    unexpected = set(current_feature_columns).difference(feature_columns)
    if unexpected:
        raise AssertionError(f"Feature contract omitted columns: {sorted(unexpected)}")

    for column in feature_columns:
        if column not in frame.columns:
            if column in {cfg.ID_COL, *METADATA_COLUMNS}:
                raise AssertionError(f"Required metadata column {column!r} is missing.")
            frame[column] = np.float32(0.0)

    ordered_targets = [column for column in TARGET_COLUMNS if column in frame.columns]
    frame = frame.loc[:, [*feature_columns, *ordered_targets]]
    return _canonicalize_prepared_dtypes(frame)


def process_single_fold(
    fold,
    df: pd.DataFrame,
    all_user_ids: pd.Series,
    *,
    output_dir: str | Path = cfg.DATA_DIR,
    user_policy: UserPolicy = "seen_only",
    expected_test_ids: pd.Series | None = None,
) -> pd.DataFrame:
    if user_policy not in {"seen_only", "all_users"}:
        raise ValueError(f"Unknown user_policy={user_policy!r}.")

    started_at = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    is_test = fold.name == "test"

    print("\n" + "=" * 70)
    print(
        f"{fold.name} | cutoff={fold.cutoff} | "
        f"target={fold.target_start}..{fold.target_end} | policy={user_policy}"
    )
    print("=" * 70)

    # This is the hard feature boundary.  Even if a feature implementation
    # forgets its own cutoff filter, it cannot see target/future events.
    # Boolean indexing already materializes the bounded history.  Avoid a
    # second defensive copy: the feature builder treats this frame as
    # read-only and performs local copies for sequence-sensitive blocks.
    hist = df.loc[df[cfg.DATE_COL] <= pd.Timestamp(fold.cutoff)]
    if not hist.empty and hist[cfg.DATE_COL].max() > pd.Timestamp(fold.cutoff):
        raise AssertionError("Feature history exceeds cutoff.")
    existing_users = _ordered_int64_ids(hist[cfg.ID_COL])

    # Test must always follow the official sample universe.  For labeled folds
    # the default preserves the historic seen-only policy; all_users is the
    # controlled cold-start experiment.
    effective_policy: UserPolicy = "all_users" if is_test else user_policy
    desired_users = (
        existing_users if effective_policy == "seen_only" else _ordered_int64_ids(all_user_ids)
    )
    print(
        f"seen users={len(existing_users):,} | output users={len(desired_users):,} "
        f"| unseen rows={len(desired_users) - desired_users.isin(existing_users).sum():,}"
    )

    observed_features = build_features_for_cutoff(
        hist,
        fold.cutoff,
        user_ids=existing_users,
    )
    features = _fill_unseen_feature_rows(
        observed_features,
        desired_users,
        existing_users,
        fold.cutoff,
    )
    # Feature engineering intentionally creates many blocks.  Add metadata in
    # one concat so pandas does not append another block to an already
    # fragmented frame (which is both noisy and expensive at production size).
    features = pd.concat(
        [
            features,
            pd.Series(fold.name, index=features.index, name="fold", dtype="string"),
        ],
        axis=1,
    ).copy()
    features[cfg.ID_COL] = features[cfg.ID_COL].astype("int64")

    if is_test:
        prepared = _canonicalize_prepared_dtypes(features)
        _validate_prepared_frame(
            prepared,
            fold_name=fold.name,
            cutoff=fold.cutoff,
            is_test=True,
            expected_test_ids=expected_test_ids,
        )
        out_path = output_dir / "test_features.parquet"
    else:
        target = build_target(
            df,
            desired_users,
            fold.target_start,
            fold.target_end,
        )[[cfg.ID_COL, cfg.TARGET_COL]]
        prepared = features.merge(
            target,
            on=cfg.ID_COL,
            how="left",
            sort=False,
            validate="one_to_one",
        )
        prepared[cfg.TARGET_COL] = (
            prepared[cfg.TARGET_COL].fillna(0).clip(lower=0).astype("float32")
        )
        prepared[cfg.TARGET_LOG_COL] = np.log1p(
            prepared[cfg.TARGET_COL]
        ).astype("float32")
        prepared = _canonicalize_prepared_dtypes(prepared)
        _validate_prepared_frame(
            prepared,
            fold_name=fold.name,
            cutoff=fold.cutoff,
            is_test=False,
        )
        out_path = output_dir / f"{fold.name}.parquet"

    prepared.to_parquet(out_path, index=False)
    print(f"Готов {fold.name} -> {out_path}")
    print(f"Строк: {len(prepared):,} | features: {len(_feature_schema(prepared)) - 3}")
    if not is_test:
        print(f"Positive target: {(prepared[cfg.TARGET_COL] > 0).mean():.4%}")
    print(f"Время: {time.time() - started_at:.1f}s")
    return prepared


def main(
    n_folds: int = cfg.N_FOLDS,
    step_days: int | None = cfg.STEP_DAYS,
    *,
    user_policy: UserPolicy = "seen_only",
    output_dir: str | Path = cfg.DATA_DIR,
    data_path: str | Path | None = None,
    sample_path: str | Path | None = None,
    expected_test_rows: int | None = None,
) -> None:
    print("=" * 70)
    print("BUILD TEMPORAL DATASET")
    print("=" * 70)

    resolved_data_path = Path(cfg.DATA_PATH if data_path is None else data_path)
    synthetic_mode = (
        resolved_data_path.name == cfg.SYNTHETIC_DATA_PATH.name
        or resolved_data_path.resolve() == cfg.SYNTHETIC_DATA_PATH.resolve()
    )
    if expected_test_rows is None:
        expected_test_rows = None if synthetic_mode else cfg.EXPECTED_SUBMISSION_ROWS

    df = load_raw(resolved_data_path, allow_future=synthetic_mode)
    df = optimize_dtypes(df)
    df[cfg.ID_COL] = df[cfg.ID_COL].astype("int64")
    df[cfg.DATE_COL] = pd.to_datetime(df[cfg.DATE_COL], errors="raise")

    sample = load_sample_submission(
        cfg.SAMPLE_SUBMISSION_PATH if sample_path is None else sample_path,
        expected_rows=expected_test_rows,
    )
    fixed_user_universe = _ordered_int64_ids(sample[cfg.ID_COL])

    cv_folds = build_cv_folds(n_folds=n_folds, step_days=step_days)
    test_fold = build_test_fold()
    validate_fold_contract(cv_folds, test_fold)

    print(f"Official/fixed user universe: {len(fixed_user_universe):,}")
    print("Temporal fold contract:")
    for fold in cv_folds:
        print(
            f"  {fold.name}: cutoff={fold.cutoff} | "
            f"target={fold.target_start}..{fold.target_end}"
        )
    print(
        f"  test: cutoff={test_fold.cutoff} | "
        f"target={test_fold.target_start}..{test_fold.target_end}"
    )

    feature_column_union: list[str] = []
    prepared_outputs: list[tuple[object, Path]] = []
    for fold in [*cv_folds, test_fold]:
        prepared = process_single_fold(
            fold,
            df,
            fixed_user_universe,
            output_dir=output_dir,
            user_policy=user_policy,
            expected_test_ids=(fixed_user_universe if fold.name == "test" else None),
        )
        for column in prepared.columns:
            if column not in TARGET_COLUMNS and column not in feature_column_union:
                feature_column_union.append(column)
        output_path = Path(output_dir) / (
            "test_features.parquet"
            if fold.name == "test"
            else f"{fold.name}.parquet"
        )
        prepared_outputs.append((fold, output_path))

    print(
        f"Aligning {len(prepared_outputs)} datasets to one "
        f"{len(feature_column_union) - 3}-feature contract ..."
    )
    reference_schema: tuple[tuple[str, str], ...] | None = None
    for fold, output_path in prepared_outputs:
        prepared = pd.read_parquet(output_path)
        prepared = _align_to_feature_contract(prepared, feature_column_union)
        _validate_prepared_frame(
            prepared,
            fold_name=fold.name,
            cutoff=fold.cutoff,
            is_test=(fold.name == "test"),
            expected_test_ids=(
                fixed_user_universe if fold.name == "test" else None
            ),
        )
        schema = _feature_schema(prepared)
        if reference_schema is None:
            reference_schema = schema
        elif schema != reference_schema:
            raise AssertionError(
                f"{fold.name}: feature columns/dtypes still differ after alignment."
            )
        prepared.to_parquet(output_path, index=False)

    print("All fold/test feature columns, order and dtypes match.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build leakage-safe temporal datasets.")
    parser.add_argument("--n-folds", type=int, default=cfg.N_FOLDS)
    parser.add_argument("--step-days", type=int, default=cfg.STEP_DAYS)
    parser.add_argument(
        "--user-policy",
        choices=("seen_only", "all_users"),
        default="seen_only",
        help="Labeled-fold population policy. Test always follows official sample IDs.",
    )
    parser.add_argument("--data-path", type=Path, default=cfg.DATA_PATH)
    parser.add_argument("--sample-path", type=Path, default=cfg.SAMPLE_SUBMISSION_PATH)
    parser.add_argument("--output-dir", type=Path, default=cfg.DATA_DIR)
    parser.add_argument(
        "--expected-test-rows",
        type=int,
        default=None,
        help="Defaults to 250000 for official data and sample length for synthetic data.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        n_folds=args.n_folds,
        step_days=args.step_days,
        user_policy=args.user_policy,
        output_dir=args.output_dir,
        data_path=args.data_path,
        sample_path=args.sample_path,
        expected_test_rows=args.expected_test_rows,
    )
