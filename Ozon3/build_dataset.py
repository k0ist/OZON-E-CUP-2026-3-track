import time

import numpy as np
import pandas as pd

import config as cfg

from data_loading import load_raw
from features import (
    build_features_for_cutoff,
    build_target,
    optimize_dtypes,
)
from time_split import build_folds


def process_single_fold(
    fold,
    df,
    all_user_ids,
):

    t0 = time.time()

    print()
    print("=" * 70)
    print(
        f"{fold.name} | cutoff={fold.cutoff}"
    )
    print("=" * 70)

    # ---------------------------------------------------------
    # Пользователи, существующие на cutoff
    # ---------------------------------------------------------

    hist = df[
        df[cfg.DATE_COL]
        <= pd.Timestamp(fold.cutoff)
    ]

    existing_users = (
        hist[cfg.ID_COL]
        .drop_duplicates()
    )

    print(
        f"Пользователей на cutoff: "
        f"{len(existing_users):,}"
    )

    # ---------------------------------------------------------
    # Features
    # ---------------------------------------------------------

    feats = build_features_for_cutoff(
        df,
        fold.cutoff,
        user_ids=existing_users,
    )

    print(
        f"Features построены: "
        f"{feats.shape}"
    )

    # ---------------------------------------------------------
    # Test
    # ---------------------------------------------------------

    if fold.name == "test":

        out_path = (
            cfg.DATA_DIR
            / "test_features.parquet"
        )

        feats.to_parquet(
            out_path,
            index=False,
        )

        print(
            f"Готов test -> {out_path}"
        )

        return

    # ---------------------------------------------------------
    # Target
    # ---------------------------------------------------------

    target = build_target(
        df,
        existing_users,
        fold.target_start,
        fold.target_end,
    )

    merged = feats.merge(
        target,
        on=cfg.ID_COL,
        how="left",
    )

    merged[cfg.TARGET_COL] = (
        merged[cfg.TARGET_COL]
        .fillna(0)
        .clip(lower=0)
        .astype("float32")
    )

    merged[cfg.TARGET_LOG_COL] = np.log1p(
        merged[cfg.TARGET_COL]
    ).astype("float32")

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    out_path = (
        cfg.DATA_DIR
        / f"{fold.name}.parquet"
    )

    merged = optimize_dtypes(
        merged
    )

    merged.to_parquet(
        out_path,
        index=False,
    )

    print(
        f"Готов {fold.name} -> "
        f"{out_path}"
    )

    print(
        f"Строк: {len(merged):,}"
    )

    print(
        f"Positive target: "
        f"{(merged[cfg.TARGET_COL] > 0).mean():.4%}"
    )

    print(
        f"Время: "
        f"{time.time() - t0:.1f}s"
    )


def main(
    n_folds=6,
    step_days=None,
):

    print("=" * 70)
    print("BUILD TEMPORAL DATASET")
    print("=" * 70)

    df = load_raw()

    df = optimize_dtypes(
        df
    )

    df[cfg.DATE_COL] = pd.to_datetime(
        df[cfg.DATE_COL]
    )

    folds = build_folds(
        n_folds=n_folds,
        step_days=step_days,
    )

    all_user_ids = (
        df[cfg.ID_COL]
        .drop_duplicates()
    )

    print(
        f"Всего пользователей: "
        f"{len(all_user_ids):,}"
    )

    for fold in folds:

        process_single_fold(
            fold,
            df,
            all_user_ids,
        )


if __name__ == "__main__":

    main(
        n_folds=6,
        step_days=None,
    )