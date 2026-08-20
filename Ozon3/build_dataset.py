import time
import pandas as pd
from joblib import Parallel, delayed

import config as cfg
from data_loading import load_raw
from time_split import build_folds
from features import build_features_for_cutoff, build_target, optimize_dtypes

def process_single_fold(fold, df, all_user_ids):
    t0 = time.time()
    print(f"\n=== Обработка {fold.name} | cutoff={fold.cutoff} ===")

    feats = build_features_for_cutoff(df, fold.cutoff, user_ids=all_user_ids)

    if fold.name != "test":
        target = build_target(df, all_user_ids, fold.target_start, fold.target_end)
        merged = feats.merge(target, on=cfg.ID_COL, how="left")
        out_path = cfg.DATA_DIR / f"{fold.name}.parquet"
        merged.to_parquet(out_path, index=False)
        print(f"Готов {fold.name} -> {out_path} ({time.time()-t0:.1f}s)")
    else:
        out_path = cfg.DATA_DIR / "test_features.parquet"
        feats.to_parquet(out_path, index=False)
        print(f"Готов test_features -> {out_path} ({time.time()-t0:.1f}s)")


def main(n_folds: int = 6, step_days: int = 20):
    df = load_raw()

    # Приводим датафрейм к правильным типам ДО передачи в фолды
    df = optimize_dtypes(df)

    folds = build_folds(n_folds=n_folds, step_days=step_days)
    all_user_ids = df[cfg.ID_COL].drop_duplicates()

    # Используем 2 параллельных процесса, чтобы не забивать шину RAM
    Parallel(n_jobs=2, backend="loky")(
        delayed(process_single_fold)(fold, df, all_user_ids) for fold in folds
    )


if __name__ == "__main__":
    main(n_folds=6, step_days=20)