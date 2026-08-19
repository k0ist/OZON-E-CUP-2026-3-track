"""
Собирает фичи+таргет для каждого CV-фолда и фичи (без таргета) для теста.
Результаты кладёт в data/ в виде parquet - чтобы не пересчитывать при
каждом эксперименте с моделью.

Запуск:
    python build_dataset.py
"""
import pandas as pd
import time

import config as cfg
from data_loading import load_raw
from time_split import build_folds
from features import build_features_for_cutoff, build_target


def main(n_folds: int = 3, step_days: int = 30):
    df = load_raw()
    folds = build_folds(n_folds=n_folds, step_days=step_days)

    all_user_ids = df[cfg.ID_COL].drop_duplicates()

    for fold in folds:
        t0 = time.time()
        print(f"\n=== {fold.name} | cutoff={fold.cutoff} | "
              f"target=[{fold.target_start}, {fold.target_end}] ===")

        feats = build_features_for_cutoff(df, fold.cutoff, user_ids=all_user_ids)
        print(f"  фичи построены: {feats.shape}, {time.time()-t0:.1f}s")

        if fold.name != "test":
            target = build_target(df, all_user_ids, fold.target_start, fold.target_end)
            merged = feats.merge(target, on=cfg.ID_COL, how="left")
            out_path = cfg.DATA_DIR / f"{fold.name}.parquet"
            merged.to_parquet(out_path, index=False)
            print(f"  сохранено -> {out_path} | "
                  f"target: mean={merged['target'].mean():.2f}, "
                  f"nonzero_share={(merged['target']>0).mean():.3f}")
        else:
            out_path = cfg.DATA_DIR / "test_features.parquet"
            feats.to_parquet(out_path, index=False)
            print(f"  сохранено -> {out_path}")


if __name__ == "__main__":
    main(n_folds=3, step_days=30)
