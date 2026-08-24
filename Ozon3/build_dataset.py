import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import config as cfg
from data_loading import load_raw
from features import build_features_for_cutoff, build_target, optimize_dtypes
from time_split import build_folds


def process_single_fold(fold, df, all_user_ids):
    t0 = time.time()
    print(f"\n=== Обработка {fold.name} | cutoff={fold.cutoff} ===")

    # Генерируем фичи на момент cutoff
    feats = build_features_for_cutoff(df, fold.cutoff, user_ids=all_user_ids)

    if fold.name != "test":
        # Генерируем таргет за целевой период
        target_df = build_target(
            df, all_user_ids, fold.target_start, fold.target_end
        )

        # Объединяем фичи и таргет
        merged = feats.merge(target_df, on=cfg.ID_COL, how="left")

        # Заполняем возможные пропуски в таргете нулями
        if cfg.TARGET_COL in merged.columns:
            merged[cfg.TARGET_COL] = merged[cfg.TARGET_COL].fillna(0.0)
        else:
            # На случай, если функция build_target возвращает колонку под другим именем
            val_col = [c for c in target_df.columns if c != cfg.ID_COL][0]
            merged[cfg.TARGET_COL] = merged[val_col].fillna(0.0)

        # ФИКС: Создаем колонку target_log, необходимую для train.py
        merged[cfg.TARGET_LOG_COL] = np.log1p(
            np.maximum(merged[cfg.TARGET_COL], 0)
        )

        out_path = cfg.DATA_DIR / f"{fold.name}.parquet"
        merged.to_parquet(out_path, index=False)
        print(
            f"Готов {fold.name} -> {out_path} | строк: {len(merged):,} ({time.time()-t0:.1f}s)"
        )
    else:
        out_path = cfg.DATA_DIR / "test_features.parquet"
        feats.to_parquet(out_path, index=False)
        print(
            f"Готов test_features -> {out_path} | строк: {len(feats):,} ({time.time()-t0:.1f}s)"
        )


def main(n_folds: int = 6, step_days: int = 20):
    df = load_raw()

    # Приводим датафрейм к правильным типам ДО передачи в фолды
    df = optimize_dtypes(df)

    folds = build_folds(n_folds=n_folds, step_days=step_days)
    all_user_ids = df[cfg.ID_COL].drop_duplicates().reset_index(drop=True)

    # Последовательная обработка фолдов (избегает дублирования памяти в RAM на Windows)
    for fold in folds:
        process_single_fold(fold, df, all_user_ids)


if __name__ == "__main__":
    main(n_folds=6, step_days=20)