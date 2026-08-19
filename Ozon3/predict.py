"""
Инференс на тест (2026-02-14 .. 2026-03-15) и сборка submission.csv.

Перед запуском:
  1) python build_dataset.py   - строит фолды + test_features.parquet
  2) python train.py           - обучает модель на всех данных до теста

Запуск:
    python predict.py --mode tweedie
"""
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb

import config as cfg
from train import get_feature_cols, load_fold, train_tweedie, train_two_stage, predict_two_stage


def main(mode: str = "tweedie"):
    # Для финального сабмита обучаем на ВСЕХ доступных CV-фолдах (максимум данных)
    fold_0 = load_fold("fold_0")
    fold_1 = load_fold("fold_1")
    fold_2 = load_fold("fold_2")
    full_train = pd.concat([fold_0, fold_1, fold_2], ignore_index=True)

    # держим маленький holdout для early stopping (последние даты из fold_2 переиспользуем,
    # либо, если хочешь честнее - пересобери отдельный fold_3 ближе к тесту в build_dataset.py)
    train_df = pd.concat([fold_0, fold_1], ignore_index=True)
    valid_df = fold_2

    test_feats = pd.read_parquet(cfg.DATA_DIR / "test_features.parquet")
    feature_cols = get_feature_cols(train_df)

    # sanity: колонки теста и трейна должны совпадать
    missing_in_test = set(feature_cols) - set(test_feats.columns)
    if missing_in_test:
        raise ValueError(f"В test_features.parquet не хватает колонок: {missing_in_test}")

    if mode == "tweedie":
        model = train_tweedie(train_df, valid_df, feature_cols)
        preds = np.clip(model.predict(test_feats[feature_cols]), 0, None)
    elif mode == "two_stage":
        clf, reg = train_two_stage(train_df, valid_df, feature_cols)
        preds = predict_two_stage(clf, reg, test_feats, feature_cols)
    else:
        raise ValueError(mode)

    submission = pd.DataFrame({
        cfg.ID_COL: test_feats[cfg.ID_COL],
        "target": preds,   # <-- поправь имя колонки под требования организаторов, если другое
    })

    out_path = cfg.SUB_DIR / f"submission_{mode}.csv"
    submission.to_csv(out_path, index=False)
    print(f"Сабмит сохранён: {out_path}")
    print(submission.describe())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["tweedie", "two_stage"], default="tweedie")
    args = parser.parse_args()
    main(mode=args.mode)
