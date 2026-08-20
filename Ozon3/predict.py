import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg
from train import (
    get_feature_cols, load_all_cv_folds, rmsle,
    train_log_target, predict_log_target, kfold_cv_log_target,
    train_log_target_fixed_rounds,
    train_tweedie, train_two_stage, predict_two_stage,
)


def get_full_user_id_list() -> pd.Series | None:
    path = Path(cfg.SAMPLE_SUBMISSION_PATH)
    if not path.exists():
        print(f"[predict] {path} не найден - беру список юзеров из train "
              "(проверь, что это все требуемые 250 000 клиентов!)")
        return None
    sample = pd.read_csv(path)
    id_col = cfg.ID_COL if cfg.ID_COL in sample.columns else sample.columns[0]
    print(f"[predict] нашёл sample_submission.csv, юзеров: {sample[id_col].nunique():,}")
    return sample[id_col]


def predict_log_target_ensemble(fold_frames, test_feats, feature_cols):
    print("=== Leave-one-fold-out CV (для ансамбля + честной оценки) ===")
    models, best_iterations, oof_score = kfold_cv_log_target(fold_frames, feature_cols)
    print(f"\nOOF RMSLE по {len(fold_frames)} фолдам: {oof_score:.5f} "
          "(это лучшая локальная оценка того, что покажет лидерборд)")

    all_preds = [predict_log_target(m, test_feats, feature_cols) for m in models]

    full_train = pd.concat(fold_frames, ignore_index=True)
    avg_rounds = max(50, int(round(np.mean(best_iterations))))
    print(f"\nДообучаю финальную модель на ВСЕХ {len(fold_frames)} фолдах "
          f"сразу, num_boost_round={avg_rounds} (среднее из CV)")
    full_model = train_log_target_fixed_rounds(full_train, feature_cols, num_boost_round=avg_rounds)
    all_preds.append(predict_log_target(full_model, test_feats, feature_cols))

    ensemble_pred = np.mean(all_preds, axis=0)
    print(f"Ансамбль из {len(all_preds)} моделей ({len(models)} CV + 1 full-data)")
    return ensemble_pred, oof_score


def main(mode: str = "log_target"):
    fold_frames = load_all_cv_folds()
    test_feats = pd.read_parquet(cfg.DATA_DIR / "test_features.parquet")
    feature_cols = get_feature_cols(fold_frames[0])

    missing_in_test = set(feature_cols) - set(test_feats.columns)
    if missing_in_test:
        raise ValueError(f"В test_features.parquet не хватает колонок: {missing_in_test}")

    if mode == "log_target":
        preds, oof_score = predict_log_target_ensemble(fold_frames, test_feats, feature_cols)
    elif mode in ("tweedie", "two_stage"):
        train_df = pd.concat(fold_frames[:-1], ignore_index=True)
        valid_df = fold_frames[-1]
        if mode == "tweedie":
            model = train_tweedie(train_df, valid_df, feature_cols)
            preds = np.clip(model.predict(test_feats[feature_cols]), 0, None)
        else:
            clf, reg = train_two_stage(train_df, valid_df, feature_cols)
            preds = predict_two_stage(clf, reg, test_feats, feature_cols)
    else:
        raise ValueError(mode)

    submission = pd.DataFrame({
        cfg.ID_COL: test_feats[cfg.ID_COL],
        "predict": preds,
    })

    full_ids = get_full_user_id_list()
    if full_ids is not None:
        full_df = pd.DataFrame({cfg.ID_COL: full_ids.values}).drop_duplicates()
        before = len(submission)
        submission = full_df.merge(submission, on=cfg.ID_COL, how="left")
        n_missing = submission["predict"].isna().sum()
        if n_missing:
            print(f"[predict] {n_missing:,} юзеров из sample_submission не нашлось "
                  "в test_features - заполняю 0.0 (нет истории => нет прогноза продаж)")
        submission["predict"] = submission["predict"].fillna(0.0)
        print(f"[predict] строк было {before:,} -> стало {len(submission):,} "
              f"(итог должен совпасть с числом строк в sample_submission)")

    submission["predict"] = submission["predict"].clip(lower=0)

    out_path = cfg.SUB_DIR / f"submission_{mode}.csv"
    submission.to_csv(out_path, index=False)
    print(f"\nСабмит сохранён: {out_path}")
    print(f"Строк: {len(submission):,}, уникальных user_id: {submission[cfg.ID_COL].nunique():,}")
    print(submission["predict"].describe())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["log_target", "tweedie", "two_stage"], default="log_target")
    args = parser.parse_args()
    main(mode=args.mode)