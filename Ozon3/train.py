import gc
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb

import config as cfg

NON_FEATURE_COLS = [cfg.ID_COL, "target", "cutoff_date"]


def rmsle(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.clip(np.asarray(y_pred, dtype=np.float64), 0, None)
    log_diff = np.log1p(y_true) - np.log1p(y_pred)
    return float(np.sqrt(np.mean(log_diff ** 2)))


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def train_log_target(train_df, valid_df, feature_cols, params_override=None):
    params = dict(
        device="cuda",
        max_bin=127,
        objective="regression",
        metric="rmse",
        learning_rate=0.03,
        num_leaves=96,
        min_data_in_leaf=120,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=1.5,
        verbosity=-1,
        seed=cfg.RANDOM_STATE,
    )
    if params_override:
        params.update(params_override)

    y_train = np.log1p(train_df["target"].clip(lower=0))
    y_valid = np.log1p(valid_df["target"].clip(lower=0))

    # Сдвиг весов в пользу платящих пользователей для минимизации ошибки RMSLE
    weights_train = np.where(train_df["target"] > 0, 1.25, 0.85)

    dtrain = lgb.Dataset(train_df[feature_cols], label=y_train, weight=weights_train)
    dvalid = lgb.Dataset(valid_df[feature_cols], label=y_valid, reference=dtrain)

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=3000,
        valid_sets=[dtrain, dvalid],
        valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)],
    )
    return model


def predict_log_target(model, df, feature_cols):
    pred_log = model.predict(df[feature_cols])
    return np.clip(np.expm1(pred_log), 0, None)


def train_log_target_fixed_rounds(train_df, feature_cols, num_boost_round, params_override=None):
    params = dict(
        device="cuda",
        max_bin=127,
        objective="regression",
        metric="rmse",
        learning_rate=0.03,
        num_leaves=96,
        min_data_in_leaf=120,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=1.5,
        verbosity=-1,
        seed=cfg.RANDOM_STATE,
    )
    if params_override:
        params.update(params_override)

    y_train = np.log1p(train_df["target"].clip(lower=0))
    weights_train = np.where(train_df["target"] > 0, 1.25, 0.85)

    dtrain = lgb.Dataset(train_df[feature_cols], label=y_train, weight=weights_train)
    model = lgb.train(params, dtrain, num_boost_round=num_boost_round)
    return model


def kfold_cv_log_target(fold_frames: list[pd.DataFrame], feature_cols, params_override=None, verbose=True):
    models, best_iterations = [], []
    oof_true, oof_pred = [], []

    n = len(fold_frames)
    for i in range(n):
        valid_df = fold_frames[i]
        train_df = pd.concat([f for j, f in enumerate(fold_frames) if j != i], ignore_index=True)

        model = train_log_target(train_df, valid_df, feature_cols, params_override=params_override)
        preds = predict_log_target(model, valid_df, feature_cols)

        fold_rmsle = rmsle(valid_df["target"].values, preds)
        if verbose:
            print(f"  [kfold_cv] fold {i}: best_iter={model.best_iteration}, valid RMSLE={fold_rmsle:.5f}")

        models.append(model)
        best_iterations.append(model.best_iteration)
        oof_true.append(valid_df["target"].values)
        oof_pred.append(preds)

        # Очистка GPU памяти между фолдами
        del train_df, model
        gc.collect()

    oof_true = np.concatenate(oof_true)
    oof_pred = np.concatenate(oof_pred)
    oof_score = rmsle(oof_true, oof_pred)

    if verbose:
        print(f"\n  [kfold_cv] OOF RMSLE: {oof_score:.5f}")

    return models, best_iterations, oof_score


def load_all_cv_folds() -> list[pd.DataFrame]:
    paths = sorted(cfg.DATA_DIR.glob("fold_*.parquet"), key=lambda p: int(p.stem.split("_")[1]))
    if not paths:
        raise FileNotFoundError(f"Файлы fold_*.parquet не найдены в {cfg.DATA_DIR}.")
    return [pd.read_parquet(p) for p in paths]


def main(mode: str = "log_target"):
    fold_frames = load_all_cv_folds()
    feature_cols = get_feature_cols(fold_frames[0])

    if mode == "log_target":
        models, best_iterations, oof_score = kfold_cv_log_target(fold_frames, feature_cols)
        print(f"\nИтоговый OOF RMSLE: {oof_score:.5f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["log_target"], default="log_target")
    args = parser.parse_args()
    main(mode=args.mode)