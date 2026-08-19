"""
Обучение baseline-модели: LightGBM с Tweedie loss.
Плюс опционально two-stage (classifier P(target>0) x regressor E[target|target>0]).

Оценка на CV фолдах через time-based валидацию (см. time_split.py).

Запуск:
    python train.py --mode tweedie
    python train.py --mode two_stage
"""
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

import config as cfg

NON_FEATURE_COLS = [cfg.ID_COL, "target", "cutoff_date"]


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


def wape(y_true, y_pred):
    """Weighted Absolute Percentage Error - устойчива к нулям, частая метрика для GMV."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    denom = np.abs(y_true).sum()
    return np.abs(y_true - y_pred).sum() / denom if denom > 0 else np.nan


def load_fold(name: str) -> pd.DataFrame:
    return pd.read_parquet(cfg.DATA_DIR / f"{name}.parquet")


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


# ---------------------------------------------------------------------------
# Вариант 1: прямая регрессия с Tweedie loss (хороший baseline "из коробки")
# ---------------------------------------------------------------------------
def train_tweedie(train_df, valid_df, feature_cols, params_override=None):
    params = dict(
        objective="tweedie",
        tweedie_variance_power=1.3,  # 1.1-1.9, обычно 1.2-1.5 для GMV-подобных таргетов
        metric="rmse",
        learning_rate=0.03,
        num_leaves=63,
        min_data_in_leaf=100,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=1.0,
        max_depth=-1,
        verbosity=-1,
        seed=cfg.RANDOM_STATE,
    )
    if params_override:
        params.update(params_override)

    dtrain = lgb.Dataset(train_df[feature_cols], label=train_df["target"])
    dvalid = lgb.Dataset(valid_df[feature_cols], label=valid_df["target"], reference=dtrain)

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=3000,
        valid_sets=[dtrain, dvalid],
        valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
    )
    return model


# ---------------------------------------------------------------------------
# Вариант 2: two-stage (classification P(target>0) + regression на log1p)
# ---------------------------------------------------------------------------
def train_two_stage(train_df, valid_df, feature_cols):
    y_train_bin = (train_df["target"] > 0).astype(int)
    y_valid_bin = (valid_df["target"] > 0).astype(int)

    clf_params = dict(
        objective="binary",
        metric="auc",
        learning_rate=0.03,
        num_leaves=63,
        min_data_in_leaf=100,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        verbosity=-1,
        seed=cfg.RANDOM_STATE,
    )
    dtrain_clf = lgb.Dataset(train_df[feature_cols], label=y_train_bin)
    dvalid_clf = lgb.Dataset(valid_df[feature_cols], label=y_valid_bin, reference=dtrain_clf)
    clf = lgb.train(
        clf_params, dtrain_clf, num_boost_round=2000,
        valid_sets=[dvalid_clf], valid_names=["valid"],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
    )

    # регрессия только на положительных таргетах, в log1p-пространстве
    pos_train = train_df[train_df["target"] > 0]
    pos_valid = valid_df[valid_df["target"] > 0]

    reg_params = dict(
        objective="regression",
        metric="rmse",
        learning_rate=0.03,
        num_leaves=63,
        min_data_in_leaf=50,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=1.0,
        verbosity=-1,
        seed=cfg.RANDOM_STATE,
    )
    dtrain_reg = lgb.Dataset(pos_train[feature_cols], label=np.log1p(pos_train["target"]))
    dvalid_reg = lgb.Dataset(pos_valid[feature_cols], label=np.log1p(pos_valid["target"]), reference=dtrain_reg)
    reg = lgb.train(
        reg_params, dtrain_reg, num_boost_round=2000,
        valid_sets=[dvalid_reg], valid_names=["valid"],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
    )
    return clf, reg


def predict_two_stage(clf, reg, df, feature_cols, threshold=0.5):
    p_buy = clf.predict(df[feature_cols])
    pred_amount = np.expm1(reg.predict(df[feature_cols]))
    pred_amount = np.clip(pred_amount, 0, None)
    # Ожидаемое значение: p(buy) * E[amount | buy]  (мягкий вариант, обычно лучше чем hard threshold)
    return p_buy * pred_amount


def main(mode: str = "tweedie"):
    # используем 2 последних CV фолда: предпоследний для train, последний для valid
    # (можно расширить до полноценного k-fold, для хакатон-baseline этого достаточно)
    fold_0 = load_fold("fold_0")
    fold_1 = load_fold("fold_1")
    fold_2 = load_fold("fold_2")

    train_df = pd.concat([fold_0, fold_1], ignore_index=True)
    valid_df = fold_2

    feature_cols = get_feature_cols(train_df)
    print(f"Число фичей: {len(feature_cols)}")
    print(f"Train: {train_df.shape}, Valid: {valid_df.shape}")

    if mode == "tweedie":
        model = train_tweedie(train_df, valid_df, feature_cols)
        preds = np.clip(model.predict(valid_df[feature_cols]), 0, None)
        model.save_model(str(cfg.DATA_DIR / "model_tweedie.txt"))
    elif mode == "two_stage":
        clf, reg = train_two_stage(train_df, valid_df, feature_cols)
        preds = predict_two_stage(clf, reg, valid_df, feature_cols)
        clf.save_model(str(cfg.DATA_DIR / "model_clf.txt"))
        reg.save_model(str(cfg.DATA_DIR / "model_reg.txt"))
    else:
        raise ValueError(mode)

    y_true = valid_df["target"].values
    print(f"\n=== Метрики на valid ({mode}) ===")
    print(f"RMSE: {rmse(y_true, preds):.4f}")
    print(f"MAE:  {mean_absolute_error(y_true, preds):.4f}")
    print(f"WAPE: {wape(y_true, preds):.4f}")
    print(f"Baseline (persistence, gmv_sum_30d): "
          f"RMSE={rmse(y_true, valid_df['gmv_sum_30d']):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["tweedie", "two_stage"], default="tweedie")
    args = parser.parse_args()
    main(mode=args.mode)
