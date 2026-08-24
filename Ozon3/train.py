import argparse
import os
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def rmsle(y_true, y_pred):
    y_true_clean = np.clip(y_true, 0, None)
    y_pred_clean = np.clip(y_pred, 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_pred_clean) - np.log1p(y_true_clean)) ** 2)))


def load_all_cv_folds():
    fold_files = sorted(list(DATA_DIR.glob("fold_*.parquet")))
    if not fold_files:
        raise FileNotFoundError(
            f"Не найдено ни одного файла fold_*.parquet в {DATA_DIR.resolve()}. Сначала запустите build_dataset.py!"
        )
    print(f"Загрузка {len(fold_files)} фолдов из {DATA_DIR}...")
    folds = [pd.read_parquet(f) for f in fold_files]
    return folds


def train_log_target_lgb(device="cpu"):
    fold_frames = load_all_cv_folds()
    num_folds = len(fold_frames)

    exclude_cols = {
        "user_id",
        "target",
        "target_log",
        "fold",
        "cutoff_date",
        "first_order_dt",
        "last_order_dt",
    }
    feature_cols = [c for c in fold_frames[0].columns if c not in exclude_cols]

    print(f"\n--- Starting K-Fold CV ({device.upper()} Mode) ---")
    print(f"Number of features: {len(feature_cols)}")

    lgb_device = "cuda" if device in ["gpu", "cuda"] else "cpu"

    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_child_samples": 50,
        "verbose": -1,
        "n_jobs": -1,
        "device": lgb_device,
    }

    oof_preds = []
    targets = []

    for val_fold_idx in range(num_folds):
        val_df = fold_frames[val_fold_idx]
        train_df = pd.concat(
            [fold_frames[i] for i in range(num_folds) if i != val_fold_idx],
            ignore_index=True,
        )

        X_train, y_train = train_df[feature_cols], train_df["target_log"]
        X_val, y_val = val_df[feature_cols], val_df["target_log"]

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)]

        try:
            model = lgb.train(
                params,
                train_data,
                num_boost_round=1500,
                valid_sets=[train_data, val_data],
                callbacks=callbacks,
            )
        except Exception as e:
            if lgb_device != "cpu":
                print(f"[Предупреждение] Ошибка GPU ({e}). Переключение на CPU...")
                params["device"] = "cpu"
                model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=1500,
                    valid_sets=[train_data, val_data],
                    callbacks=callbacks,
                )
            else:
                raise e

        val_preds_log = model.predict(X_val)
        val_preds_gmv = np.expm1(np.clip(val_preds_log, 0, None))

        fold_rmsle = rmsle(val_df["target"].values, val_preds_gmv)
        print(f"Fold {val_fold_idx} RMSLE (GMV): {fold_rmsle:.5f}")

        oof_preds.extend(val_preds_gmv)
        targets.extend(val_df["target"].values)

        model_path = MODELS_DIR / f"model_fold_{val_fold_idx}.txt"
        model.save_model(str(model_path))

    overall_rmsle = rmsle(np.array(targets), np.array(oof_preds))
    print("\n" + "=" * 40)
    print(f"ИТОГОВЫЙ OOF RMSLE (Log-Target LightGBM): {overall_rmsle:.5f}")
    print("=" * 40)


def main(mode, device):
    if mode == "log_target":
        train_log_target_lgb(device=device)
    else:
        raise ValueError(f"Неизвестный режим: {mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обучение моделей LightGBM")
    parser.add_argument(
        "--mode",
        choices=["log_target"],
        default="log_target",
        help="Режим обучения",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu", "cuda"],
        default="cpu",
        help="Устройство для вычислений (cpu, gpu или cuda)",
    )
    args = parser.parse_args()
    main(mode=args.mode, device=args.device)