import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import config as cfg

DATA_DIR = cfg.DATA_DIR
MODELS_DIR = cfg.DATA_DIR / "models"
OOF_DIR = cfg.DATA_DIR / "oof"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OOF_DIR.mkdir(parents=True, exist_ok=True)


def load_best_params():
    path = DATA_DIR / "best_params.json"
    if path.exists():
        with open(path) as f:
            params = json.load(f)
        print(f"[train] загружены гиперпараметры из tune.py: {path}")
        return params
    return None


def rmsle(
    y_true,
    y_pred,
):

    y_true = np.clip(
        y_true,
        0,
        None,
    )

    y_pred = np.clip(
        y_pred,
        0,
        None,
    )

    return float(
        np.sqrt(
            np.mean(
                (
                    np.log1p(y_pred)
                    - np.log1p(y_true)
                )
                ** 2
            )
        )
    )


def load_folds():

    files = sorted(
        DATA_DIR.glob(
            "fold_*.parquet"
        )
    )

    if not files:
        raise FileNotFoundError(
            "Нет fold_*.parquet. "
            "Сначала запусти build_dataset.py"
        )

    folds = []

    for f in files:

        df = pd.read_parquet(f)

        print(
            f"{f.name}: "
            f"{len(df):,} строк"
        )

        folds.append(df)

    return folds


def get_features(
    folds,
):

    exclude = {
        "user_id",
        "target",
        "target_log",
        "fold",
        "cutoff_date",
        "first_order_dt",
        "last_order_dt",
    }

    return [
        c for c in folds[0].columns
        if c not in exclude
        and pd.api.types.is_numeric_dtype(
            folds[0][c]
        )
    ]


def train_single_model(
    train_df,
    val_df,
    feature_cols,
    params,
):

    X_train = train_df[
        feature_cols
    ]

    y_train = train_df[
        "target_log"
    ]

    X_val = val_df[
        feature_cols
    ]

    y_val = val_df[
        "target_log"
    ]

    train_data = lgb.Dataset(
        X_train,
        label=y_train,
        free_raw_data=False,
    )

    val_data = lgb.Dataset(
        X_val,
        label=y_val,
        reference=train_data,
        free_raw_data=False,
    )

    callbacks = [
        lgb.early_stopping(
            stopping_rounds=100,
            verbose=False,
        ),
        lgb.log_evaluation(
            period=200
        ),
    ]

    model = lgb.train(
        params,
        train_data,
        num_boost_round=5000,
        valid_sets=[
            train_data,
            val_data,
        ],
        valid_names=[
            "train",
            "valid",
        ],
        callbacks=callbacks,
    )

    return model


def train_temporal_cv(
    device="cpu",
):

    folds = load_folds()

    feature_cols = get_features(
        folds
    )

    print()
    print("=" * 70)
    print("TEMPORAL LIGHTGBM CV")
    print("=" * 70)

    print(
        f"Количество признаков: "
        f"{len(feature_cols)}"
    )

    # ---------------------------------------------------------
    # Основная конфигурация
    # ---------------------------------------------------------

    lgb_device = (
        "cuda"
        if device in ("gpu", "cuda")
        else "cpu"
    )

    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",

        "learning_rate": 0.02,

        "num_leaves": 63,
        "max_depth": -1,

        "min_data_in_leaf": 100,

        "feature_fraction": 0.80,
        "bagging_fraction": 0.80,
        "bagging_freq": 1,

        "lambda_l1": 0.05,
        "lambda_l2": 1.0,

        "max_bin": 255,

        "verbosity": -1,
        "n_jobs": -1,

        "device": lgb_device,
    }

    best_params = load_best_params()
    if best_params:
        params.update(best_params)

    oof_predictions = []
    oof_targets = []
    oof_ids = []

    models = []

    # ---------------------------------------------------------
    # Temporal CV
    # ---------------------------------------------------------

    for val_idx in range(1, len(folds)):

        val_df = folds[val_idx]

        train_parts = folds[
            :val_idx
        ]

        train_df = pd.concat(
            train_parts,
            ignore_index=True,
        )

        print()
        print("-" * 70)
        print(
            f"VALIDATION FOLD: "
            f"fold_{val_idx}"
        )

        print(
            f"Train: "
            f"{len(train_df):,}"
        )

        print(
            f"Validation: "
            f"{len(val_df):,}"
        )

        try:

            model = train_single_model(
                train_df,
                val_df,
                feature_cols,
                params,
            )

        except Exception as e:

            if lgb_device != "cpu":

                print(
                    "[WARNING] GPU "
                    "LightGBM failed:"
                )

                print(e)

                print(
                    "Переключаемся на CPU..."
                )

                params["device"] = "cpu"

                model = train_single_model(
                    train_df,
                    val_df,
                    feature_cols,
                    params,
                )

            else:

                raise

        pred_log = model.predict(
            val_df[feature_cols],
            num_iteration=model.best_iteration,
        )

        pred = np.expm1(
            np.clip(
                pred_log,
                0,
                None,
            )
        )

        score = rmsle(
            val_df["target"].values,
            pred,
        )

        print()
        print(
            f"fold_{val_idx} RMSLE = "
            f"{score:.6f}"
        )

        print(
            f"best_iteration = "
            f"{model.best_iteration}"
        )

        model_path = (
            MODELS_DIR
            / f"model_fold_{val_idx}.txt"
        )

        model.save_model(
            str(model_path)
        )

        print(
            f"Модель сохранена: "
            f"{model_path}"
        )

        oof_predictions.extend(
            pred
        )

        oof_targets.extend(
            val_df["target"].values
        )

        oof_ids.extend(
            val_df["user_id"].values
        )

        models.append(model)

    # ---------------------------------------------------------
    # OOF
    # ---------------------------------------------------------

    oof_predictions = np.asarray(
        oof_predictions
    )

    oof_targets = np.asarray(
        oof_targets
    )

    overall = rmsle(
        oof_targets,
        oof_predictions,
    )

    print()
    print("=" * 70)
    print(
        f"ИТОГОВЫЙ TEMPORAL OOF RMSLE: "
        f"{overall:.6f}"
    )
    print("=" * 70)

    oof_df = pd.DataFrame(
        {
            "user_id": oof_ids,
            "target": oof_targets,
            "pred": oof_predictions,
        }
    )

    oof_path = (
        OOF_DIR
        / "oof_lgbm.csv"
    )

    oof_df.to_csv(
        oof_path,
        index=False,
    )

    print(
        f"OOF сохранён: "
        f"{oof_path}"
    )

    # ---------------------------------------------------------
    # Feature importance
    # ---------------------------------------------------------

    importance = np.zeros(
        len(feature_cols)
    )

    for model in models:

        importance += (
            model.feature_importance(
                importance_type="gain"
            )
        )

    importance /= len(models)

    fi = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": importance,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    fi_path = (
        DATA_DIR
        / "lgbm_feature_importances.csv"
    )

    fi.to_csv(
        fi_path,
        index=False,
    )

    print(
        f"Feature importance "
        f"сохранён: {fi_path}"
    )

    print()
    print("Top-30 features:")
    print(
        fi.head(30).to_string(
            index=False
        )
    )

    return (
        overall,
        models,
        feature_cols,
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["log_target"],
        default="log_target",
    )

    parser.add_argument(
        "--device",
        choices=[
            "cpu",
            "gpu",
            "cuda",
        ],
        default="cpu",
    )

    args = parser.parse_args()

    if args.mode != "log_target":
        raise ValueError(
            "Поддерживается только "
            "log_target"
        )

    train_temporal_cv(
        device=args.device
    )


if __name__ == "__main__":
    main()