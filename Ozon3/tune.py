import argparse
import json

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.integration import LightGBMPruningCallback

import config as cfg
from train import load_folds, get_features, rmsle


def walk_forward_oof_rmsle(params, folds, feature_cols, trial=None,
                            num_boost_round=1500, early_stopping=50,
                            n_val_folds=2):
    """
    Оценивает только n_val_folds ПОСЛЕДНИХ шагов walk-forward (а не все 5) -
    для тюнинга этого достаточно и в 2-3 раза быстрее. Финальную модель на
    ВСЕХ шагах обучайте отдельно через train.py с уже подобранными params.
    """
    val_indices = range(len(folds) - n_val_folds, len(folds))
    oof_pred, oof_true = [], []

    for val_idx in val_indices:
        train_df = pd.concat(folds[:val_idx], ignore_index=True)
        val_df = folds[val_idx]

        dtrain = lgb.Dataset(train_df[feature_cols], label=train_df[cfg.TARGET_LOG_COL], free_raw_data=False)
        dval = lgb.Dataset(val_df[feature_cols], label=val_df[cfg.TARGET_LOG_COL], reference=dtrain, free_raw_data=False)

        callbacks = [lgb.early_stopping(early_stopping, verbose=False), lgb.log_evaluation(0)]
        if trial is not None:
            callbacks.append(LightGBMPruningCallback(trial, "rmse", valid_name="valid"))

        model = lgb.train(
            params, dtrain, num_boost_round=num_boost_round,
            valid_sets=[dval], valid_names=["valid"], callbacks=callbacks,
        )

        pred_log = model.predict(val_df[feature_cols], num_iteration=model.best_iteration)
        pred = np.expm1(np.clip(pred_log, 0, None))
        oof_pred.append(pred)
        oof_true.append(val_df[cfg.TARGET_COL].values)

    oof_pred = np.concatenate(oof_pred)
    oof_true = np.concatenate(oof_true)
    return rmsle(oof_true, oof_pred)


def objective(trial, folds, feature_cols, n_val_folds):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "n_jobs": -1,
        "seed": cfg.RANDOM_STATE,
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 300, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "max_bin": trial.suggest_categorical("max_bin", [127, 255]),
    }
    score = walk_forward_oof_rmsle(params, folds, feature_cols, trial=trial, n_val_folds=n_val_folds)
    print(f"  trial {trial.number}: RMSLE={score:.5f}, params={trial.params}")
    return score


def main(n_trials=40, sample_frac=0.3, timeout=None, n_val_folds=2):
    folds = load_folds()
    feature_cols = get_features(folds)

    if sample_frac < 1.0:
        folds = [f.sample(frac=sample_frac, random_state=cfg.RANDOM_STATE) for f in folds]
        print(f"Сэмплирую {sample_frac:.0%} каждого фолда для ускорения тюнинга")

    print(f"Фолдов: {len(folds)} (валидируем на последних {n_val_folds}), "
          f"фичей: {len(feature_cols)}, trials: {n_trials}")

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=cfg.RANDOM_STATE),
        pruner=MedianPruner(n_warmup_steps=200),
    )
    study.optimize(
        lambda t: objective(t, folds, feature_cols, n_val_folds),
        n_trials=n_trials, timeout=timeout,
    )

    print(f"\nЛучший OOF RMSLE (на {n_val_folds} последних фолдах, sample_frac={sample_frac}): {study.best_value:.5f}")
    print(json.dumps(study.best_params, indent=2, ensure_ascii=False))

    out_path = cfg.DATA_DIR / "best_params.json"
    with open(out_path, "w") as f:
        json.dump(study.best_params, f, indent=2, ensure_ascii=False)
    print(f"Сохранено -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--sample-frac", type=float, default=0.3)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--n-val-folds", type=int, default=2)
    args = parser.parse_args()
    main(n_trials=args.n_trials, sample_frac=args.sample_frac, timeout=args.timeout, n_val_folds=args.n_val_folds)