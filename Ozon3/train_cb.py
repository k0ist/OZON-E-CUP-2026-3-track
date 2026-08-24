import os
from pathlib import Path
import warnings
from catboost import CatBoostRegressor
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

DATA_DIR = Path('data')
SUB_DIR = Path('submissions')
SUB_DIR.mkdir(parents=True, exist_ok=True)


def rmsle(y_true, y_pred):
  y_true_clean = np.clip(y_true, 0, None)
  y_pred_clean = np.clip(y_pred, 0, None)
  return float(
      np.sqrt(np.mean((np.log1p(y_pred_clean) - np.log1p(y_true_clean)) ** 2))
  )


print('=== Обучение CatBoost на фолдах ===')
fold_files = sorted(list(DATA_DIR.glob('fold_*.parquet')))
test_df = pd.read_parquet(DATA_DIR / 'test_features.parquet')
folds = [pd.read_parquet(f) for f in fold_files]

exclude_cols = {
    'user_id',
    'target',
    'target_log',
    'fold',
    'cutoff_date',
    'first_order_dt',
    'last_order_dt',
}
feature_cols = [c for c in folds[0].columns if c not in exclude_cols]

oof_preds = []
targets = []
test_preds_list = []

cb_params = {
    'iterations': 1500,
    'learning_rate': 0.04,
    'depth': 6,
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'random_seed': 42,
    'verbose': 200,
    'task_type': 'CPU',  # Смените на 'GPU', если есть NVIDIA карта
}

for val_fold_idx, val_df in enumerate(folds):
  print(f'\n--- Fold {val_fold_idx} ---')
  train_df = pd.concat(
      [folds[i] for i in range(len(folds)) if i != val_fold_idx],
      ignore_index=True,
  )

  X_train, y_train = (
      train_df[feature_cols].fillna(0),
      train_df['target_log'].values,
  )
  X_val, y_val = val_df[feature_cols].fillna(0), val_df['target_log'].values
  X_test = test_df[feature_cols].fillna(0)

  model = CatBoostRegressor(**cb_params)
  model.fit(
      X_train,
      y_train,
      eval_set=(X_val, y_val),
      early_stopping_rounds=100,
      use_best_model=True,
  )

  val_preds_log = model.predict(X_val)
  val_preds_gmv = np.expm1(np.clip(val_preds_log, 0, None))

  oof_preds.extend(val_preds_gmv)
  targets.extend(val_df['target'].values)

  test_preds_log = model.predict(X_test)
  test_preds_list.append(np.expm1(np.clip(test_preds_log, 0, None)))

overall_rmsle = rmsle(np.array(targets), np.array(oof_preds))
print(f'\nИтоговый OOF RMSLE (CatBoost): {overall_rmsle:.5f}')

final_cb_preds = np.mean(test_preds_list, axis=0)
sub_cb = pd.DataFrame({'user_id': test_df['user_id'], 'predict': final_cb_preds})
sub_cb.to_csv(SUB_DIR / 'submission_catboost.csv', index=False)
print(f'Сабмит сохранен в: {SUB_DIR / "submission_catboost.csv"}')