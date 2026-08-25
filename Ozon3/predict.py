import argparse
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
SUB_DIR = Path("submissions")
SUB_DIR.mkdir(parents=True, exist_ok=True)


def predict_log_target(device="cpu"):
    test_path = DATA_DIR / "test_features.parquet"
    if not test_path.exists():
        raise FileNotFoundError(f"Файл {test_path} не найден!")

    print(f"Загрузка тестовых данных из {test_path}...")
    test_df = pd.read_parquet(test_path)

    exclude_cols = {
        "user_id",
        "target",
        "target_log",
        "fold",
        "cutoff_date",
        "first_order_dt",
        "last_order_dt",
    }
    feature_cols = [c for c in test_df.columns if c not in exclude_cols]

    model_files = sorted(list(MODELS_DIR.glob("model_fold_*.txt")))
    if not model_files:
        raise FileNotFoundError(
            f"Модели не найдены в {MODELS_DIR}. Сначала запустите train.py!"
        )

    print(f"Найдено моделей: {len(model_files)}")

    test_preds_list = []

    for model_path in model_files:
        print(f"Инференс модели: {model_path.name}")
        model = lgb.Booster(model_file=str(model_path))

        preds_log = model.predict(test_df[feature_cols])
        preds_gmv = np.expm1(np.clip(preds_log, 0, None))
        test_preds_list.append(preds_gmv)

    final_preds = np.mean(test_preds_list, axis=0)

    submission = pd.DataFrame(
        {"user_id": test_df["user_id"], "predict": final_preds}
    )

    out_path = SUB_DIR / "submission_log_target.csv"
    submission.to_csv(out_path, index=False)
    print("\n" + "=" * 40)
    print(f"Сабмит успешно сохранен в: {out_path}")
    print(f"Всего строк: {len(submission)}")
    print("Первые 5 строк:")
    print(submission.head())
    print("=" * 40)


def main(mode, device):
    if mode in ["log_target", "tweedie", "two_stage"]:
        predict_log_target(device=device)
    else:
        raise ValueError(f"Неизвестный режим: {mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Инференс моделей LightGBM")
    parser.add_argument(
        "--mode",
        choices=["log_target", "tweedie", "two_stage"],
        default="log_target",
        help="Режим инференса",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu", "cuda"],
        default="cpu",
        help="Устройство для инференса (cpu, gpu или cuda)",
    )
    args = parser.parse_args()
    main(mode=args.mode, device=args.device)