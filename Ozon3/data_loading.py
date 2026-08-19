"""
Загрузка данных.

ВАЖНО: данные - разреженные (строка есть только в дни с активностью).
Мы НЕ делаем reindex на полный календарь для всех юзеров - это раздует
30M строк до ~100M. Все временные агрегаты считаем через groupby +
фильтрацию по датам, а "пропущенные" дни учитываем аналитически
(gap-фичи, recency), а не материализуем их как нулевые строки.
"""
import pandas as pd
import numpy as np
from pathlib import Path

import config as cfg


def load_raw(path: str | Path = cfg.DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    print(f"[load_raw] читаю {path} ...")

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        # dtype-подсказки экономят память на 30M строк
        dtype_map = {c: "int32" for c in cfg.FLAG_COLS + cfg.COUNT_COLS}
        dtype_map.update({c: "float32" for c in cfg.GMV_COLS})
        df = pd.read_csv(
            path,
            parse_dates=[cfg.DATE_COL],
            dtype=dtype_map,
        )
    else:
        raise ValueError(f"Неизвестный формат файла: {path.suffix}")

    print(f"[load_raw] строк: {len(df):,}, юзеров: {df[cfg.ID_COL].nunique():,}")
    print(f"[load_raw] период: {df[cfg.DATE_COL].min()} .. {df[cfg.DATE_COL].max()}")

    # компактный dtype для user_id, если это int
    if pd.api.types.is_integer_dtype(df[cfg.ID_COL]):
        df[cfg.ID_COL] = df[cfg.ID_COL].astype("int32")

    df = df.sort_values([cfg.ID_COL, cfg.DATE_COL]).reset_index(drop=True)
    return df


def basic_sanity_checks(df: pd.DataFrame) -> None:
    """Быстрые проверки на старте EDA."""
    print("\n=== Sanity checks ===")
    print("Пропуски по колонкам:\n", df.isna().sum())
    print("\nДубликаты (user_id, event_date):",
          df.duplicated(subset=[cfg.ID_COL, cfg.DATE_COL]).sum())
    print("\nОтрицательные gmv:", (df["gmv"] < 0).sum() if "gmv" in df else "n/a")
    print("\nСредняя плотность записей на юзера:",
          (len(df) / df[cfg.ID_COL].nunique()))
    total_days = (cfg.HIST_END - cfg.HIST_START).days + 1
    print(f"Всего календарных дней в истории: {total_days}")
    print("Если бы данные были dense: "
          f"{df[cfg.ID_COL].nunique() * total_days:,} строк "
          "(вот почему НЕ делаем reindex).")


def get_all_user_ids(df: pd.DataFrame) -> pd.Series:
    return df[cfg.ID_COL].drop_duplicates().reset_index(drop=True)


if __name__ == "__main__":
    df = load_raw()
    basic_sanity_checks(df)
