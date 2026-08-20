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
        dtype_map = {c: "int32" for c in cfg.FLAG_COLS + cfg.COUNT_COLS}
        dtype_map.update({c: "float32" for c in cfg.GMV_COLS})
        df = pd.read_csv(
            path,
            parse_dates=[cfg.DATE_COL],
            dtype=dtype_map,
        )
    else:
        raise ValueError(f"Неизвестный формат файла: {path.suffix}")

    if not pd.api.types.is_datetime64_any_dtype(df[cfg.DATE_COL]):
        df[cfg.DATE_COL] = pd.to_datetime(df[cfg.DATE_COL])

    print(f"[Загрузка] строк: {len(df):,}, юзеров: {df[cfg.ID_COL].nunique():,}")
    print(f"[Загрузка] период: {df[cfg.DATE_COL].min()} .. {df[cfg.DATE_COL].max()}")

    if pd.api.types.is_integer_dtype(df[cfg.ID_COL]):
        df[cfg.ID_COL] = df[cfg.ID_COL].astype("int32")

    df = df.sort_values([cfg.ID_COL, cfg.DATE_COL]).reset_index(drop=True)
    return df


def basic_checks(df: pd.DataFrame) -> None:
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
    basic_checks(df)