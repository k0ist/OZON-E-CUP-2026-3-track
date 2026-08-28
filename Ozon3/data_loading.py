from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg


RAW_REQUIRED_COLUMNS = (
    cfg.ID_COL,
    cfg.DATE_COL,
    *cfg.ALL_NUMERIC_COLS,
)


def _as_int64_ids(values: pd.Series, *, source: str) -> pd.Series:
    if values.isna().any():
        raise ValueError(f"{source}: {cfg.ID_COL} contains missing values.")
    numeric = pd.to_numeric(values, errors="raise")
    numeric_array = numeric.to_numpy()
    if not np.isfinite(numeric_array).all():
        raise ValueError(f"{source}: {cfg.ID_COL} contains non-finite values.")
    if not np.equal(numeric_array, np.floor(numeric_array)).all():
        raise ValueError(f"{source}: {cfg.ID_COL} must contain integer values.")
    int64_info = np.iinfo(np.int64)
    if numeric.min() < int64_info.min or numeric.max() > int64_info.max:
        raise OverflowError(f"{source}: {cfg.ID_COL} does not fit into int64.")
    return numeric.astype("int64")


def validate_raw_schema(
    df: pd.DataFrame,
    *,
    source: str = "raw data",
    allow_future: bool = False,
) -> None:
    missing = [column for column in RAW_REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{source}: missing required columns: {missing}")
    if df.empty:
        raise ValueError(f"{source}: dataset is empty.")
    if df[cfg.ID_COL].isna().any() or df[cfg.DATE_COL].isna().any():
        raise ValueError(f"{source}: user_id/event_date must not contain missing values.")
    if not pd.api.types.is_datetime64_any_dtype(df[cfg.DATE_COL]):
        raise TypeError(f"{source}: {cfg.DATE_COL} must be datetime64 after parsing.")

    lower_bound = pd.Timestamp(cfg.HIST_START)
    upper_bound = pd.Timestamp(cfg.TARGET_END if allow_future else cfg.HIST_END)
    min_date = df[cfg.DATE_COL].min()
    max_date = df[cfg.DATE_COL].max()
    if min_date < lower_bound or max_date > upper_bound:
        raise ValueError(
            f"{source}: event period {min_date}..{max_date} is outside "
            f"allowed {lower_bound}..{upper_bound}."
        )

    for column in cfg.ALL_NUMERIC_COLS:
        series = df[column]
        if not pd.api.types.is_numeric_dtype(series):
            raise TypeError(f"{source}: {column} must be numeric, got {series.dtype}.")
        if series.isna().any() or not np.isfinite(series.to_numpy()).all():
            raise ValueError(f"{source}: {column} contains NaN or infinite values.")


def load_raw(
    path: str | Path | None = None,
    *,
    allow_future: bool | None = None,
) -> pd.DataFrame:
    path = Path(cfg.DATA_PATH if path is None else path)
    if not path.is_file():
        candidates = ", ".join(str(candidate) for candidate in cfg.TRAIN_DATA_CANDIDATES)
        raise FileNotFoundError(
            f"Training data not found: {path}. Expected one of: {candidates}."
        )
    if path.suffix.lower() not in {".parquet", ".csv"}:
        raise ValueError(f"Unsupported training-data format: {path.suffix}")

    print(f"[load_raw] читаю {path} ...")
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        dtype_map = {column: "int32" for column in cfg.FLAG_COLS + cfg.COUNT_COLS}
        dtype_map.update({column: "float32" for column in cfg.GMV_COLS})
        df = pd.read_csv(path, parse_dates=[cfg.DATE_COL], dtype=dtype_map)

    missing_before_parse = [
        column for column in RAW_REQUIRED_COLUMNS if column not in df.columns
    ]
    if missing_before_parse:
        raise ValueError(f"{path}: missing required columns: {missing_before_parse}")

    df[cfg.DATE_COL] = pd.to_datetime(df[cfg.DATE_COL], errors="raise")
    df[cfg.ID_COL] = _as_int64_ids(df[cfg.ID_COL], source=str(path))
    if allow_future is None:
        allow_future = path.resolve() == cfg.SYNTHETIC_DATA_PATH.resolve()
    validate_raw_schema(df, source=str(path), allow_future=allow_future)

    print(
        f"[Загрузка] строк: {len(df):,}, юзеров: {df[cfg.ID_COL].nunique():,}"
    )
    print(
        f"[Загрузка] период: {df[cfg.DATE_COL].min()} .. {df[cfg.DATE_COL].max()}"
    )

    # The feature builders use explicit groupby operations and sort the only
    # sequence-sensitive purchase-gap slice themselves.  Sorting all 30M raw
    # rows here creates a large second in-memory frame without changing any
    # feature semantics, so preserve the input order.
    return df


def load_sample_submission(
    path: str | Path | None = None,
    *,
    expected_rows: int | None = cfg.EXPECTED_SUBMISSION_ROWS,
) -> pd.DataFrame:
    path = Path(cfg.SAMPLE_SUBMISSION_PATH if path is None else path)
    if not path.is_file():
        candidates = ", ".join(
            str(candidate) for candidate in cfg.SAMPLE_SUBMISSION_CANDIDATES
        )
        raise FileNotFoundError(
            f"Official sample submission not found: {path}. Expected one of: {candidates}."
        )

    sample = pd.read_csv(path)
    expected_columns = [cfg.ID_COL, "predict"]
    if sample.columns.tolist() != expected_columns:
        raise ValueError(
            f"{path}: expected exact columns {expected_columns}, got {sample.columns.tolist()}. "
            "Do not use the unrelated submissions/sample_submission.csv artifact."
        )
    if expected_rows is not None and len(sample) != expected_rows:
        raise ValueError(f"{path}: expected {expected_rows:,} rows, got {len(sample):,}.")

    sample[cfg.ID_COL] = _as_int64_ids(sample[cfg.ID_COL], source=str(path))
    if sample[cfg.ID_COL].duplicated().any():
        raise ValueError(f"{path}: duplicate {cfg.ID_COL} values.")
    if sample["predict"].isna().any() or not np.isfinite(sample["predict"]).all():
        raise ValueError(f"{path}: predict contains NaN or infinite values.")
    return sample


def load_fold(fold_idx: int | str, data_dir: str | Path = cfg.DATA_DIR) -> pd.DataFrame:
    """Load a prepared labeled fold."""

    data_dir = Path(data_dir)
    if isinstance(fold_idx, int) or str(fold_idx).isdigit():
        path = data_dir / f"fold_{fold_idx}.parquet"
    else:
        path = data_dir / f"{fold_idx}.parquet"
    if not path.is_file():
        raise FileNotFoundError(
            f"Fold file not found: {path}. Run build_dataset.py first."
        )
    return pd.read_parquet(path)


def load_test(data_dir: str | Path = cfg.DATA_DIR) -> pd.DataFrame:
    """Load prepared test features."""

    path = Path(data_dir) / "test_features.parquet"
    if not path.is_file():
        raise FileNotFoundError(
            f"Test feature file not found: {path}. Run build_dataset.py first."
        )
    return pd.read_parquet(path)


def basic_checks(df: pd.DataFrame) -> None:
    print("\n=== Sanity checks ===")
    print("Пропуски по колонкам:\n", df.isna().sum())
    print(
        "\nДубликаты (user_id, event_date):",
        df.duplicated(subset=[cfg.ID_COL, cfg.DATE_COL]).sum(),
    )
    print(
        "\nОтрицательные gmv:", (df["gmv"] < 0).sum() if "gmv" in df else "n/a"
    )
    print(
        "\nСредняя плотность записей на юзера:",
        len(df) / df[cfg.ID_COL].nunique(),
    )
    total_days = (cfg.HIST_END - cfg.HIST_START).days + 1
    print(f"Всего календарных дней в истории: {total_days}")
    print(
        "Если бы данные были dense: "
        f"{df[cfg.ID_COL].nunique() * total_days:,} строк "
        "(вот почему НЕ делаем reindex)."
    )


def get_all_user_ids(df: pd.DataFrame) -> pd.Series:
    return df[cfg.ID_COL].drop_duplicates().astype("int64").reset_index(drop=True)


if __name__ == "__main__":
    raw = load_raw()
    basic_checks(raw)
