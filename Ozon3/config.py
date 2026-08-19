from pathlib import Path
import datetime as dt
import os


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
SUB_DIR = PROJECT_DIR / "submissions"
UPLOADS_DIR = PROJECT_DIR / "uploads"

for _d in (DATA_DIR, SUB_DIR, UPLOADS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DATA_PATH = UPLOADS_DIR / "train.parquet"
if os.environ.get("ECUP_USE_SYNTHETIC") == "1":
    DATA_PATH = DATA_DIR / "synthetic_train.parquet"

SAMPLE_SUBMISSION_PATH = UPLOADS_DIR / "sample_submission.csv"

HIST_START = dt.date(2025, 1, 1)
HIST_END = dt.date(2026, 2, 13)
TARGET_START = dt.date(2026, 2, 14)
TARGET_END = dt.date(2026, 3, 15)
TARGET_LEN_DAYS = (TARGET_END - TARGET_START).days + 1

ID_COL = "user_id"
DATE_COL = "event_date"

FLAG_COLS = [
    "search", "cat",
    "has_search_to_cart", "has_search_to_ord",
    "has_cat_to_cart", "has_cat_to_ord",
]
COUNT_COLS = [
    "search_to_cart", "search_to_ord",
    "cat_to_cart", "cat_to_ord",
    "to_cart", "to_ord", "searches",
]
GMV_COLS = ["gmv_search", "gmv_cat", "gmv"]

ALL_NUMERIC_COLS = FLAG_COLS + COUNT_COLS + GMV_COLS

LOOKBACKS = [7, 14, 30, 60, 90, 180, 365]

RANDOM_STATE = 42