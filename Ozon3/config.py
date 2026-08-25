import datetime as dt
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
SUB_DIR = PROJECT_DIR / "submissions"
UPLOADS_DIR = PROJECT_DIR / "uploads"

for _d in (DATA_DIR, SUB_DIR, UPLOADS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DATA_PATH = UPLOADS_DIR / "train.parquet"
if not DATA_PATH.exists() and (UPLOADS_DIR / "train.csv").exists():
    DATA_PATH = UPLOADS_DIR / "train.csv"

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
TARGET_COL = "target"
TARGET_LOG_COL = "target_log"

N_FOLDS = 6
STEP_DAYS = 30

NN_TOP_K_FEATURES = 100
NN_HIDDEN_DIM = 256
NN_NUM_BLOCKS = 2
NN_DROPOUT = 0.15
NN_BATCH_SIZE = 4096
NN_EPOCHS = 80
NN_LR = 1e-3
NN_WEIGHT_DECAY = 1e-4
NN_PATIENCE = 8

FLAG_COLS = [
    "search",
    "cat",
    "has_search_to_cart",
    "has_search_to_ord",
    "has_cat_to_cart",
    "has_cat_to_ord",
]
COUNT_COLS = [
    "search_to_cart",
    "search_to_ord",
    "cat_to_cart",
    "cat_to_ord",
    "to_cart",
    "to_ord",
    "searches",
]
GMV_COLS = ["gmv_search", "gmv_cat", "gmv"]

ALL_NUMERIC_COLS = FLAG_COLS + COUNT_COLS + GMV_COLS

LOOKBACKS = [7, 14, 30, 60, 90, 180, 365]

RANDOM_STATE = 42