import datetime as dt
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
SUB_DIR = PROJECT_DIR / "submissions"
UPLOADS_DIR = PROJECT_DIR / "uploads"
MODELS_DIR = DATA_DIR / "models"
OOF_DIR = DATA_DIR / "oof"
REPORTS_DIR = PROJECT_DIR / "reports"

# Only output directories are created automatically.  ``uploads`` is an input
# directory and must not be silently created when the competition files are
# missing.
for _d in (DATA_DIR, SUB_DIR, MODELS_DIR, OOF_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

TRAIN_DATA_CANDIDATES = (
    UPLOADS_DIR / "train.parquet",
    UPLOADS_DIR / "train.csv",
)
SAMPLE_SUBMISSION_CANDIDATES = (
    UPLOADS_DIR / "sample_submit.csv",
    UPLOADS_DIR / "sample_submission.csv",
)


def _first_existing(candidates: tuple[Path, ...]) -> Path:
    """Return the first existing candidate, or the preferred path.

    Existence is validated by the loader so importing config remains safe for
    the synthetic-data generator.
    """

    return next((path for path in candidates if path.is_file()), candidates[0])


DATA_PATH = _first_existing(TRAIN_DATA_CANDIDATES)
SAMPLE_SUBMISSION_PATH = _first_existing(SAMPLE_SUBMISSION_CANDIDATES)
SYNTHETIC_DATA_PATH = DATA_DIR / "synthetic_train.parquet"
SYNTHETIC_SAMPLE_SUBMISSION_PATH = DATA_DIR / "synthetic_sample_submit.csv"

if os.environ.get("ECUP_USE_SYNTHETIC") == "1":
    DATA_PATH = SYNTHETIC_DATA_PATH
    SAMPLE_SUBMISSION_PATH = SYNTHETIC_SAMPLE_SUBMISSION_PATH

HIST_START = dt.date(2025, 1, 1)
HIST_END = dt.date(2026, 2, 13)
TARGET_START = dt.date(2026, 2, 14)
TARGET_END = dt.date(2026, 3, 15)
TARGET_LEN_DAYS = (TARGET_END - TARGET_START).days + 1
EXPECTED_SUBMISSION_ROWS = 250_000

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
BTYD_BACKEND = os.environ.get("ECUP_BTYD_BACKEND", "fallback")
if BTYD_BACKEND not in {"fallback", "lifetimes"}:
    raise ValueError(
        "ECUP_BTYD_BACKEND must be either 'fallback' or 'lifetimes', "
        f"got {BTYD_BACKEND!r}."
    )
