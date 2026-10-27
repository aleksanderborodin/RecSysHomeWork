from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW, INTERIM, PROCESSED = DATA / "raw", DATA / "interim", DATA / "processed"
ARTIFACTS = ROOT / "artifacts"
FIGURES = ARTIFACTS / "figures"
TUNING = ARTIFACTS / "tuning"
TEST = ARTIFACTS / "test"

SEED = 42
K = 10
PRIMARY_METRIC = f"ndcg@{K}"
POSITIVE_RATING = 4


SOURCE_PAGE = "https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html"
BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads"
FILES = {
    "interactions": f"{BASE_URL}/byGenre/goodreads_interactions_poetry.json.gz",
    "books": f"{BASE_URL}/byGenre/goodreads_books_poetry.json.gz",
    "reviews": f"{BASE_URL}/byGenre/goodreads_reviews_poetry.json.gz",
    "authors": f"{BASE_URL}/goodreads_book_authors.json.gz",
}


def _utc(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


MIN_DATE = _utc("2007-01-01")
VAL_START = _utc("2017-03-01")
TEST_START = _utc("2017-06-01")
TEST_END = _utc("2017-11-01")


KCORE_USER = 15
KCORE_ITEM = 15


GRID_POP = {"window_days": [None, 30, 90, 180, 365]}
GRID_ITEMKNN = {"k": [50, 100, 200, 500], "shrinkage": [0, 10, 50, 200], "source": ["all", "positive"]}
GRID_EASE = {"reg": [50, 100, 200, 500, 1000, 2000, 5000], "source": ["all", "positive"]}
GRID_IALS = {"factors": [32, 64, 128, 256], "reg": [0.1, 1, 10, 100, 300, 1000], "alpha": [1, 2, 5, 10, 20, 50]}


GRID_SIGNAL = {"tau": [None, 0.25, 0.75], "alpha_neg": [None, 10, 40], "w_shelf": [0.3, 1.0],
               "half_life": [180, 365, 730, 1095, 1825], "alpha_rev": [0, 5]}
SIGNAL_OFF = {"tau": None, "alpha_neg": None, "w_shelf": 1.0, "half_life": None, "alpha_rev": 0}

GRID_FM = {"feature_set": ["ids+item+user"], "factors": [64], "lr": [1e-3, 3e-3], "reg": [1e-5, 1e-4, 1e-3]}
FM_FEATURE_SETS = ["ids", "ids+item", "ids+item+user"]
FM_CHECKPOINTS = [1, 2, 5, 10, 20, 30, 40, 50, 60]
IALS_ITERS = 15
