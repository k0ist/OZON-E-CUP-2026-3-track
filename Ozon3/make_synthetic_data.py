"""Generate deterministic, sparse data for end-to-end pipeline smoke tests.

Rows after ``HIST_END`` are deliberate leakage canaries: test features must be
identical whether those future rows are present or removed.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg


DEFAULT_N_USERS = 2_000
DEFAULT_COLD_START_USERS = 25
COLUMNS = [cfg.ID_COL, cfg.DATE_COL, *cfg.ALL_NUMERIC_COLS]


def _make_event_row(
    user_id: int,
    event_date: pd.Timestamp,
    rng: np.random.Generator,
    *,
    force_purchase: bool = False,
) -> tuple:
    search = int(rng.integers(0, 2))
    cat = int(rng.integers(0, 2))
    searches = int(rng.poisson(3)) if search else 0
    to_cart = int(rng.poisson(1)) if (search or cat) else 0
    if force_purchase:
        to_ord = int(rng.poisson(1)) + 1
    else:
        to_ord = int(rng.poisson(0.3)) if rng.random() < 0.15 else 0
    gmv = float(to_ord * rng.uniform(500, 5_000)) if to_ord > 0 else 0.0

    has_search_to_cart = int(to_cart > 0 and search)
    has_search_to_ord = int(to_ord > 0 and search)
    has_cat_to_cart = int(to_cart > 0 and cat)
    has_cat_to_ord = int(to_ord > 0 and cat)
    search_to_cart = to_cart if search else 0
    search_to_ord = to_ord if search else 0
    cat_to_cart = to_cart if cat else 0
    cat_to_ord = to_ord if cat else 0
    gmv_search = gmv * 0.6 if search else 0.0
    gmv_cat = gmv * 0.4 if cat else 0.0

    return (
        user_id,
        event_date,
        search,
        cat,
        has_search_to_cart,
        has_search_to_ord,
        has_cat_to_cart,
        has_cat_to_ord,
        search_to_cart,
        search_to_ord,
        cat_to_cart,
        cat_to_ord,
        to_cart,
        to_ord,
        searches,
        gmv_search,
        gmv_cat,
        gmv,
    )


def generate_synthetic_data(
    *,
    n_users: int = DEFAULT_N_USERS,
    cold_start_users: int = DEFAULT_COLD_START_USERS,
    seed: int = cfg.RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if n_users < 1 or cold_start_users < 0:
        raise ValueError("n_users must be positive and cold_start_users non-negative.")

    rng = np.random.default_rng(seed)
    dates = pd.date_range(cfg.HIST_START, cfg.TARGET_END, freq="D")
    rows: list[tuple] = []

    for user_id in range(n_users):
        probability_active = rng.beta(1.5, 8)
        active_dates = dates[rng.random(len(dates)) < probability_active]
        for event_date in active_dates:
            rows.append(_make_event_row(user_id, event_date, rng))

    # These IDs are known from the sample submission but have no history at the
    # test cutoff.  They exercise all_users and exact sample-order alignment.
    cold_ids = range(n_users, n_users + cold_start_users)
    cold_dates = pd.date_range(cfg.TARGET_START, periods=3, freq="7D")
    for user_id in cold_ids:
        for event_date in cold_dates:
            rows.append(
                _make_event_row(
                    user_id,
                    event_date,
                    rng,
                    force_purchase=True,
                )
            )

    raw = pd.DataFrame(rows, columns=COLUMNS)
    raw[cfg.ID_COL] = raw[cfg.ID_COL].astype("int64")
    raw[cfg.DATE_COL] = pd.to_datetime(raw[cfg.DATE_COL])

    sample_ids = np.arange(n_users + cold_start_users, dtype="int64")
    rng.shuffle(sample_ids)
    sample = pd.DataFrame(
        {
            cfg.ID_COL: sample_ids,
            "predict": np.zeros(len(sample_ids), dtype="float64"),
        }
    )

    if cold_start_users:
        cold_mask = raw[cfg.ID_COL].isin(list(cold_ids))
        if (raw.loc[cold_mask, cfg.DATE_COL] <= pd.Timestamp(cfg.HIST_END)).any():
            raise AssertionError("Synthetic cold-start users unexpectedly have history.")
    return raw, sample


def main(
    *,
    output_dir: str | Path = cfg.DATA_DIR,
    n_users: int = DEFAULT_N_USERS,
    cold_start_users: int = DEFAULT_COLD_START_USERS,
    seed: int = cfg.RANDOM_STATE,
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw, sample = generate_synthetic_data(
        n_users=n_users,
        cold_start_users=cold_start_users,
        seed=seed,
    )

    raw_path = output_dir / cfg.SYNTHETIC_DATA_PATH.name
    sample_path = output_dir / cfg.SYNTHETIC_SAMPLE_SUBMISSION_PATH.name
    raw.to_parquet(raw_path, index=False)
    sample.to_csv(sample_path, index=False)
    print(
        f"Synthetic raw -> {raw_path} | rows={len(raw):,} | "
        f"users={raw[cfg.ID_COL].nunique():,}"
    )
    print(
        f"Synthetic sample -> {sample_path} | rows={len(sample):,} | "
        f"cold-start users={cold_start_users:,}"
    )
    return raw_path, sample_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=cfg.DATA_DIR)
    parser.add_argument("--n-users", type=int, default=DEFAULT_N_USERS)
    parser.add_argument(
        "--cold-start-users", type=int, default=DEFAULT_COLD_START_USERS
    )
    parser.add_argument("--seed", type=int, default=cfg.RANDOM_STATE)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        output_dir=args.output_dir,
        n_users=args.n_users,
        cold_start_users=args.cold_start_users,
        seed=args.seed,
    )
