"""Прописал генерацию синтетического датасета, чтобы прогнать всё через пайплайн и

убедиться в отсутствии багов до загрузки реальных данных.

Смоук тест, короче
"""

import datetime as dt
import numpy as np
import pandas as pd

import config as cfg

rng = np.random.default_rng(cfg.RANDOM_STATE)

N_USERS = 2000
# Генерируем даты до TARGET_END, чтобы проверить правильность работы расчета таргета
dates = pd.date_range(cfg.HIST_START, cfg.TARGET_END, freq="D")

rows = []
for uid in range(N_USERS):
    p_active = rng.beta(1.5, 8)
    active_mask = rng.random(len(dates)) < p_active
    active_dates = dates[active_mask]
    if len(active_dates) == 0:
        continue

    for d in active_dates:
        search = rng.integers(0, 2)
        cat = rng.integers(0, 2)
        searches = rng.poisson(3) if search else 0
        to_cart = rng.poisson(1) if (search or cat) else 0

        buy_prob = 0.15
        to_ord = rng.poisson(0.3) if rng.random() < buy_prob else 0
        gmv = float(to_ord * rng.uniform(500, 5000)) if to_ord > 0 else 0.0

        # Флаги транзакций
        has_search_to_cart = int(to_cart > 0 and search)
        has_search_to_ord = int(to_ord > 0 and search)
        has_cat_to_cart = int(to_cart > 0 and cat)
        has_cat_to_ord = int(to_ord > 0 and cat)

        # Счетчики транзакций
        search_to_cart = to_cart if search else 0
        search_to_ord = to_ord if search else 0
        cat_to_cart = to_cart if cat else 0
        cat_to_ord = to_ord if cat else 0

        # GMV разрез
        gmv_search = gmv * 0.6 if search else 0.0
        gmv_cat = gmv * 0.4 if cat else 0.0

        rows.append((
            uid,
            d,
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
        ))

# Полный и явный порядок колонок согласно config.py
cols = (
    [cfg.ID_COL, cfg.DATE_COL]
    + cfg.FLAG_COLS
    + [
        "search_to_cart",
        "search_to_ord",
        "cat_to_cart",
        "cat_to_ord",
        "to_cart",
        "to_ord",
        "searches",
        "gmv_search",
        "gmv_cat",
        "gmv",
    ]
)

df = pd.DataFrame(rows, columns=cols)

# Сохраняем синтетику
out_path = cfg.DATA_DIR / "synthetic_train.parquet"
df.to_parquet(out_path, index=False)
print(
    f"Синтетика сохранена: {out_path}, строк: {len(df):,}, юзеров: {df[cfg.ID_COL].nunique():,}"
)