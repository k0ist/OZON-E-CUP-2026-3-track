"""
Прописал генерацию синтетического датасета, чтобы
прогнать всё через пайплайн и убедиться в отсутствии багов до загрузки
реальных данных.

Смоук тест, короче
"""
import numpy as np
import pandas as pd
import datetime as dt

import config as cfg

rng = np.random.default_rng(cfg.RANDOM_STATE)

N_USERS = 2000
dates = pd.date_range(cfg.HIST_START, cfg.HIST_END, freq="D")

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
        rows.append((
            uid, d, search, cat,
            int(to_cart > 0 and search), int(to_ord > 0 and search),
            int(to_cart > 0 and cat), int(to_ord > 0 and cat),
            to_cart if search else 0, to_ord if search else 0,
            to_cart if cat else 0, to_ord if cat else 0,
            gmv * 0.6, gmv * 0.4,
            to_cart, to_ord, gmv, searches,
        ))

cols = [cfg.ID_COL, cfg.DATE_COL] + cfg.FLAG_COLS + [
    "search_to_cart", "search_to_ord", "cat_to_cart", "cat_to_ord",
    "gmv_search", "gmv_cat", "to_cart", "to_ord", "gmv", "searches",
]
df = pd.DataFrame(rows, columns=cols)
out_path = cfg.DATA_DIR / "synthetic_train.parquet"
df.to_parquet(out_path, index=False)
print(f"Синтетика сохранена: {out_path}, строк: {len(df):,}, юзеров: {df[cfg.ID_COL].nunique()}")
