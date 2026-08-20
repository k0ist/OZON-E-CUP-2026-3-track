import numpy as np
import pandas as pd
import datetime as dt

import config as cfg
from btyd_features import calculate_btyd_features


def _to_ts(d: dt.date) -> pd.Timestamp:
    return pd.Timestamp(d)


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Оптимизация типов данных для экономии VRAM/RAM в 2 раза."""
    for col in df.columns:
        if col == cfg.ID_COL:
            continue
        if df[col].dtype == "float64":
            df[col] = df[col].astype("float32")
        elif df[col].dtype == "int64":
            df[col] = df[col].astype("int32")
    return df


def build_features_for_cutoff(
        df: pd.DataFrame,
        cutoff: dt.date,
        user_ids: pd.Series | None = None,
) -> pd.DataFrame:
    cutoff_ts = _to_ts(cutoff)
    hist = df[df[cfg.DATE_COL] <= cutoff_ts]

    if user_ids is None:
        user_ids = hist[cfg.ID_COL].drop_duplicates()

    # Собираем список всех частей датасета (блоки колонок)
    feature_blocks = []

    # Индексный датафрейм
    base = pd.DataFrame({cfg.ID_COL: user_ids.values}).drop_duplicates().set_index(cfg.ID_COL)
    feature_blocks.append(base)

    # 1) Recency / Lifetime
    last_seen = hist.groupby(cfg.ID_COL)[cfg.DATE_COL].max().rename("last_active_date")
    first_seen = hist.groupby(cfg.ID_COL)[cfg.DATE_COL].min().rename("first_active_date")
    n_active_days_total = hist.groupby(cfg.ID_COL)[cfg.DATE_COL].nunique().rename("n_active_days_total")

    life = pd.concat([last_seen, first_seen, n_active_days_total], axis=1)
    life["recency_days"] = (cutoff_ts - life["last_active_date"]).dt.days
    life["tenure_days"] = (cutoff_ts - life["first_active_date"]).dt.days + 1
    life["activity_density_total"] = life["n_active_days_total"] / life["tenure_days"].clip(lower=1)
    life = life.drop(columns=["last_active_date", "first_active_date"])
    feature_blocks.append(life)

    # 2) Purchases Recency & Gaps
    purchases = hist[hist["gmv"] > 0]
    if len(purchases):
        last_purch = purchases.groupby(cfg.ID_COL)[cfg.DATE_COL].max().rename("last_purchase_date")
        purch_rec = (cutoff_ts - last_purch).dt.days.rename("recency_purchase_days").to_frame()
        feature_blocks.append(purch_rec)

        p = purchases.sort_values([cfg.ID_COL, cfg.DATE_COL]).copy()
        p["prev_purchase_date"] = p.groupby(cfg.ID_COL)[cfg.DATE_COL].shift(1)
        p["gap"] = (p[cfg.DATE_COL] - p["prev_purchase_date"]).dt.days
        gap_stats = p.groupby(cfg.ID_COL)["gap"].agg(
            mean_purchase_gap="mean", std_purchase_gap="std", n_purchase_days="count"
        )
        feature_blocks.append(gap_stats)

    # 3) Lookback aggregations
    agg_cols = cfg.COUNT_COLS + cfg.GMV_COLS
    lookback_dfs = []
    for L in cfg.LOOKBACKS:
        window_start = cutoff_ts - pd.Timedelta(days=L - 1)
        w = hist[hist[cfg.DATE_COL] >= window_start]
        if len(w) > 0:
            g = w.groupby(cfg.ID_COL)[agg_cols].sum()
            g[f"n_active_days_{L}d"] = w.groupby(cfg.ID_COL)[cfg.DATE_COL].nunique()
            if (w["gmv"] > 0).any():
                g[f"n_purchase_days_{L}d"] = w[w["gmv"] > 0].groupby(cfg.ID_COL)[cfg.DATE_COL].nunique()
            else:
                g[f"n_purchase_days_{L}d"] = 0

            g.columns = [f"{c}_sum_{L}d" if c in agg_cols else c for c in g.columns]
            lookback_dfs.append(g)

    if lookback_dfs:
        all_lookbacks = pd.concat(lookback_dfs, axis=1)
        feature_blocks.append(all_lookbacks)

    # 4) BTYD
    btyd_feats = calculate_btyd_features(df, cutoff).set_index(cfg.ID_COL)
    feature_blocks.append(btyd_feats)

    # 5) EMA Signals
    ema_blocks = []
    for w in [7, 30, 90]:
        w_start = cutoff_ts - pd.Timedelta(days=w - 1)
        sub = hist[hist[cfg.DATE_COL] >= w_start].copy()
        if len(sub) > 0:
            sub["decay_weight"] = np.exp(-0.05 * (cutoff_ts - sub[cfg.DATE_COL]).dt.days)
            sub["weighted_gmv"] = sub["gmv"] * sub["decay_weight"]
            ema_g = sub.groupby(cfg.ID_COL)["weighted_gmv"].sum().rename(f"ema_gmv_{w}d")
            ema_blocks.append(ema_g)
    if ema_blocks:
        feature_blocks.append(pd.concat(ema_blocks, axis=1))

    # СБОРКА ВСЕХ ФИЧЕЙ В ОДИН ДАТАФРЕЙМ БЕЗ MERGE
    feats = pd.concat(feature_blocks, axis=1).reset_index()

    # 6) Ratios (выполняем один раз над цельным датафреймом)
    for L in cfg.LOOKBACKS:
        gmv_col = f"gmv_sum_{L}d"
        to_ord_col = f"to_ord_sum_{L}d"
        to_cart_col = f"to_cart_sum_{L}d"
        searches_col = f"searches_sum_{L}d"
        active_days_col = f"n_active_days_{L}d"

        if gmv_col in feats.columns and active_days_col in feats.columns:
            feats[f"gmv_per_active_day_{L}d"] = feats[gmv_col] / feats[active_days_col].clip(lower=1)
            feats[f"avg_order_value_{L}d"] = feats[gmv_col] / feats[to_ord_col].clip(lower=1)
            feats[f"cart_to_order_rate_{L}d"] = feats[to_ord_col] / feats[to_cart_col].clip(lower=1)
            feats[f"search_to_cart_rate_{L}d"] = feats[to_cart_col] / feats[searches_col].clip(lower=1)
            feats[f"activity_density_{L}d"] = feats[active_days_col] / L

    # 7) Missing Values & Downcasting
    fill_zero_cols = [c for c in feats.columns if c.endswith(("d", "total")) and c != cfg.ID_COL]
    for c in fill_zero_cols:
        if c in feats.columns and feats[c].dtype != object:
            feats[c] = feats[c].fillna(0)

    max_possible_recency = (cutoff_ts - _to_ts(cfg.HIST_START)).days + 1
    feats["recency_days"] = feats["recency_days"].fillna(max_possible_recency)
    if "recency_purchase_days" in feats.columns:
        feats["recency_purchase_days"] = feats["recency_purchase_days"].fillna(max_possible_recency)
    if "mean_purchase_gap" in feats.columns:
        feats["mean_purchase_gap"] = feats["mean_purchase_gap"].fillna(-1)
        feats["std_purchase_gap"] = feats["std_purchase_gap"].fillna(-1)

    feats["cutoff_date"] = cutoff_ts

    # Дефрагментируем память полностью один раз
    feats = feats.copy()
    return optimize_dtypes(feats)


def build_target(
    df: pd.DataFrame,
    user_ids: pd.Series,
    target_start: dt.date,
    target_end: dt.date,
) -> pd.DataFrame:
    t_start_ts, t_end_ts = _to_ts(target_start), _to_ts(target_end)
    window = df[(df[cfg.DATE_COL] >= t_start_ts) & (df[cfg.DATE_COL] <= t_end_ts)]
    target = window.groupby(cfg.ID_COL)["gmv"].sum().rename("target").reset_index()

    base = pd.DataFrame({cfg.ID_COL: user_ids.values}).drop_duplicates()
    target = base.merge(target, on=cfg.ID_COL, how="left")
    target["target"] = target["target"].fillna(0.0).astype("float32")
    return target