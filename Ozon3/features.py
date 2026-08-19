"""
Построение фичей на юзер-уровне относительно cutoff-даты.

Все фичи считаются ТОЛЬКО по строкам с event_date <= cutoff.
Это гарантирует отсутствие лика из таргет-периода.

Данные разреженные -> вместо reindex на полный календарь используем:
  - groupby + агрегации по срезам [cutoff - lookback + 1, cutoff]
  - gap-фичи через diff() дат внутри groupby('user_id')
  - recency = (cutoff - last_active_date).days
"""
import numpy as np
import pandas as pd
import datetime as dt

import config as cfg


def _to_ts(d: dt.date) -> pd.Timestamp:
    return pd.Timestamp(d)


def build_features_for_cutoff(
    df: pd.DataFrame,
    cutoff: dt.date,
    user_ids: pd.Series | None = None,
) -> pd.DataFrame:
    """
    df: полный сырой лог (event_date, user_id, ...)
    cutoff: последняя допустимая дата истории (включительно)
    user_ids: список юзеров, для которых нужно построить фичи
              (если None - берём всех, кто встречался в истории до cutoff)

    Возвращает DataFrame с одной строкой на user_id и набором фичей.
    """
    cutoff_ts = _to_ts(cutoff)
    hist = df[df[cfg.DATE_COL] <= cutoff_ts]

    if user_ids is None:
        user_ids = hist[cfg.ID_COL].drop_duplicates()

    base = pd.DataFrame({cfg.ID_COL: user_ids.values}).drop_duplicates()
    feats = base.copy()

    # -----------------------------------------------------------------
    # 1) Recency / lifetime фичи (по всей доступной истории юзера)
    # -----------------------------------------------------------------
    last_seen = hist.groupby(cfg.ID_COL)[cfg.DATE_COL].max().rename("last_active_date")
    first_seen = hist.groupby(cfg.ID_COL)[cfg.DATE_COL].min().rename("first_active_date")
    n_active_days_total = hist.groupby(cfg.ID_COL)[cfg.DATE_COL].nunique().rename("n_active_days_total")

    life = pd.concat([last_seen, first_seen, n_active_days_total], axis=1).reset_index()
    life["recency_days"] = (cutoff_ts - life["last_active_date"]).dt.days
    life["tenure_days"] = (cutoff_ts - life["first_active_date"]).dt.days + 1
    life["activity_density_total"] = life["n_active_days_total"] / life["tenure_days"].clip(lower=1)
    life = life.drop(columns=["last_active_date", "first_active_date"])
    feats = feats.merge(life, on=cfg.ID_COL, how="left")

    # -----------------------------------------------------------------
    # 2) Последняя дата покупки (recency по покупкам, не только визитам)
    # -----------------------------------------------------------------
    purchases = hist[hist["gmv"] > 0]
    if len(purchases):
        last_purch = purchases.groupby(cfg.ID_COL)[cfg.DATE_COL].max().rename("last_purchase_date")
        purch_recency = last_purch.reset_index()
        purch_recency["recency_purchase_days"] = (cutoff_ts - purch_recency["last_purchase_date"]).dt.days
        purch_recency = purch_recency.drop(columns=["last_purchase_date"])
        feats = feats.merge(purch_recency, on=cfg.ID_COL, how="left")
    else:
        feats["recency_purchase_days"] = np.nan

    # -----------------------------------------------------------------
    # 3) Межпокупочные интервалы (frequency, для BTYD-подобных фичей)
    # -----------------------------------------------------------------
    if len(purchases):
        p = purchases.sort_values([cfg.ID_COL, cfg.DATE_COL]).copy()
        p["prev_purchase_date"] = p.groupby(cfg.ID_COL)[cfg.DATE_COL].shift(1)
        p["gap"] = (p[cfg.DATE_COL] - p["prev_purchase_date"]).dt.days
        gap_stats = p.groupby(cfg.ID_COL)["gap"].agg(
            mean_purchase_gap="mean", std_purchase_gap="std", n_purchase_days="count"
        ).reset_index()
        feats = feats.merge(gap_stats, on=cfg.ID_COL, how="left")
    else:
        feats["mean_purchase_gap"] = np.nan
        feats["std_purchase_gap"] = np.nan
        feats["n_purchase_days"] = 0

    # -----------------------------------------------------------------
    # 4) Лукбэк-агрегаты: суммы/средние по окнам [cutoff-L+1, cutoff]
    # -----------------------------------------------------------------
    agg_cols = cfg.COUNT_COLS + cfg.GMV_COLS
    for L in cfg.LOOKBACKS:
        window_start = cutoff_ts - pd.Timedelta(days=L - 1)
        w = hist[hist[cfg.DATE_COL] >= window_start]
        if len(w) == 0:
            g = pd.DataFrame({cfg.ID_COL: []})
        else:
            g = w.groupby(cfg.ID_COL)[agg_cols].sum()
            g[f"n_active_days_{L}d"] = w.groupby(cfg.ID_COL)[cfg.DATE_COL].nunique()
            g[f"n_purchase_days_{L}d"] = (
                w[w["gmv"] > 0].groupby(cfg.ID_COL)[cfg.DATE_COL].nunique()
                if (w["gmv"] > 0).any() else 0
            )
            g = g.rename(columns={c: f"{c}_sum_{L}d" for c in agg_cols})
            g = g.reset_index()
        feats = feats.merge(g, on=cfg.ID_COL, how="left")

    # -----------------------------------------------------------------
    # 5) Производные (rate/conversion) фичи по каждому лукбэку
    # -----------------------------------------------------------------
    for L in cfg.LOOKBACKS:
        gmv_col = f"gmv_sum_{L}d"
        to_ord_col = f"to_ord_sum_{L}d"
        to_cart_col = f"to_cart_sum_{L}d"
        searches_col = f"searches_sum_{L}d"
        active_days_col = f"n_active_days_{L}d"

        if gmv_col in feats:
            feats[f"gmv_per_active_day_{L}d"] = (
                feats[gmv_col] / feats[active_days_col].clip(lower=1)
            )
            feats[f"avg_order_value_{L}d"] = (
                feats[gmv_col] / feats[to_ord_col].clip(lower=1)
            )
            feats[f"cart_to_order_rate_{L}d"] = (
                feats[to_ord_col] / feats[to_cart_col].clip(lower=1)
            )
            feats[f"search_to_cart_rate_{L}d"] = (
                feats[to_cart_col] / feats[searches_col].clip(lower=1)
            )
            feats[f"activity_density_{L}d"] = feats[active_days_col] / L

    # -----------------------------------------------------------------
    # 6) Momentum: сравнение "последние 7д" vs "предыдущие 7д" (14д окно)
    # -----------------------------------------------------------------
    if "gmv_sum_7d" in feats and "gmv_sum_14d" in feats:
        prev7_gmv = (feats["gmv_sum_14d"] - feats["gmv_sum_7d"]).clip(lower=0)
        feats["gmv_momentum_7v7"] = feats["gmv_sum_7d"] - prev7_gmv
        feats["gmv_momentum_ratio_7v7"] = feats["gmv_sum_7d"] / (prev7_gmv + 1.0)

    if "to_ord_sum_7d" in feats and "to_ord_sum_14d" in feats:
        prev7_ord = (feats["to_ord_sum_14d"] - feats["to_ord_sum_7d"]).clip(lower=0)
        feats["order_momentum_7v7"] = feats["to_ord_sum_7d"] - prev7_ord

    # -----------------------------------------------------------------
    # 7) Заполнение пропусков
    # -----------------------------------------------------------------
    fill_zero_cols = [c for c in feats.columns if c.endswith(("d", "total")) and c != cfg.ID_COL]
    for c in fill_zero_cols:
        if c in feats.columns and feats[c].dtype != object:
            feats[c] = feats[c].fillna(0)

    # recency для тех, кто вообще не встречался - большое число (никогда не были активны)
    max_possible_recency = (cutoff_ts - _to_ts(cfg.HIST_START)).days + 1
    feats["recency_days"] = feats["recency_days"].fillna(max_possible_recency)
    feats["recency_purchase_days"] = feats["recency_purchase_days"].fillna(max_possible_recency)
    feats["mean_purchase_gap"] = feats["mean_purchase_gap"].fillna(-1)  # маркер "нет покупок"
    feats["std_purchase_gap"] = feats["std_purchase_gap"].fillna(-1)

    feats["cutoff_date"] = cutoff_ts
    feats = feats.copy()  # de-fragment после множества merge/insert
    return feats


def build_target(
    df: pd.DataFrame,
    user_ids: pd.Series,
    target_start: dt.date,
    target_end: dt.date,
) -> pd.DataFrame:
    """Считает target = sum(gmv) за [target_start, target_end] на каждого юзера из user_ids."""
    t_start_ts, t_end_ts = _to_ts(target_start), _to_ts(target_end)
    window = df[(df[cfg.DATE_COL] >= t_start_ts) & (df[cfg.DATE_COL] <= t_end_ts)]
    target = window.groupby(cfg.ID_COL)["gmv"].sum().rename("target").reset_index()

    base = pd.DataFrame({cfg.ID_COL: user_ids.values}).drop_duplicates()
    target = base.merge(target, on=cfg.ID_COL, how="left")
    target["target"] = target["target"].fillna(0.0)
    return target
