import datetime as dt
import numpy as np
import pandas as pd

from btyd_features import calculate_btyd_features
import config as cfg


def _to_ts(d: dt.date) -> pd.Timestamp:
    return pd.Timestamp(d)


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Оптимизация типов данных для экономии VRAM/RAM."""
    for col in df.columns:
        if col in (cfg.ID_COL, "cutoff_date"):
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

    # Формируем базовый индекс всех пользователей
    base_index = pd.Index(user_ids.unique(), name=cfg.ID_COL)
    feature_blocks = []

    # 1) Recency / Lifetime
    last_seen = hist.groupby(cfg.ID_COL)[cfg.DATE_COL].max()
    first_seen = hist.groupby(cfg.ID_COL)[cfg.DATE_COL].min()
    n_active_days_total = hist.groupby(cfg.ID_COL)[cfg.DATE_COL].nunique()

    life = pd.DataFrame(index=base_index)
    life["recency_days"] = (cutoff_ts - last_seen).dt.days
    life["tenure_days"] = (cutoff_ts - first_seen).dt.days + 1
    life["n_active_days_total"] = n_active_days_total
    life["activity_density_total"] = life["n_active_days_total"] / life[
        "tenure_days"
    ].clip(lower=1)
    feature_blocks.append(life)

    # 2) Purchases Recency & Gaps
    purchases = hist[hist["gmv"] > 0]
    purch_block = pd.DataFrame(index=base_index)
    if len(purchases):
        last_purch = purchases.groupby(cfg.ID_COL)[cfg.DATE_COL].max()
        purch_block["recency_purchase_days"] = (cutoff_ts - last_purch).dt.days

        p = purchases.sort_values([cfg.ID_COL, cfg.DATE_COL]).copy()
        p["prev_purchase_date"] = p.groupby(cfg.ID_COL)[cfg.DATE_COL].shift(1)
        p["gap"] = (p[cfg.DATE_COL] - p["prev_purchase_date"]).dt.days
        gap_stats = p.groupby(cfg.ID_COL)["gap"].agg(
            mean_purchase_gap="mean",
            std_purchase_gap="std",
            n_purchase_days="count",
        )
        purch_block = purch_block.join(gap_stats, how="left")

    feature_blocks.append(purch_block)

    # 3) Lookback aggregations
    agg_cols = cfg.COUNT_COLS + cfg.GMV_COLS
    lookback_dfs = []
    for L in cfg.LOOKBACKS:
        window_start = cutoff_ts - pd.Timedelta(days=L - 1)
        w = hist[hist[cfg.DATE_COL] >= window_start]

        g = pd.DataFrame(index=base_index)
        if len(w) > 0:
            sum_g = w.groupby(cfg.ID_COL)[agg_cols].sum()
            act_days = (
                w.groupby(cfg.ID_COL)[cfg.DATE_COL]
                .nunique()
                .rename(f"n_active_days_{L}d")
            )

            if (w["gmv"] > 0).any():
                purch_days = (
                    w[w["gmv"] > 0]
                    .groupby(cfg.ID_COL)[cfg.DATE_COL]
                    .nunique()
                    .rename(f"n_purchase_days_{L}d")
                )
            else:
                purch_days = pd.Series(0, index=base_index, name=f"n_purchase_days_{L}d")

            # Изменяем имена суммовых колонок
            sum_g.columns = [f"{c}_sum_{L}d" for c in sum_g.columns]

            g = g.join(sum_g, how="left").join(act_days, how="left").join(purch_days, how="left")

        lookback_dfs.append(g)

    if lookback_dfs:
        all_lookbacks = pd.concat(lookback_dfs, axis=1)
        feature_blocks.append(all_lookbacks)

    # 4) BTYD
    btyd_feats = calculate_btyd_features(df, cutoff).set_index(cfg.ID_COL)
    feature_blocks.append(btyd_feats.reindex(base_index))

    # 5) EMA Signals
    ema_block = pd.DataFrame(index=base_index)
    for w in [7, 30, 90]:
        w_start = cutoff_ts - pd.Timedelta(days=w - 1)
        sub = hist[hist[cfg.DATE_COL] >= w_start].copy()
        if len(sub) > 0:
            sub["decay_weight"] = np.exp(
                -0.05 * (cutoff_ts - sub[cfg.DATE_COL]).dt.days
            )
            sub["weighted_gmv"] = sub["gmv"] * sub["decay_weight"]
            ema_g = (
                sub.groupby(cfg.ID_COL)["weighted_gmv"]
                .sum()
                .rename(f"ema_gmv_{w}d")
            )
            ema_block[f"ema_gmv_{w}d"] = ema_g
    feature_blocks.append(ema_block)

    # Сборка всех фичей в один датафрейм
    feats = pd.concat(feature_blocks, axis=1).reset_index()

    # 6) Ratios (выполняем над цельным датафреймом)
    for L in cfg.LOOKBACKS:
        gmv_col = f"gmv_sum_{L}d"
        to_ord_col = f"to_ord_sum_{L}d"
        to_cart_col = f"to_cart_sum_{L}d"
        searches_col = f"searches_sum_{L}d"
        active_days_col = f"n_active_days_{L}d"

        if gmv_col in feats.columns and active_days_col in feats.columns:
            feats[f"gmv_per_active_day_{L}d"] = feats[gmv_col] / feats[
                active_days_col
            ].clip(lower=1)
            feats[f"avg_order_value_{L}d"] = feats[gmv_col] / feats[
                to_ord_col
            ].clip(lower=1)
            feats[f"cart_to_order_rate_{L}d"] = feats[to_ord_col] / feats[
                to_cart_col
            ].clip(lower=1)
            feats[f"search_to_cart_rate_{L}d"] = feats[to_cart_col] / feats[
                searches_col
            ].clip(lower=1)
            feats[f"activity_density_{L}d"] = feats[active_days_col] / L

    # 7) Missing Values & Downcasting
    fill_zero_cols = [
        c
        for c in feats.columns
        if c.endswith(("d", "total")) and c != cfg.ID_COL
    ]
    for c in fill_zero_cols:
        if c in feats.columns and feats[c].dtype != object:
            feats[c] = feats[c].fillna(0)

    max_possible_recency = (cutoff_ts - _to_ts(cfg.HIST_START)).days + 1
    if "recency_days" in feats.columns:
        feats["recency_days"] = feats["recency_days"].fillna(max_possible_recency)
    if "recency_purchase_days" in feats.columns:
        feats["recency_purchase_days"] = feats["recency_purchase_days"].fillna(
            max_possible_recency
        )
    if "mean_purchase_gap" in feats.columns:
        feats["mean_purchase_gap"] = feats["mean_purchase_gap"].fillna(-1)
        feats["std_purchase_gap"] = feats["std_purchase_gap"].fillna(-1)

    feats["cutoff_date"] = cutoff_ts

    # Дефрагментируем память
    feats = feats.copy()
    return optimize_dtypes(feats)


def build_target(
        df: pd.DataFrame,
        user_ids: pd.Series,
        target_start: dt.date,
        target_end: dt.date,
) -> pd.DataFrame:
    """Генерирует целевую переменную target и ее логарифм target_log."""
    t_start_ts, t_end_ts = _to_ts(target_start), _to_ts(target_end)
    window = df[(df[cfg.DATE_COL] >= t_start_ts) & (df[cfg.DATE_COL] <= t_end_ts)]
    target = window.groupby(cfg.ID_COL)["gmv"].sum().rename(cfg.TARGET_COL).reset_index()

    base = pd.DataFrame({cfg.ID_COL: user_ids.unique()})
    target = base.merge(target, on=cfg.ID_COL, how="left")
    target[cfg.TARGET_COL] = target[cfg.TARGET_COL].fillna(0.0).astype("float32")

    # ФИКС: Автоматически добавляем target_log
    target[cfg.TARGET_LOG_COL] = np.log1p(
        np.maximum(target[cfg.TARGET_COL], 0.0)
    ).astype("float32")

    return target