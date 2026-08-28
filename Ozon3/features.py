import datetime as dt

import numpy as np
import pandas as pd

from btyd_features import calculate_btyd_features
import config as cfg


def _to_ts(d: dt.date) -> pd.Timestamp:
    return pd.Timestamp(d)


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:

    for col in df.columns:

        if col in (cfg.ID_COL, "cutoff_date"):
            continue

        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].astype("float32")

        elif pd.api.types.is_integer_dtype(df[col]):
            # Не трогаем user_id
            df[col] = df[col].astype("int32")

    return df


def safe_div(a, b):
    return a / np.maximum(b, 1e-6)


def build_features_for_cutoff(
    df: pd.DataFrame,
    cutoff: dt.date,
    user_ids: pd.Series | None = None,
    include_unseen: bool = False,
) -> pd.DataFrame:

    cutoff_ts = _to_ts(cutoff)

    # build_dataset.py passes an already cutoff-bounded frame.  Reuse it to
    # avoid duplicating tens of millions of rows; keep the defensive filter
    # for direct callers that pass a wider history.
    if df.empty or df[cfg.DATE_COL].max() <= cutoff_ts:
        hist = df
    else:
        hist = df.loc[df[cfg.DATE_COL] <= cutoff_ts]

    # ---------------------------------------------------------
    # 0. Пользователи, существовавшие к cutoff
    # ---------------------------------------------------------

    existing_users = pd.Index(
        hist[cfg.ID_COL].drop_duplicates(),
        name=cfg.ID_COL,
    )

    if user_ids is None:

        user_ids = (
            hist[cfg.ID_COL]
            .drop_duplicates()
            .reset_index(drop=True)
        )

    else:

        user_ids = pd.Series(user_ids).drop_duplicates()

        # seen_only остаётся основной train-policy. Для test и отдельного
        # controlled experiment можно сохранить всю заданную популяцию;
        # индикатор seen_before_anchor отличает cold-start строки от нулевой
        # активности уже известного пользователя.
        if not include_unseen:
            user_ids = user_ids[user_ids.isin(existing_users)]

    base_index = pd.Index(
        user_ids.unique(),
        name=cfg.ID_COL,
    )

    blocks = []

    # =========================================================
    # 1. BASIC LIFETIME
    # =========================================================

    last_seen = hist.groupby(cfg.ID_COL)[cfg.DATE_COL].max()
    first_seen = hist.groupby(cfg.ID_COL)[cfg.DATE_COL].min()

    active_days = (
        hist.groupby(cfg.ID_COL)[cfg.DATE_COL]
        .nunique()
    )

    life = pd.DataFrame(index=base_index)

    life["seen_before_anchor"] = base_index.isin(existing_users).astype("int8")

    life["recency_days"] = (
        cutoff_ts - last_seen
    ).dt.days

    life["tenure_days"] = (
        cutoff_ts - first_seen
    ).dt.days + 1

    life["n_active_days_total"] = active_days

    life["activity_density_total"] = safe_div(
        life["n_active_days_total"],
        life["tenure_days"],
    )

    # log versions
    life["log_tenure_days"] = np.log1p(
        life["tenure_days"].clip(lower=0)
    )

    life["log_recency_days"] = np.log1p(
        life["recency_days"].clip(lower=0)
    )

    blocks.append(life)

    # =========================================================
    # 2. PURCHASE HISTORY
    # =========================================================

    purchases = hist[hist["gmv"] > 0].copy()

    purch = pd.DataFrame(index=base_index)

    if len(purchases):

        last_purchase = (
            purchases.groupby(cfg.ID_COL)[cfg.DATE_COL]
            .max()
        )

        first_purchase = (
            purchases.groupby(cfg.ID_COL)[cfg.DATE_COL]
            .min()
        )

        n_purchase_days = (
            purchases.groupby(cfg.ID_COL)[cfg.DATE_COL]
            .nunique()
        )

        total_gmv = (
            purchases.groupby(cfg.ID_COL)["gmv"]
            .sum()
        )

        purch["recency_purchase_days"] = (
            cutoff_ts - last_purchase
        ).dt.days

        purch["first_purchase_days"] = (
            cutoff_ts - first_purchase
        ).dt.days

        purch["n_purchase_days"] = n_purchase_days

        purch["total_purchase_gmv"] = total_gmv

        purch["purchase_frequency"] = safe_div(
            n_purchase_days,
            life["tenure_days"].reindex(n_purchase_days.index),
        )

        # gaps
        p = purchases[
            [cfg.ID_COL, cfg.DATE_COL]
        ].drop_duplicates().sort_values(
            [cfg.ID_COL, cfg.DATE_COL]
        )

        p["prev_date"] = (
            p.groupby(cfg.ID_COL)[cfg.DATE_COL]
            .shift(1)
        )

        p["gap"] = (
            p[cfg.DATE_COL] - p["prev_date"]
        ).dt.days

        gaps = (
            p.groupby(cfg.ID_COL)["gap"]
            .agg(
                mean_purchase_gap="mean",
                std_purchase_gap="std",
                min_purchase_gap="min",
                max_purchase_gap="max",
            )
        )

        purch = purch.join(gaps)

    blocks.append(purch)

    # =========================================================
    # 3. LOOKBACK FEATURES
    # =========================================================

    agg_cols = cfg.COUNT_COLS + cfg.GMV_COLS

    lookbacks = {}

    for L in cfg.LOOKBACKS:

        start = cutoff_ts - pd.Timedelta(days=L - 1)

        w = hist[
            hist[cfg.DATE_COL] >= start
        ]

        g = pd.DataFrame(index=base_index)

        if len(w):

            sums = (
                w.groupby(cfg.ID_COL)[agg_cols]
                .sum()
            )

            sums.columns = [
                f"{c}_sum_{L}d"
                for c in sums.columns
            ]

            g = g.join(sums)

            active = (
                w.groupby(cfg.ID_COL)[cfg.DATE_COL]
                .nunique()
                .rename(
                    f"n_active_days_{L}d"
                )
            )

            g = g.join(active)

            purchase_w = w[w["gmv"] > 0]

            if len(purchase_w):

                p_days = (
                    purchase_w.groupby(cfg.ID_COL)[
                        cfg.DATE_COL
                    ]
                    .nunique()
                    .rename(
                        f"n_purchase_days_{L}d"
                    )
                )

                g = g.join(p_days)

        lookbacks[L] = g

        blocks.append(g)

    # Exact previous 14-day block for a compact disjoint trend group.
    # Other previous windows can be derived exactly from the cumulative
    # 7/14/30/60/90-day aggregates below.
    prev14 = hist[
        (hist[cfg.DATE_COL] >= cutoff_ts - pd.Timedelta(days=27))
        & (hist[cfg.DATE_COL] <= cutoff_ts - pd.Timedelta(days=14))
    ]
    prev14_block = pd.DataFrame(index=base_index)
    trend_base_cols = ["gmv", "to_ord", "searches"]
    if len(prev14):
        prev14_sums = prev14.groupby(cfg.ID_COL)[trend_base_cols].sum()
        prev14_sums.columns = [f"{c}_sum_prev_14d" for c in trend_base_cols]
        prev14_block = prev14_block.join(prev14_sums)
    blocks.append(prev14_block)

    # =========================================================
    # 4. BTYD
    # =========================================================

    btyd = calculate_btyd_features(
        df,
        cutoff,
        user_ids=base_index,
    )

    btyd = btyd.set_index(cfg.ID_COL)

    blocks.append(
        btyd.reindex(base_index)
    )

    # =========================================================
    # 5. EMA
    # =========================================================

    ema = pd.DataFrame(index=base_index)

    for w_days in [7, 14, 30, 60, 90]:

        start = cutoff_ts - pd.Timedelta(
            days=w_days - 1
        )

        sub = hist[
            hist[cfg.DATE_COL] >= start
        ].copy()

        if len(sub):

            age = (
                cutoff_ts - sub[cfg.DATE_COL]
            ).dt.days

            sub["decay"] = np.exp(
                -np.log(2) * age / max(w_days, 1)
            )

            sub["weighted_gmv"] = (
                sub["gmv"] * sub["decay"]
            )

            value = (
                sub.groupby(cfg.ID_COL)[
                    "weighted_gmv"
                ]
                .sum()
                .rename(
                    f"ema_gmv_{w_days}d"
                )
            )

            ema = ema.join(value)

    blocks.append(ema)

    # =========================================================
    # 6. ASSEMBLE
    # =========================================================

    feats = pd.concat(
        blocks,
        axis=1,
    )

    feats = feats.reset_index()

    # =========================================================
    # 7. RATIOS
    # =========================================================

    for L in cfg.LOOKBACKS:

        def col(name):
            return f"{name}_sum_{L}d"

        gmv = col("gmv")

        orders = col("to_ord")

        cart = col("to_cart")

        searches = col("searches")

        active = f"n_active_days_{L}d"

        purchase_days = f"n_purchase_days_{L}d"

        if gmv in feats.columns:

            if active in feats.columns:

                feats[
                    f"gmv_per_active_day_{L}d"
                ] = safe_div(
                    feats[gmv],
                    feats[active],
                )

            if orders in feats.columns:

                feats[
                    f"avg_order_value_{L}d"
                ] = safe_div(
                    feats[gmv],
                    feats[orders],
                )

            if purchase_days in feats.columns:

                feats[
                    f"gmv_per_purchase_day_{L}d"
                ] = safe_div(
                    feats[gmv],
                    feats[purchase_days],
                )

        if orders in feats.columns and cart in feats.columns:

            feats[
                f"cart_to_order_rate_{L}d"
            ] = safe_div(
                feats[orders],
                feats[cart],
            )

        if orders in feats.columns and searches in feats.columns:

            feats[
                f"search_to_order_rate_{L}d"
            ] = safe_div(
                feats[orders],
                feats[searches],
            )

        if cart in feats.columns and searches in feats.columns:

            feats[
                f"search_to_cart_rate_{L}d"
            ] = safe_div(
                feats[cart],
                feats[searches],
            )

        if active in feats.columns:

            feats[
                f"activity_density_{L}d"
            ] = safe_div(
                feats[active],
                L,
            )

            if purchase_days in feats.columns:

                feats[
                    f"purchase_day_share_{L}d"
                ] = safe_div(
                    feats[purchase_days],
                    feats[active],
                )

    # =========================================================
    # 8. CROSS-WINDOW DYNAMICS
    # =========================================================

    windows = cfg.LOOKBACKS

    for short, long in [
        (7, 30),
        (14, 30),
        (30, 60),
        (30, 90),
        (60, 180),
        (90, 365),
    ]:

        if (
            f"gmv_sum_{short}d" in feats.columns
            and f"gmv_sum_{long}d" in feats.columns
        ):

            feats[
                f"gmv_ratio_{short}_{long}"
            ] = safe_div(
                feats[f"gmv_sum_{short}d"],
                feats[f"gmv_sum_{long}d"],
            )

        if (
            f"to_ord_sum_{short}d" in feats.columns
            and f"to_ord_sum_{long}d" in feats.columns
        ):

            feats[
                f"orders_ratio_{short}_{long}"
            ] = safe_div(
                feats[f"to_ord_sum_{short}d"],
                feats[f"to_ord_sum_{long}d"],
            )

        if (
            f"n_purchase_days_{short}d" in feats.columns
            and f"n_purchase_days_{long}d" in feats.columns
        ):

            feats[
                f"purchase_days_ratio_{short}_{long}"
            ] = safe_div(
                feats[
                    f"n_purchase_days_{short}d"
                ],
                feats[
                    f"n_purchase_days_{long}d"
                ],
            )

    # =========================================================
    # 9. RECENT VS OLD
    # =========================================================

    if (
        "gmv_sum_30d" in feats.columns
        and "gmv_sum_90d" in feats.columns
    ):

        feats["gmv_old_60d"] = (
            feats["gmv_sum_90d"]
            - feats["gmv_sum_30d"]
        ).clip(lower=0)

        feats["gmv_recent_share"] = safe_div(
            feats["gmv_sum_30d"],
            feats["gmv_sum_90d"],
        )

    if (
        "to_ord_sum_30d" in feats.columns
        and "to_ord_sum_90d" in feats.columns
    ):

        feats["orders_old_60d"] = (
            feats["to_ord_sum_90d"]
            - feats["to_ord_sum_30d"]
        ).clip(lower=0)

        feats["orders_recent_share"] = safe_div(
            feats["to_ord_sum_30d"],
            feats["to_ord_sum_90d"],
        )

    # =========================================================
    # 10. DISJOINT TEMPORAL TRENDS
    # =========================================================

    # Consolidate blocks accumulated by the ratio section before adding the
    # compact trend group. This avoids pandas block fragmentation on full data.
    feats = feats.copy()

    for signal in trend_base_cols:
        recent_7 = f"{signal}_sum_7d"
        recent_14 = f"{signal}_sum_14d"
        recent_30 = f"{signal}_sum_30d"
        total_60 = f"{signal}_sum_60d"
        total_90 = f"{signal}_sum_90d"
        previous_14 = f"{signal}_sum_prev_14d"

        if recent_7 in feats and recent_14 in feats:
            prev = (feats[recent_14] - feats[recent_7]).clip(lower=0)
            feats[f"{signal}_sum_prev_7d"] = prev
            feats[f"{signal}_delta_7_vs_prev7"] = (
                feats[recent_7] - prev
            ) / 7.0
            feats[f"{signal}_ratio_7_vs_prev7"] = safe_div(
                feats[recent_7] + 1.0,
                prev + 1.0,
            )

        if recent_14 in feats and previous_14 in feats:
            feats[f"{signal}_delta_14_vs_prev14"] = (
                feats[recent_14] - feats[previous_14]
            ) / 14.0
            feats[f"{signal}_ratio_14_vs_prev14"] = safe_div(
                feats[recent_14] + 1.0,
                feats[previous_14] + 1.0,
            )

        if recent_30 in feats and total_60 in feats:
            prev_30 = (feats[total_60] - feats[recent_30]).clip(lower=0)
            feats[f"{signal}_sum_prev_30d"] = prev_30
            feats[f"{signal}_delta_30_vs_prev30"] = (
                feats[recent_30] - prev_30
            ) / 30.0
            feats[f"{signal}_ratio_30_vs_prev30"] = safe_div(
                feats[recent_30] + 1.0,
                prev_30 + 1.0,
            )

        if recent_30 in feats and total_60 in feats and total_90 in feats:
            prev_30 = (feats[total_60] - feats[recent_30]).clip(lower=0)
            older_30 = (feats[total_90] - feats[total_60]).clip(lower=0)
            prev_60 = (feats[total_90] - feats[recent_30]).clip(lower=0)
            feats[f"{signal}_sum_prev_60d"] = prev_60
            feats[f"{signal}_delta_30_vs_prev60"] = (
                feats[recent_30] / 30.0 - prev_60 / 60.0
            )
            feats[f"{signal}_slope_3x30"] = (
                feats[recent_30] - older_30
            ) / 60.0
            feats[f"{signal}_acceleration_3x30"] = (
                feats[recent_30] - 2.0 * prev_30 + older_30
            ) / 30.0

    # =========================================================
    # 11. LOG FEATURES
    # =========================================================

    feats = feats.copy()
    numeric_cols = feats.select_dtypes(
        include=[np.number]
    ).columns

    log_features = {}

    for c in numeric_cols:

        if c == cfg.ID_COL:
            continue

        if (
            "log_" not in c
            and not c.startswith("target")
            and not any(
                token in c
                for token in ("delta_", "slope_", "acceleration_")
            )
        ):

            log_features[f"log1p_{c}"] = np.log1p(
                np.clip(
                    feats[c].astype("float64"),
                    0,
                    None,
                )
            ).astype("float32")

    if log_features:
        feats = pd.concat(
            [feats, pd.DataFrame(log_features, index=feats.index)],
            axis=1,
        )

    # The returned frame is reused by dataset assembly; consolidate it once so
    # adding metadata does not trigger thousands of fragmented pandas blocks.
    feats = feats.copy()

    # =========================================================
    # 12. MISSING VALUES
    # =========================================================

    for c in feats.columns:

        if c in (cfg.ID_COL, "cutoff_date"):
            continue

        if pd.api.types.is_numeric_dtype(
            feats[c]
        ):

            feats[c] = feats[c].replace(
                [np.inf, -np.inf],
                np.nan,
            )

            feats[c] = feats[c].fillna(0)

    feats = pd.concat(
        [
            feats,
            pd.Series(cutoff_ts, index=feats.index, name="cutoff_date"),
        ],
        axis=1,
    )

    return optimize_dtypes(
        feats.copy()
    )


def feature_group_for_column(column: str) -> str:
    """Stable, target-free grouping used by ablations and diagnostics."""
    raw = column.removeprefix("log1p_")
    if raw.startswith("btyd_"):
        return "btyd"
    if any(token in raw for token in ("delta_", "ratio_7_vs", "ratio_14_vs", "ratio_30_vs", "slope_", "acceleration_", "sum_prev_")):
        return "trend"
    if "_rate_" in raw or "_ratio_" in raw or "_share_" in raw or raw.endswith("_share"):
        return "ratio"
    if any(token in raw for token in ("recency", "tenure", "purchase_frequency", "n_purchase_days", "n_active_days_total")):
        return "rfm"
    if any(token in raw for token in ("180d", "365d", "total_purchase", "purchase_gap")):
        return "long_term"
    if raw.startswith("ema_"):
        return "trend"
    return "window_aggregate"


def get_feature_groups(feature_columns) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for column in feature_columns:
        groups.setdefault(feature_group_for_column(column), []).append(column)
    return groups


def build_target(
    df: pd.DataFrame,
    user_ids: pd.Series,
    target_start: dt.date,
    target_end: dt.date,
) -> pd.DataFrame:

    start = _to_ts(target_start)
    end = _to_ts(target_end)

    window = df[
        (df[cfg.DATE_COL] >= start)
        & (df[cfg.DATE_COL] <= end)
    ]

    target = (
        window.groupby(cfg.ID_COL)["gmv"]
        .sum()
        .rename(cfg.TARGET_COL)
        .reset_index()
    )

    base = pd.DataFrame(
        {
            cfg.ID_COL:
                pd.Series(user_ids).unique()
        }
    )

    target = base.merge(
        target,
        on=cfg.ID_COL,
        how="left",
    )

    target[cfg.TARGET_COL] = (
        target[cfg.TARGET_COL]
        .fillna(0)
        .clip(lower=0)
        .astype("float32")
    )

    target[cfg.TARGET_LOG_COL] = np.log1p(
        target[cfg.TARGET_COL]
    ).astype("float32")

    return target
