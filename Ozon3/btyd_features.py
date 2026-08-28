import datetime as dt

import numpy as np
import pandas as pd

import config as cfg

try:
    from lifetimes import (
        BetaGeoFitter,
        GammaGammaFitter,
    )

    HAS_LIFETIMES = True

except ImportError:

    HAS_LIFETIMES = False


def calculate_btyd_features(
    df: pd.DataFrame,
    cutoff: dt.date,
    user_ids=None,
    backend: str | None = None,
) -> pd.DataFrame:

    cutoff_ts = pd.Timestamp(cutoff)

    hist = df[
        df[cfg.DATE_COL] <= cutoff_ts
    ]

    if user_ids is None:

        all_users = (
            hist[cfg.ID_COL]
            .drop_duplicates()
        )

    else:

        all_users = pd.Series(
            user_ids
        ).drop_duplicates()

    result = pd.DataFrame(
        {
            cfg.ID_COL: all_users
        }
    )

    orders = hist[
        hist["gmv"] > 0
    ].copy()

    backend = backend or getattr(cfg, "BTYD_BACKEND", "fallback")
    if backend not in {"fallback", "lifetimes"}:
        raise ValueError(
            f"Unknown BTYD backend={backend!r}; expected 'fallback' or 'lifetimes'."
        )

    if len(orders) == 0:

        for column in (
            "btyd_p_alive",
            "btyd_exp_orders_30d",
            "btyd_exp_gmv_30d",
            "btyd_log_exp_orders",
            "btyd_log_exp_gmv",
            "btyd_order_value",
            "btyd_total_gmv",
        ):
            result[column] = 0.0

        return result

    orders[cfg.DATE_COL] = pd.to_datetime(
        orders[cfg.DATE_COL]
    )

    # BTYD transactions are unique positive-GMV purchase days.  The raw
    # competition table is sparse daily activity, not an order-level table.
    orders = orders.sort_values([cfg.ID_COL, cfg.DATE_COL])
    first_purchase = orders.groupby(cfg.ID_COL)[cfg.DATE_COL].transform("min")
    repeat_orders = orders[orders[cfg.DATE_COL] > first_purchase]
    repeat_monetary = repeat_orders.groupby(cfg.ID_COL)["gmv"].mean()

    rfm = (
        orders.groupby(cfg.ID_COL)
        .agg(
            first_order=(
                cfg.DATE_COL,
                "min",
            ),
            last_order=(
                cfg.DATE_COL,
                "max",
            ),
            n_events=(
                cfg.DATE_COL,
                "nunique",
            ),
            mean_purchase_day_gmv=(
                "gmv",
                "mean",
            ),
            total_gmv=(
                "gmv",
                "sum",
            ),
        )
        .reset_index()
    )

    rfm["frequency"] = (
        rfm["n_events"] - 1
    ).clip(lower=0)

    rfm["recency"] = (
        rfm["last_order"]
        - rfm["first_order"]
    ).dt.days.clip(lower=0)

    rfm["T"] = (
        cutoff_ts
        - rfm["first_order"]
    ).dt.days.clip(lower=0)

    rfm["monetary_value"] = rfm[cfg.ID_COL].map(repeat_monetary)
    rfm["monetary_value"] = rfm["monetary_value"].fillna(
        rfm["mean_purchase_day_gmv"]
    )

    if (rfm["recency"] > rfm["T"]).any():
        raise AssertionError("BTYD invariant violated: recency must not exceed T")

    # =========================================================
    # BASIC BTYD
    # =========================================================

    if backend == "lifetimes":

        if not HAS_LIFETIMES:
            raise RuntimeError(
                "BTYD_BACKEND='lifetimes', but the lifetimes package is not installed."
            )

        try:

            bgf = BetaGeoFitter(
                penalizer_coef=0.01
            )

            bgf.fit(
                rfm["frequency"],
                rfm["recency"],
                rfm["T"],
            )

            rfm["btyd_p_alive"] = (
                bgf.conditional_probability_alive(
                    rfm["frequency"],
                    rfm["recency"],
                    rfm["T"],
                )
            )

            rfm["btyd_exp_orders_30d"] = (
                bgf
                .conditional_expected_number_of_purchases_up_to_time(
                    30,
                    rfm["frequency"],
                    rfm["recency"],
                    rfm["T"],
                )
            )

            rfm["btyd_exp_gmv_30d"] = 0.0

            repeat = (
                rfm["frequency"] > 0
            ) & (
                rfm["monetary_value"] > 0
            )

            if repeat.sum() >= 20:

                ggf = GammaGammaFitter(
                    penalizer_coef=0.01
                )

                ggf.fit(
                    rfm.loc[repeat, "frequency"],
                    rfm.loc[
                        repeat,
                        "monetary_value"
                    ],
                )

                exp_value = (
                    ggf
                    .conditional_expected_average_profit(
                        rfm["frequency"],
                        rfm["monetary_value"],
                    )
                )

                rfm[
                    "btyd_exp_gmv_30d"
                ] = (
                    rfm[
                        "btyd_exp_orders_30d"
                    ]
                    * exp_value
                )

            else:

                rfm[
                    "btyd_exp_gmv_30d"
                ] = (
                    rfm[
                        "btyd_exp_orders_30d"
                    ]
                    * rfm[
                        "monetary_value"
                    ]
                )

        except Exception as exc:
            # A silent per-fold fallback changes feature semantics and makes
            # OOF irreproducible. Fail loudly instead.
            raise RuntimeError("lifetimes BTYD fit failed") from exc

    else:

        rfm = _fallback_btyd(
            rfm
        )

    # =========================================================
    # EXTRA BTYD FEATURES
    # =========================================================

    rfm["btyd_log_exp_orders"] = np.log1p(
        np.clip(
            rfm[
                "btyd_exp_orders_30d"
            ],
            0,
            None,
        )
    )

    rfm["btyd_log_exp_gmv"] = np.log1p(
        np.clip(
            rfm[
                "btyd_exp_gmv_30d"
            ],
            0,
            None,
        )
    )

    rfm["btyd_order_value"] = (
        rfm["monetary_value"]
    )

    rfm["btyd_total_gmv"] = (
        rfm["total_gmv"]
    )

    cols = [
        cfg.ID_COL,
        "btyd_p_alive",
        "btyd_exp_orders_30d",
        "btyd_exp_gmv_30d",
        "btyd_log_exp_orders",
        "btyd_log_exp_gmv",
        "btyd_order_value",
        "btyd_total_gmv",
    ]

    result = result.merge(
        rfm[cols],
        on=cfg.ID_COL,
        how="left",
    )

    numeric = [
        c for c in cols
        if c != cfg.ID_COL
    ]

    result[numeric] = (
        result[numeric]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0)
    )

    return result


def _fallback_btyd(
    rfm: pd.DataFrame,
) -> pd.DataFrame:

    gap = (
        rfm["T"]
        - rfm["recency"]
    ).clip(lower=0)

    rfm["btyd_p_alive"] = np.exp(
        -0.02 * gap
    )

    lambda_est = (
        rfm["frequency"] + 1
    ) / (
        rfm["T"] + 30
    )

    rfm["btyd_exp_orders_30d"] = (
        rfm["btyd_p_alive"]
        * lambda_est
        * 30
    )

    rfm["btyd_exp_gmv_30d"] = (
        rfm["btyd_exp_orders_30d"]
        * rfm["monetary_value"]
    )

    return rfm
