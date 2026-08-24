import datetime as dt
import numpy as np
import pandas as pd

import config as cfg

try:
    from lifetimes import BetaGeoFitter, GammaGammaFitter

    HAS_LIFETIMES = True
except ImportError:
    HAS_LIFETIMES = False


def calculate_btyd_features(df: pd.DataFrame, cutoff: dt.date) -> pd.DataFrame:
    """Расчет BTYD (BG/NBD + Gamma-Gamma) показателей на срез cutoff.

    Восстанавливает вероятности активности P(Alive) и ожидаемый GMV.
    """
    cutoff_ts = pd.Timestamp(cutoff)

    # 1. Получаем полная база всех уникальных ID из df (чтобы не потерять "беззаказных")
    all_users = pd.DataFrame({cfg.ID_COL: df[cfg.ID_COL].unique()})

    # 2. Фильтруем историю до среза cutoff
    hist = df[df[cfg.DATE_COL] <= cutoff_ts]

    # Расчет ведем по фактам покупок (gmv > 0)
    orders = hist[hist["gmv"] > 0].copy()

    if len(orders) == 0:
        all_users["btyd_p_alive"] = 0.0
        all_users["btyd_exp_orders_30d"] = 0.0
        all_users["btyd_exp_gmv_30d"] = 0.0
        return all_users

    # Гарантируем datetime тип
    orders[cfg.DATE_COL] = pd.to_datetime(orders[cfg.DATE_COL])

    # 3. Агрегируем RFM фичи
    rfm = (
        orders.groupby(cfg.ID_COL)
        .agg(
            first_order=(cfg.DATE_COL, "min"),
            last_order=(cfg.DATE_COL, "max"),
            n_events=(cfg.DATE_COL, "nunique"),
            monetary_value=("gmv", "mean"),
        )
        .reset_index()
    )

    rfm["frequency"] = (rfm["n_events"] - 1).clip(lower=0)
    rfm["recency"] = (rfm["last_order"] - rfm["first_order"]).dt.days.clip(
        lower=0
    )
    rfm["T"] = (cutoff_ts - rfm["first_order"]).dt.days.clip(lower=0)

    # 4. Расчет BTYD показателей
    if HAS_LIFETIMES:
        try:
            bgf = BetaGeoFitter(penalizer_coef=0.01)
            bgf.fit(rfm["frequency"], rfm["recency"], rfm["T"])

            rfm["btyd_p_alive"] = bgf.conditional_probability_alive(
                rfm["frequency"], rfm["recency"], rfm["T"]
            )
            rfm["btyd_exp_orders_30d"] = (
                bgf.conditional_expected_number_of_purchases_up_to_time(
                    30, rfm["frequency"], rfm["recency"], rfm["T"]
                )
            )

            idx_repeat = (rfm["frequency"] > 0) & (rfm["monetary_value"] > 0)
            rfm["btyd_exp_gmv_30d"] = 0.0

            if idx_repeat.sum() > 10:
                ggf = GammaGammaFitter(penalizer_coef=0.01)
                ggf.fit(
                    rfm.loc[idx_repeat, "frequency"],
                    rfm.loc[idx_repeat, "monetary_value"],
                )

                exp_monetary = ggf.conditional_expected_average_profit(
                    rfm["frequency"], rfm["monetary_value"]
                )
                rfm["btyd_exp_gmv_30d"] = (
                    rfm["btyd_exp_orders_30d"] * exp_monetary
                )
            else:
                rfm["btyd_exp_gmv_30d"] = (
                    rfm["btyd_exp_orders_30d"] * rfm["monetary_value"]
                )
        except Exception:
            rfm = _fallback_btyd(rfm)
    else:
        rfm = _fallback_btyd(rfm)

    # 5. Соединяем со всей базой пользователей, чтобы не терять нули
    res_cols = [
        cfg.ID_COL,
        "btyd_p_alive",
        "btyd_exp_orders_30d",
        "btyd_exp_gmv_30d",
    ]
    result = all_users.merge(rfm[res_cols], on=cfg.ID_COL, how="left")

    # Заполняем пропуски для пользователей без заказов
    result["btyd_p_alive"] = result["btyd_p_alive"].fillna(0.0)
    result["btyd_exp_orders_30d"] = result["btyd_exp_orders_30d"].fillna(0.0)
    result["btyd_exp_gmv_30d"] = result["btyd_exp_gmv_30d"].fillna(0.0)

    return result


def _fallback_btyd(rfm: pd.DataFrame) -> pd.DataFrame:
    """Аналитическая аппроксимация при отсутствии/сбое библиотеки lifetimes."""
    gap = (rfm["T"] - rfm["recency"]).clip(lower=0)
    rfm["btyd_p_alive"] = np.exp(-0.02 * gap)
    lambda_est = (rfm["frequency"] + 1) / (rfm["T"] + 30)
    rfm["btyd_exp_orders_30d"] = rfm["btyd_p_alive"] * lambda_est * 30
    rfm["btyd_exp_gmv_30d"] = rfm["btyd_exp_orders_30d"] * rfm["monetary_value"]
    return rfm