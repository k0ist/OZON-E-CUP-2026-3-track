import numpy as np
import pandas as pd
import datetime as dt
import config as cfg

try:
    from lifetimes import BetaGeoFitter, GammaGammaFitter

    HAS_LIFETIMES = True
except ImportError:
    HAS_LIFETIMES = False


def calculate_btyd_features(df: pd.DataFrame, cutoff: dt.date) -> pd.DataFrame:
    """
    Расчет BTYD (BG/NBD + Gamma-Gamma) показателей на срез cutoff.
    Восстанавливает вероятности активности P(Alive) и ожидаемый GMV.
    """
    cutoff_ts = pd.Timestamp(cutoff)
    hist = df[df[cfg.DATE_COL] <= cutoff_ts]

    # Расчет ведем по фактам покупок
    orders = hist[hist["gmv"] > 0].copy()

    if len(orders) == 0:
        base = pd.DataFrame({cfg.ID_COL: df[cfg.ID_COL].unique()})
        base["btyd_p_alive"] = 0.0
        base["btyd_exp_orders_30d"] = 0.0
        base["btyd_exp_gmv_30d"] = 0.0
        return base

    rfm = orders.groupby(cfg.ID_COL).agg(
        frequency=("event_date", lambda x: x.nunique() - 1),
        recency=("event_date", lambda x: (x.max() - x.min()).days),
        T=("event_date", lambda x: (cutoff_ts - x.min()).days),
        monetary_value=("gmv", "mean")
    ).reset_index()

    rfm["recency"] = rfm["recency"].clip(lower=0)
    rfm["T"] = rfm["T"].clip(lower=0)
    rfm["frequency"] = rfm["frequency"].clip(lower=0)

    if HAS_LIFETIMES:
        try:
            bgf = BetaGeoFitter(penalizer_coef=0.01)
            bgf.fit(rfm['frequency'], rfm['recency'], rfm['T'])

            rfm['btyd_p_alive'] = bgf.conditional_probability_alive(
                rfm['frequency'], rfm['recency'], rfm['T']
            )
            rfm['btyd_exp_orders_30d'] = bgf.conditional_expected_number_of_purchases_up_to_time(
                30, rfm['frequency'], rfm['recency'], rfm['T']
            )

            idx_repeat = rfm['frequency'] > 0
            rfm['btyd_exp_gmv_30d'] = 0.0
            if idx_repeat.sum() > 10:
                ggf = GammaGammaFitter(penalizer_coef=0.01)
                ggf.fit(rfm.loc[idx_repeat, 'frequency'], rfm.loc[idx_repeat, 'monetary_value'])

                exp_monetary = ggf.conditional_expected_average_profit(
                    rfm['frequency'], rfm['monetary_value']
                )
                rfm['btyd_exp_gmv_30d'] = rfm['btyd_exp_orders_30d'] * exp_monetary
        except Exception:
            rfm = _fallback_btyd(rfm)
    else:
        rfm = _fallback_btyd(rfm)

    return rfm[[cfg.ID_COL, "btyd_p_alive", "btyd_exp_orders_30d", "btyd_exp_gmv_30d"]]


def _fallback_btyd(rfm: pd.DataFrame) -> pd.DataFrame:
    """Аналитическая аппроксимация при отсутствии библиотеки lifetimes."""
    gap = (rfm["T"] - rfm["recency"]).clip(lower=0)
    rfm["btyd_p_alive"] = np.exp(-0.02 * gap)
    lambda_est = (rfm["frequency"] + 1) / (rfm["T"] + 30)
    rfm["btyd_exp_orders_30d"] = rfm["btyd_p_alive"] * lambda_est * 30
    rfm["btyd_exp_gmv_30d"] = rfm["btyd_exp_orders_30d"] * rfm["monetary_value"]
    return rfm