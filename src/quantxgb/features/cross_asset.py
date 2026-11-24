"""Cross-asset and market-wide features (PRD §8.2)."""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from quantxgb.config import FeaturesConfig
from quantxgb.features.registry import FeatureSpec

R = pl.col("lr")


def _rolling_corr(a: pl.Expr, b: pl.Expr, w: int) -> pl.Expr:
    cov = (a * b).rolling_mean(w) - a.rolling_mean(w) * b.rolling_mean(w)
    va = (a * a).rolling_mean(w) - a.rolling_mean(w) ** 2
    vb = (b * b).rolling_mean(w) - b.rolling_mean(w) ** 2
    return cov / (va * vb).sqrt()


def exprs(cfg: FeaturesConfig) -> list[pl.Expr]:
    """Needs ``ret_k`` (asset), ``mkt_ret_k`` and ``mkt_lr`` / ``rates_lr`` columns."""
    c = cfg.cross_asset
    out = [
        (pl.col(f"ret_{k}") - pl.col(f"mkt_ret_{k}")).alias(f"rel_mkt_{k}")
        for k in c.rel_market_lookbacks
    ]
    w = c.corr_lookback
    out.append(_rolling_corr(R, pl.col("mkt_lr"), w).over("ticker").alias(f"corr_mkt_{w}"))
    out.append(_rolling_corr(R, pl.col("rates_lr"), w).over("ticker").alias(f"corr_rates_{w}"))
    return out


def rel_class_exprs(cfg: FeaturesConfig) -> list[pl.Expr]:
    """Leave-one-out asset-class mean on each date (null if the asset is alone in its class)."""
    out = []
    for k in cfg.cross_asset.rel_class_lookbacks:
        x = pl.col(f"ret_{k}")
        grp = ["date", "asset_class"]
        n = x.count().over(grp)
        loo = (x.sum().over(grp) - x) / (n - 1)
        out.append(pl.when(n > 1).then(x - loo).alias(f"rel_class_{k}"))
    return out


def average_correlation(lr_wide: np.ndarray, window: int) -> np.ndarray:
    """Mean off-diagonal correlation over trailing windows (assets with full windows only)."""
    t_len, _ = lr_wide.shape
    out = np.full(t_len, np.nan)
    for t in range(window - 1, t_len):
        win = lr_wide[t - window + 1 : t + 1]
        ok = ~np.isnan(win).any(axis=0)
        sub = win[:, ok]
        if sub.shape[1] < 2:
            continue
        sd = sub.std(axis=0)
        sub = sub[:, sd > 0]
        if sub.shape[1] < 2:
            continue
        corr = np.corrcoef(sub, rowvar=False)
        n = corr.shape[0]
        out[t] = (corr.sum() - n) / (n * (n - 1))
    return out


def market_frame(panel: pl.DataFrame, cfg: FeaturesConfig, market: str) -> pl.DataFrame:
    """Date-level features from the market proxy and the whole universe."""
    mk = cfg.market
    m = panel.filter(pl.col("ticker") == market).sort("date")
    if m.height == 0:
        raise ValueError(f"market proxy {market} not in price panel")
    tr = pl.col("tr_index")
    mkt = m.select(
        "date",
        pl.col("lr").alias("market_ret_1d"),
        (pl.col("lr").rolling_std(mk.market_vol_lookback) * math.sqrt(252.0)).alias(
            f"market_vol_{mk.market_vol_lookback}"
        ),
        pl.when(tr.rolling_mean(mk.trend_sma).is_not_null())
        .then((tr > tr.rolling_mean(mk.trend_sma)).cast(pl.Float64))
        .alias(f"market_above_sma{mk.trend_sma}"),
    )
    wide = (
        panel.select("date", "ticker", "lr")
        .pivot(on="ticker", index="date", values="lr")
        .sort("date")
    )
    arr = wide.drop("date").cast(pl.Float64).fill_null(np.nan).to_numpy()
    avg = average_correlation(arr, mk.avg_corr_lookback)
    ac = pl.DataFrame({"date": wide["date"], f"avg_corr_{mk.avg_corr_lookback}": avg}).with_columns(
        pl.col(f"avg_corr_{mk.avg_corr_lookback}").fill_nan(None)
    )
    return mkt.join(ac, on="date", how="full", coalesce=True).sort("date")


def specs(cfg: FeaturesConfig) -> list[FeatureSpec]:
    c, mk = cfg.cross_asset, cfg.market
    w = c.corr_lookback
    out = [
        FeatureSpec(
            f"rel_mkt_{k}",
            "cross_asset",
            f"ret_{k} − ret_{k}(M)",
            k,
            "Performance relative to the equity market proxy",
        )
        for k in c.rel_market_lookbacks
    ]
    out += [
        FeatureSpec(
            f"rel_class_{k}",
            "cross_asset",
            f"ret_{k} − leave-one-out mean ret_{k} of same asset class on t",
            k,
            "Within-asset-class relative strength (null for single-asset classes)",
            scope="cross_sectional",
        )
        for k in c.rel_class_lookbacks
    ]
    out += [
        FeatureSpec(
            f"corr_mkt_{w}",
            "cross_asset",
            f"corr(r, r_M) over {w}",
            w,
            "Diversification role vs. equities",
        ),
        FeatureSpec(
            f"corr_rates_{w}",
            "cross_asset",
            f"corr(r, r_rates proxy) over {w}",
            w,
            "Diversification role vs. duration",
        ),
        FeatureSpec(
            "market_ret_1d",
            "market",
            "r_M,t",
            1,
            "Market daily return (HMM regime input)",
            scope="market",
            model_input=False,
        ),
        FeatureSpec(
            f"market_vol_{mk.market_vol_lookback}",
            "market",
            f"std(r_M over {mk.market_vol_lookback}) · √252",
            mk.market_vol_lookback,
            "Market-wide volatility regime",
            scope="market",
        ),
        FeatureSpec(
            f"market_above_sma{mk.trend_sma}",
            "market",
            f"1[TR_M > SMA_{mk.trend_sma}(TR_M)]",
            mk.trend_sma,
            "Market trend regime",
            scope="market",
        ),
        FeatureSpec(
            f"avg_corr_{mk.avg_corr_lookback}",
            "market",
            f"mean pairwise corr of universe r over {mk.avg_corr_lookback}",
            mk.avg_corr_lookback,
            "Diversification breakdown / risk-on-risk-off intensity",
            scope="market",
        ),
    ]
    return out
