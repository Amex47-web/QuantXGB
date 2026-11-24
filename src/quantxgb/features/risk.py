"""Risk features (PRD §8.2)."""

from __future__ import annotations

import numpy as np
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view

from quantxgb.config import FeaturesConfig
from quantxgb.features.registry import FeatureSpec

R = pl.col("lr")
M = pl.col("mkt_lr")
TR = pl.col("tr_index")


def _rolling_cov(a: pl.Expr, b: pl.Expr, w: int) -> pl.Expr:
    return (a * b).rolling_mean(w) - a.rolling_mean(w) * b.rolling_mean(w)


def exprs(cfg: FeaturesConfig) -> list[pl.Expr]:
    r = cfg.risk
    out = [
        (_rolling_cov(R, M, k) / _rolling_cov(M, M, k)).over("ticker").alias(f"beta_mkt_{k}")
        for k in r.beta_lookbacks
    ]
    d = r.drawdown_lookback
    out.append((TR / TR.rolling_max(d) - 1.0).over("ticker").alias(f"dd_{d}"))
    m = r.moments_lookback
    out.append(R.rolling_skew(m, bias=False).over("ticker").alias(f"skew_{m}"))
    out.append(R.rolling_kurtosis(m, fisher=True, bias=False).over("ticker").alias(f"kurt_{m}"))
    return out


def rolling_max_drawdown(values: np.ndarray, window: int) -> np.ndarray:
    """Max drawdown (≤ 0) of ``values`` within each trailing window; NaN if window has NaN."""
    out = np.full(values.shape[0], np.nan)
    if values.shape[0] < window:
        return out
    win = sliding_window_view(values, window)
    run_max = np.maximum.accumulate(win, axis=1)
    dd = (win / run_max - 1.0).min(axis=1)
    dd[np.isnan(win).any(axis=1)] = np.nan
    out[window - 1 :] = dd
    return out


def max_drawdown_frame(panel: pl.DataFrame, cfg: FeaturesConfig) -> pl.DataFrame:
    w = cfg.risk.max_drawdown_lookback
    parts = []
    for (ticker,), g in panel.group_by(["ticker"], maintain_order=True):
        vals = g["tr_index"].cast(pl.Float64).fill_null(np.nan).to_numpy()
        parts.append(
            pl.DataFrame(
                {
                    "ticker": [ticker] * g.height,
                    "date": g["date"],
                    f"mdd_{w}": rolling_max_drawdown(vals, w),
                }
            )
        )
    return pl.concat(parts).with_columns(pl.col(f"mdd_{w}").fill_nan(None))


def specs(cfg: FeaturesConfig) -> list[FeatureSpec]:
    r = cfg.risk
    out = [
        FeatureSpec(
            f"beta_mkt_{k}",
            "risk",
            f"cov(r, r_M) / var(r_M) over {k}",
            k,
            "Systematic exposure to the market proxy",
        )
        for k in r.beta_lookbacks
    ]
    d, w, m = r.drawdown_lookback, r.max_drawdown_lookback, r.moments_lookback
    out += [
        FeatureSpec(
            f"dd_{d}",
            "risk",
            f"TR_t / max(TR over {d}) − 1",
            d,
            "Current drawdown state (distance from the 1-year high)",
        ),
        FeatureSpec(
            f"mdd_{w}",
            "risk",
            f"min over window of TR / running max(TR) − 1, window {w}",
            w,
            "Recent tail behaviour / crash history",
        ),
        FeatureSpec(
            f"skew_{m}",
            "risk",
            f"sample skewness of r over {m}",
            m,
            "Asymmetry of returns (lottery / crash risk)",
        ),
        FeatureSpec(
            f"kurt_{m}",
            "risk",
            f"sample excess kurtosis of r over {m}",
            m,
            "Fat-tailedness of recent returns",
        ),
    ]
    return out
