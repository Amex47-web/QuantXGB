"""Trend features (PRD §8.2)."""

from __future__ import annotations

import polars as pl

from quantxgb.config import FeaturesConfig
from quantxgb.features.registry import FeatureSpec

TR = pl.col("tr_index")


def _ema(span: int) -> pl.Expr:
    return TR.ewm_mean(span=span, adjust=False, min_samples=span, ignore_nulls=False)


def exprs(cfg: FeaturesConfig) -> list[pl.Expr]:
    t = cfg.trend
    out = [(TR / _ema(k) - 1.0).over("ticker").alias(f"px_ema_{k}") for k in t.ema_spans]
    f, s = t.crossover["fast"], t.crossover["slow"]
    out.append((_ema(f) / _ema(s) - 1.0).over("ticker").alias(f"ema_{f}_{s}"))
    w = t.slope_lookback
    # OLS slope of ln TR on the session index over w sessions: cov(x, y) / var(x).
    x = pl.col("sess_idx").cast(pl.Float64)
    y = TR.log()
    cov = (x * y).rolling_mean(w) - x.rolling_mean(w) * y.rolling_mean(w)
    var = (x * x).rolling_mean(w) - x.rolling_mean(w) ** 2
    out.append((cov / var * 252.0).over("ticker").alias("_slope_ann"))
    u = t.up_fraction_lookback
    out.append(
        (pl.col("lr") > 0).cast(pl.Float64).rolling_mean(u).over("ticker").alias(f"up_frac_{u}")
    )
    return out


def slope_expr(cfg: FeaturesConfig) -> pl.Expr:
    w = cfg.trend.slope_lookback
    return (pl.col("_slope_ann") / pl.col("vol_60")).alias(f"slope_{w}")


def specs(cfg: FeaturesConfig) -> list[FeatureSpec]:
    t = cfg.trend
    f, s = t.crossover["fast"], t.crossover["slow"]
    out = [
        FeatureSpec(
            f"px_ema_{k}",
            "trend",
            f"TR_t / EMA_{k}(TR)_t − 1",
            k,
            "Distance from trend; trend-following signal",
        )
        for k in t.ema_spans
    ]
    out += [
        FeatureSpec(
            f"ema_{f}_{s}",
            "trend",
            f"EMA_{f}(TR) / EMA_{s}(TR) − 1",
            s,
            "Moving-average crossover state",
        ),
        FeatureSpec(
            f"slope_{t.slope_lookback}",
            "trend",
            f"252·β̂ / vol_60, β̂ = OLS slope of ln TR on session index over {t.slope_lookback}",
            t.slope_lookback,
            "Trend strength normalised by risk (t-stat-like)",
        ),
        FeatureSpec(
            f"up_frac_{t.up_fraction_lookback}",
            "trend",
            f"share of sessions with r > 0 over {t.up_fraction_lookback}",
            t.up_fraction_lookback,
            "Trend consistency (path smoothness)",
        ),
    ]
    return out
