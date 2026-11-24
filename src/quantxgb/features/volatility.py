"""Volatility features (PRD §8.2)."""

from __future__ import annotations

import math

import polars as pl

from quantxgb.config import FeaturesConfig
from quantxgb.features.registry import FeatureSpec

R = pl.col("lr")
ANN = math.sqrt(252.0)


def exprs(cfg: FeaturesConfig) -> list[pl.Expr]:
    v = cfg.volatility
    out = [
        (R.rolling_std(k) * ANN).over("ticker").alias(f"vol_{k}")
        for k in sorted(required_vol_lookbacks(cfg))
    ]
    d = v.downside_lookback
    out.append(
        ((pl.min_horizontal(R, pl.lit(0.0)) ** 2).rolling_mean(d).sqrt() * ANN)
        .over("ticker")
        .alias(f"downvol_{d}")
    )
    p = v.parkinson_lookback
    hl = (pl.col("high") / pl.col("low")).log() ** 2
    out.append(
        ((hl.rolling_mean(p) / (4.0 * math.log(2.0))).sqrt() * ANN)
        .over("ticker")
        .alias(f"parkinson_{p}")
    )
    return out


def ratio_expr(cfg: FeaturesConfig) -> pl.Expr:
    s, lg = cfg.volatility.ratio["short"], cfg.volatility.ratio["long"]
    return (pl.col(f"vol_{s}") / pl.col(f"vol_{lg}")).alias(f"vol_ratio_{s}_{lg}")


def required_vol_lookbacks(cfg: FeaturesConfig) -> set[int]:
    v = cfg.volatility
    return set(v.lookbacks) | {v.ratio["short"], v.ratio["long"], 60}


def specs(cfg: FeaturesConfig) -> list[FeatureSpec]:
    v = cfg.volatility
    s, lg = v.ratio["short"], v.ratio["long"]
    out = [
        FeatureSpec(
            f"vol_{k}",
            "volatility",
            f"std(r over {k}) · √252",
            k,
            "Risk level; low-volatility effect and volatility clustering",
        )
        for k in sorted(required_vol_lookbacks(cfg))
    ]
    out += [
        FeatureSpec(
            f"downvol_{v.downside_lookback}",
            "volatility",
            f"√mean(min(r,0)²) over {v.downside_lookback} · √252 (semi-deviation)",
            v.downside_lookback,
            "Downside risk, which investors price differently from upside variance",
        ),
        FeatureSpec(
            f"vol_ratio_{s}_{lg}",
            "volatility",
            f"vol_{s} / vol_{lg}",
            lg,
            "Volatility regime shift for the asset (rising vs. falling risk)",
        ),
        FeatureSpec(
            f"parkinson_{v.parkinson_lookback}",
            "volatility",
            f"√(mean(ln(H/L)²) / (4 ln 2)) over {v.parkinson_lookback} · √252",
            v.parkinson_lookback,
            "Range-based volatility, more efficient than close-to-close",
        ),
    ]
    return out
