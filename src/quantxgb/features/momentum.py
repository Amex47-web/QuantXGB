"""Momentum features (PRD §8.2)."""

from __future__ import annotations

import math

import polars as pl

from quantxgb.config import FeaturesConfig
from quantxgb.features.registry import FeatureSpec

TR = pl.col("tr_index")


def exprs(cfg: FeaturesConfig) -> list[pl.Expr]:
    m = cfg.momentum
    out = [(TR / TR.shift(k) - 1.0).over("ticker").alias(f"ret_{k}") for k in m.return_lookbacks]
    long, short = m.skip_month["long"], m.skip_month["short"]
    out.append((TR.shift(short) / TR.shift(long) - 1.0).over("ticker").alias("mom_12_1"))
    return out


def voladj_exprs(cfg: FeaturesConfig) -> list[pl.Expr]:
    """Needs ``ret_k`` and ``vol_60`` already computed."""
    out = []
    for k in cfg.momentum.voladj_lookbacks:
        denom = pl.col("vol_60") * math.sqrt(k / 252.0)
        out.append((pl.col(f"ret_{k}") / denom).alias(f"ret_{k}_voladj"))
    return out


def xsrank_columns(cfg: FeaturesConfig) -> list[str]:
    return [f"ret_{k}" for k in cfg.momentum.xsrank_lookbacks] + ["mom_12_1"]


def specs(cfg: FeaturesConfig) -> list[FeatureSpec]:
    m = cfg.momentum
    out = [
        FeatureSpec(
            f"ret_{k}",
            "momentum",
            f"TR_t / TR_(t-{k}) − 1",
            k,
            "Time-series momentum / short-term reversal over the window",
        )
        for k in m.return_lookbacks
    ]
    long, short = m.skip_month["long"], m.skip_month["short"]
    out.append(
        FeatureSpec(
            "mom_12_1",
            "momentum",
            f"TR_(t-{short}) / TR_(t-{long}) − 1",
            long,
            "Classic 12-1 momentum, skipping the most recent month to avoid short-term reversal",
        )
    )
    out += [
        FeatureSpec(
            f"ret_{k}_voladj",
            "momentum",
            f"ret_{k} / (vol_60 · √({k}/252))",
            max(k, 60),
            "Risk-adjusted momentum: return per unit of volatility (comparable across classes)",
        )
        for k in m.voladj_lookbacks
    ]
    out += [
        FeatureSpec(
            f"{c}_xsrank",
            "momentum",
            f"percentile rank of {c} among eligible assets on date t",
            0,
            "Relative strength — the quantity a cross-sectional ranker ultimately orders",
            scope="cross_sectional",
        )
        for c in xsrank_columns(cfg)
    ]
    return out
