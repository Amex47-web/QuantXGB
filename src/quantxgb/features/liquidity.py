"""Liquidity features (PRD §8.2)."""

from __future__ import annotations

import polars as pl

from quantxgb.config import FeaturesConfig
from quantxgb.features.registry import FeatureSpec

DVOL = pl.col("close") * pl.col("volume")


def exprs(cfg: FeaturesConfig) -> list[pl.Expr]:
    q = cfg.liquidity
    d, z, a = q.dollar_volume_lookback, q.volume_z_lookback, q.amihud_lookback
    lv = pl.col("volume").log1p()
    safe_dvol = pl.when(DVOL > 0).then(DVOL)
    return [
        DVOL.rolling_mean(d).log().over("ticker").alias(f"log_dvol_{d}"),
        ((lv - lv.rolling_mean(z)) / lv.rolling_std(z)).over("ticker").alias(f"volume_z_{z}"),
        ((pl.col("ret").abs() / safe_dvol).rolling_mean(a) * 1e9)
        .over("ticker")
        .alias(f"amihud_{a}"),
    ]


def specs(cfg: FeaturesConfig) -> list[FeatureSpec]:
    q = cfg.liquidity
    d, z, a = q.dollar_volume_lookback, q.volume_z_lookback, q.amihud_lookback
    return [
        FeatureSpec(
            f"log_dvol_{d}",
            "liquidity",
            f"ln(mean(close · volume) over {d})",
            d,
            "Tradability / size of the market for the asset",
        ),
        FeatureSpec(
            f"volume_z_{z}",
            "liquidity",
            f"(ln(1+V_t) − mean_{z}) / std_{z} of ln(1+V)",
            z,
            "Abnormal volume: attention and flow pressure",
        ),
        FeatureSpec(
            f"amihud_{a}",
            "liquidity",
            f"mean(|ret| / (close · volume)) over {a} · 1e9",
            a,
            "Amihud illiquidity: price impact per dollar traded",
        ),
    ]
