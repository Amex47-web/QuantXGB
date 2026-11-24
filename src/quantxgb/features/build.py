"""Feature-matrix builder (PRD FR-FEAT-01..04).

Pipeline:
1. daily panel per ticker on the master calendar → causal (trailing) features;
2. sample decision dates; keep eligible assets (≥ ``min_history_sessions`` returns);
3. within-date cross-sectional transforms on eligible assets only;
4. join date-level market features and as-of macro features.

Everything in step 1 uses only sessions ≤ t; this is enforced by the truncation and
future-perturbation tests in ``tests/leakage``.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import polars as pl

from quantxgb.config import AppConfig, FeaturesConfig
from quantxgb.features import cross_asset, liquidity, macro, momentum, risk, trend, volatility
from quantxgb.features.registry import FeatureSpec
from quantxgb.utils.calendar import weekly_decision_dates
from quantxgb.utils.io import get_logger

log = get_logger(__name__)

ID_COLUMNS = ["date", "ticker", "asset_class"]


def all_specs(cfg: FeaturesConfig) -> list[FeatureSpec]:
    return (
        momentum.specs(cfg)
        + volatility.specs(cfg)
        + trend.specs(cfg)
        + liquidity.specs(cfg)
        + risk.specs(cfg)
        + cross_asset.specs(cfg)
        + macro.specs(cfg)
    )


def model_feature_names(cfg: FeaturesConfig, available: list[str] | None = None) -> list[str]:
    names = [s.name for s in all_specs(cfg) if s.model_input]
    return [n for n in names if available is None or n in available]


def prepare_panel(prices: pl.DataFrame) -> pl.DataFrame:
    cols = ["ticker", "asset_class", "date", "close", "high", "low", "volume", "ret", "tr_index"]
    p = prices.select(cols).sort(["ticker", "date"])
    return p.with_columns(
        pl.col("tr_index").log().diff().over("ticker").alias("lr"),
        pl.int_range(pl.len()).over("ticker").alias("sess_idx"),
    )


def daily_asset_features(
    prices: pl.DataFrame, cfg: FeaturesConfig, market: str, rates: str
) -> pl.DataFrame:
    p = prepare_panel(prices)
    tickers = set(p["ticker"].unique().to_list())
    if market not in tickers:
        raise ValueError(f"market proxy {market} missing from prices")
    mkt = p.filter(pl.col("ticker") == market).select("date", pl.col("lr").alias("mkt_lr"))
    p = p.join(mkt, on="date", how="left")
    if rates in tickers:
        rts = p.filter(pl.col("ticker") == rates).select("date", pl.col("lr").alias("rates_lr"))
        p = p.join(rts, on="date", how="left")
    else:
        log.warning("rates proxy %s missing; corr_rates feature will be null", rates)
        p = p.with_columns(pl.lit(None, dtype=pl.Float64).alias("rates_lr"))
    p = p.sort(["ticker", "date"])

    stage_a = (
        momentum.exprs(cfg)
        + volatility.exprs(cfg)
        + trend.exprs(cfg)
        + liquidity.exprs(cfg)
        + risk.exprs(cfg)
    )
    p = p.with_columns(stage_a)
    p = p.with_columns(
        [*momentum.voladj_exprs(cfg), volatility.ratio_expr(cfg), trend.slope_expr(cfg)]
    )
    rel_k = cfg.cross_asset.rel_market_lookbacks
    mret = p.filter(pl.col("ticker") == market).select(
        "date", *[pl.col(f"ret_{k}").alias(f"mkt_ret_{k}") for k in rel_k]
    )
    p = p.join(mret, on="date", how="left").sort(["ticker", "date"])
    p = p.with_columns(cross_asset.exprs(cfg))
    p = p.join(risk.max_drawdown_frame(p, cfg), on=["ticker", "date"], how="left")
    p = p.with_columns(
        pl.col("ret").is_not_null().cum_sum().over("ticker").alias("n_hist"),
    ).with_columns(
        ((pl.col("n_hist") >= cfg.min_history_sessions) & pl.col("close").is_not_null()).alias(
            "eligible"
        )
    )
    # replace ±inf (e.g. log of 0 volume) with null so models treat them as missing
    num = [c for c, t in p.schema.items() if t == pl.Float64]
    return p.with_columns(
        [pl.when(pl.col(c).is_infinite()).then(None).otherwise(pl.col(c)).alias(c) for c in num]
    )


def decision_dates_for(sessions: list[dt.date], frequency: str) -> list[dt.date]:
    if frequency == "daily":
        return list(sessions)
    return weekly_decision_dates(sessions)


@dataclass
class FeatureBuild:
    matrix: pl.DataFrame  # date, ticker, asset_class, <features>
    market_daily: pl.DataFrame  # date-level daily market features (HMM inputs)
    feature_names: list[str]  # model inputs present in matrix
    specs: list[FeatureSpec]


def build_features(
    prices: pl.DataFrame,
    macro_pit: pl.DataFrame | None,
    cfg: AppConfig,
    sessions: list[dt.date],
    decision_dates: list[dt.date] | None = None,
) -> FeatureBuild:
    fc = cfg.features
    market, rates = cfg.data.market_proxy, cfg.data.rates_proxy
    daily = daily_asset_features(prices, fc, market, rates)
    if decision_dates is None:
        # Decision dates come from the exchange calendar (known in advance), never from the
        # data: "last session of the week" must not depend on whether later bars exist.
        decision_dates = decision_dates_for(sessions, cfg.labels.signal_frequency)

    spec_list = all_specs(fc)
    asset_cols = [s.name for s in spec_list if s.scope == "asset" and s.name in daily.columns]
    helper = [f"ret_{k}" for k in fc.cross_asset.rel_class_lookbacks] + momentum.xsrank_columns(fc)
    keep = list(dict.fromkeys(ID_COLUMNS + asset_cols + helper))
    sampled = daily.filter(pl.col("date").is_in(decision_dates) & pl.col("eligible")).select(keep)
    sampled = sampled.with_columns(cross_asset.rel_class_exprs(fc))
    xs_cols = momentum.xsrank_columns(fc)
    sampled = sampled.with_columns(
        [
            (pl.col(c).rank("average").over("date") / pl.col(c).count().over("date")).alias(
                f"{c}_xsrank"
            )
            for c in xs_cols
        ]
    )

    mkt_daily = cross_asset.market_frame(daily, fc, market)
    mkt_cols = [c for c in mkt_daily.columns if c != "date"]
    sampled = sampled.join(mkt_daily.select(["date", *mkt_cols]), on="date", how="left")

    mac = macro.macro_frame(
        macro_pit,
        fc,
        sorted(set(sampled["date"].to_list())),
        sessions,
        cfg.data.availability.revised_macro_lag_sessions,
    )
    sampled = sampled.join(mac, on="date", how="left")

    present = set(sampled.columns)
    names = [s.name for s in spec_list if s.model_input and s.name in present]
    missing = [s.name for s in spec_list if s.model_input and s.name not in present]
    if missing:
        log.warning("features not built (missing inputs): %s", ", ".join(missing))
    matrix = sampled.select(ID_COLUMNS + names).sort(["date", "ticker"])
    return FeatureBuild(matrix=matrix, market_daily=mkt_daily, feature_names=names, specs=spec_list)
