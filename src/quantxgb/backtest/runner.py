"""Backtest orchestration (PRD Stage 5): all strategies × cost scenarios, metrics,
bootstrap CIs, Deflated Sharpe Ratio and PBO."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from quantxgb.backtest.engine import simulate
from quantxgb.backtest.metrics import performance
from quantxgb.backtest.strategies import (
    Context,
    EqualWeight,
    FixedWeights,
    InverseVol,
    MomentumTopK,
    OptimisedScores,
    Strategy,
)
from quantxgb.config import AppConfig
from quantxgb.evaluation.stats import bootstrap_ci, deflated_sharpe_ratio, pbo_cscv, sharpe_ratio
from quantxgb.tracking.trial_log import TrialLog
from quantxgb.utils.io import get_logger, write_json, write_parquet_atomic

log = get_logger(__name__)


def returns_wide(prices: pl.DataFrame) -> pl.DataFrame:
    return (
        prices.select("date", "ticker", "ret")
        .pivot(on="ticker", index="date", values="ret")
        .sort("date")
    )


def risk_free_daily(macro: pl.DataFrame | None, series: str | None) -> pl.DataFrame | None:
    """Daily risk-free return from an annualised percentage yield, keyed by availability."""
    if macro is None or series is None:
        return None
    s = macro.filter(pl.col("series_id") == series)
    if s.height == 0:
        return None
    return (
        s.sort("available_on")
        .unique(subset="available_on", keep="last", maintain_order=True)
        .select(pl.col("available_on").alias("date"), (pl.col("value") / 100.0 / 252.0).alias("rf"))
    )


def rebalance_dates(signal_dates: list[dt.date], every: int) -> list[dt.date]:
    return sorted(signal_dates)[::every]


def build_strategies(ctx: Context, predictions: pl.DataFrame) -> list[Strategy]:
    b = ctx.cfg.backtest.benchmarks
    strategies: list[Strategy] = []
    for model in sorted(predictions["model"].unique().to_list()):
        strategies.append(OptimisedScores(ctx, model, predictions.filter(pl.col("model") == model)))
    strategies += [
        EqualWeight(ctx),
        FixedWeights(ctx, b.sixty_forty, name="sixty_forty"),
        FixedWeights(ctx, {b.market: 1.0}, name="market"),
        InverseVol(ctx, b.inverse_vol_feature),
        MomentumTopK(ctx, b.momentum_top_k, ctx.cfg.model.baselines.momentum_feature),
    ]
    return strategies


@dataclass
class BacktestOutput:
    metrics: dict[str, Any]
    daily: pl.DataFrame
    weights: pl.DataFrame


def run_backtests(
    cfg: AppConfig,
    features: pl.DataFrame,
    prices: pl.DataFrame,
    macro: pl.DataFrame | None,
    predictions: pl.DataFrame,
    signal_dates: list[dt.date],
    sessions: list[dt.date],
    end: dt.date,
    out_dir: Path,
    quantiles: pl.DataFrame | None = None,
    trial_log: TrialLog | None = None,
    run_id: str = "",
    label: str = "oos",
) -> BacktestOutput:
    bt = cfg.backtest
    rets = returns_wide(prices)
    rf = risk_free_daily(macro, bt.risk_free_series)
    ctx = Context(cfg, features, rets, quantiles)
    schedule = rebalance_dates(signal_dates, bt.timing.rebalance_every_n_signals)
    strategies = build_strategies(ctx, predictions)
    costs = sorted(set(bt.costs_bps) | {bt.base_cost_bps})

    daily_parts, weight_parts = [], []
    results: dict[str, dict[str, Any]] = {}
    for cost in costs:
        for s in strategies:
            res = simulate(
                rets,
                schedule,
                s.target,
                sessions,
                cost,
                delay=bt.timing.execution_delay_sessions,
                end=end,
                rf_daily=rf,
            )
            d = res.daily.with_columns(
                pl.lit(s.name).alias("strategy"), pl.lit(cost).alias("cost_bps")
            )
            daily_parts.append(d)
            if cost == bt.base_cost_bps:
                weight_parts.append(res.weights.with_columns(pl.lit(s.name).alias("strategy")))
            results.setdefault(s.name, {})[f"{cost:g}"] = {
                "events": res.events[:20],
                "missing_returns": res.missing_returns,
            }
    daily = pl.concat(daily_parts)
    weights = pl.concat(weight_parts) if weight_parts else pl.DataFrame()

    rf_arr_cache: dict[int, np.ndarray] = {}

    def rf_for(d: pl.DataFrame) -> np.ndarray:
        key = d.height
        if rf is None:
            return np.zeros(d.height)
        if key not in rf_arr_cache:
            j = d.select("date").join_asof(rf.sort("date"), on="date", strategy="backward")
            rf_arr_cache[key] = j["rf"].fill_null(0.0).to_numpy()
        return rf_arr_cache[key]

    metrics: dict[str, Any] = {
        "label": label,
        "schedule": [str(schedule[0]), str(schedule[-1])],
        "n_rebalances": len(schedule),
        "end": str(end),
        "strategies": {},
    }
    bc = cfg.validation.bootstrap
    for s in strategies:
        per_cost = {}
        for cost in costs:
            d = daily.filter((pl.col("strategy") == s.name) & (pl.col("cost_bps") == cost)).sort(
                "date"
            )
            bench = daily.filter(
                (pl.col("strategy") == "equal_weight") & (pl.col("cost_bps") == cost)
            ).sort("date")
            perf = performance(
                d, rf_for(d), bench["ret"].to_numpy() if s.name != "equal_weight" else None
            )
            if cost == bt.base_cost_bps:
                ex = d["ret"].to_numpy() - rf_for(d)
                _, lo, hi = bootstrap_ci(
                    ex,
                    sharpe_ratio,
                    bc.mean_block_length,
                    bc.n_resamples,
                    bc.confidence,
                    seed=bt.seed,
                )
                perf.update({"sharpe_ci_low": lo, "sharpe_ci_high": hi})
            per_cost[f"{cost:g}"] = perf
        metrics["strategies"][s.name] = {
            "is_baseline": s.is_baseline,
            "by_cost": per_cost,
            **results[s.name],
        }
        if isinstance(s, OptimisedScores):
            metrics["strategies"][s.name]["optimizer_relaxations"] = s.relax_log[:50]
            metrics["strategies"][s.name]["n_relaxations"] = len(s.relax_log)

    base = f"{bt.base_cost_bps:g}"
    if trial_log is not None and label == "oos":
        for name, m in metrics["strategies"].items():
            trial_log.log(
                "strategy",
                run_id,
                strategy=name,
                is_baseline=m["is_baseline"],
                cost_bps=bt.base_cost_bps,
                sr_per_period=m["by_cost"][base].get("sr_per_period"),
                active_sr_per_period=m["by_cost"][base].get("active_sr_per_period"),
                config_hash=cfg.section_hash("features", "labels", "model", "backtest"),
            )

    def srs(key: str) -> np.ndarray:
        if trial_log is not None:
            return np.array(trial_log.strategy_srs(key=key), dtype=float)
        return np.array(
            [
                m["by_cost"][base].get(key, np.nan)
                for m in metrics["strategies"].values()
                if not m["is_baseline"]
            ],
            dtype=float,
        )

    bench_d = daily.filter(
        (pl.col("strategy") == "equal_weight") & (pl.col("cost_bps") == bt.base_cost_bps)
    ).sort("date")
    for name, m in metrics["strategies"].items():
        if m["is_baseline"]:
            continue
        d = daily.filter(
            (pl.col("strategy") == name) & (pl.col("cost_bps") == bt.base_cost_bps)
        ).sort("date")
        if not d["date"].equals(bench_d["date"]):
            raise AssertionError("strategy and benchmark return dates are misaligned")
        # absolute: is the Sharpe ratio > 0 after deflating for all trials?
        m["deflated_sharpe"] = deflated_sharpe_ratio(
            d["ret"].to_numpy() - rf_for(d), srs("sr_per_period")
        )
        # active vs. equal weight: is there alpha over the naive benchmark after deflation?
        m["deflated_sharpe_active"] = deflated_sharpe_ratio(
            d["ret"].to_numpy() - bench_d["ret"].to_numpy(), srs("active_sr_per_period")
        )
    wide = (
        daily.filter(pl.col("cost_bps") == bt.base_cost_bps)
        .pivot(on="strategy", index="date", values="ret")
        .sort("date")
        .drop_nulls()
    )
    metrics["pbo"] = pbo_cscv(wide.drop("date").to_numpy())
    metrics["pbo"]["strategies"] = [c for c in wide.columns if c != "date"]

    out_dir.mkdir(parents=True, exist_ok=True)
    write_parquet_atomic(daily, out_dir / "daily.parquet")
    if weights.height:
        write_parquet_atomic(weights, out_dir / "weights.parquet")
    write_json(metrics, out_dir / "metrics.json")
    return BacktestOutput(metrics, daily, weights)
