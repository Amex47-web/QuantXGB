"""Portfolio performance metrics (PRD §9.5)."""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from quantxgb.evaluation.stats import probabilistic_sharpe_ratio


def max_drawdown(nav: np.ndarray) -> float:
    peak = np.maximum.accumulate(nav)
    return float((nav / peak - 1.0).min()) if nav.size else float("nan")


def performance(
    daily: pl.DataFrame,
    rf: np.ndarray | None = None,
    benchmark: np.ndarray | None = None,
    periods: int = 252,
) -> dict[str, float]:
    r = daily["ret"].to_numpy()
    n = r.size
    if n < 2:
        return {"n_days": float(n)}
    rf = np.zeros(n) if rf is None else rf
    ex = r - rf
    years = n / periods
    nav = daily["nav"].to_numpy()
    base = nav[0] / (1.0 + r[0])  # NAV before the first row's return/cost
    growth = np.r_[1.0, nav / base]
    total = growth[-1]
    cagr = total ** (1 / years) - 1 if total > 0 else float("nan")
    vol = r.std(ddof=1) * math.sqrt(periods)
    sharpe = ex.mean() / ex.std(ddof=1) * math.sqrt(periods) if ex.std(ddof=1) > 0 else float("nan")
    downside = np.sqrt(np.mean(np.minimum(ex, 0.0) ** 2)) * math.sqrt(periods)
    sortino = ex.mean() * periods / downside if downside > 0 else float("nan")
    mdd = max_drawdown(growth)
    var95 = float(np.quantile(r, 0.05))
    cvar95 = float(r[r <= var95].mean())
    out = {
        "cagr": float(cagr),
        "ann_vol": float(vol),
        "sharpe": float(sharpe),
        "sr_per_period": float(ex.mean() / ex.std(ddof=1)) if ex.std(ddof=1) > 0 else float("nan"),
        "sortino": float(sortino),
        "max_drawdown": mdd,
        "calmar": float(cagr / abs(mdd)) if mdd < 0 else float("nan"),
        "var_95_daily": var95,
        "cvar_95_daily": cvar95,
        "turnover_annual": float(daily["turnover"].to_numpy().sum()) / years,
        "cost_drag_annual": float(daily["cost"].to_numpy().sum()) / years,
        "psr_vs_0": probabilistic_sharpe_ratio(ex, 0.0),
        "n_days": float(n),
    }
    if benchmark is not None and benchmark.size == n:
        active = r - benchmark
        te = active.std(ddof=1) * math.sqrt(periods)
        out["tracking_error"] = float(te)
        out["information_ratio"] = float(active.mean() * periods / te) if te > 0 else float("nan")
        sd_a = active.std(ddof=1)
        out["active_sr_per_period"] = float(active.mean() / sd_a) if sd_a > 0 else float("nan")
    return out
