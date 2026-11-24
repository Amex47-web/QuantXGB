"""Event-driven daily backtest engine (PRD §10.4, FR-BT-01..03) — critical module.

Timing: a target computed from information at the close of signal date t is executed at
the close of session t + delay. Between executions, holdings drift with returns.

Per session d (weights w are fractions of NAV at the previous close, cash = 1 − Σw):

    gross_d = Σ_i w_i r_i,d + (1 − Σ w) · rf_d
    NAV_d   = NAV_{d−1} · (1 + gross_d)
    w_i     ← w_i (1 + r_i,d) / (1 + gross_d)                        (drift)
    if d is an execution date:
        cost_d = c · Σ_i |w*_i − w_i|                                (one-way, on traded notional)
        NAV_d  ← NAV_d · (1 − cost_d);  w ← w*

The accounting identity NAV_d = NAV_{d−1}(1 + gross_d)(1 − cost_d) is asserted.
"""

from __future__ import annotations

import bisect
import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import polars as pl

TargetFn = Callable[[dt.date, dict[str, float]], dict[str, float]]


@dataclass
class BacktestResult:
    daily: pl.DataFrame  # date, nav, ret, gross, cost, turnover
    weights: pl.DataFrame  # exec_date, signal_date, ticker, weight
    events: list[str] = field(default_factory=list)
    missing_returns: int = 0


def simulate(
    returns: pl.DataFrame,
    signal_dates: list[dt.date],
    target_fn: TargetFn,
    sessions: list[dt.date],
    cost_bps: float,
    delay: int = 1,
    end: dt.date | None = None,
    rf_daily: pl.DataFrame | None = None,
) -> BacktestResult:
    """``returns``: wide frame ``date, <ticker>...`` of daily simple total returns."""
    if delay < 1:
        raise ValueError("execution delay must be >= 1 session (no same-bar execution)")
    returns = returns.sort("date")
    tickers = [c for c in returns.columns if c != "date"]
    col = {t: i for i, t in enumerate(tickers)}
    dates = returns["date"].to_list()
    r = returns.select(tickers).cast(pl.Float64).fill_null(np.nan).to_numpy()
    rf = np.zeros(len(dates))
    if rf_daily is not None and rf_daily.height:
        j = pl.DataFrame({"date": dates}).join_asof(
            rf_daily.sort("date"), on="date", strategy="backward"
        )
        rf = j["rf"].fill_null(0.0).to_numpy()

    exec_for: dict[dt.date, dt.date] = {}
    for s in sorted(signal_dates):
        i = bisect.bisect_right(sessions, s) + delay - 1
        if i < len(sessions):
            exec_for[sessions[i]] = s
    if not exec_for:
        raise ValueError("no executable signal dates")
    start = min(exec_for)
    last = end or dates[-1]

    w = np.zeros(len(tickers))
    nav = 1.0
    c = cost_bps / 1e4
    rows: list[tuple[dt.date, float, float, float, float, float]] = []
    wrows: list[tuple[dt.date, dt.date, str, float]] = []
    pending: dict[str, float] | None = None
    missing = 0
    events: list[str] = []
    for k, d in enumerate(dates):
        if d < start or d > last:
            continue
        rd = r[k]
        held = w != 0
        bad = held & np.isnan(rd)
        if bad.any():
            missing += int(bad.sum())
            events.append(
                f"{d}: missing return for held {[tickers[i] for i in np.where(bad)[0]]} (0 used)"
            )
        rd = np.where(np.isnan(rd), 0.0, rd)
        gross = float(w @ rd + (1.0 - w.sum()) * rf[k])
        prev_nav = nav
        nav *= 1.0 + gross
        w = w * (1.0 + rd) / (1.0 + gross) if (1.0 + gross) > 0 else w
        cost = turnover = 0.0
        if d in exec_for:
            signal = exec_for[d]
            current = {tickers[i]: float(w[i]) for i in np.where(w != 0)[0]}
            pending = target_fn(signal, current)
            target = np.zeros(len(tickers))
            for t, x in pending.items():
                if t not in col:
                    raise KeyError(f"target ticker {t} has no return series")
                target[col[t]] = x
            if target.sum() > 1 + 1e-6 or (target < -1e-9).any():
                raise ValueError(f"invalid target weights on {d}: sum={target.sum():.6f}")
            turnover = float(np.abs(target - w).sum())
            cost = c * turnover
            nav *= 1.0 - cost
            w = target
            wrows.extend((d, signal, t, float(x)) for t, x in pending.items() if x > 1e-10)
        expected = prev_nav * (1.0 + gross) * (1.0 - cost)
        if not np.isclose(nav, expected, rtol=1e-12, atol=0):
            raise AssertionError(f"accounting identity violated on {d}")
        rows.append((d, nav, nav / prev_nav - 1.0, gross, cost, turnover))

    daily = pl.DataFrame(
        rows, schema=["date", "nav", "ret", "gross", "cost", "turnover"], orient="row"
    ).with_columns(pl.col("date").cast(pl.Date))
    weights = (
        pl.DataFrame(wrows, schema=["exec_date", "signal_date", "ticker", "weight"], orient="row")
        if wrows
        else pl.DataFrame(
            schema={
                "exec_date": pl.Date,
                "signal_date": pl.Date,
                "ticker": pl.String,
                "weight": pl.Float64,
            }
        )
    )
    return BacktestResult(daily, weights, events, missing)
