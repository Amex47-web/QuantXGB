"""Backtest timing and accounting (FR-BT-01..03)."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from quantxgb.backtest.engine import simulate
from quantxgb.backtest.metrics import performance

D = [dt.date(2021, 1, 4) + dt.timedelta(days=i) for i in range(10)]


def _returns() -> pl.DataFrame:
    a = [0.0, 0.01, 0.10, -0.02, 0.03, 0.0, 0.01, 0.02, -0.01, 0.0]
    b = [0.0, 0.0, 0.0, 0.01, 0.01, 0.01, 0.0, 0.0, 0.0, 0.0]
    return pl.DataFrame({"date": D, "A": a, "B": b})


def test_execution_delay_no_same_bar() -> None:
    # signal at D[1]; executed at the close of D[2]; first return earned on D[3]
    res = simulate(_returns(), [D[1]], lambda t, cur: {"A": 1.0}, D, cost_bps=0, delay=1)
    daily = res.daily
    assert daily["date"][0] == D[2]
    assert daily["ret"][0] == 0.0  # the +10% on D[2] is NOT earned (position opened at that close)
    assert daily["ret"][1] == pytest.approx(-0.02)
    with pytest.raises(ValueError):
        simulate(_returns(), [D[1]], lambda t, cur: {"A": 1.0}, D, cost_bps=0, delay=0)


def test_costs_drift_and_accounting() -> None:
    calls: list[tuple[dt.date, dict[str, float]]] = []

    def target(t: dt.date, cur: dict[str, float]) -> dict[str, float]:
        calls.append((t, cur))
        return {"A": 0.5, "B": 0.5}

    res = simulate(_returns(), [D[0], D[4]], target, D, cost_bps=10, delay=1)
    daily = res.daily
    assert daily["cost"][0] == pytest.approx(0.001 * 1.0)  # from cash: turnover 1
    # second signal sees drifted weights, not the stale target
    cur = calls[1][1]
    assert cur["A"] != pytest.approx(0.5)
    assert sum(cur.values()) == pytest.approx(1.0)
    nav = daily["nav"].to_numpy()
    gross, cost = daily["gross"].to_numpy(), daily["cost"].to_numpy()
    rebuilt = np.cumprod((1 + gross) * (1 - cost))
    np.testing.assert_allclose(nav, rebuilt, rtol=1e-12)
    assert res.weights.filter(pl.col("signal_date") == D[4]).height == 2


def test_invalid_targets_rejected() -> None:
    with pytest.raises(ValueError):
        simulate(_returns(), [D[0]], lambda t, c: {"A": 0.8, "B": 0.5}, D, cost_bps=0)
    with pytest.raises(KeyError):
        simulate(_returns(), [D[0]], lambda t, c: {"ZZZ": 1.0}, D, cost_bps=0)


def test_cash_earns_risk_free() -> None:
    rf = pl.DataFrame({"date": [D[0]], "rf": [0.0001]})
    res = simulate(_returns(), [D[0]], lambda t, c: {"A": 0.5}, D, cost_bps=0, rf_daily=rf)
    # executed at the close of D[1]; first holding day D[2]: half in A (+10%), half cash at rf
    row = res.daily.row(1, named=True)
    assert row["date"] == D[2]
    assert row["gross"] == pytest.approx(0.5 * 0.10 + 0.5 * 0.0001)
    # next day uses drifted weights: A's weight grew after its +10% day
    w_a = 0.5 * 1.10 / (1 + row["gross"])
    assert res.daily.row(2, named=True)["gross"] == pytest.approx(w_a * -0.02 + (1 - w_a) * 0.0001)


def test_performance_metrics() -> None:
    res = simulate(_returns(), [D[0]], lambda t, c: {"A": 1.0}, D, cost_bps=0)
    p = performance(res.daily)
    growth = np.prod(1 + res.daily["ret"].to_numpy())
    assert p["cagr"] == pytest.approx(growth ** (252 / res.daily.height) - 1)
    assert p["max_drawdown"] <= 0
