from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from quantxgb.evaluation.signal_metrics import (
    ic_summary,
    newey_west_tstat,
    pinball_loss,
    quantile_metrics,
    rank_ic_by_date,
)
from quantxgb.evaluation.stats import (
    bootstrap_ci,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    pbo_cscv,
    probabilistic_sharpe_ratio,
    stationary_bootstrap_indices,
)


def test_rank_ic_perfect_and_inverse() -> None:
    d = dt.date(2020, 1, 1)
    df = pl.DataFrame(
        {
            "date": [d] * 5 + [d + dt.timedelta(1)] * 5,
            "score": list(range(5)) * 2,
            "fwd_xs_ret": [0.1, 0.2, 0.3, 0.4, 0.5] + [0.5, 0.4, 0.3, 0.2, 0.1],
        }
    )
    ic = rank_ic_by_date(df)["ic"].to_list()
    assert ic == pytest.approx([1.0, -1.0])
    s = ic_summary(rank_ic_by_date(df), 1)
    assert s["ic_mean"] == pytest.approx(0.0)


def test_newey_west_reduces_t_for_autocorrelated_series() -> None:
    rng = np.random.default_rng(0)
    e = rng.normal(size=2000)
    x = 0.05 + np.convolve(e, np.ones(10) / 10, mode="same")
    assert abs(newey_west_tstat(x, 10)) < abs(newey_west_tstat(x, 0))


def test_pinball_and_coverage() -> None:
    y = np.array([0.0, 1.0])
    assert pinball_loss(y, np.array([0.5, 0.5]), 0.5) == pytest.approx(0.25)
    qm = quantile_metrics(y, np.array([[-1, 0, 2], [-1, 0, 0.5]], dtype=float), [0.05, 0.5, 0.95])
    assert qm["coverage"] == 0.5


def test_stationary_bootstrap_shapes_and_ci() -> None:
    rng = np.random.default_rng(0)
    idx = stationary_bootstrap_indices(100, 10, 50, rng)
    assert idx.shape == (50, 100) and idx.min() >= 0 and idx.max() < 100
    x = rng.normal(0.1, 1, 500)
    est, lo, hi = bootstrap_ci(x, np.mean, 10, 500, 0.95)
    assert lo < est < hi


def test_psr_and_dsr() -> None:
    rng = np.random.default_rng(1)
    good = rng.normal(0.002, 0.01, 1000)
    assert probabilistic_sharpe_ratio(good, 0.0) > 0.99
    assert expected_max_sharpe(np.array([0.1])) == 0.0
    few = deflated_sharpe_ratio(good, np.array([0.0, 0.05, 0.1]))["dsr"]
    many = deflated_sharpe_ratio(good, rng.normal(0.05, 0.08, 500))["dsr"]
    assert many < few  # more trials → harder to pass


def test_pbo_detects_overfitting_and_skill() -> None:
    rng = np.random.default_rng(2)
    noise = rng.normal(0, 0.01, (800, 10))
    assert 0.2 < pbo_cscv(noise, 8)["pbo"] < 0.8  # no skill → selection is a coin flip-ish
    skill = noise.copy()
    skill[:, 0] += 0.003
    assert pbo_cscv(skill, 8)["pbo"] < 0.05
