from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from quantxgb.config import ConstraintsConfig, OptimizerConfig
from quantxgb.portfolio.covariance import estimate_covariance
from quantxgb.portfolio.expected_returns import grinold_mu, trailing_ic, zscore
from quantxgb.portfolio.optimizer import optimize, two_piece_normal_scenarios

CLASSES = ["eq", "eq", "eq", "rates", "rates", "credit", "gold", "re"]


def _sigma(n: int = 8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.01, (300, n))
    return estimate_covariance(r, "ledoit_wolf") * 20


@pytest.mark.parametrize("method", ["sample", "ledoit_wolf", "ewma"])
def test_covariance_psd(method: str) -> None:
    r = np.random.default_rng(0).normal(0, 0.01, (100, 5))
    c = estimate_covariance(r, method)
    assert np.allclose(c, c.T) and np.linalg.eigvalsh(c).min() > -1e-12


def test_mean_variance_respects_constraints() -> None:
    mu = np.array([0.05, 0.04, 0.03, 0.0, 0.0, 0.0, 0.0, 0.0])
    cons = ConstraintsConfig(w_max=0.25, asset_class_max=0.5, turnover_max=0.5)
    res = optimize(mu, np.zeros(8), CLASSES, cons, OptimizerConfig(), sigma=_sigma())
    w = res.weights
    assert res.relaxed == []
    assert w.sum() == pytest.approx(1.0)
    assert (w >= -1e-9).all() and (w <= 0.25 + 1e-6).all()
    assert w[:3].sum() <= 0.5 + 1e-6  # equity class cap binds


def test_turnover_cap_and_relaxation() -> None:
    cons = ConstraintsConfig(w_max=0.25, asset_class_max=1.0, turnover_max=0.2)
    prev = np.array([0.25, 0.25, 0.25, 0.25, 0, 0, 0, 0])
    mu = np.array([0, 0, 0, 0, 0.1, 0.1, 0.1, 0.1])
    res = optimize(mu, prev, CLASSES, cons, OptimizerConfig(risk_aversion=0.0), sigma=_sigma())
    assert np.abs(res.weights - prev).sum() <= 0.2 + 1e-5
    # forced sale of 0.5 in a dropped asset exceeds the cap → turnover cap is relaxed, not violated silently
    res2 = optimize(
        mu[:4],
        np.array([0.25, 0.25, 0, 0]),
        CLASSES[:4],
        ConstraintsConfig(w_max=0.5, asset_class_max=1.0, turnover_max=0.2),
        OptimizerConfig(),
        sigma=_sigma(4),
        forced_sell=0.5,
    )
    assert "turnover_cap" in res2.relaxed


def test_cvar_optimizer_runs() -> None:
    rng = np.random.default_rng(0)
    scen = rng.normal(0.0, 0.05, (500, 8))
    scen[:, 0] -= 0.2 * (rng.random(500) < 0.05)  # asset 0 has a fat left tail
    res = optimize(
        np.zeros(8),
        np.zeros(8),
        CLASSES,
        ConstraintsConfig(asset_class_max=1.0),
        OptimizerConfig(type="cvar"),
        scenarios=scen,
    )
    assert res.weights.sum() == pytest.approx(1.0)
    assert res.weights[0] < 0.125  # tail asset under-weighted vs. equal weight


def test_two_piece_normal_matches_quantiles() -> None:
    q_lo, q_mid, q_hi = np.array([-0.1]), np.array([0.01]), np.array([0.05])
    s = two_piece_normal_scenarios(q_lo, q_mid, q_hi, np.eye(1), 200_000, -1.6449, 1.6449, 0)
    assert np.quantile(s, 0.05) == pytest.approx(-0.1, abs=3e-3)
    assert np.quantile(s, 0.5) == pytest.approx(0.01, abs=2e-3)
    assert np.quantile(s, 0.95) == pytest.approx(0.05, abs=3e-3)


def test_grinold_mu_and_trailing_ic_point_in_time() -> None:
    assert np.all(grinold_mu(np.array([1.0, 2, 3]), np.array([0.2] * 3), float("nan"), 20) == 0)
    assert np.all(grinold_mu(np.array([1.0, 2, 3]), np.array([0.2] * 3), -0.05, 20) == 0)
    mu = grinold_mu(np.array([1.0, 2, 3]), np.array([0.2] * 3), 0.05, 252)
    np.testing.assert_allclose(mu, 0.05 * 0.2 * zscore(np.array([1.0, 2, 3])))
    d = [dt.date(2020, 1, i) for i in range(1, 6)]
    ic = pl.DataFrame({"date": d, "ic": [0.1, 0.2, 0.3, 0.4, 0.5]})
    ends = pl.DataFrame({"date": d, "label_end": [dt.date(2020, 1, i + 10) for i in range(1, 6)]})
    # at Jan 12 only the first two labels (ending Jan 11, 12) are realised
    assert trailing_ic(ic, ends, dt.date(2020, 1, 12), 2) == pytest.approx(0.15)
    assert np.isnan(trailing_ic(ic, ends, dt.date(2020, 1, 11), 2))
