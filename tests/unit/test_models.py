from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from quantxgb.config import HMMConfig, QuantileConfig, RankerConfig, RegimeRulesConfig
from quantxgb.models.alpha_ranker import XGBAlphaRanker
from quantxgb.models.base import query_ids
from quantxgb.models.quantile_risk import XGBQuantileRisk
from quantxgb.models.regime_detector import HMMRegime, RuleRegime


def _panel(n_dates: int = 120, n_assets: int = 10, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    d0 = dt.date(2015, 1, 2)
    rows = []
    for i in range(n_dates):
        x = rng.normal(size=n_assets)
        y = 0.8 * x + rng.normal(scale=0.5, size=n_assets)
        ranks = y.argsort().argsort()
        for j in range(n_assets):
            rows.append(
                (
                    d0 + dt.timedelta(days=7 * i),
                    f"T{j}",
                    x[j],
                    rng.normal(),
                    y[j] / 10,
                    int(ranks[j] * 5 // n_assets),
                )
            )
    return pl.DataFrame(
        rows, schema=["date", "ticker", "f1", "noise", "fwd_ret", "fwd_grade"], orient="row"
    )


def test_query_ids_require_sorted_dates() -> None:
    df = _panel(3, 2)
    assert len(np.unique(query_ids(df))) == 3
    with pytest.raises(ValueError):
        query_ids(df.reverse())


def test_ranker_learns_signal_and_early_stops() -> None:
    df = _panel()
    train, valid = (
        df.filter(pl.col("date") < dt.date(2016, 6, 1)),
        df.filter(pl.col("date") >= dt.date(2016, 6, 1)),
    )
    cfg = RankerConfig(
        params={"n_estimators": 300, "learning_rate": 0.1, "max_depth": 2}, early_stopping_rounds=20
    )
    m = XGBAlphaRanker(cfg, seed=1).fit(train, ["f1", "noise"], valid=valid)
    assert m.best_iteration is not None and m.best_iteration < 300
    s = m.predict(valid)
    corr = np.corrcoef(s, valid["f1"].to_numpy())[0, 1]
    assert corr > 0.8
    contrib = m.contributions(valid)
    assert contrib.shape == (valid.height, 3)
    np.testing.assert_allclose(
        contrib.sum(axis=1),
        m.model.predict(  # type: ignore[union-attr]
            valid.select("f1", "noise").to_numpy(),
            output_margin=True,
            iteration_range=m._iteration_range(),
        ),
        rtol=1e-4,
        atol=1e-4,
    )
    fresh = m.with_n_estimators(10)
    assert fresh.model is None and fresh.params["n_estimators"] == 10


def test_quantile_model_is_non_crossing_and_ordered() -> None:
    df = _panel()
    m = XGBQuantileRisk(QuantileConfig(params={"n_estimators": 50, "max_depth": 2})).fit(
        df, ["f1", "noise"]
    )
    q = m.predict(df)
    assert q.shape == (df.height, 3)
    assert (np.diff(q, axis=1) >= 0).all()
    y = df["fwd_ret"].to_numpy()
    cover = ((y >= q[:, 0]) & (y <= q[:, 2])).mean()
    assert 0.75 < cover <= 1.0


def _daily(n: int = 600, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    vol = np.where(np.arange(n) % 200 < 100, 0.1, 0.3) + rng.normal(0, 0.01, n)
    ret = rng.normal(0, vol / 16)
    d0 = dt.date(2010, 1, 1)
    return pl.DataFrame(
        {
            "date": [d0 + dt.timedelta(days=i) for i in range(n)],
            "market_ret_1d": ret,
            "market_vol_20": vol,
        }
    )


def test_hmm_filtered_probabilities_do_not_use_the_future() -> None:
    daily = _daily()
    hmm = HMMRegime(HMMConfig(n_iter=50)).fit(daily.head(400))
    full = hmm.filtered_probabilities(daily)
    cut = daily["date"][450]
    trunc = hmm.filtered_probabilities(daily.filter(pl.col("date") <= cut))
    a = full.filter(pl.col("date") <= cut).drop("date").to_numpy()
    np.testing.assert_allclose(a, trunc.drop("date").to_numpy(), rtol=0, atol=1e-12)
    probs = full.drop("date").to_numpy()
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)
    # state 0 is the calm state by construction
    hi = (
        full.join(daily, on="date")
        .group_by(pl.col("hmm_p_state_1") > 0.5)
        .agg(pl.col("market_vol_20").mean())
    )
    calm = hi.filter(~pl.col("hmm_p_state_1"))["market_vol_20"][0]
    stressed = hi.filter(pl.col("hmm_p_state_1"))["market_vol_20"][0]
    assert stressed > calm


def test_rule_regime_threshold_from_training_only() -> None:
    train = pl.DataFrame(
        {
            "date": [dt.date(2020, 1, i + 1) for i in range(30)],
            "market_vol_20": np.linspace(0.1, 0.4, 30),
            "market_above_sma200": [1.0] * 30,
        }
    )
    r = RuleRegime(RegimeRulesConfig(vol_feature="missing")).fit(train)
    assert r.vol_feature == "market_vol_20"
    assert r.threshold == pytest.approx(np.quantile(np.linspace(0.1, 0.4, 30), 0.8))
    test = pl.DataFrame(
        {
            "date": [dt.date(2021, 1, 1), dt.date(2021, 1, 2)],
            "market_vol_20": [0.05, 0.9],
            "market_above_sma200": [0.0, 1.0],
        }
    )
    out = r.predict(test)
    assert out["regime"].to_list() == ["downtrend/normal_vol", "uptrend/high_vol"]
