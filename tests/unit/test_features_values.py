"""Hand-checked feature values (owner should extend these with real-data spot checks)."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from quantxgb.config import AppConfig
from quantxgb.features.build import build_features, daily_asset_features
from quantxgb.features.risk import rolling_max_drawdown
from quantxgb.synthetic import SyntheticData


@pytest.fixture(scope="module")
def daily(small_cfg: AppConfig, small_data: SyntheticData) -> pl.DataFrame:
    return daily_asset_features(small_data.prices, small_cfg.features, "MKT", "RATES")


def _one(daily: pl.DataFrame, ticker: str, idx: int) -> dict[str, float]:
    return daily.filter(pl.col("ticker") == ticker).sort("date").row(idx, named=True)


def test_momentum_and_volatility_by_hand(daily: pl.DataFrame) -> None:
    g = daily.filter(pl.col("ticker") == "A05").sort("date")
    tr = g["tr_index"].to_numpy()
    lr = np.diff(np.log(tr))
    i = 400
    row = g.row(i, named=True)
    assert row["ret_20"] == pytest.approx(tr[i] / tr[i - 20] - 1)
    assert row["mom_12_1"] == pytest.approx(tr[i - 21] / tr[i - 252] - 1)
    window = lr[i - 60 : i]  # lr[k] is the return from k to k+1 → returns ending at i
    assert row["vol_60"] == pytest.approx(window.std(ddof=1) * math.sqrt(252))
    assert row["ret_60_voladj"] == pytest.approx(
        row["ret_60"] / (row["vol_60"] * math.sqrt(60 / 252))
    )
    assert row["dd_252"] == pytest.approx(tr[i] / tr[i - 251 : i + 1].max() - 1)


def test_beta_by_hand(daily: pl.DataFrame) -> None:
    a = daily.filter(pl.col("ticker") == "A05").sort("date")
    m = daily.filter(pl.col("ticker") == "MKT").sort("date")
    j = a.join(m.select("date", pl.col("lr").alias("m")), on="date")
    i = 500
    x = j["lr"].to_numpy()[i - 59 : i + 1]
    y = j["m"].to_numpy()[i - 59 : i + 1]
    beta = np.cov(x, y, ddof=0)[0, 1] / np.var(y)
    assert j.row(i, named=True)["beta_mkt_60"] == pytest.approx(beta, rel=1e-6)


def test_rolling_max_drawdown() -> None:
    v = np.array([1.0, 2.0, 1.0, 1.5, 3.0, 1.5])
    out = rolling_max_drawdown(v, 3)
    assert np.isnan(out[:2]).all()
    assert out[2] == pytest.approx(-0.5)  # 2 → 1
    assert out[3] == pytest.approx(-0.5)
    assert out[5] == pytest.approx(-0.5)  # 3 → 1.5
    v[3] = np.nan
    assert np.isnan(rolling_max_drawdown(v, 3)[3:6]).all()


def test_cross_sectional_features(small_cfg: AppConfig, small_data: SyntheticData) -> None:
    fb = build_features(small_data.prices, small_data.macro, small_cfg, small_data.sessions)
    m = fb.matrix
    d = m["date"].unique().sort()[200]
    day = m.filter(pl.col("date") == d)
    # percentile ranks lie in (0, 1] and the max equals 1
    assert day["ret_20_xsrank"].max() == pytest.approx(1.0)
    assert day["ret_20_xsrank"].min() > 0
    # leave-one-out asset-class mean: single-member classes are null
    counts = day.group_by("asset_class").len()
    singles = counts.filter(pl.col("len") == 1)["asset_class"].to_list()
    assert day.filter(pl.col("asset_class").is_in(singles))["rel_class_20"].null_count() == len(
        singles
    )
    # every asset on a decision date shares the market-wide features
    assert day["avg_corr_60"].n_unique() == 1
    # eligibility: no asset with fewer than min_history returns
    first = (
        small_data.prices.filter(pl.col("ret").is_not_null())
        .group_by("ticker")
        .agg(pl.col("date").min())
    )
    late = first.sort("date").tail(1).row(0, named=True)
    rows = m.filter(pl.col("ticker") == late["ticker"]).sort("date")
    n_before = small_data.prices.filter(
        (pl.col("ticker") == late["ticker"])
        & (pl.col("date") <= rows["date"][0])
        & pl.col("ret").is_not_null()
    ).height
    assert n_before >= small_cfg.features.min_history_sessions


def test_no_infinite_values(small_cfg: AppConfig, small_data: SyntheticData) -> None:
    fb = build_features(small_data.prices, small_data.macro, small_cfg, small_data.sessions)
    arr = fb.matrix.select(fb.feature_names).cast(pl.Float64).to_numpy()
    assert not np.isinf(arr).any()
