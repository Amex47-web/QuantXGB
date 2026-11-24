"""FR-PIT-03 / FR-PIT-04: every feature at date t must depend only on data available at t.

* Truncation: features computed on data truncated at t equal those computed on the full
  data for all dates <= t.
* Future perturbation: randomly perturbing every price / macro value after t leaves all
  features at dates <= t unchanged.

Both tests cover every model feature (asset, cross-sectional, market and macro).
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from quantxgb.config import AppConfig
from quantxgb.features.build import build_features
from quantxgb.synthetic import SyntheticData


def _compare(a: pl.DataFrame, b: pl.DataFrame, cols: list[str], cutoff: dt.date) -> list[str]:
    a = a.filter(pl.col("date") <= cutoff).sort(["date", "ticker"])
    b = b.filter(pl.col("date") <= cutoff).sort(["date", "ticker"])
    assert a.select("date", "ticker").equals(b.select("date", "ticker")), "row sets differ"
    bad = []
    for c in cols:
        x = a[c].cast(pl.Float64).to_numpy()
        y = b[c].cast(pl.Float64).to_numpy()
        same_nan = np.isnan(x) == np.isnan(y)
        close = np.isclose(np.nan_to_num(x), np.nan_to_num(y), rtol=1e-9, atol=1e-12)
        if not (same_nan.all() and close.all()):
            bad.append(c)
    return bad


def _cutoffs(data: SyntheticData) -> list[dt.date]:
    d = sorted(data.prices["date"].unique().to_list())
    return [d[int(len(d) * q)] for q in (0.45, 0.7, 0.9)]


@pytest.fixture(scope="module")
def full_build(small_cfg: AppConfig, small_data: SyntheticData):  # type: ignore[no-untyped-def]
    return build_features(small_data.prices, small_data.macro, small_cfg, small_data.sessions)


def test_feature_set_is_complete(full_build) -> None:  # type: ignore[no-untyped-def]
    # every configured model feature must have been built on synthetic data
    assert len(full_build.feature_names) >= 50
    assert full_build.matrix.height > 1000


@pytest.mark.parametrize("q", [0, 1, 2])
def test_truncation_invariance(
    small_cfg: AppConfig,
    small_data: SyntheticData,
    full_build,
    q: int,  # type: ignore[no-untyped-def]
) -> None:
    cutoff = _cutoffs(small_data)[q]
    prices = small_data.prices.filter(pl.col("date") <= cutoff)
    macro = small_data.macro.filter(pl.col("available_on") <= cutoff)
    trunc = build_features(prices, macro, small_cfg, small_data.sessions)
    bad = _compare(full_build.matrix, trunc.matrix, full_build.feature_names, cutoff)
    assert not bad, f"features change when future data is removed: {bad}"


@pytest.mark.parametrize("q", [0, 2])
def test_future_perturbation_invariance(
    small_cfg: AppConfig,
    small_data: SyntheticData,
    full_build,
    q: int,  # type: ignore[no-untyped-def]
) -> None:
    cutoff = _cutoffs(small_data)[q]
    rng = np.random.default_rng(q)
    after = pl.col("date") > cutoff
    noise = pl.Series(rng.lognormal(0, 0.2, small_data.prices.height))
    p = small_data.prices.with_columns(noise.alias("_n"))
    perturbed = p.with_columns(
        [
            pl.when(after).then(pl.col(c) * pl.col("_n")).otherwise(pl.col(c)).alias(c)
            for c in ("open", "high", "low", "close", "tr_index", "volume")
        ]
        + [pl.when(after).then(pl.col("ret") * 3 + 0.01).otherwise(pl.col("ret")).alias("ret")]
    ).drop("_n")
    m = small_data.macro
    mnoise = pl.Series(rng.normal(0, 5, m.height))
    macro = m.with_columns(
        pl.when(pl.col("available_on") > cutoff)
        .then(pl.col("value") + mnoise)
        .otherwise(pl.col("value"))
        .alias("value")
    )
    pert = build_features(perturbed, macro, small_cfg, small_data.sessions)
    bad = _compare(full_build.matrix, pert.matrix, full_build.feature_names, cutoff)
    assert not bad, f"features at <= t react to perturbations after t: {bad}"
