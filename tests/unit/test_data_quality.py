from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

from quantxgb.validation.data_quality import check_macro, check_prices, run_data_quality


def _prices(n: int = 300) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    d0 = dt.date(2020, 1, 1)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
    df = pl.DataFrame(
        {
            "ticker": ["X"] * n,
            "date": [d0 + dt.timedelta(days=i) for i in range(n)],
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(n, 1e6),
            "splits": np.zeros(n),
        }
    )
    return df.with_columns(
        (pl.col("close") / pl.col("close").shift(1) - 1).alias("ret"),
        (pl.col("close") / pl.col("close").shift(1) - 1).alias("adj_ret"),
        pl.col("date").alias("available_on"),
    )


def _checks(df: pl.DataFrame) -> dict[str, str]:
    return {i.check: i.severity for i in check_prices(df)}


def test_clean_data_passes() -> None:
    assert _checks(_prices()) == {}


def test_each_check_fires() -> None:
    p = _prices()
    idx = pl.int_range(pl.len())
    bad = p.with_columns(
        pl.when(idx == 10).then(-1.0).otherwise(pl.col("close")).alias("close"),
        pl.when(idx == 20).then(pl.col("high") * 0.5).otherwise(pl.col("high")).alias("high"),
        pl.when(idx == 150).then(0.5).otherwise(pl.col("ret")).alias("ret"),
        pl.when(idx == 160)
        .then(pl.col("adj_ret") + 0.02)
        .otherwise(pl.col("adj_ret"))
        .alias("adj_ret"),
        pl.when(idx == 170).then(2.0).otherwise(pl.col("splits")).alias("splits"),
        pl.when(idx == 180)
        .then(pl.col("date") - dt.timedelta(days=1))
        .otherwise(pl.col("available_on"))
        .alias("available_on"),
    )
    stale = bad.with_columns(
        pl.when(idx.is_between(200, 204)).then(55.0).otherwise(pl.col("close")).alias("close")
    )
    missing = stale.with_columns(
        pl.when(idx.is_between(250, 260)).then(None).otherwise(pl.col("close")).alias("close")
    )
    got = _checks(pl.concat([missing, missing.head(1)]))
    assert got["duplicates"] == "fail"
    assert got["non_positive_price"] == "fail"
    assert got["ohlc_consistency"] == "fail"
    assert got["outlier_return"] == "flag"
    assert got["adjustment_reconciliation"] == "flag"
    assert got["split_events"] in ("flag", "warn")
    assert got["availability_sanity"] == "fail"
    assert got["stale_price"] == "flag"
    assert got["calendar_completeness"] == "fail"


def test_macro_checks_and_report() -> None:
    d = dt.date(2020, 1, 1)
    macro = pl.DataFrame(
        {
            "series_id": ["CPI", "CPI", "DGS10"],
            "observed_on": [d, d, d],
            "value": [1.0, 1.0, 2.0],
            "realtime_start": [d - dt.timedelta(days=5), d - dt.timedelta(days=5), None],
            "realtime_end": [d, d, None],
            "revised": [True, True, False],
            "available_on": [d, d, d - dt.timedelta(days=1)],
        }
    )
    checks = {(i.check, i.severity) for i in check_macro(macro)}
    assert ("duplicates", "fail") in checks
    assert ("availability_sanity", "fail") in checks
    assert ("vintage_integrity", "fail") in checks
    report = run_data_quality(_prices(), macro)
    assert not report.passed
    md = report.to_markdown()
    assert "**Result: FAIL**" in md and "first bar per ticker" in md
    assert report.to_frame().height == len(report.issues)
