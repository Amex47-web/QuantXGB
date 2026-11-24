from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from quantxgb.config import MacroFeatureSpec
from quantxgb.validation.point_in_time import (
    LeakageError,
    asof_join_macro,
    assert_available,
    macro_feature_table,
)


def _sessions() -> list[dt.date]:
    d0 = dt.date(2020, 1, 1)
    return [d0 + dt.timedelta(days=i) for i in range(400)]


def _cpi() -> pl.DataFrame:
    """Two observations, each first released then revised."""
    rows = [
        # obs Jan: released Feb 10 (100), revised Mar 10 (101)
        ("2020-01-01", "2020-02-10", "2020-03-09", 100.0),
        ("2020-01-01", "2020-03-10", "9999-12-31", 101.0),
        # obs Feb: released Mar 10 (102)
        ("2020-02-01", "2020-03-10", "9999-12-31", 102.0),
    ]
    df = pl.DataFrame(
        rows, schema=["observed_on", "realtime_start", "realtime_end", "value"], orient="row"
    )
    return df.with_columns(
        pl.col("observed_on").str.to_date(),
        pl.col("realtime_start").str.to_date(),
        pl.col("realtime_end").str.to_date(),
        pl.lit("CPI").alias("series_id"),
        pl.lit(True).alias("revised"),
    ).with_columns(pl.col("realtime_start").alias("available_on"))


def test_vintage_consistent_change_uses_only_the_published_vintage() -> None:
    spec = MacroFeatureSpec(name="cpi_chg", series="CPI", transform="diff", periods=1)
    lvl = MacroFeatureSpec(name="cpi", series="CPI", transform="level")
    t = macro_feature_table(_cpi(), [spec, lvl], _sessions(), revised_lag_sessions=0)
    wide = asof_join_macro([dt.date(2020, 2, 9), dt.date(2020, 2, 20), dt.date(2020, 3, 15)], t)
    rows = wide.sort("date").to_dicts()
    assert rows[0]["cpi"] is None  # nothing published yet
    assert rows[1]["cpi"] == 100.0  # first release, not the later revision
    assert rows[1]["cpi_chg"] is None  # only one observation in that vintage
    assert rows[2]["cpi"] == 102.0
    assert rows[2]["cpi_chg"] == pytest.approx(102.0 - 101.0)  # revised Jan from the same vintage


def test_release_lag_shifts_availability() -> None:
    lvl = MacroFeatureSpec(name="cpi", series="CPI", transform="level")
    t = macro_feature_table(_cpi(), [lvl], _sessions(), revised_lag_sessions=1)
    wide = asof_join_macro([dt.date(2020, 2, 10), dt.date(2020, 2, 11)], t).sort("date")
    assert wide["cpi"].to_list() == [None, 100.0]


def test_assert_available_raises() -> None:
    df = pl.DataFrame({"d": [dt.date(2020, 1, 1)], "a": [dt.date(2020, 1, 2)]})
    with pytest.raises(LeakageError):
        assert_available(df, "d", "a")
    assert_available(df.with_columns(pl.col("d").alias("a")), "d", "a")
