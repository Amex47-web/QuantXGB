from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import polars as pl
import pytest

from quantxgb.ingestion.base import RawStore
from quantxgb.ingestion.fred_provider import FredError, FredProvider
from quantxgb.ingestion.normalize import normalize_macro, normalize_prices
from quantxgb.ingestion.yfinance_provider import yahoo_frame_to_polars
from quantxgb.utils.calendar import business_day_sessions


def _raw(n: int = 6, dividends: dict[int, float] | None = None) -> pl.DataFrame:
    sess = business_day_sessions(dt.date(2021, 1, 4), dt.date(2021, 1, 4) + dt.timedelta(days=20))[
        :n
    ]
    close = np.array([100, 101, 99, 100, 102, 103][:n], dtype=float)
    div = np.zeros(n)
    for i, v in (dividends or {}).items():
        div[i] = v
    return pl.DataFrame(
        {
            "date": sess,
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "adj_close": close,
            "volume": np.full(n, 1e6),
            "dividends": div,
            "capital_gains": np.zeros(n),
            "splits": np.zeros(n),
        }
    )


def _norm(raw: pl.DataFrame, sessions: list[dt.date]) -> pl.DataFrame:
    df, _ = normalize_prices(
        raw, "X", "eq", sessions, source="t", snapshot_date="s", snapshot_sha256="h"
    )
    return df


def test_total_return_includes_distributions() -> None:
    raw = _raw(dividends={3: 2.0})
    sess = raw["date"].to_list()
    df = _norm(raw, sess)
    r = df["ret"].to_list()
    assert r[0] is None
    assert r[3] == pytest.approx((100 + 2.0) / 99 - 1)
    tr = df["tr_index"].to_numpy()
    assert tr[0] == 1.0
    assert tr[-1] == pytest.approx(np.prod([1 + x for x in r[1:]]))


def test_capital_gains_count_as_distributions() -> None:
    raw = _raw().with_columns(pl.Series("capital_gains", [0, 0, 1.0, 0, 0, 0], dtype=pl.Float64))
    df = _norm(raw, raw["date"].to_list())
    assert df["ret"][2] == pytest.approx((99 + 1.0) / 101 - 1)


def test_missing_session_kept_as_null_and_return_spans_gap() -> None:
    raw = _raw()
    sess = raw["date"].to_list()
    df = _norm(raw.filter(pl.col("date") != sess[2]), sess)
    assert df.height == 6
    assert df["close"][2] is None and df["ret"][2] is None and df["tr_index"][2] is None
    # the next bar's return is measured against the last valid close (two sessions)
    assert df["ret"][3] == pytest.approx(100 / 101 - 1)
    assert df["tr_index"][3] == pytest.approx(df["tr_index"][1] * 100 / 101)


def test_off_calendar_bars_are_reported() -> None:
    raw = _raw()
    sess = [d for d in raw["date"].to_list() if d != raw["date"][1]]
    df, off = normalize_prices(
        raw, "X", "eq", sess, source="t", snapshot_date="s", snapshot_sha256="h"
    )
    assert off == [raw["date"][1]]
    assert df.height == 5


def test_yahoo_frame_conversion_uses_new_york_dates() -> None:
    idx = pd.DatetimeIndex(["2024-03-19 00:00", "2024-03-20 00:00"]).tz_localize("America/New_York")
    hist = pd.DataFrame(
        {
            "Open": [1.0, 2],
            "High": [1.0, 2],
            "Low": [1.0, 2],
            "Close": [1.0, 2],
            "Adj Close": [1.0, 2],
            "Volume": [10, 20],
            "Dividends": [0.0, 0.1],
            "Stock Splits": [0.0, 0.0],
        },
        index=idx,
    )
    df = yahoo_frame_to_polars(hist)
    assert df["date"].to_list() == [dt.date(2024, 3, 19), dt.date(2024, 3, 20)]
    assert df["capital_gains"].to_list() == [0.0, 0.0]  # absent column → zeros


def test_raw_store_is_immutable_and_hash_verified(tmp_path: Path) -> None:
    store = RawStore(tmp_path)
    df = pl.DataFrame({"a": [1, 2]})
    rec = store.write("p", "X", df, {}, snapshot_date=dt.date(2024, 1, 1))
    again = store.write("p", "X", df, {}, snapshot_date=dt.date(2024, 1, 1))
    assert again == rec
    assert store.read(rec).equals(df)
    (tmp_path / rec.path).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        store.read(rec)


def test_fred_pagination_missing_values_and_key_redaction() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.path.endswith("/series/observations"):
            offset = int(request.url.params["offset"])
            obs = [
                {
                    "realtime_start": "2020-02-13",
                    "realtime_end": "9999-12-31",
                    "date": f"2020-0{offset + 1}-01",
                    "value": "." if offset == 1 else "1.5",
                }
            ]
            return httpx.Response(200, json={"count": 2, "offset": offset, "observations": obs})
        return httpx.Response(400, text="bad request for api_key=SECRETKEY")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    fred = FredProvider(
        "SECRETKEY", base_url="https://x/fred", page_limit=1, max_per_minute=10_000, client=client
    )
    df = fred.fetch_series("CPI", all_vintages=True)
    assert df.height == 2 and len(calls) == 2
    assert calls[0].url.params["realtime_start"] == "1776-07-04"
    with pytest.raises(FredError) as exc:
        fred.fetch_metadata("CPI")
    assert "SECRETKEY" not in str(exc.value)


def test_fred_requires_key() -> None:
    with pytest.raises(FredError):
        FredProvider("")


def test_macro_availability_rules() -> None:
    sess = business_day_sessions(dt.date(2020, 1, 1), dt.date(2020, 3, 31))
    raw = pl.DataFrame(
        {
            "realtime_start": ["2020-02-13", "2020-03-12"],
            "realtime_end": ["2020-03-11", "9999-12-31"],
            "date": ["2020-01-01", "2020-01-01"],
            "value": ["100.0", "100.5"],
        }
    )
    rev = normalize_macro(
        raw,
        "CPI",
        True,
        sess,
        daily_lag_sessions=1,
        revised_lag_sessions=1,
        source="t",
        snapshot_date="s",
        snapshot_sha256="h",
    )
    # vintage published 2020-02-13 (Thu) → usable at close of the next session (Fri 14th)
    assert rev["available_on"].to_list() == [dt.date(2020, 2, 14), dt.date(2020, 3, 13)]
    daily = pl.DataFrame(
        {
            "realtime_start": ["x"] * 2,
            "realtime_end": ["x"] * 2,
            "date": ["2020-01-03", "2020-01-06"],
            "value": ["1.0", "."],
        }
    ).with_columns(
        pl.lit("2020-11-01").alias("realtime_start"), pl.lit("2020-11-01").alias("realtime_end")
    )
    d = normalize_macro(
        daily,
        "DGS10",
        False,
        sess,
        daily_lag_sessions=1,
        revised_lag_sessions=1,
        source="t",
        snapshot_date="s",
        snapshot_sha256="h",
    )
    assert d.height == 1  # "." dropped
    assert d["available_on"][0] == dt.date(2020, 1, 6)  # Fri obs → Mon
    assert d["realtime_start"][0] is None


def test_manifest_records_are_json(tmp_path: Path) -> None:
    store = RawStore(tmp_path)
    store.write("p", "X", pl.DataFrame({"a": [1]}), {"start": dt.date(2020, 1, 1)})
    line = (tmp_path / "manifest.jsonl").read_text().strip()
    assert json.loads(line)["params"]["start"] == "2020-01-01"


def test_calendar_covers_full_history() -> None:
    from quantxgb.utils.calendar import trading_sessions

    s = trading_sessions(dt.date(1995, 1, 1), dt.date.today() + dt.timedelta(days=60))
    assert s[0] == dt.date(1995, 1, 3)
    # known closures: 9/11, Hurricane Sandy, national day of mourning 2025-01-09
    for closed in (dt.date(2001, 9, 11), dt.date(2012, 10, 29), dt.date(2025, 1, 9)):
        assert closed not in s
    assert dt.date(2001, 9, 17) in s
