"""Normalise raw snapshots into curated point-in-time tables (PRD FR-DATA-03/05, §7.3-7.4).

Curated prices (one row per ticker x session, from the ticker's first to last bar):

    ret_t      = (close_t + dist_t) / close_{prev valid} - 1     dist = dividends + capital_gains
    tr_index_t = prod_{s<=t} (1 + ret_s),  tr_index = 1 on the first bar

``close`` is Yahoo's split-adjusted close, and distributions are on the same basis, so no
split ratio is applied (applying one would double-count the split). Sessions with no bar
are kept with null prices so that rolling windows measured in sessions stay aligned.

Curated macro (bitemporal, long):
    series_id, observed_on, value, realtime_start, realtime_end, revised, available_on

``available_on`` is the first session at whose close the value may be used.
"""

from __future__ import annotations

import datetime as dt

import polars as pl

from quantxgb.utils.calendar import shift_to_session

PRICE_COLUMNS = [
    "ticker",
    "asset_class",
    "date",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "dividends",
    "capital_gains",
    "splits",
    "ret",
    "tr_index",
    "adj_ret",
    "available_on",
    "source",
    "snapshot_date",
    "snapshot_sha256",
]


def normalize_prices(
    raw: pl.DataFrame,
    ticker: str,
    asset_class: str,
    sessions: list[dt.date],
    *,
    source: str,
    snapshot_date: str,
    snapshot_sha256: str,
    price_lag_sessions: int = 0,
) -> tuple[pl.DataFrame, list[dt.date]]:
    """Return (curated frame, list of raw dates that are not calendar sessions)."""
    session_set = set(sessions)
    raw = raw.sort("date")
    off_calendar = [d for d in raw["date"].to_list() if d not in session_set]
    raw = raw.filter(pl.col("date").is_in(sessions))
    if raw.height == 0:
        return pl.DataFrame(schema={c: pl.Null for c in PRICE_COLUMNS}), off_calendar

    dates = raw["date"].to_list()
    first, last = min(dates), max(dates)
    grid = pl.DataFrame({"date": [s for s in sessions if first <= s <= last]}).with_columns(
        pl.col("date").cast(pl.Date)
    )
    df = grid.join(raw, on="date", how="left").sort("date")

    has_bar = pl.col("close").is_not_null()
    dist = pl.col("dividends").fill_null(0.0) + pl.col("capital_gains").fill_null(0.0)
    prev_close = pl.col("close").forward_fill().shift(1)
    prev_adj = pl.col("adj_close").forward_fill().shift(1)
    df = df.with_columns(
        pl.when(has_bar).then((pl.col("close") + dist) / prev_close - 1.0).alias("ret"),
        pl.when(has_bar).then(pl.col("adj_close") / prev_adj - 1.0).alias("adj_ret"),
    )
    df = df.with_columns(
        pl.when(has_bar)
        .then((pl.col("ret").fill_null(0.0).log1p()).cum_sum().exp())
        .alias("tr_index")
    )
    if price_lag_sessions == 0:
        available = pl.col("date")
        df = df.with_columns(available.alias("available_on"))
    else:
        df = df.with_columns(
            shift_to_session(df["date"], sessions, price_lag_sessions, strictly_after=True).alias(
                "available_on"
            )
        )
    df = df.with_columns(
        pl.lit(ticker).alias("ticker"),
        pl.lit(asset_class).alias("asset_class"),
        pl.lit(source).alias("source"),
        pl.lit(snapshot_date).alias("snapshot_date"),
        pl.lit(snapshot_sha256).alias("snapshot_sha256"),
    )
    return df.select(PRICE_COLUMNS), off_calendar


MACRO_COLUMNS = [
    "series_id",
    "observed_on",
    "value",
    "realtime_start",
    "realtime_end",
    "revised",
    "available_on",
    "source",
    "snapshot_date",
    "snapshot_sha256",
]


def normalize_macro(
    raw: pl.DataFrame,
    series_id: str,
    revised: bool,
    sessions: list[dt.date],
    *,
    daily_lag_sessions: int,
    revised_lag_sessions: int,
    source: str,
    snapshot_date: str,
    snapshot_sha256: str,
) -> pl.DataFrame:
    df = raw.with_columns(
        pl.col("date").str.to_date().alias("observed_on"),
        pl.when(pl.col("value") == ".")
        .then(None)
        .otherwise(pl.col("value"))
        .cast(pl.Float64, strict=False)
        .alias("value"),
        pl.col("realtime_start").str.to_date(),
        pl.col("realtime_end").str.to_date(),
    ).filter(pl.col("value").is_not_null())

    if revised:
        lag, anchor = revised_lag_sessions, df["realtime_start"]
    else:
        lag, anchor = daily_lag_sessions, df["observed_on"]
        # Latest-only download: realtime columns describe "today", not history.
        df = df.with_columns(
            pl.lit(None, dtype=pl.Date).alias("realtime_start"),
            pl.lit(None, dtype=pl.Date).alias("realtime_end"),
        )
    available = shift_to_session(anchor, sessions, lag, strictly_after=lag > 0)
    df = df.with_columns(
        available.alias("available_on"),
        pl.lit(series_id).alias("series_id"),
        pl.lit(revised).alias("revised"),
        pl.lit(source).alias("source"),
        pl.lit(snapshot_date).alias("snapshot_date"),
        pl.lit(snapshot_sha256).alias("snapshot_sha256"),
    ).filter(pl.col("available_on").is_not_null())
    return df.select(MACRO_COLUMNS).sort(["series_id", "observed_on", "available_on"])
