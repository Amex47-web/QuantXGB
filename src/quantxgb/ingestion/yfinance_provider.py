"""Yahoo Finance daily bars via ``yfinance`` (PRD FR-DATA-01).

Facts about the source that the rest of the pipeline relies on (verified against
yfinance 1.8 output):

* ``Close`` is **split-adjusted but not dividend-adjusted**. Yahoo does not provide a
  truly unadjusted close, so "raw" here means "as served by Yahoo, split-adjusted".
* ``Dividends`` and ``Capital Gains`` are per-share cash distributions on the ex-date,
  on the same split-adjusted basis as ``Close``.
* ``Adj Close`` additionally back-adjusts for distributions; we keep it only for
  reconciliation and compute our own total-return index (see ``normalize.py``).

Terms: yfinance is unofficial and Yahoo data is for personal research use. Do not
redistribute downloaded data.
"""

from __future__ import annotations

import datetime as dt

import polars as pl
from tenacity import retry, stop_after_attempt, wait_exponential

from quantxgb.ingestion.base import RateLimiter

_COLUMNS = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
    "Dividends": "dividends",
    "Capital Gains": "capital_gains",
    "Stock Splits": "splits",
}


class EmptyDownloadError(RuntimeError):
    pass


class YFinanceProvider:
    name = "yfinance"

    def __init__(self, max_per_minute: float = 30.0) -> None:
        self._limiter = RateLimiter(max_per_minute)

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, max=30), reraise=True)
    def fetch_daily(self, ticker: str, start: dt.date, end: dt.date | None) -> pl.DataFrame:
        import yfinance as yf

        self._limiter.wait()
        # yfinance treats `end` as exclusive.
        end_excl = (end + dt.timedelta(days=1)) if end else None
        hist = yf.Ticker(ticker).history(
            start=start.isoformat(),
            end=end_excl.isoformat() if end_excl else None,
            auto_adjust=False,
            actions=True,
            repair=False,
        )
        if hist is None or hist.empty:
            raise EmptyDownloadError(f"no data returned for {ticker}")
        return yahoo_frame_to_polars(hist)


def yahoo_frame_to_polars(hist: object) -> pl.DataFrame:
    """Convert a yfinance history DataFrame (tz-aware index) to the raw schema."""
    import pandas as pd

    assert isinstance(hist, pd.DataFrame)
    idx = hist.index
    if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
        idx = idx.tz_convert("America/New_York")
    dates = [ts.date() for ts in idx]
    data: dict[str, object] = {"date": dates}
    for src, dst in _COLUMNS.items():
        if src in hist.columns:
            data[dst] = hist[src].astype("float64").to_numpy()
        elif dst in ("dividends", "capital_gains", "splits"):
            data[dst] = [0.0] * len(dates)
        else:
            raise KeyError(f"yfinance frame missing required column {src!r}")
    df = pl.DataFrame(data).with_columns(pl.col("date").cast(pl.Date))
    return df.sort("date").unique(subset="date", keep="last", maintain_order=True)
