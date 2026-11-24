"""FRED / ALFRED client (PRD FR-DATA-04).

* ``all_vintages=True`` requests ``realtime_start=1776-07-04`` / ``realtime_end=9999-12-31``,
  which returns every historical version of every observation (ALFRED). Each row is a
  (realtime_start, realtime_end, date, value) tuple: the value of observation ``date`` as
  it was published between ``realtime_start`` and ``realtime_end`` (inclusive).
* FRED encodes missing values as ``"."``; values are kept as strings in raw snapshots.
* The API key is never logged; HTTP errors are re-raised with the key redacted.

Terms: free API key; attribution required; some series carry third-party copyright —
series ``notes`` are stored in the manifest so the owner can review them.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import httpx
import polars as pl
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from quantxgb.ingestion.base import RateLimiter

ALFRED_REALTIME_START = "1776-07-04"
ALFRED_REALTIME_END = "9999-12-31"


class FredError(RuntimeError):
    pass


class _Retryable(Exception):
    pass


class FredProvider:
    name = "fred"

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.stlouisfed.org/fred",
        observation_start: dt.date = dt.date(1990, 1, 1),
        max_per_minute: float = 100,
        page_limit: int = 100_000,
        client: httpx.Client | None = None,
    ) -> None:
        if not api_key:
            raise FredError("FRED_API_KEY is not set (put it in .env; see .env.example)")
        self._key = api_key
        self._base = base_url.rstrip("/")
        self._obs_start = observation_start
        self._limiter = RateLimiter(max_per_minute)
        self._page_limit = page_limit
        self._client = client or httpx.Client(timeout=60.0)

    def _redact(self, text: str) -> str:
        return text.replace(self._key, "***")

    @retry(
        retry=retry_if_exception_type(_Retryable),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, max=60),
        reraise=True,
    )
    def _get(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        self._limiter.wait()
        q = {**params, "api_key": self._key, "file_type": "json"}
        try:
            resp = self._client.get(f"{self._base}/{endpoint}", params=q)
        except httpx.TransportError as exc:
            raise _Retryable(self._redact(str(exc))) from None
        if resp.status_code in (429, 500, 502, 503, 504):
            raise _Retryable(f"FRED HTTP {resp.status_code}")
        if resp.status_code != 200:
            raise FredError(f"FRED HTTP {resp.status_code}: {self._redact(resp.text[:300])}")
        payload: dict[str, Any] = resp.json()
        return payload

    def fetch_series(self, series_id: str, all_vintages: bool) -> pl.DataFrame:
        params: dict[str, Any] = {
            "series_id": series_id,
            "observation_start": self._obs_start.isoformat(),
            "sort_order": "asc",
            "limit": self._page_limit,
        }
        if all_vintages:
            params["realtime_start"] = ALFRED_REALTIME_START
            params["realtime_end"] = ALFRED_REALTIME_END
        rows: list[dict[str, str]] = []
        offset = 0
        while True:
            page = self._get("series/observations", {**params, "offset": offset})
            obs = page.get("observations", [])
            rows.extend(obs)
            count = int(page.get("count", len(rows)))
            offset += len(obs)
            if not obs or offset >= count:
                break
        if not rows:
            raise FredError(f"no observations for {series_id}")
        return pl.DataFrame(
            {
                "realtime_start": [r["realtime_start"] for r in rows],
                "realtime_end": [r["realtime_end"] for r in rows],
                "date": [r["date"] for r in rows],
                "value": [r["value"] for r in rows],
            }
        )

    def fetch_metadata(self, series_id: str) -> dict[str, Any]:
        payload = self._get("series", {"series_id": series_id})
        seriess = payload.get("seriess", [])
        if not seriess:
            return {}
        s = seriess[0]
        keep = ("id", "title", "frequency", "units", "seasonal_adjustment", "notes", "last_updated")
        return {k: s.get(k) for k in keep}
