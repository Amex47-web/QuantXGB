"""Provider protocols and the immutable raw-snapshot store (PRD FR-DATA-02, §7.5).

Raw files are written once under ``data/raw/<provider>/<snapshot_date>/<name>.parquet``
and recorded in ``data/raw/manifest.jsonl`` with their SHA-256. They are never edited;
re-downloading creates a new dated snapshot so that provider-side history revisions
become visible as diffs between snapshots.
"""

from __future__ import annotations

import datetime as dt
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import polars as pl

from quantxgb.utils.io import sha256_file, write_parquet_atomic


class PriceProvider(Protocol):
    name: str

    def fetch_daily(self, ticker: str, start: dt.date, end: dt.date | None) -> pl.DataFrame:
        """Return raw daily bars with columns
        ``date, open, high, low, close, adj_close, volume, dividends, capital_gains, splits``."""
        ...


class MacroProvider(Protocol):
    name: str

    def fetch_series(self, series_id: str, all_vintages: bool) -> pl.DataFrame:
        """Return ``realtime_start, realtime_end, date, value`` (value as string)."""
        ...

    def fetch_metadata(self, series_id: str) -> dict[str, Any]: ...


@dataclass(frozen=True)
class SnapshotRecord:
    provider: str
    name: str
    path: str
    sha256: str
    rows: int
    snapshot_date: str
    ingested_at: str
    params: dict[str, Any]


class RawStore:
    def __init__(self, raw_dir: Path) -> None:
        self.raw_dir = raw_dir
        self.manifest = raw_dir / "manifest.jsonl"

    def write(
        self,
        provider: str,
        name: str,
        df: pl.DataFrame,
        params: dict[str, Any],
        snapshot_date: dt.date | None = None,
    ) -> SnapshotRecord:
        snap = (snapshot_date or dt.date.today()).isoformat()
        path = self.raw_dir / provider / snap / f"{name}.parquet"
        if path.exists():
            # Immutable: never overwrite an existing snapshot file.
            existing = self.latest(provider, name)
            if existing is not None and existing.snapshot_date == snap:
                return existing
            raise FileExistsError(f"raw snapshot exists and is not in manifest: {path}")
        write_parquet_atomic(df, path)
        rec = SnapshotRecord(
            provider=provider,
            name=name,
            path=str(path.relative_to(self.raw_dir)),
            sha256=sha256_file(path),
            rows=df.height,
            snapshot_date=snap,
            ingested_at=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            params=params,
        )
        self.manifest.parent.mkdir(parents=True, exist_ok=True)
        with self.manifest.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(rec), default=str) + "\n")
        return rec

    def records(self) -> list[SnapshotRecord]:
        if not self.manifest.exists():
            return []
        out = []
        for line in self.manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(SnapshotRecord(**json.loads(line)))
        return out

    def latest(self, provider: str, name: str) -> SnapshotRecord | None:
        recs = [r for r in self.records() if r.provider == provider and r.name == name]
        return max(recs, key=lambda r: (r.snapshot_date, r.ingested_at)) if recs else None

    def read(self, rec: SnapshotRecord, verify: bool = True) -> pl.DataFrame:
        path = self.raw_dir / rec.path
        if verify and sha256_file(path) != rec.sha256:
            raise ValueError(f"raw snapshot hash mismatch (file modified?): {path}")
        return pl.read_parquet(path)


class RateLimiter:
    """Minimum-interval limiter (simple and sufficient for sequential downloads)."""

    def __init__(self, max_per_minute: float) -> None:
        self.interval = 60.0 / max_per_minute
        self._last = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        delay = self._last + self.interval - now
        if delay > 0:
            time.sleep(delay)
        self._last = time.monotonic()
