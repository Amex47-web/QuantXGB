"""Read-only access to run artefacts for the API and dashboard (never trains or downloads)."""

from __future__ import annotations

import datetime as dt
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import polars as pl

from quantxgb.utils.io import read_json


def artifacts_root() -> Path:
    env = os.environ.get("QUANTXGB_ARTIFACTS")
    if env:
        return Path(env).resolve()
    from quantxgb.config import load_config

    return load_config().artifacts_dir


class ArtifactStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or artifacts_root()

    # --------------------------------------------------------------- runs
    def runs(self) -> list[str]:
        d = self.root / "runs"
        return (
            sorted(p.name for p in d.glob("*") if (p / "manifest.json").exists())
            if d.exists()
            else []
        )

    def run_dir(self, run_id: str | None = None) -> Path:
        runs = self.runs()
        if not runs:
            raise FileNotFoundError(f"no runs under {self.root}")
        rid = run_id or runs[-1]
        if rid not in runs:
            raise FileNotFoundError(f"unknown run {rid}")
        return self.root / "runs" / rid

    def json(self, run_id: str | None, name: str) -> Any:
        p = self.run_dir(run_id) / name
        return read_json(p) if p.exists() else None

    def frame(self, run_id: str | None, name: str) -> pl.DataFrame | None:
        p = self.run_dir(run_id) / name
        return _read(str(p), p.stat().st_mtime) if p.exists() else None

    # ------------------------------------------------------------ helpers
    def latest_live(self, run_id: str | None) -> tuple[dt.date, Path] | None:
        live = self.run_dir(run_id) / "live"
        if not live.exists():
            return None
        dates = sorted(p.name for p in live.iterdir() if p.is_dir())
        return (dt.date.fromisoformat(dates[-1]), live / dates[-1]) if dates else None

    def prediction_dates(self, run_id: str | None) -> list[dt.date]:
        p = self.frame(run_id, "predictions.parquet")
        return sorted(p["date"].unique().to_list()) if p is not None else []

    def rankings(self, run_id: str | None, date: dt.date | None) -> tuple[dt.date, pl.DataFrame]:
        """Scores at ``date`` (OOS predictions) or the latest live scoring if date is None."""
        if date is None:
            live = self.latest_live(run_id)
            if live is not None:
                return live[0], pl.read_parquet(live[1] / "scores.parquet")
            dates = self.prediction_dates(run_id)
            if not dates:
                raise FileNotFoundError("no predictions")
            date = dates[-1]
        p = self.frame(run_id, "predictions.parquet")
        assert p is not None
        out = p.filter(pl.col("date") == date)
        if out.height == 0:
            live_scores = self.run_dir(run_id) / "live" / date.isoformat() / "scores.parquet"
            if live_scores.exists():
                out = pl.read_parquet(live_scores)
        return date, out

    def quantiles(self, run_id: str | None, date: dt.date) -> pl.DataFrame:
        live = self.run_dir(run_id) / "live" / date.isoformat() / "quantiles.parquet"
        if live.exists():
            return pl.read_parquet(live)
        q = self.frame(run_id, "quantiles.parquet")
        return q.filter(pl.col("date") == date) if q is not None else pl.DataFrame()

    def regimes(self, run_id: str | None) -> pl.DataFrame:
        r = self.frame(run_id, "regimes.parquet")
        frames = [r] if r is not None else []
        live = self.latest_live(run_id)
        if live is not None and (live[1] / "regimes.parquet").exists():
            frames.append(pl.read_parquet(live[1] / "regimes.parquet"))
        if not frames:
            return pl.DataFrame()
        return (
            pl.concat(frames, how="diagonal_relaxed")
            .unique(subset="date", keep="last")
            .sort("date")
        )

    def shap(self, run_id: str | None, date: dt.date, ticker: str) -> pl.DataFrame:
        live = self.run_dir(run_id) / "live" / date.isoformat() / "shap.parquet"
        src = pl.read_parquet(live) if live.exists() else self.frame(run_id, "shap_oos.parquet")
        if src is None:
            return pl.DataFrame()
        row = src.filter((pl.col("date") == date) & (pl.col("ticker") == ticker))
        if row.height == 0:
            return pl.DataFrame()
        feats = [c for c in row.columns if c not in ("date", "ticker", "fold", "_bias")]
        vals = row.select(feats).row(0)
        return pl.DataFrame(
            {"feature": feats, "contribution": [float(v) if v is not None else 0.0 for v in vals]}
        ).sort(pl.col("contribution").abs(), descending=True)

    def data_quality_markdown(self) -> str | None:
        p = self.root / "reports" / "data_quality.md"
        return p.read_text(encoding="utf-8") if p.exists() else None


@lru_cache(maxsize=64)
def _read(path: str, mtime: float) -> pl.DataFrame:
    return pl.read_parquet(path)
