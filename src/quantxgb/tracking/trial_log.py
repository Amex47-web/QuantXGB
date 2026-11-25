"""Global trial log (PRD §9.6, FR-VAL-03/06).

Every walk-forward run, Optuna trial and backtested strategy configuration is appended to
``artifacts/tracking/trials.jsonl``. The Deflated Sharpe Ratio uses the per-period Sharpe
ratios of all *backtested strategy configurations* recorded here, so re-running variants
silently is impossible: each run raises the bar for the next.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any, Literal

TrialKind = Literal["walk_forward", "optuna", "strategy"]


class TrialLog:
    def __init__(self, artifacts_dir: Path) -> None:
        self.path = artifacts_dir / "tracking" / "trials.jsonl"

    def log(self, kind: TrialKind, run_id: str, **fields: Any) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rec = {
            "kind": kind,
            "run_id": run_id,
            "ts": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")

    def records(self, kind: TrialKind | None = None) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        recs = [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return [r for r in recs if kind is None or r["kind"] == kind]

    def count(self, kind: TrialKind | None = None) -> int:
        return len(self.records(kind))

    def strategy_srs(
        self, exclude_baselines: bool = True, key: str = "sr_per_period"
    ) -> list[float]:
        """Per-period SR (``sr_per_period``, or ``active_sr_per_period`` vs. equal weight)
        of every model strategy configuration ever backtested."""
        out = []
        for r in self.records("strategy"):
            if exclude_baselines and r.get("is_baseline"):
                continue
            sr = r.get(key)
            if sr is not None:
                out.append(float(sr))
        return out
