"""Run registry: ids, directories, manifests."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

from quantxgb.config import AppConfig
from quantxgb.utils.io import git_sha, read_json, write_json

RESEARCH_SECTIONS = ("data", "features", "labels", "model", "validation", "backtest")


def runs_root(cfg: AppConfig) -> Path:
    return cfg.artifacts_dir / "runs"


def new_run_id(cfg: AppConfig) -> str:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{cfg.section_hash(*RESEARCH_SECTIONS)[:8]}"


def resolve_run(cfg: AppConfig, run: str) -> Path:
    root = runs_root(cfg)
    if run == "latest":
        dirs = sorted(p for p in root.glob("*") if (p / "manifest.json").exists())
        if not dirs:
            raise FileNotFoundError("no runs yet — run `quantxgb train` first")
        return dirs[-1]
    path = root / run
    if not (path / "manifest.json").exists():
        raise FileNotFoundError(f"run {run} not found under {root}")
    return path


def write_manifest(cfg: AppConfig, run_dir: Path, run_id: str, **extra: Any) -> dict[str, Any]:
    manifest = {
        "run_id": run_id,
        "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "git_sha": git_sha(cfg.root),
        "config_hash": cfg.section_hash(*RESEARCH_SECTIONS),
        "config_status": {s: getattr(cfg, s).status for s in RESEARCH_SECTIONS},
        "config": {s: getattr(cfg, s).model_dump(mode="json") for s in RESEARCH_SECTIONS},
        **extra,
    }
    write_json(manifest, run_dir / "manifest.json")
    return manifest


def read_manifest(run_dir: Path) -> dict[str, Any]:
    m: dict[str, Any] = read_json(run_dir / "manifest.json")
    return m


def list_runs(cfg: AppConfig) -> list[dict[str, Any]]:
    out = []
    for p in sorted(runs_root(cfg).glob("*")):
        if (p / "manifest.json").exists():
            m = read_manifest(p)
            metrics = read_json(p / "metrics.json") if (p / "metrics.json").exists() else {}
            out.append(
                {
                    "run_id": m["run_id"],
                    "created": m["created"],
                    "hypothesis": m.get("hypothesis", ""),
                    "shuffle_labels": m.get("shuffle_labels", False),
                    "has_backtest": (p / "backtest" / "metrics.json").exists(),
                    "ic": {k: v.get("ic_mean") for k, v in metrics.get("models", {}).items()},
                }
            )
    return out
