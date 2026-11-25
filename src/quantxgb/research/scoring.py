"""Scoring with the final (development-period) models: live scoring and the one-shot
holdout evaluation (PRD FR-VAL-08)."""

from __future__ import annotations

import datetime as dt
import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from quantxgb.config import AppConfig
from quantxgb.evaluation.signal_metrics import ic_summary, quantile_metrics, rank_ic_by_date
from quantxgb.research.dataset import Dataset
from quantxgb.research.runs import RESEARCH_SECTIONS, read_manifest
from quantxgb.research.walk_forward import (
    FoldModels,
    date_level,
    quantile_col,
    sessions_per_signal,
)
from quantxgb.utils.io import read_json, write_json, write_parquet_atomic


def load_final(run_dir: Path) -> FoldModels:
    with (run_dir / "models" / "final.pkl").open("rb") as fh:
        fm = pickle.load(fh)
    assert isinstance(fm, FoldModels)
    return fm


def score_rows(
    fm: FoldModels, ds: Dataset, rows: pl.DataFrame, cfg: AppConfig
) -> dict[str, pl.DataFrame]:
    """Scores, quantiles, regimes and SHAP contributions for arbitrary decision-date rows."""
    macro_cols = [m.name for m in cfg.features.macro if m.name in ds.frame.columns]
    dates_df = date_level(ds.frame.filter(pl.col("date") <= rows["date"].max()), macro_cols)
    daily = ds.market_daily.filter(pl.col("date") <= rows["date"].max())
    reg = fm.regime_frame(dates_df, daily)
    hmm_cols = fm.hmm_feature_names()
    rows = rows.sort(["date", "ticker"])
    if hmm_cols:
        rows = rows.join(reg.select("date", *hmm_cols), on="date", how="left")
    scores, quant = fm.score(rows)
    contrib = fm.ranker.contributions(rows)
    shap = rows.select("date", "ticker").with_columns(
        [pl.Series(f, contrib[:, j]) for j, f in enumerate(fm.features)]
        + [pl.Series("_bias", contrib[:, -1])]
    )
    reg = reg.filter(pl.col("date").is_in(rows["date"].unique().implode()))
    return {"scores": scores, "quantiles": quant, "regimes": reg, "shap": shap}


def score_live(run_dir: Path, ds: Dataset, cfg: AppConfig, date: dt.date | None = None) -> Path:
    fm = load_final(run_dir)
    d = date or max(ds.frame["date"].to_list())
    rows = ds.frame.filter(pl.col("date") == d)
    if rows.height == 0:
        raise ValueError(f"no eligible rows on {d}")
    out = score_rows(fm, ds, rows, cfg)
    live = run_dir / "live" / d.isoformat()
    for name, df in out.items():
        write_parquet_atomic(df, live / f"{name}.parquet")
    write_json(
        {
            "date": d.isoformat(),
            "scored_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "in_holdout_period": d >= cfg.backtest.splits.holdout.start,
        },
        live / "meta.json",
    )
    return live


class HoldoutLocked(RuntimeError):
    pass


def holdout_lock_path(cfg: AppConfig) -> Path:
    return cfg.artifacts_dir / "holdout" / "LOCK.json"


def check_holdout_preconditions(cfg: AppConfig, run_dir: Path, override_reason: str | None) -> None:
    if not cfg.all_frozen:
        raise HoldoutLocked(
            "all configs must have `status: frozen` before the holdout can be evaluated "
            "(methodology freeze, PRD §15)"
        )
    manifest = read_manifest(run_dir)
    if manifest["config_hash"] != cfg.section_hash(*RESEARCH_SECTIONS):
        raise HoldoutLocked("current configs differ from the run's configs — retrain first")
    if manifest.get("shuffle_labels"):
        raise HoldoutLocked("cannot evaluate the holdout with a shuffled-label sanity run")
    lock = holdout_lock_path(cfg)
    if lock.exists() and not override_reason:
        prev = read_json(lock)
        raise HoldoutLocked(
            f"holdout already evaluated by run {prev['run_id']} on {prev['evaluated_at']}. "
            "Re-running requires --override-reason, which is logged permanently."
        )


def evaluate_holdout(
    run_dir: Path,
    ds: Dataset,
    prices: pl.DataFrame,
    macro: pl.DataFrame | None,
    cfg: AppConfig,
    override_reason: str | None = None,
) -> dict[str, Any]:
    from quantxgb.backtest.runner import run_backtests

    check_holdout_preconditions(cfg, run_dir, override_reason)
    h = cfg.backtest.splits.holdout
    rows = ds.frame.filter(pl.col("date").is_between(h.start, h.end, closed="both"))
    if rows.height == 0:
        raise ValueError("no data in the holdout period")
    fm = load_final(run_dir)
    out = score_rows(fm, ds, rows, cfg)
    labels = rows.select(
        "date",
        "ticker",
        "asset_class",
        "fwd_ret",
        "fwd_xs_ret",
        "fwd_grade",
        "label_start",
        "label_end",
    )
    preds = out["scores"].join(labels, on=["date", "ticker"], how="left")
    quant = out["quantiles"].join(labels, on=["date", "ticker"], how="left")

    sps = sessions_per_signal(ds.dates, ds.sessions)
    nw = cfg.validation.ic_newey_west_lags or max(1, math.ceil(cfg.labels.horizon / sps))
    metrics: dict[str, Any] = {
        "models": {},
        "override_reason": override_reason,
        "newey_west_lags": nw,
    }
    for (model,), g in preds.group_by(["model"]):
        metrics["models"][str(model)] = ic_summary(rank_ic_by_date(g), nw)
    lab = quant.filter(pl.col("fwd_ret").is_not_null())
    if lab.height:
        cols = [quantile_col(a) for a in cfg.model.quantile.alphas]
        metrics["quantile"] = quantile_metrics(
            lab["fwd_ret"].to_numpy(), lab.select(cols).to_numpy(), cfg.model.quantile.alphas
        )
    hdir = cfg.artifacts_dir / "holdout" / run_dir.name
    write_parquet_atomic(preds, hdir / "predictions.parquet")
    write_parquet_atomic(quant, hdir / "quantiles.parquet")
    write_parquet_atomic(out["regimes"], hdir / "regimes.parquet")

    dev_preds = pl.read_parquet(run_dir / "predictions.parquet").drop("fold")
    all_preds = pl.concat([dev_preds, preds.select(dev_preds.columns)], how="vertical")
    last_data = prices["date"].max()
    end = min(h.end, last_data) if isinstance(last_data, dt.date) else h.end
    bt = run_backtests(
        cfg,
        ds.frame,
        prices,
        macro,
        all_preds,
        sorted(rows["date"].unique().to_list()),
        ds.sessions,
        end,
        hdir / "backtest",
        quantiles=quant,
        label="holdout",
    )
    metrics["backtest"] = bt.metrics
    write_json(metrics, hdir / "metrics.json")
    write_json(
        {
            "run_id": run_dir.name,
            "evaluated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "override_reason": override_reason,
        },
        holdout_lock_path(cfg),
    )
    with (cfg.artifacts_dir / "holdout" / "history.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "run_id": run_dir.name,
                    "override_reason": override_reason,
                    "at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                }
            )
            + "\n"
        )
    return metrics


def mean_abs_shap(shap: pl.DataFrame) -> pl.DataFrame:
    feats = [c for c in shap.columns if c not in ("date", "ticker", "_bias")]
    vals = np.abs(shap.select(feats).to_numpy()).mean(axis=0)
    return pl.DataFrame({"feature": feats, "mean_abs_shap": vals}).sort(
        "mean_abs_shap", descending=True
    )
