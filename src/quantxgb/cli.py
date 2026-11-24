"""QuantXGB command line (PRD §6.4).

quantxgb data ingest | data build
quantxgb features build | features dictionary
quantxgb train --hypothesis "..."
quantxgb backtest --run latest
quantxgb report --run latest
quantxgb score --run latest
quantxgb holdout --run <id>          # one-shot, guarded
quantxgb runs
quantxgb demo                        # synthetic end-to-end, no API keys
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
from typing import Annotated

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import typer
from rich.console import Console
from rich.table import Table

from quantxgb.config import AppConfig, load_config

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)
data_app = typer.Typer(no_args_is_help=True, help="Download and curate data.")
features_app = typer.Typer(no_args_is_help=True, help="Build feature matrices.")
app.add_typer(data_app, name="data")
app.add_typer(features_app, name="features")
console = Console()

ConfigOpt = Annotated[
    Path | None, typer.Option("--config-dir", help="Directory with YAML configs.")
]


def _cfg(config_dir: Path | None) -> AppConfig:
    return load_config(config_dir)


def _status_warning(cfg: AppConfig) -> None:
    if not cfg.all_frozen:
        console.print(
            "[yellow]Note: configs are `proposed`, not `frozen` — results are exploratory.[/]"
        )


# ----------------------------------------------------------------------------- data
@data_app.command("ingest")
def data_ingest(
    config_dir: ConfigOpt = None,
    prices: Annotated[bool, typer.Option(help="Download prices (yfinance).")] = True,
    macro: Annotated[bool, typer.Option(help="Download FRED/ALFRED macro series.")] = True,
) -> None:
    """Download raw snapshots (immutable, dated) for the configured universe."""
    from quantxgb.ingestion.pipeline import default_providers, ingest_macro, ingest_prices

    cfg = _cfg(config_dir)
    price_p, macro_p = default_providers(cfg)
    if prices:
        n = ingest_prices(cfg, price_p)
        console.print(f"[green]prices: {n} tickers snapshotted[/]")
    if macro:
        if macro_p is None:
            console.print("[yellow]FRED_API_KEY not set — macro skipped (see .env.example)[/]")
        else:
            n = ingest_macro(cfg, macro_p)
            console.print(f"[green]macro: {n} series snapshotted[/]")


@data_app.command("build")
def data_build(config_dir: ConfigOpt = None) -> None:
    """Normalise the latest raw snapshots into curated PIT tables + data-quality report."""
    from quantxgb.ingestion.pipeline import build_curated

    cfg = _cfg(config_dir)
    report = build_curated(cfg)
    path = cfg.artifacts_dir / "reports" / "data_quality.md"
    colour = "green" if report.passed else "red"
    console.print(f"[{colour}]data quality: {'PASS' if report.passed else 'FAIL'}[/] → {path}")
    flags = [i for i in report.issues if i.severity == "flag"]
    if flags:
        console.print(f"[yellow]{len(flags)} items flagged for manual review by the owner[/]")
    if not report.passed:
        raise typer.Exit(1)


# ------------------------------------------------------------------------- features
@features_app.command("build")
def features_build(
    config_dir: ConfigOpt = None,
    allow_dq_fail: Annotated[bool, typer.Option(help="Build even if data quality failed.")] = False,
) -> None:
    """Build the feature matrix + labels for the current config."""
    from quantxgb.ingestion.pipeline import calendar_sessions
    from quantxgb.ingestion.store import CuratedStore
    from quantxgb.research.dataset import assemble, save_dataset
    from quantxgb.validation.data_quality import run_data_quality

    cfg = _cfg(config_dir)
    store = CuratedStore(cfg.curated_dir)
    prices, macro = store.prices(), store.macro()
    dq = run_data_quality(prices, macro)
    if not dq.passed and not allow_dq_fail:
        console.print("[red]data quality FAIL — fix the data or pass --allow-dq-fail[/]")
        raise typer.Exit(1)
    ds, fb = assemble(prices, macro, cfg, calendar_sessions(cfg))
    out = save_dataset(ds, fb, cfg, store.data_hash())
    console.print(f"[green]{ds.frame.height} rows × {len(ds.feature_names)} features → {out}[/]")


@features_app.command("dictionary")
def features_dictionary(config_dir: ConfigOpt = None) -> None:
    """Write research/feature_dictionary.md from the feature registry."""
    from quantxgb.features.build import all_specs
    from quantxgb.features.registry import render_dictionary

    cfg = _cfg(config_dir)
    path = cfg.root / "research" / "feature_dictionary.md"
    path.write_text(render_dictionary(all_specs(cfg.features)))
    console.print(f"[green]→ {path}[/]")


# ------------------------------------------------------------------------- research
FAMILIES = [
    "momentum",
    "volatility",
    "trend",
    "liquidity",
    "risk",
    "cross_asset",
    "market",
    "macro",
    "regime",
]


def _train(
    cfg: AppConfig,
    hypothesis: str,
    shuffle_labels: bool,
    mlflow: bool,
    max_folds: int | None,
    drop_families: list[str] | None = None,
) -> Path:
    from quantxgb.features.build import all_specs
    from quantxgb.research.dataset import load_dataset
    from quantxgb.research.runs import new_run_id, runs_root, write_manifest
    from quantxgb.research.walk_forward import run_walk_forward
    from quantxgb.tracking.mlflow_utils import Tracker
    from quantxgb.tracking.trial_log import TrialLog

    ds, meta = load_dataset(cfg)
    drop = sorted(set(drop_families or []))
    unknown = set(drop) - set(FAMILIES)
    if unknown:
        raise typer.BadParameter(
            f"unknown feature families {sorted(unknown)}; choose from {FAMILIES}"
        )
    if drop:
        family = {s.name: s.family for s in all_specs(cfg.features)}
        ds.feature_names = [f for f in ds.feature_names if family.get(f) not in drop]
        if "regime" in drop:
            regime = cfg.model.regime.model_copy(update={"add_regime_features": False})
            cfg = cfg.model_copy(update={"model": cfg.model.model_copy(update={"regime": regime})})
    run_id = new_run_id(cfg)
    run_dir = runs_root(cfg) / run_id
    run_dir.mkdir(parents=True)
    write_manifest(
        cfg,
        run_dir,
        run_id,
        hypothesis=hypothesis,
        shuffle_labels=shuffle_labels,
        drop_families=drop,
        n_features=len(ds.feature_names),
        data_hash=meta.get("data_hash"),
        dataset_hash=meta.get("dataset_hash"),
    )
    tracker = Tracker(cfg.artifacts_dir, enabled=mlflow)
    tags = {
        "config_hash": cfg.section_hash("features", "labels", "model", "validation"),
        "data_hash": str(meta.get("data_hash")),
        "shuffle_labels": str(shuffle_labels),
    }
    with tracker.run(run_id, tags):
        tracker.params(
            {
                "model": cfg.model.model_dump(mode="json"),
                "labels": cfg.labels.model_dump(mode="json"),
                "validation": cfg.validation.model_dump(mode="json"),
            }
        )
        run_walk_forward(
            ds,
            cfg,
            run_id,
            run_dir,
            tracker,
            TrialLog(cfg.artifacts_dir),
            shuffle_labels=shuffle_labels,
            max_folds=max_folds,
        )
    return run_dir


@app.command()
def train(
    hypothesis: Annotated[
        str, typer.Option(help="Pre-registered hypothesis for this run (required).")
    ],
    config_dir: ConfigOpt = None,
    shuffle_labels: Annotated[
        bool, typer.Option(help="Sanity run: permute labels within dates.")
    ] = False,
    mlflow: Annotated[bool, typer.Option(help="Log to local MLflow.")] = True,
    max_folds: Annotated[int | None, typer.Option(help="Debug: limit number of folds.")] = None,
    drop_family: Annotated[
        list[str] | None,
        typer.Option(
            help="Ablation: drop a feature family (repeatable). momentum|volatility|trend|"
            "liquidity|risk|cross_asset|market|macro|regime"
        ),
    ] = None,
) -> None:
    """Walk-forward training/evaluation on the development period."""
    cfg = _cfg(config_dir)
    _status_warning(cfg)
    if not hypothesis.strip():
        raise typer.BadParameter("a pre-registered hypothesis is required")
    run_dir = _train(cfg, hypothesis, shuffle_labels, mlflow, max_folds, drop_family)
    console.print(f"[green]run complete → {run_dir}[/]")
    _print_signal(run_dir)


def _print_signal(run_dir: Path) -> None:
    from quantxgb.utils.io import read_json

    met = read_json(run_dir / "metrics.json")
    t = Table(title=f"OOS signal quality — {run_dir.name}")
    for c in ("model", "IC mean", "95% CI", "t (NW)", "NDCG@5"):
        t.add_column(c)
    for name, s in met["models"].items():
        t.add_row(
            name,
            f"{s['ic_mean']:+.4f}",
            f"[{s['ic_mean_ci_low']:+.4f}, {s['ic_mean_ci_high']:+.4f}]",
            f"{s['ic_tstat_nw']:+.2f}",
            f"{s['ndcg_at_5']:.3f}",
        )
    console.print(t)


def _backtest(cfg: AppConfig, run_dir: Path) -> Path:
    import polars as pl

    from quantxgb.backtest.runner import run_backtests
    from quantxgb.ingestion.store import CuratedStore
    from quantxgb.research.dataset import load_dataset
    from quantxgb.tracking.trial_log import TrialLog
    from quantxgb.utils.calendar import trading_sessions

    ds, _ = load_dataset(cfg)
    store = CuratedStore(cfg.curated_dir)
    preds = pl.read_parquet(run_dir / "predictions.parquet")
    quant = pl.read_parquet(run_dir / "quantiles.parquet")
    dates = sorted(preds["date"].unique().to_list())
    h0 = cfg.backtest.splits.holdout.start
    sess = [s for s in ds.sessions if s < h0]
    end = min(sess[-1], trading_sessions(dates[-1], h0, cfg.data.calendar)[-1])
    run_backtests(
        cfg,
        ds.frame,
        store.prices().filter(pl.col("date") < h0),
        store.macro(),
        preds,
        dates,
        ds.sessions,
        end,
        run_dir / "backtest",
        quantiles=quant,
        trial_log=TrialLog(cfg.artifacts_dir),
        run_id=run_dir.name,
    )
    return run_dir / "backtest"


@app.command()
def backtest(
    run: Annotated[str, typer.Option(help="Run id or 'latest'.")] = "latest",
    config_dir: ConfigOpt = None,
) -> None:
    """Backtest every strategy × cost scenario on a run's OOS predictions."""
    from quantxgb.research.runs import resolve_run

    cfg = _cfg(config_dir)
    out = _backtest(cfg, resolve_run(cfg, run))
    console.print(f"[green]backtest → {out}[/]")


@app.command()
def report(
    run: Annotated[str, typer.Option(help="Run id or 'latest'.")] = "latest",
    config_dir: ConfigOpt = None,
) -> None:
    """Write the run's Markdown report (numbers only)."""
    from quantxgb.reporting.report import write_run_report
    from quantxgb.research.runs import resolve_run

    cfg = _cfg(config_dir)
    path = write_run_report(resolve_run(cfg, run))
    console.print(f"[green]→ {path}[/]")


@app.command()
def score(
    run: Annotated[str, typer.Option(help="Run id or 'latest'.")] = "latest",
    date: Annotated[str | None, typer.Option(help="YYYY-MM-DD (default: latest).")] = None,
    config_dir: ConfigOpt = None,
) -> None:
    """Score a decision date with the run's final models (live view; not an evaluation)."""
    from quantxgb.research.dataset import load_dataset
    from quantxgb.research.runs import resolve_run
    from quantxgb.research.scoring import score_live

    cfg = _cfg(config_dir)
    ds, _ = load_dataset(cfg)
    out = score_live(resolve_run(cfg, run), ds, cfg, dt.date.fromisoformat(date) if date else None)
    console.print(f"[green]→ {out}[/]")


@app.command()
def holdout(
    run: Annotated[str, typer.Option(help="Run id to evaluate (no 'latest' — be explicit).")],
    override_reason: Annotated[
        str | None, typer.Option(help="Required to re-run; logged permanently.")
    ] = None,
    config_dir: ConfigOpt = None,
) -> None:
    """ONE-SHOT evaluation on the final holdout. Requires frozen configs."""
    from quantxgb.ingestion.store import CuratedStore
    from quantxgb.research.dataset import load_dataset
    from quantxgb.research.runs import resolve_run
    from quantxgb.research.scoring import HoldoutLocked, evaluate_holdout

    cfg = _cfg(config_dir)
    if run == "latest":
        raise typer.BadParameter("pass an explicit run id for the holdout")
    ds, _ = load_dataset(cfg)
    store = CuratedStore(cfg.curated_dir)
    try:
        evaluate_holdout(
            resolve_run(cfg, run), ds, store.prices(), store.macro(), cfg, override_reason
        )
    except HoldoutLocked as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(2) from None
    console.print(f"[green]holdout evaluated → {cfg.artifacts_dir / 'holdout' / run}[/]")


@app.command()
def runs(config_dir: ConfigOpt = None) -> None:
    """List runs."""
    from quantxgb.research.runs import list_runs

    cfg = _cfg(config_dir)
    t = Table(title="Runs")
    for c in ("run_id", "created", "backtest", "sanity", "XGB IC", "hypothesis"):
        t.add_column(c)
    for r in list_runs(cfg):
        xgb = next((v for k, v in r["ic"].items() if k.startswith("xgb")), None)
        t.add_row(
            r["run_id"],
            r["created"],
            "yes" if r["has_backtest"] else "-",
            "shuffled" if r["shuffle_labels"] else "",
            f"{xgb:+.4f}" if xgb is not None else "-",
            (r["hypothesis"] or "")[:60],
        )
    console.print(t)


@app.command()
def demo(
    signal: Annotated[float, typer.Option(help="Injected signal strength (0 = pure noise).")] = 1.0,
    config_dir: ConfigOpt = None,
) -> None:
    """End-to-end run on SYNTHETIC data (no API keys) into data/demo and artifacts/demo."""
    from quantxgb.ingestion.base import RawStore
    from quantxgb.ingestion.pipeline import build_curated
    from quantxgb.reporting.report import write_run_report
    from quantxgb.research.dataset import assemble, save_dataset
    from quantxgb.research.scoring import score_live
    from quantxgb.synthetic import (
        SyntheticSpec,
        generate_raw_macro,
        generate_raw_prices,
        synthetic_config,
    )
    from quantxgb.utils.calendar import business_day_sessions

    base = _cfg(config_dir)
    spec = SyntheticSpec(signal=signal)
    cfg = synthetic_config(base, spec)
    root = base.root
    paths = cfg.data.paths.model_copy(
        update={
            "raw": root / "data/demo/raw",
            "curated": root / "data/demo/curated",
            "features": root / "data/demo/features",
            "artifacts": root / "artifacts/demo",
        }
    )
    cfg = cfg.model_copy(update={"data": cfg.data.model_copy(update={"paths": paths})})
    console.print("[bold]Synthetic demo — numbers are NOT research evidence.[/]")
    sessions = [s for s in business_day_sessions(spec.start, spec.end)]
    raw = RawStore(cfg.raw_dir)
    snap = dt.date(2000, 1, 1) + dt.timedelta(days=int(signal * 1000))
    for t, df in generate_raw_prices(spec, sessions, cfg.data.universe).items():
        raw.write("yfinance", t, df, {"synthetic": True, "signal": signal}, snapshot_date=snap)
    for sid, df in generate_raw_macro(spec, sessions).items():
        raw.write("fred", sid, df, {"synthetic": True}, snapshot_date=snap)
    report_dq = build_curated(cfg)
    console.print(f"data quality: {'PASS' if report_dq.passed else 'FAIL'}")
    from quantxgb.ingestion.pipeline import calendar_sessions
    from quantxgb.ingestion.store import CuratedStore

    store = CuratedStore(cfg.curated_dir)
    ds, fb = assemble(store.prices(), store.macro(), cfg, calendar_sessions(cfg, spec.start))
    save_dataset(ds, fb, cfg, store.data_hash())
    run_dir = _train(cfg, f"Synthetic demo (signal={signal})", False, True, None)
    _print_signal(run_dir)
    _backtest(cfg, run_dir)
    score_live(run_dir, ds, cfg)
    path = write_run_report(run_dir)
    console.print(f"[green]demo complete → {path}[/]")
    console.print("Dashboard: QUANTXGB_ARTIFACTS=artifacts/demo uv run streamlit run app/Home.py")


if __name__ == "__main__":
    app()
