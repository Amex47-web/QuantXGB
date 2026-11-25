"""The full operational workflow on synthetic raw snapshots, through the same functions the
CLI uses: raw store → curated → features → train (MLflow + tuning) → backtest → report →
live score → one-shot holdout (lock + override)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from quantxgb.cli import _backtest, _train
from quantxgb.config import AppConfig
from quantxgb.ingestion.base import RawStore
from quantxgb.ingestion.pipeline import build_curated, calendar_sessions
from quantxgb.ingestion.store import CuratedStore
from quantxgb.reporting.report import write_run_report
from quantxgb.research.dataset import assemble, load_dataset, save_dataset
from quantxgb.research.runs import list_runs, resolve_run
from quantxgb.research.scoring import HoldoutLocked, evaluate_holdout, score_live
from quantxgb.synthetic import (
    SyntheticSpec,
    generate_raw_macro,
    generate_raw_prices,
    synthetic_config,
)
from quantxgb.tracking.trial_log import TrialLog
from quantxgb.utils.calendar import business_day_sessions

pytestmark = pytest.mark.slow


def _frozen(cfg: AppConfig) -> AppConfig:
    upd = {
        s: getattr(cfg, s).model_copy(update={"status": "frozen"})
        for s in ("data", "features", "labels", "model", "validation", "backtest")
    }
    return cfg.model_copy(update=upd)


@pytest.fixture(scope="module")
def workflow(base_config: AppConfig, tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    tmp = tmp_path_factory.mktemp("wf")
    spec = SyntheticSpec(n_assets=8, start=dt.date(2004, 1, 1), end=dt.date(2012, 12, 31), seed=2)
    cfg = synthetic_config(base_config, spec)
    paths = cfg.data.paths.model_copy(
        update={
            "raw": tmp / "raw",
            "curated": tmp / "curated",
            "features": tmp / "features",
            "artifacts": tmp / "artifacts",
        }
    )
    tuning = cfg.model.tuning.model_copy(update={"enabled": True, "n_trials": 2})
    cfg = cfg.model_copy(
        update={
            "data": cfg.data.model_copy(update={"paths": paths}),
            "model": cfg.model.model_copy(update={"tuning": tuning}),
        }
    )
    cfg = _frozen(cfg)
    sessions = business_day_sessions(spec.start, spec.end)
    raw = RawStore(cfg.raw_dir)
    for t, df in generate_raw_prices(spec, sessions, cfg.data.universe).items():
        raw.write("yfinance", t, df, {})
    for sid, df in generate_raw_macro(spec, sessions).items():
        raw.write("fred", sid, df, {})
    dq = build_curated(cfg)
    store = CuratedStore(cfg.curated_dir)
    ds, fb = assemble(store.prices(), store.macro(), cfg, calendar_sessions(cfg, spec.start))
    save_dataset(ds, fb, cfg, store.data_hash())
    run_dir = _train(cfg, "workflow test", False, True, 2)
    return cfg, dq, run_dir


def test_curated_and_duckdb(workflow) -> None:  # type: ignore[no-untyped-def]
    import duckdb

    cfg, dq, _ = workflow
    assert dq.passed
    assert (cfg.artifacts_dir / "reports" / "data_quality.md").exists()
    db = cfg.curated_dir.parent / "quantxgb.duckdb"
    with duckdb.connect(str(db), read_only=True) as con:
        n = con.execute("select count(distinct ticker) from prices").fetchone()
        m = con.execute("select count(distinct series_id) from macro_pit").fetchone()
    assert n == (8,) and m == (3,)
    ds, meta = load_dataset(cfg)
    assert meta["data_hash"] != "none" and len(ds.feature_names) > 40


def test_training_logged(workflow) -> None:  # type: ignore[no-untyped-def]
    cfg, _, run_dir = workflow
    log = TrialLog(cfg.artifacts_dir)
    assert log.count("optuna") == 4  # 2 folds × 2 trials
    assert log.count("walk_forward") == 1
    assert (cfg.artifacts_dir / "mlflow.db").exists()
    assert resolve_run(cfg, "latest") == run_dir
    runs = list_runs(cfg)
    assert runs[0]["hypothesis"] == "workflow test"


def test_backtest_report_and_score(workflow) -> None:  # type: ignore[no-untyped-def]
    cfg, _, run_dir = workflow
    _backtest(cfg, run_dir)
    assert TrialLog(cfg.artifacts_dir).count("strategy") >= 9
    report = write_run_report(run_dir).read_text(encoding="utf-8")
    for section in (
        "Out-of-sample signal quality",
        "Backtest",
        "Overfitting diagnostics",
        "Interpretation (owner)",
    ):
        assert section in report
    ds, _ = load_dataset(cfg)
    live = score_live(run_dir, ds, cfg)
    assert (live / "meta.json").exists()


def test_holdout_is_one_shot(workflow) -> None:  # type: ignore[no-untyped-def]
    cfg, _, run_dir = workflow
    ds, _ = load_dataset(cfg)
    store = CuratedStore(cfg.curated_dir)
    metrics = evaluate_holdout(run_dir, ds, store.prices(), store.macro(), cfg)
    assert "xgb_ndcg" in metrics["models"] and "backtest" in metrics
    assert (cfg.artifacts_dir / "holdout" / "LOCK.json").exists()
    with pytest.raises(HoldoutLocked, match="already evaluated"):
        evaluate_holdout(run_dir, ds, store.prices(), store.macro(), cfg)
    evaluate_holdout(run_dir, ds, store.prices(), store.macro(), cfg, override_reason="test rerun")
    hist = (
        (cfg.artifacts_dir / "holdout" / "history.jsonl").read_text(encoding="utf-8").splitlines()
    )
    assert len(hist) == 2 and "test rerun" in hist[1]


def test_holdout_rejects_config_drift(workflow, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    cfg, _, run_dir = workflow
    drifted = cfg.model_copy(update={"labels": cfg.labels.model_copy(update={"horizon": 5})})
    ds, _ = load_dataset(cfg)
    store = CuratedStore(cfg.curated_dir)
    with pytest.raises(HoldoutLocked, match="differ"):
        evaluate_holdout(run_dir, ds, store.prices(), store.macro(), drifted)
