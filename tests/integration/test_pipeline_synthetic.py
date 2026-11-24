"""End-to-end on synthetic data (FR-VAL-07, NFR-04).

* Injected signal → the pipeline must find significantly positive OOS IC.
* Pure noise / shuffled labels → OOS IC must be indistinguishable from 0.
  A positive IC here would indicate leakage somewhere in the pipeline.
"""

from __future__ import annotations

import datetime as dt
import itertools
from pathlib import Path

import polars as pl
import pytest

from quantxgb.backtest.runner import run_backtests
from quantxgb.config import AppConfig
from quantxgb.research.dataset import assemble
from quantxgb.research.scoring import HoldoutLocked, check_holdout_preconditions, score_live
from quantxgb.research.walk_forward import run_walk_forward
from quantxgb.synthetic import SyntheticSpec, make_synthetic, synthetic_config
from quantxgb.tracking.trial_log import TrialLog

pytestmark = pytest.mark.slow


def _setup(base: AppConfig, signal: float, tmp: Path, seed: int = 11):  # type: ignore[no-untyped-def]
    spec = SyntheticSpec(
        n_assets=10, start=dt.date(2004, 1, 1), end=dt.date(2013, 12, 31), signal=signal, seed=seed
    )
    cfg = synthetic_config(base, spec)
    paths = cfg.data.paths.model_copy(update={"artifacts": tmp / "artifacts"})
    cfg = cfg.model_copy(update={"data": cfg.data.model_copy(update={"paths": paths})})
    sd = make_synthetic(cfg, spec)
    ds, _ = assemble(sd.prices, sd.macro, cfg, sd.sessions)
    return cfg, sd, ds


@pytest.fixture(scope="module")
def signal_run(base_config: AppConfig, tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    tmp = tmp_path_factory.mktemp("signal")
    cfg, sd, ds = _setup(base_config, 1.0, tmp)
    res = run_walk_forward(ds, cfg, "sig", tmp / "run", trial_log=TrialLog(cfg.artifacts_dir))
    return cfg, sd, ds, res


def test_signal_is_found(signal_run) -> None:  # type: ignore[no-untyped-def]
    _, _, _, res = signal_run
    xgb = res.metrics["models"]["xgb_ndcg"]
    assert xgb["ic_mean_ci_low"] > 0.05
    assert xgb["ic_tstat_nw"] > 3


def test_predictions_are_out_of_sample_and_pre_holdout(signal_run) -> None:  # type: ignore[no-untyped-def]
    cfg, _, _, res = signal_run
    p = res.predictions
    assert p["date"].max() <= cfg.backtest.splits.development.end
    lab = p.filter(pl.col("fwd_ret").is_not_null())
    assert lab["label_end"].max() < cfg.backtest.splits.holdout.start
    folds = res.fold_metrics
    for a, b in itertools.pairwise(folds):
        assert a["test_end"] < b["test_start"]
    assert (res.run_dir / "shap_oos.parquet").exists()
    assert (res.run_dir / "models" / "final.pkl").exists()


def test_backtest_and_live_scoring(signal_run, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    cfg, sd, ds, res = signal_run
    dates = sorted(res.predictions["date"].unique().to_list())
    out = run_backtests(
        cfg,
        ds.frame,
        sd.prices,
        sd.macro,
        res.predictions,
        dates,
        sd.sessions,
        cfg.backtest.splits.development.end,
        tmp_path / "bt",
        quantiles=res.quantiles,
    )
    s = out.metrics["strategies"]
    assert {"opt_xgb_ndcg", "equal_weight", "momentum_topk"} <= set(s)
    xgb = s["opt_xgb_ndcg"]["by_cost"]["10"]
    ew = s["equal_weight"]["by_cost"]["10"]
    assert xgb["sharpe"] > ew["sharpe"]
    for cost in ("5", "10", "20"):
        assert s["opt_xgb_ndcg"]["by_cost"][cost]["turnover_annual"] > 0
    # higher costs never help
    assert s["opt_xgb_ndcg"]["by_cost"]["20"]["cagr"] < s["opt_xgb_ndcg"]["by_cost"]["5"]["cagr"]
    live = score_live(res.run_dir, ds, cfg)
    assert (live / "scores.parquet").exists() and (live / "shap.parquet").exists()


def test_holdout_requires_frozen_configs(signal_run) -> None:  # type: ignore[no-untyped-def]
    cfg, _, _, res = signal_run
    from quantxgb.research.runs import write_manifest

    write_manifest(cfg, res.run_dir, "sig")
    with pytest.raises(HoldoutLocked, match="frozen"):
        check_holdout_preconditions(cfg, res.run_dir, None)


@pytest.mark.parametrize(("signal", "shuffle"), [(0.0, False), (1.0, True)])
def test_no_edge_without_signal(
    base_config: AppConfig, tmp_path: Path, signal: float, shuffle: bool
) -> None:
    cfg, _, ds = _setup(base_config, signal, tmp_path, seed=5)
    res = run_walk_forward(ds, cfg, "noise", tmp_path / "run", shuffle_labels=shuffle)
    for name, m in res.metrics["models"].items():
        assert m["ic_mean_ci_low"] < 0 < m["ic_mean_ci_high"] or abs(m["ic_mean"]) < 0.03, (
            f"{name}: IC {m['ic_mean']:.4f} without signal suggests leakage"
        )
