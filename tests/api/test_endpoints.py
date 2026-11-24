from __future__ import annotations

import datetime as dt

import pytest
from fastapi.testclient import TestClient

from quantxgb.config import AppConfig
from quantxgb.research.dataset import assemble
from quantxgb.research.runs import write_manifest
from quantxgb.research.scoring import score_live
from quantxgb.research.walk_forward import run_walk_forward
from quantxgb.synthetic import SyntheticSpec, make_synthetic, synthetic_config

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def client(base_config: AppConfig, tmp_path_factory: pytest.TempPathFactory) -> TestClient:
    from quantxgb.backtest.runner import run_backtests

    tmp = tmp_path_factory.mktemp("api")
    spec = SyntheticSpec(n_assets=8, start=dt.date(2004, 1, 1), end=dt.date(2011, 12, 31))
    cfg = synthetic_config(base_config, spec)
    paths = cfg.data.paths.model_copy(update={"artifacts": tmp})
    cfg = cfg.model_copy(update={"data": cfg.data.model_copy(update={"paths": paths})})
    sd = make_synthetic(cfg, spec)
    ds, _ = assemble(sd.prices, sd.macro, cfg, sd.sessions)
    run_dir = tmp / "runs" / "r1"
    run_dir.mkdir(parents=True)
    write_manifest(cfg, run_dir, "r1", hypothesis="api test")
    res = run_walk_forward(ds, cfg, "r1", run_dir, max_folds=2)
    dates = sorted(res.predictions["date"].unique().to_list())
    run_backtests(
        cfg,
        ds.frame,
        sd.prices,
        sd.macro,
        res.predictions,
        dates,
        sd.sessions,
        max(dates),
        run_dir / "backtest",
    )
    score_live(run_dir, ds, cfg)
    import os

    os.environ["QUANTXGB_ARTIFACTS"] = str(tmp)
    from quantxgb.api.main import app

    return TestClient(app)


def test_health_and_runs(client: TestClient) -> None:
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["latest_run"] == "r1"
    assert client.get("/runs").json()[0]["hypothesis"] == "api test"
    assert len(client.get("/universe").json()) == 8


def test_rankings_risk_regime_explain(client: TestClient) -> None:
    r = client.get("/rankings").json()
    assert r["rankings"][0]["rank"] == 1
    d = r["date"]
    risk = client.get("/risk", params={"date": d}).json()
    assert {"ticker", "q05", "q50", "q95"} <= set(risk[0])
    assert all(x["q05"] <= x["q50"] <= x["q95"] for x in risk)
    assert "regime" in client.get("/regime").json()
    t = r["rankings"][0]["ticker"]
    e = client.get(f"/explain/{t}", params={"date": d}).json()
    assert len(e["contributions"]) == 10


def test_portfolio_backtest_experiments(client: TestClient) -> None:
    p = client.get("/portfolio").json()
    assert abs(sum(w["weight"] for w in p["weights"]) - 1) < 1e-6
    assert "strategies" in client.get("/backtests/r1").json()
    assert client.get("/backtests/nope").status_code == 404
    assert client.get("/experiments").json()[0]["run_id"] == "r1"
    assert client.get("/data-health").status_code == 404
