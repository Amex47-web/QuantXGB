"""Read-only FastAPI service over precomputed artefacts (PRD §11.1, FR-API-01).

uv run uvicorn quantxgb.api.main:app --port 8000
"""

from __future__ import annotations

import datetime as dt
import os
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from quantxgb.serving.store import ArtifactStore

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

app = FastAPI(
    title="QuantXGB API",
    description="Read-only research artefacts. Research only — not investment advice.",
    version="0.1.0",
)


def store() -> ArtifactStore:
    return ArtifactStore()


class Health(BaseModel):
    status: str
    artifacts: str
    runs: int
    latest_run: str | None


class Ranking(BaseModel):
    ticker: str
    model: str
    score: float | None
    rank: int


class RankingsResponse(BaseModel):
    run_id: str
    date: dt.date
    rankings: list[Ranking]


def _run_id(s: ArtifactStore, run_id: str | None) -> str:
    try:
        return s.run_dir(run_id).name
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from None


@app.get("/health", response_model=Health)
def health() -> Health:
    s = store()
    runs = s.runs()
    return Health(
        status="ok", artifacts=str(s.root), runs=len(runs), latest_run=runs[-1] if runs else None
    )


@app.get("/runs")
def runs() -> list[dict[str, Any]]:
    s = store()
    out = []
    for r in s.runs():
        m = s.json(r, "manifest.json") or {}
        out.append(
            {
                "run_id": r,
                "created": m.get("created"),
                "hypothesis": m.get("hypothesis"),
                "shuffle_labels": m.get("shuffle_labels", False),
            }
        )
    return out


@app.get("/universe")
def universe(run_id: str | None = None) -> list[dict[str, Any]]:
    s = store()
    m = s.json(_run_id(s, run_id), "manifest.json") or {}
    return list(m.get("config", {}).get("data", {}).get("universe", []))


@app.get("/rankings", response_model=RankingsResponse)
def rankings(
    run_id: str | None = None, date: dt.date | None = None, model: str = Query("xgb_ndcg")
) -> RankingsResponse:
    s = store()
    rid = _run_id(s, run_id)
    try:
        d, df = s.rankings(rid, date)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from None
    df = df.filter(df["model"] == model).sort("score", descending=True, nulls_last=True)
    if df.height == 0:
        raise HTTPException(404, f"no scores for model {model} on {d}")
    items = [
        Ranking(ticker=r["ticker"], model=r["model"], score=r["score"], rank=i + 1)
        for i, r in enumerate(df.iter_rows(named=True))
    ]
    return RankingsResponse(run_id=rid, date=d, rankings=items)


@app.get("/risk")
def risk(date: dt.date, run_id: str | None = None) -> list[dict[str, Any]]:
    s = store()
    q = s.quantiles(_run_id(s, run_id), date)
    if q.height == 0:
        raise HTTPException(404, f"no quantile forecasts on {date}")
    cols = [c for c in q.columns if c in ("ticker",) or (c.startswith("q") and c[1:].isdigit())]
    return list(q.select(cols).to_dicts())


@app.get("/regime")
def regime(date: dt.date | None = None, run_id: str | None = None) -> dict[str, Any]:
    s = store()
    r = s.regimes(_run_id(s, run_id))
    if r.height == 0:
        raise HTTPException(404, "no regimes")
    r = r.filter(r["date"] <= date) if date else r
    if r.height == 0:
        raise HTTPException(404, f"no regime at or before {date}")
    return dict(r.tail(1).to_dicts()[0])


@app.get("/portfolio")
def portfolio(
    run_id: str | None = None, strategy: str = "opt_xgb_ndcg", date: dt.date | None = None
) -> dict[str, Any]:
    s = store()
    rid = _run_id(s, run_id)
    w = s.frame(rid, "backtest/weights.parquet")
    if w is None:
        raise HTTPException(404, "no backtest for this run")
    w = w.filter(w["strategy"] == strategy)
    if date:
        w = w.filter(w["exec_date"] <= date)
    if w.height == 0:
        raise HTTPException(404, "no weights")
    last = w["exec_date"].max()
    cur = w.filter(w["exec_date"] == last).sort("weight", descending=True)
    return {
        "run_id": rid,
        "strategy": strategy,
        "exec_date": last,
        "weights": cur.select("ticker", "weight").to_dicts(),
    }


@app.get("/explain/{ticker}")
def explain(ticker: str, date: dt.date, run_id: str | None = None, top: int = 10) -> dict[str, Any]:
    s = store()
    e = s.shap(_run_id(s, run_id), date, ticker)
    if e.height == 0:
        raise HTTPException(404, f"no explanation for {ticker} on {date}")
    return {
        "ticker": ticker,
        "date": date,
        "note": "TreeSHAP attribution in ranking-score units; not causal.",
        "contributions": e.head(top).to_dicts(),
    }


@app.get("/backtests/{run_id}")
def backtest(run_id: str) -> dict[str, Any]:
    s = store()
    m = s.json(_run_id(s, run_id), "backtest/metrics.json")
    if m is None:
        raise HTTPException(404, "no backtest for this run")
    return dict(m)


@app.get("/experiments")
def experiments() -> list[dict[str, Any]]:
    s = store()
    out = []
    for r in s.runs():
        met = s.json(r, "metrics.json") or {}
        out.append(
            {"run_id": r, **{f"ic_{k}": v.get("ic_mean") for k, v in met.get("models", {}).items()}}
        )
    return out


@app.get("/data-health")
def data_health() -> dict[str, Any]:
    md = store().data_quality_markdown()
    if md is None:
        raise HTTPException(404, "no data-quality report")
    return {"markdown": md, "passed": "**Result: PASS**" in md}
