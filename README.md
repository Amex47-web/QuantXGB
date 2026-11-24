# QuantXGB

**Point-in-time, regime-aware quantitative research & portfolio intelligence — built around XGBoost.**

A local-first research platform that ranks a multi-asset ETF universe with XGBoost
learning-to-rank, forecasts return distributions with XGBoost quantile regression, detects
market regimes, and turns signals into constrained portfolios. Results come from purged,
embargoed walk-forward validation, with leakage tests, overfitting statistics and an
untouched holdout. Total cost: ₹0 (open-source tools, free data, laptop CPU).

> Research only — not investment advice. See [`research/limitations.md`](research/limitations.md).

Full specification: [`PRD.md`](PRD.md) · Methodology as implemented:
[`research/methodology.md`](research/methodology.md).

## Quick start

```bash
brew install libomp            # macOS only: OpenMP runtime for XGBoost
uv sync                        # Python 3.12 environment from uv.lock
make demo                      # end-to-end on SYNTHETIC data, no API keys (~1 min)
QUANTXGB_ARTIFACTS=artifacts/demo make app   # dashboard on the demo run
```

Synthetic numbers show that the pipeline works. They are not research evidence.

## Real data

```bash
cp .env.example .env           # add a free FRED_API_KEY
make data                      # yfinance + FRED/ALFRED → raw snapshots → curated → DQ report
# review artifacts/reports/data_quality.md and research/research_spec.md §3 (owner)
make features                  # feature matrix + labels
make sanity                    # shuffled labels: IC must be ≈ 0
make train H="H1: cross-asset momentum + macro features rank 20d returns (see experiments.md E-002)"
make backtest report           # strategies × costs, DSR, PBO → artifacts/runs/<id>/report.md
make score                     # live scoring of the latest date with the final models
make app                       # Streamlit dashboard     (http://localhost:8501)
make api                       # FastAPI (read-only)     (http://localhost:8000/docs)
make mlflow                    # MLflow UI               (http://localhost:5000)
```

Feature-family ablation: `uv run quantxgb train --hypothesis "..." --drop-family macro`.

One-shot holdout (only after you set every config to `status: frozen`):
`uv run quantxgb holdout --run <run_id>`.

## Architecture

```
configs/*.yaml (typed, status: proposed|frozen)
      │
ingestion/  yfinance · FRED/ALFRED → immutable raw snapshots → curated PIT tables (Parquet + DuckDB)
validation/ data quality · point-in-time as-of joins · walk-forward splits (purge+embargo) · holdout guard
features/   momentum · volatility · trend · liquidity · risk · cross-asset · market · macro (registry)
labels/     forward returns aligned to t+1 execution, excess returns, ranking grades
models/     XGB ranker · XGB quantile · momentum / ridge / RF baselines · rule + HMM regimes · Optuna
research/   walk-forward runner · final models · live scoring · one-shot holdout
portfolio/  Grinold μ · Ledoit–Wolf/EWMA Σ · mean-variance & CVaR optimiser (CVXPY)
backtest/   event-driven engine (t+1, drift, costs, accounting checks) · strategies · metrics
evaluation/ IC, NDCG, pinball/coverage · stationary bootstrap · DSR · PBO
tracking/   MLflow (local sqlite) · global trial log
serving/ api/ app/   read-only artefact store → FastAPI · Streamlit
```

## Guarantees enforced by tests

| Guarantee | Test |
|---|---|
| Every feature is unchanged by truncating or perturbing data after t | `tests/leakage/test_features_point_in_time.py` |
| Labels start at t+1, gaps → null | `tests/leakage/test_labels.py` |
| Revised macro data uses the vintage published at the time | `tests/unit/test_point_in_time.py` |
| Training labels end before each test block (+ embargo); holdout untouched | `tests/unit/test_splits.py`, integration |
| No same-bar execution; NAV accounting identity | `tests/unit/test_backtest_engine.py` |
| HMM regime probabilities are filtered, not smoothed | `tests/unit/test_models.py` |
| Signal found when present; IC ≈ 0 on noise and on shuffled labels | `tests/integration/test_pipeline_synthetic.py` |

```bash
make check     # ruff + mypy + pytest
```

## Data terms

Yahoo Finance data via yfinance is for personal research use and must not be
redistributed. FRED data requires attribution: *Source: Federal Reserve Bank of St. Louis,
FRED / ALFRED.* No data, artefacts or keys are committed (`.gitignore`, pre-commit hook).
