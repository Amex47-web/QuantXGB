# Methodology (as implemented)

This file describes what the code does, with pointers to the implementing module. It is
factual; the owner's justification of each choice belongs in `research_spec.md`.

## Data

* **Prices** — Yahoo Finance via yfinance (`ingestion/yfinance_provider.py`). Yahoo's `Close`
  is split-adjusted but not distribution-adjusted; dividends and capital gains are on the
  same basis. Total return is computed in-house (`ingestion/normalize.py`):
  `ret_t = (close_t + div_t + capgain_t) / close_prev_valid − 1`, `TR = Π(1 + ret)`.
  Yahoo's `Adj Close` is used only for reconciliation (flag if daily returns differ > 0.5%).
* **Raw snapshots** are immutable, dated, and SHA-256 hashed (`ingestion/base.py`).
* **Calendar** — NYSE sessions from `exchange_calendars`. Each ticker is placed on the
  calendar grid from its first to its last bar; missing sessions stay null so rolling
  windows count sessions correctly.
* **Macro** — FRED for non-revised daily market series (available at the close of the next
  session); ALFRED vintages for revised series (available at the close of the session after
  the vintage's `realtime_start`).

## Point-in-time construction

* `available_on` = first session at whose close a value may be used.
* Macro transforms are **vintage-consistent** (`validation/point_in_time.py`): for each
  vintage the series is rebuilt exactly as published, transformed, and only its latest
  observation is kept, so the as-of join never mixes revised history with first releases.
* Decision dates = last session of each ISO week **from the exchange calendar**, never
  inferred from the data (found and fixed by the truncation test).
* Leakage tests (`tests/leakage`) recompute **every model feature** after (a) truncating
  all data at t and (b) randomly perturbing all data after t, and require identical values
  at dates ≤ t.

## Labels (`labels/forward_returns.py`)

`fwd_ret(t) = TR(t+1+h) / TR(t+1) − 1` with h = 20 sessions: the label starts at the
execution close (t+1). Excess return = minus the equal-weight cross-sectional mean of
eligible assets; ranking grades = within-date quintiles 0–4.

## Validation (`validation/splits.py`)

* Expanding walk-forward over the development period; first test block starts 5 years
  after the start; 6-month test blocks.
* Training rows must satisfy `label_end < session[idx(test_start) − embargo]` (purging +
  5-session embargo). Early stopping uses a purged inner split (last 20% of training dates),
  after which the model is refit on the full fold with the selected number of trees.
* `development_view` removes holdout dates and nulls any label reaching into the holdout;
  `assert_no_holdout` is called on every development dataset and on the final fit.

## Models

| Model | Module | Target |
|---|---|---|
| XGBoost ranker (`rank:ndcg`, date query groups) | `models/alpha_ranker.py` | grade 0–4 |
| XGBoost multi-quantile (`reg:quantileerror`, α = .05/.5/.95) | `models/quantile_risk.py` | fwd_ret |
| Momentum (12-1) | `models/baselines.py` | — |
| Ridge on within-date z-scores | `models/baselines.py` | fwd_xs_ret |
| Random forest | `models/baselines.py` | fwd_xs_ret |
| Rule regimes (vol threshold from training window × market trend) | `models/regime_detector.py` | — |
| Gaussian HMM, forward-filtered probabilities only | `models/regime_detector.py` | — |

## Evaluation

* Signal: Spearman rank IC per decision date, ICIR, Newey–West t (lags = ⌈h / sessions per signal⌉),
  stationary-bootstrap CI, NDCG@5, top-k hit rate, quintile spread, IC by regime.
* Quantiles: pinball loss per α, coverage of [Q05, Q95], width, crossing rate, against
  unconditional training-fold quantiles.
* Backtest (`backtest/engine.py`): signal at close t, execution at close t+1, holdings drift,
  one-way costs on traded notional, accounting identity asserted every session.
* Expected returns: Grinold `μ = IC̄ · σ_h · z` with IC̄ = mean IC of past OOS predictions
  whose labels were realised by t (clipped at 0; μ = 0 with < 26 realised dates).
* Optimiser: CVXPY mean-variance (Ledoit–Wolf Σ) or Rockafellar–Uryasev CVaR (historical or
  quantile-copula scenarios), long-only, w_max, asset-class caps, turnover cap/penalty, with
  logged relaxations.
* Overfitting: Deflated Sharpe Ratio (absolute and active vs. equal weight) using every
  strategy configuration in the trial log; PBO via CSCV across all strategies of a run.
* Sanity checks: synthetic noise and shuffled labels must give IC ≈ 0
  (`tests/integration/test_pipeline_synthetic.py`, `make sanity` on real data).
