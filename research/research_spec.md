# Research specification (Stage 0) — OWNER DOCUMENT

> **Owner:** you. Claude Code must not fill in or change the decisions below.
> The proposed defaults live in `configs/*.yaml` with `status: proposed` and mirror PRD §4.
> When you have decided, edit the configs, set `status: frozen` in each file, record the
> decision in PRD §18.2, and commit. The holdout command refuses to run until every config
> is frozen.

## 1. Hypothesis (in your own words)

_Why is H1 (PRD §4.1) worth testing? What would convince you it is false?_

- H1:
- H2:
- H3:
- Kill criteria (decided now, before seeing results):

## 2. Decisions to freeze

| # | Decision | Proposed default (configs) | Your decision | Rationale |
|---|---|---|---|---|
| Q1 | Universe (20 ETFs) | `data.yaml → universe` | | |
| Q2 | Primary horizon | 20 sessions (`labels.yaml`) | | |
| Q3 | Rebalance frequency | every 4 weekly signals | | |
| Q4 | Execution price | close of t+1 | | |
| Q5 | Holdout window | 2024-11-01 → 2026-10-30 | | |
| Q6 | Asset-class neutralisation | not implemented as primary (experiment) | | |
| Q7 | Constraints | w_max 0.25 / class 0.50 / turnover 0.50 | | |
| Q8 | Alpha Vantage eligibility | not used | | |
| Q10 | Risk-free | DGS3MO (PIT) | | |
| — | Cost scenarios | 5 / 10 / 20 bps one-way | | |
| — | Macro release lag | +1 session after ALFRED `realtime_start` | | |
| — | Tuning | disabled; 40 trials when enabled | | |

## 3. Data verification log (Stage 1 exit criterion)

Verify against the original source and record what you checked:

| Date checked | Item | Source checked | Result |
|---|---|---|---|
| | 10 random price bars | Yahoo website / other provider | |
| | 3 dividend / capital-gain events | fund sponsor site | |
| | XLK split 2025-12-05 handled | data_quality report | |
| | 5 macro vintages (CPI, UNRATE) | ALFRED web UI | |
| | Inception dates vs `approx_inception` | sponsor site | |

## 4. Feature hand-checks (Stage 2 exit criterion)

Hand-calculate at least 3 features on 3 dates from `data/features/<hash>/dataset.parquet`.

| Feature | Date | Ticker | Your value | Pipeline value |
|---|---|---|---|---|
