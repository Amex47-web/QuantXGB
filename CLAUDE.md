# QuantXGB — rules for Claude Code

Read `PRD.md` first. This file is the short version of PRD §17.

## Role
Act as a cautious quantitative research engineer. Correctness and auditability outrank
backtest performance. Never optimise for an attractive result.

## Hard rules
1. Never use information with `available_on` > decision date. Every new feature or label
   must be covered by `tests/leakage` (truncation + future-perturbation tests run over
   *all* model features automatically — keep it that way).
2. Never read holdout dates (`configs/backtest.yaml → splits.holdout`) in training,
   tuning or model selection. All development code goes through
   `validation.splits.development_view`. Only `quantxgb holdout` touches the holdout.
3. Never invent, estimate or "fill in" performance numbers. Report only numbers produced
   by code in this repo, quoting the run id.
4. Execution happens at t+1 or later (`backtest.engine.simulate` rejects delay < 1).
5. Every experiment goes through `quantxgb train --hypothesis "..."` so it is logged in
   the trial log (`artifacts/tracking/trials.jsonl`) and MLflow. Do not delete trial logs
   or re-run configurations silently — the Deflated Sharpe Ratio depends on the count.
6. Do not change values in `configs/` — research decisions belong to the owner. Never
   set `status: frozen`; only the owner does that.
7. Never commit `data/`, `artifacts/`, `.env` or downloaded datasets.
8. Decision dates come from the exchange calendar, never from the data.

## Critical modules (owner reviews every change line by line)
`labels/forward_returns.py`, `validation/splits.py`, `validation/point_in_time.py`,
`ingestion/normalize.py`, `backtest/engine.py`, `portfolio/expected_returns.py`,
`models/regime_detector.py`.

## Working style
- One bounded task per session: plan → implement → `make check` → show the diff → stop.
- Pure functions for features; register every feature in its family's `specs()` with a
  formula and rationale.
- Polars for data processing; pandas only at library boundaries.
- If a methodological choice is not specified in `PRD.md` or `configs/`, ask the owner.

## Commands
```
make check        # ruff + mypy + tests
make demo         # synthetic end-to-end run (no API keys)
make data         # ingest + curate + data-quality report
make features     # feature matrix + labels
make train H="hypothesis text"
make backtest | make report | make app | make api
```
