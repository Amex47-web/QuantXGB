# Experiments log — OWNER DOCUMENT

Rules (PRD §9.6):

1. **Pre-register** each experiment below *before* running it; pass the same text to
   `quantxgb train --hypothesis "..."`.
2. Record the run id and the headline numbers from `artifacts/runs/<id>/report.md` —
   including failed and disappointing runs.
3. Do not re-run a configuration silently: every run is counted in
   `artifacts/tracking/trials.jsonl` and raises the Deflated Sharpe bar.
4. Tag `v-methodology-freeze` before `quantxgb holdout`, which can be run once.

Suggested first runs: (1) `make sanity` (shuffled labels, IC must be ≈ 0); (2) the
baseline configuration; (3) one ablation per feature family (`--drop-family`).

---

## Template

### E-001 — <title>
- **Date / run id:**
- **Hypothesis (pre-registered):**
- **Change vs. previous run:**
- **Expected outcome and kill criterion:**
- **Result (copy numbers from report.md):**
- **Interpretation:**
- **Decision:**
