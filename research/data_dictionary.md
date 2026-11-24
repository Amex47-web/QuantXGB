# Data dictionary

## `data/curated/prices.parquet` — one row per (ticker, session) from first to last bar

| Column | Type | Meaning |
|---|---|---|
| ticker, asset_class | str | from `configs/data.yaml` |
| date | date | NYSE session |
| open, high, low, close | f64 | Yahoo prices, **split-adjusted** (null on sessions without a bar) |
| adj_close | f64 | Yahoo split+distribution-adjusted close (reconciliation only) |
| volume | f64 | split-adjusted shares |
| dividends, capital_gains | f64 | cash per share on the ex-date (same basis as close) |
| splits | f64 | split ratio on the split date (0 otherwise) |
| ret | f64 | in-house daily total return vs. the previous valid close |
| tr_index | f64 | total-return index (1.0 on first bar) |
| adj_ret | f64 | return implied by `adj_close` |
| available_on | date | first session at whose close the bar may be used |
| source, snapshot_date, snapshot_sha256 | str | provenance |

## `data/curated/macro_pit.parquet` — bitemporal, long

| Column | Type | Meaning |
|---|---|---|
| series_id | str | FRED id |
| observed_on | date | period the value describes |
| value | f64 | value (FRED "." rows dropped) |
| realtime_start, realtime_end | date | ALFRED vintage validity (null for non-revised series) |
| revised | bool | whether vintages were downloaded |
| available_on | date | first session at whose close the value may be used |
| source, snapshot_date, snapshot_sha256 | str | provenance |

## `data/features/<config-hash>/dataset.parquet` — one row per (decision date, eligible ticker)

`date, ticker, asset_class`, every model feature (see `feature_dictionary.md`), and labels:

| Column | Meaning |
|---|---|
| fwd_ret | `TR(t+1+h)/TR(t+1) − 1` |
| fwd_xs_ret | fwd_ret minus the cross-sectional mean on t |
| fwd_grade | within-date quintile 0–4 |
| label_start, label_end | first / last session whose close enters the label |

## Run artefacts — `artifacts/runs/<run_id>/`

| File | Content |
|---|---|
| manifest.json | config snapshot + hashes, git sha, hypothesis |
| predictions.parquet | OOS scores of every model (long) + labels + fold |
| quantiles.parquet | OOS q05/q50/q95 |
| regimes.parquet | rule regimes + filtered HMM probabilities per test date |
| shap_oos.parquet | TreeSHAP of the fold model behind each OOS prediction |
| feature_importance.parquet | mean \|SHAP\| per feature per fold |
| metrics.json, fold_metrics.json, folds.json | evaluation |
| models/final.pkl | models fitted on the full development period (live scoring only) |
| backtest/ | daily.parquet, weights.parquet, metrics.json |
| live/<date>/ | scores, quantiles, regimes, shap for `quantxgb score` |
| report.md | numbers-only report |
