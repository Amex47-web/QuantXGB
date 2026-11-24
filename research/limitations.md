# Known limitations

Technical limitations known from the implementation. The owner should extend this with
findings from data verification and experiments.

## Data and licensing
- **yfinance is unofficial**; Yahoo data is for personal research use. Downloaded data must
  not be redistributed, so it is git-ignored and every user downloads their own.
- Yahoo can **silently revise history**. Raw snapshots are immutable and dated, so a
  revision shows up as a difference between snapshots, but the code does not diff them
  automatically.
- Yahoo provides no truly unadjusted close: `Close` is split-adjusted. Distributions are
  assumed to be on the same split-adjusted basis (verify on real split events).
- **FRED attribution** is required; some FRED series carry third-party copyright. Series
  `notes` are saved to `data/raw/fred/series_metadata.json` for review.
- Non-revised FRED series (yields, VIX, spreads) are assumed never to be revised; only the
  latest values are downloaded for them.
- ALFRED vintages begin at different dates per series. Observations published before the
  first vintage become "available" only from that first vintage (see the DQ report).

## Universe and bias
- Static 20-ETF universe chosen with hindsight: all of these funds survived, so some
  **survivorship / selection bias** remains. Results do not generalise to "the market".
- Unbalanced panel: assets enter after 252 sessions of history, so early cross-sections are
  smaller (HYG limits the full universe to ~2008-04 onwards).

## Methodology
- Costs are fixed one-way bps on traded notional (5/10/20). Daily OHLCV cannot support a
  spread or market-impact model.
- Execution at the close of t+1 assumes the closing price can be achieved.
- Cash earns DGS3MO (PIT) when available, otherwise 0.
- Grinold μ uses IC̄ from past OOS predictions; with fewer than 26 realised dates μ = 0
  and the optimiser only minimises risk + turnover.
- HMM states are ordered by mean volatility; giving them economic labels is an
  interpretation.
- Market-wide (macro / regime) features are constant within a ranking group and can only
  act through interactions with asset-level features.
- PBO is computed across the strategies of one run (models and baselines), not across
  every configuration ever tried; DSR uses the full trial log.
- SHAP values explain the ranking score (ordinal units) and are not causal.
- **Docker image build not verified** in the development environment (daemon not running).
