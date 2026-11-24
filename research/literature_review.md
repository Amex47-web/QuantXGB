# Literature review — OWNER DOCUMENT

Write your own summaries after reading the sources (PRD Appendix A). Claude can explain
mathematics or reproduce results on request, but these notes should be yours.

| Source | Key idea | How it is used in QuantXGB | Your notes |
|---|---|---|---|
| Bailey, Borwein, López de Prado & Zhu — *The Probability of Backtest Overfitting* | CSCV / PBO | `evaluation/stats.py::pbo_cscv` | |
| Bailey & López de Prado — *The Deflated Sharpe Ratio* | Correcting SR for multiple trials | `evaluation/stats.py::deflated_sharpe_ratio` | |
| López de Prado — *Advances in Financial ML* (ch. 7) | Purging and embargo | `validation/splits.py` | |
| Grinold & Kahn — *Active Portfolio Management* | IC, alpha = IC·σ·z | `portfolio/expected_returns.py` | |
| Rockafellar & Uryasev — *Optimization of CVaR* | CVaR as an LP | `portfolio/optimizer.py` | |
| Ledoit & Wolf — *Honey, I Shrunk the Sample Covariance Matrix* | Shrinkage | `portfolio/covariance.py` | |
| Politis & Romano (1994) — *The Stationary Bootstrap* | Block bootstrap CIs | `evaluation/stats.py` | |
| Lundberg & Lee — *A Unified Approach to Interpreting Model Predictions* | SHAP | `models/alpha_ranker.py::contributions` | |
| XGBoost docs — learning to rank, quantile regression | Objectives | `models/` | |
| FRED/ALFRED docs — real-time periods | Vintages | `ingestion/fred_provider.py` | |
