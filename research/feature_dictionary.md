# Feature dictionary

_Generated from `quantxgb.features` — do not edit by hand. Regenerate with `quantxgb features dictionary`._

Notation: `TR` total-return index, `r` daily log return of TR, `σ_k` annualised std of `r` over k sessions, `M` market proxy, all windows trailing and inclusive of t.

## momentum

| Feature | Formula | Lookback | Scope | Rationale |
|---|---|---|---|---|
| `ret_5` | TR_t / TR_(t-5) − 1 | 5 | asset | Time-series momentum / short-term reversal over the window |
| `ret_10` | TR_t / TR_(t-10) − 1 | 10 | asset | Time-series momentum / short-term reversal over the window |
| `ret_20` | TR_t / TR_(t-20) − 1 | 20 | asset | Time-series momentum / short-term reversal over the window |
| `ret_60` | TR_t / TR_(t-60) − 1 | 60 | asset | Time-series momentum / short-term reversal over the window |
| `ret_120` | TR_t / TR_(t-120) − 1 | 120 | asset | Time-series momentum / short-term reversal over the window |
| `ret_252` | TR_t / TR_(t-252) − 1 | 252 | asset | Time-series momentum / short-term reversal over the window |
| `mom_12_1` | TR_(t-21) / TR_(t-252) − 1 | 252 | asset | Classic 12-1 momentum, skipping the most recent month to avoid short-term reversal |
| `ret_20_voladj` | ret_20 / (vol_60 · √(20/252)) | 60 | asset | Risk-adjusted momentum: return per unit of volatility (comparable across classes) |
| `ret_60_voladj` | ret_60 / (vol_60 · √(60/252)) | 60 | asset | Risk-adjusted momentum: return per unit of volatility (comparable across classes) |
| `ret_120_voladj` | ret_120 / (vol_60 · √(120/252)) | 120 | asset | Risk-adjusted momentum: return per unit of volatility (comparable across classes) |
| `ret_252_voladj` | ret_252 / (vol_60 · √(252/252)) | 252 | asset | Risk-adjusted momentum: return per unit of volatility (comparable across classes) |
| `ret_20_xsrank` | percentile rank of ret_20 among eligible assets on date t | 0 | cross_sectional | Relative strength — the quantity a cross-sectional ranker ultimately orders |
| `ret_60_xsrank` | percentile rank of ret_60 among eligible assets on date t | 0 | cross_sectional | Relative strength — the quantity a cross-sectional ranker ultimately orders |
| `ret_120_xsrank` | percentile rank of ret_120 among eligible assets on date t | 0 | cross_sectional | Relative strength — the quantity a cross-sectional ranker ultimately orders |
| `ret_252_xsrank` | percentile rank of ret_252 among eligible assets on date t | 0 | cross_sectional | Relative strength — the quantity a cross-sectional ranker ultimately orders |
| `mom_12_1_xsrank` | percentile rank of mom_12_1 among eligible assets on date t | 0 | cross_sectional | Relative strength — the quantity a cross-sectional ranker ultimately orders |

## volatility

| Feature | Formula | Lookback | Scope | Rationale |
|---|---|---|---|---|
| `vol_20` | std(r over 20) · √252 | 20 | asset | Risk level; low-volatility effect and volatility clustering |
| `vol_60` | std(r over 60) · √252 | 60 | asset | Risk level; low-volatility effect and volatility clustering |
| `vol_120` | std(r over 120) · √252 | 120 | asset | Risk level; low-volatility effect and volatility clustering |
| `downvol_60` | √mean(min(r,0)²) over 60 · √252 (semi-deviation) | 60 | asset | Downside risk, which investors price differently from upside variance |
| `vol_ratio_20_120` | vol_20 / vol_120 | 120 | asset | Volatility regime shift for the asset (rising vs. falling risk) |
| `parkinson_20` | √(mean(ln(H/L)²) / (4 ln 2)) over 20 · √252 | 20 | asset | Range-based volatility, more efficient than close-to-close |

## trend

| Feature | Formula | Lookback | Scope | Rationale |
|---|---|---|---|---|
| `px_ema_20` | TR_t / EMA_20(TR)_t − 1 | 20 | asset | Distance from trend; trend-following signal |
| `px_ema_50` | TR_t / EMA_50(TR)_t − 1 | 50 | asset | Distance from trend; trend-following signal |
| `px_ema_200` | TR_t / EMA_200(TR)_t − 1 | 200 | asset | Distance from trend; trend-following signal |
| `ema_20_50` | EMA_20(TR) / EMA_50(TR) − 1 | 50 | asset | Moving-average crossover state |
| `slope_60` | 252·β̂ / vol_60, β̂ = OLS slope of ln TR on session index over 60 | 60 | asset | Trend strength normalised by risk (t-stat-like) |
| `up_frac_20` | share of sessions with r > 0 over 20 | 20 | asset | Trend consistency (path smoothness) |

## liquidity

| Feature | Formula | Lookback | Scope | Rationale |
|---|---|---|---|---|
| `log_dvol_20` | ln(mean(close · volume) over 20) | 20 | asset | Tradability / size of the market for the asset |
| `volume_z_60` | (ln(1+V_t) − mean_60) / std_60 of ln(1+V) | 60 | asset | Abnormal volume: attention and flow pressure |
| `amihud_20` | mean(|ret| / (close · volume)) over 20 · 1e9 | 20 | asset | Amihud illiquidity: price impact per dollar traded |

## risk

| Feature | Formula | Lookback | Scope | Rationale |
|---|---|---|---|---|
| `beta_mkt_60` | cov(r, r_M) / var(r_M) over 60 | 60 | asset | Systematic exposure to the market proxy |
| `beta_mkt_252` | cov(r, r_M) / var(r_M) over 252 | 252 | asset | Systematic exposure to the market proxy |
| `dd_252` | TR_t / max(TR over 252) − 1 | 252 | asset | Current drawdown state (distance from the 1-year high) |
| `mdd_120` | min over window of TR / running max(TR) − 1, window 120 | 120 | asset | Recent tail behaviour / crash history |
| `skew_60` | sample skewness of r over 60 | 60 | asset | Asymmetry of returns (lottery / crash risk) |
| `kurt_60` | sample excess kurtosis of r over 60 | 60 | asset | Fat-tailedness of recent returns |

## cross_asset

| Feature | Formula | Lookback | Scope | Rationale |
|---|---|---|---|---|
| `rel_mkt_20` | ret_20 − ret_20(M) | 20 | asset | Performance relative to the equity market proxy |
| `rel_mkt_60` | ret_60 − ret_60(M) | 60 | asset | Performance relative to the equity market proxy |
| `rel_mkt_120` | ret_120 − ret_120(M) | 120 | asset | Performance relative to the equity market proxy |
| `rel_class_20` | ret_20 − leave-one-out mean ret_20 of same asset class on t | 20 | cross_sectional | Within-asset-class relative strength (null for single-asset classes) |
| `rel_class_60` | ret_60 − leave-one-out mean ret_60 of same asset class on t | 60 | cross_sectional | Within-asset-class relative strength (null for single-asset classes) |
| `corr_mkt_60` | corr(r, r_M) over 60 | 60 | asset | Diversification role vs. equities |
| `corr_rates_60` | corr(r, r_rates proxy) over 60 | 60 | asset | Diversification role vs. duration |

## market

| Feature | Formula | Lookback | Scope | Rationale |
|---|---|---|---|---|
| `market_ret_1d` | r_M,t | 1 | market | Market daily return (HMM regime input) |
| `market_vol_20` | std(r_M over 20) · √252 | 20 | market | Market-wide volatility regime |
| `market_above_sma200` | 1[TR_M > SMA_200(TR_M)] | 200 | market | Market trend regime |
| `avg_corr_60` | mean pairwise corr of universe r over 60 | 60 | market | Diversification breakdown / risk-on-risk-off intensity |

## macro

| Feature | Formula | Lookback | Scope | Rationale |
|---|---|---|---|---|
| `y10` | DGS10: x_t (obs = series observations, vintage-consistent) | 0 | macro | Macro state from DGS10 |
| `y10_chg_20` | DGS10: x_t − x_(t−20) (obs = series observations, vintage-consistent) | 20 | macro | Macro state from DGS10 |
| `slope_10_2` | T10Y2Y: x_t (obs = series observations, vintage-consistent) | 0 | macro | Macro state from T10Y2Y |
| `slope_10_3m` | T10Y3M: x_t (obs = series observations, vintage-consistent) | 0 | macro | Macro state from T10Y3M |
| `credit_spread` | BAA10Y: x_t (obs = series observations, vintage-consistent) | 0 | macro | Macro state from BAA10Y |
| `credit_chg_20` | BAA10Y: x_t − x_(t−20) (obs = series observations, vintage-consistent) | 20 | macro | Macro state from BAA10Y |
| `vix` | VIXCLS: x_t (obs = series observations, vintage-consistent) | 0 | macro | Macro state from VIXCLS |
| `vix_pct_252` | VIXCLS: share of last 252 obs ≤ x_t (obs = series observations, vintage-consistent) | 252 | macro | Macro state from VIXCLS |
| `fed_funds` | DFF: x_t (obs = series observations, vintage-consistent) | 0 | macro | Macro state from DFF |
| `cpi_yoy` | CPIAUCSL: x_t / x_(t−12) − 1 (obs = series observations, vintage-consistent) | 12 | macro | Macro state from CPIAUCSL |
| `unrate_chg_3m` | UNRATE: x_t − x_(t−3) (obs = series observations, vintage-consistent) | 3 | macro | Macro state from UNRATE |
| `indpro_yoy` | INDPRO: x_t / x_(t−12) − 1 (obs = series observations, vintage-consistent) | 12 | macro | Macro state from INDPRO |
| `nfci` | NFCI: x_t (obs = series observations, vintage-consistent) | 0 | macro | Macro state from NFCI |
