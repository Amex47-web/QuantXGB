"""QuantXGB research dashboard — overview page.

uv run streamlit run app/Home.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import polars as pl
import streamlit as st

from common import footer, setup

store, run = setup("Overview")
manifest = store.json(run, "manifest.json") or {}
metrics = store.json(run, "metrics.json") or {}

c1, c2, c3, c4 = st.columns(4)
c1.metric("Run", run or "-")
date, ranks = store.rankings(run, None)
c2.metric("Latest scored date", str(date))
reg = store.regimes(run)
if reg.height:
    last = reg.filter(pl.col("date") <= date).tail(1).to_dicts()
    c3.metric("Regime", (last[0].get("regime") if last else None) or "n/a")
xgb = next((v for k, v in metrics.get("models", {}).items() if k.startswith("xgb")), {})
c4.metric(
    "OOS rank IC (XGB)",
    f"{xgb.get('ic_mean', float('nan')):+.4f}",
    help="Mean Spearman IC over walk-forward test dates (development period)",
)

st.caption(f"Hypothesis: {manifest.get('hypothesis') or '—'}")

left, right = st.columns(2)
with left:
    st.subheader("Top-ranked assets (XGB ranker)")
    top = (
        ranks.filter(pl.col("model").str.starts_with("xgb")).sort("score", descending=True).head(5)
    )
    st.dataframe(top.select("ticker", "score"), hide_index=True, width="stretch")
with right:
    st.subheader("Latest model portfolio")
    w = store.frame(run, "backtest/weights.parquet")
    if w is not None and w.height:
        w = w.filter(pl.col("strategy").str.starts_with("opt_xgb"))
        last = w.filter(pl.col("exec_date") == w["exec_date"].max()).sort("weight", descending=True)
        st.caption(f"Executed {last['exec_date'][0]} (backtest, base cost)")
        st.dataframe(last.select("ticker", "weight"), hide_index=True, width="stretch")
    else:
        st.info("No backtest yet — run `quantxgb backtest`.")

st.subheader("Key questions")
st.markdown(
    "- **Rankings / Risk** — which assets rank highest and what are their 5%/50%/95% forecasts\n"
    "- **Regime** — current market state and how skill varies by regime\n"
    "- **Portfolio / Backtest** — allocation under constraints and out-of-sample performance after costs\n"
    "- **Explainability** — which features drove a prediction (attribution, not causation)\n"
    "- **Experiments** — every run, including failures; **Data health** — data-quality report"
)
footer()
