from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import plotly.express as px
import polars as pl
import streamlit as st

from common import footer, setup

store, run = setup("Market regime")
reg = store.regimes(run)
if reg.height == 0:
    st.info("No regimes in this run.")
    st.stop()
last = reg.tail(1).to_dicts()[0]
st.metric(
    "Latest regime",
    last.get("regime") or "n/a",
    help="Rule-based: trend × volatility; thresholds fitted per training window",
)

st.subheader("Rule-based regime timeline")
fig = px.scatter(reg.drop_nulls("regime").to_pandas(), x="date", y="regime", color="regime")
st.plotly_chart(fig, width="stretch")

hmm_cols = [c for c in reg.columns if c.startswith("hmm_p_state_")]
if hmm_cols:
    st.subheader("HMM filtered state probabilities")
    st.caption(
        "Forward-filtered P(state | data up to t). State 0 = lowest mean volatility. "
        "Naming the states is the researcher's interpretation."
    )
    long = reg.select("date", *hmm_cols).unpivot(
        index="date", variable_name="state", value_name="p"
    )
    st.plotly_chart(px.area(long.to_pandas(), x="date", y="p", color="state"), width="stretch")

st.subheader("Out-of-sample IC by regime")
metrics = store.json(run, "metrics.json") or {}
rows = [
    {"model": m, "regime": r, "ic_mean": v["ic_mean"], "n_dates": v["n_dates"]}
    for m, d in metrics.get("by_regime", {}).items()
    for r, v in d.items()
]
if rows:
    st.dataframe(
        pl.DataFrame(rows).pivot(on="regime", index="model", values="ic_mean"), hide_index=True
    )
footer()
