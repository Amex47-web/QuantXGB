from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import plotly.express as px
import polars as pl
import streamlit as st

from common import footer, setup
from quantxgb.evaluation.signal_metrics import rank_ic_by_date

store, run = setup("Rankings")
preds = store.frame(run, "predictions.parquet")
if preds is None:
    st.stop()
models = sorted(preds["model"].unique().to_list())
model = st.selectbox("Model", models, index=models.index("xgb_ndcg") if "xgb_ndcg" in models else 0)
dates = store.prediction_dates(run)
live = store.latest_live(run)
options = ([live[0]] if live else []) + dates[::-1]
date = st.selectbox(
    "Decision date",
    options,
    format_func=lambda d: f"{d}{' (live)' if live and d == live[0] else ''}",
)

_, df = store.rankings(run, date)
df = df.filter(pl.col("model") == model).sort("score", descending=True)
df = df.with_columns(pl.int_range(1, pl.len() + 1).alias("rank"))
cols = [c for c in ("rank", "ticker", "score", "fwd_xs_ret") if c in df.columns]
st.dataframe(df.select(cols), hide_index=True, width="stretch")
st.caption(
    "Scores are ordinal (within-date ordering), not return forecasts. "
    "`fwd_xs_ret` is the realised excess return (shown only for past OOS dates)."
)

st.subheader("Rank IC over time (out of sample)")
ic = rank_ic_by_date(preds.filter(pl.col("model") == model))
if ic.height:
    ic = ic.with_columns(pl.col("ic").rolling_mean(26, min_samples=8).alias("rolling_26"))
    fig = px.bar(ic.to_pandas(), x="date", y="ic", opacity=0.35)
    fig.add_scatter(x=ic["date"].to_list(), y=ic["rolling_26"].to_list(), name="26-date mean")
    fig.add_hline(y=0, line_dash="dot")
    st.plotly_chart(fig, width="stretch")
footer()
