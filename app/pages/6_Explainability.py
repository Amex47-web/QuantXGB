from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import streamlit as st

from common import footer, setup

store, run = setup("Explainability")
st.info(
    "TreeSHAP attributions explain the *model's ranking score*. They are not causal effects, and "
    "correlated features share credit arbitrarily."
)

imp = store.frame(run, "feature_importance.parquet")
if imp is not None:
    g = (
        imp.group_by("feature")
        .agg(pl.col("mean_abs_shap").mean())
        .sort("mean_abs_shap", descending=True)
        .head(20)
    )
    st.subheader("Global importance (mean |SHAP| over OOS rows, averaged across folds)")
    st.plotly_chart(
        px.bar(
            g.sort("mean_abs_shap").to_pandas(), x="mean_abs_shap", y="feature", orientation="h"
        ),
        width="stretch",
    )
    by_fold = imp.pivot(on="fold", index="feature", values="mean_abs_shap")
    with st.expander("Importance stability across folds"):
        st.dataframe(by_fold, hide_index=True)

st.subheader("Local explanation")
dates = store.prediction_dates(run)
live = store.latest_live(run)
options = ([live[0]] if live else []) + dates[::-1]
date = st.selectbox("Date", options)
_, ranks = store.rankings(run, date)
tickers = sorted(ranks["ticker"].unique().to_list())
ticker = st.selectbox("Asset", tickers)
e = store.shap(run, date, ticker)
if e.height:
    top = e.head(15).reverse()
    fig = go.Figure(
        go.Waterfall(orientation="h", y=top["feature"].to_list(), x=top["contribution"].to_list())
    )
    fig.update_layout(title=f"{ticker} on {date}: top contributions to the ranking score")
    st.plotly_chart(fig, width="stretch")
else:
    st.warning("No SHAP values stored for this date/asset.")
footer()
