from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import plotly.express as px
import polars as pl
import streamlit as st

from common import footer, setup

store, run = setup("Portfolio")
w = store.frame(run, "backtest/weights.parquet")
if w is None:
    st.info("No backtest for this run — `quantxgb backtest`.")
    st.stop()
strategies = sorted(w["strategy"].unique().to_list())
default = next((i for i, s in enumerate(strategies) if s.startswith("opt_xgb")), 0)
strategy = st.selectbox("Strategy", strategies, index=default)
ws = w.filter(pl.col("strategy") == strategy)
dates = sorted(ws["exec_date"].unique().to_list())
d = st.select_slider("Execution date", dates, value=dates[-1])
cur = ws.filter(pl.col("exec_date") == d).sort("weight", descending=True)

manifest = store.json(run, "manifest.json") or {}
universe = manifest.get("config", {}).get("data", {}).get("universe", [])
classes = {u["ticker"]: u["asset_class"] for u in universe}
cons = manifest.get("config", {}).get("backtest", {}).get("constraints", {})
cur = cur.with_columns(pl.col("ticker").replace_strict(classes, default="?").alias("asset_class"))

c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(
        px.bar(
            cur.to_pandas(),
            x="ticker",
            y="weight",
            color="asset_class",
            title=f"Weights executed {d}",
        ),
        width="stretch",
    )
with c2:
    by_class = (
        cur.group_by("asset_class").agg(pl.col("weight").sum()).sort("weight", descending=True)
    )
    st.dataframe(by_class, hide_index=True, width="stretch")
    st.caption(
        f"Constraints: w_max={cons.get('w_max')}, asset-class cap={cons.get('asset_class_max')}, "
        f"turnover cap={cons.get('turnover_max')} per rebalance (long-only={cons.get('long_only')})."
    )
    st.metric("Largest position", f"{cur['weight'].max():.1%}")

st.subheader("Allocation through time")
st.plotly_chart(px.area(ws.to_pandas(), x="exec_date", y="weight", color="ticker"), width="stretch")
footer()
