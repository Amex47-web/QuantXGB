from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import plotly.graph_objects as go
import polars as pl
import streamlit as st

from common import footer, setup

store, run = setup("Risk forecasts")
q_all = store.frame(run, "quantiles.parquet")
dates = store.prediction_dates(run)
live = store.latest_live(run)
options = ([live[0]] if live else []) + dates[::-1]
if not options:
    st.stop()
date = st.selectbox("Decision date", options)
q = store.quantiles(run, date)
qcols = sorted(c for c in q.columns if c.startswith("q") and c[1:].isdigit())
if q.height and len(qcols) >= 3:
    lo, mid, hi = qcols[0], qcols[len(qcols) // 2], qcols[-1]
    q = q.sort(mid, descending=True)
    fig = go.Figure()
    fig.add_bar(
        x=q["ticker"].to_list(),
        y=(q[hi] - q[lo]).to_list(),
        base=q[lo].to_list(),
        name=f"{lo}–{hi} interval",
        opacity=0.4,
    )
    fig.add_scatter(
        x=q["ticker"].to_list(), y=q[mid].to_list(), mode="markers", name=f"{mid} (median)"
    )
    fig.update_layout(yaxis_tickformat=".1%", title="Forward-return distribution forecast")
    st.plotly_chart(fig, width="stretch")

st.subheader("Calibration history")
metrics = store.json(run, "metrics.json") or {}
qm = metrics.get("quantile", {})
c1, c2, c3 = st.columns(3)
c1.metric(
    "Coverage (nominal)",
    f"{qm.get('coverage', float('nan')):.1%}",
    f"nominal {qm.get('nominal_coverage', float('nan')):.0%}",
    delta_color="off",
)
c2.metric(
    "Pinball 5% (model / baseline)",
    f"{qm.get('pinball_0.05', float('nan')):.5f}",
    f"{qm.get('baseline_pinball_0.05', float('nan')):.5f}",
    delta_color="off",
)
c3.metric(
    "Interval width (model / baseline)",
    f"{qm.get('interval_width', float('nan')):.3f}",
    f"{qm.get('baseline_interval_width', float('nan')):.3f}",
    delta_color="off",
)
if q_all is not None and len(qcols) >= 3:
    lab = q_all.filter(pl.col("fwd_ret").is_not_null()).with_columns(
        ((pl.col("fwd_ret") >= pl.col(qcols[0])) & (pl.col("fwd_ret") <= pl.col(qcols[-1]))).alias(
            "in"
        )
    )
    cov = (
        lab.group_by("date")
        .agg(pl.col("in").mean())
        .sort("date")
        .with_columns(pl.col("in").rolling_mean(26, min_samples=8).alias("rolling"))
    )
    fig = go.Figure()
    fig.add_scatter(x=cov["date"].to_list(), y=cov["rolling"].to_list(), name="rolling coverage")
    fig.add_hline(y=qm.get("nominal_coverage", 0.9), line_dash="dot")
    fig.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig, width="stretch")
if qm.get("coverage_by_regime"):
    st.dataframe(
        pl.DataFrame(
            {
                "regime": list(qm["coverage_by_regime"]),
                "coverage": list(qm["coverage_by_regime"].values()),
            }
        ),
        hide_index=True,
    )
footer()
