from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import plotly.express as px
import polars as pl
import streamlit as st

from common import footer, setup

store, run = setup("Backtest (out of sample)")
daily = store.frame(run, "backtest/daily.parquet")
bt = store.json(run, "backtest/metrics.json")
if daily is None or bt is None:
    st.info("No backtest for this run — `quantxgb backtest`.")
    st.stop()
costs = sorted(daily["cost_bps"].unique().to_list())
cost = st.radio(
    "Cost scenario (bps, one-way)",
    costs,
    index=costs.index(10.0) if 10.0 in costs else 0,
    horizontal=True,
)
strategies = sorted(daily["strategy"].unique().to_list())
chosen = st.multiselect("Strategies", strategies, default=strategies)
d = daily.filter((pl.col("cost_bps") == cost) & pl.col("strategy").is_in(chosen)).sort("date")
d = d.with_columns((pl.col("nav") / pl.col("nav").cum_max().over("strategy") - 1).alias("drawdown"))

st.plotly_chart(
    px.line(d.to_pandas(), x="date", y="nav", color="strategy", log_y=True, title="NAV (log)"),
    width="stretch",
)
st.plotly_chart(
    px.area(d.to_pandas(), x="date", y="drawdown", color="strategy", title="Drawdown"),
    width="stretch",
)
roll = d.with_columns(
    (pl.col("ret").rolling_mean(252) / pl.col("ret").rolling_std(252) * math.sqrt(252))
    .over("strategy")
    .alias("rolling_sharpe")
)
st.plotly_chart(
    px.line(
        roll.to_pandas(),
        x="date",
        y="rolling_sharpe",
        color="strategy",
        title="Rolling 1-year Sharpe",
    ),
    width="stretch",
)

st.subheader("Metrics")
key = f"{cost:g}"
rows = []
for name, s in bt["strategies"].items():
    p = s["by_cost"][key]
    rows.append(
        {
            "strategy": name,
            "baseline": s["is_baseline"],
            "CAGR": p.get("cagr"),
            "vol": p.get("ann_vol"),
            "Sharpe": p.get("sharpe"),
            "Sharpe CI low": p.get("sharpe_ci_low"),
            "Sharpe CI high": p.get("sharpe_ci_high"),
            "Sortino": p.get("sortino"),
            "MaxDD": p.get("max_drawdown"),
            "turnover/yr": p.get("turnover_annual"),
            "cost drag/yr": p.get("cost_drag_annual"),
            "IR vs EW": p.get("information_ratio"),
            "DSR": (s.get("deflated_sharpe") or {}).get("dsr"),
            "DSR active": (s.get("deflated_sharpe_active") or {}).get("dsr"),
        }
    )
st.dataframe(
    pl.DataFrame(rows).sort(["baseline", "Sharpe"], descending=[False, True]),
    hide_index=True,
    width="stretch",
)
pbo = bt.get("pbo", {})
st.metric(
    "Probability of Backtest Overfitting (CSCV)",
    f"{pbo.get('pbo', float('nan')):.2f}",
    help="Share of CSCV splits where the in-sample best strategy ranks at/below the OOS median",
)
st.caption(
    "Sharpe CIs: stationary block bootstrap at the base cost. DSR uses every strategy configuration "
    "ever logged in the trial log."
)
footer()
