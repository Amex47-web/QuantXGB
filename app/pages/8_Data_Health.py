from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import streamlit as st

from common import footer, setup

store, run = setup("Data health")
md = store.data_quality_markdown()
if md is None:
    st.info("No data-quality report yet — run `quantxgb data build`.")
else:
    st.markdown(md)
m = store.json(run, "manifest.json") or {}
st.subheader("Provenance of this run")
st.json({k: m.get(k) for k in ("data_hash", "dataset_hash", "git_sha", "config_hash")})
st.subheader("Known limitations")
st.markdown(
    "- Yahoo Finance via yfinance: unofficial, personal research use, history can be revised (raw snapshots are kept).\n"
    "- `Close` is split-adjusted by Yahoo; total return is computed in-house from close + distributions.\n"
    "- Non-revised FRED series are assumed unrevised; revised series use ALFRED vintages.\n"
    "- Static ETF universe chosen with hindsight; no delisted instruments.\n"
    "- Daily data cannot support a realistic execution-cost model; costs are stress assumptions."
)
footer()
