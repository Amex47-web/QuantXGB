"""Shared helpers for the Streamlit dashboard (reads artefacts only)."""

from __future__ import annotations

import os

import streamlit as st

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from quantxgb.serving.store import ArtifactStore

DISCLAIMER = (
    "Research only, not investment advice. Execution assumed at the close of t+1; costs are fixed "
    "bps stress assumptions, not measured spreads. ETF universe chosen with hindsight "
    "(residual survivorship bias)."
)


def setup(title: str) -> tuple[ArtifactStore, str | None]:
    st.set_page_config(page_title=f"QuantXGB · {title}", layout="wide")
    store = ArtifactStore()
    runs = store.runs()
    st.sidebar.markdown("### QuantXGB")
    st.sidebar.caption(f"Artefacts: `{store.root}`")
    if not runs:
        st.warning("No runs found. Run `quantxgb demo` or `quantxgb train` first.")
        st.stop()
    run = st.sidebar.selectbox("Run", runs[::-1], index=0)
    manifest = store.json(run, "manifest.json") or {}
    if manifest.get("shuffle_labels"):
        st.sidebar.error("Sanity run with shuffled labels")
    unfrozen = [k for k, v in manifest.get("config_status", {}).items() if v != "frozen"]
    if unfrozen:
        st.sidebar.warning("Configs not frozen — exploratory run")
    st.title(title)
    return store, run


def footer() -> None:
    st.divider()
    st.caption(DISCLAIMER)
