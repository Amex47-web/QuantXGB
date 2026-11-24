from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import polars as pl
import streamlit as st

from common import footer, setup

store, run = setup("Experiments")
rows = []
for r in store.runs():
    m = store.json(r, "manifest.json") or {}
    met = store.json(r, "metrics.json") or {}
    row = {
        "run_id": r,
        "created": m.get("created"),
        "hypothesis": m.get("hypothesis"),
        "dropped families": ", ".join(m.get("drop_families") or []),
        "shuffled labels": m.get("shuffle_labels", False),
    }
    for k, v in met.get("models", {}).items():
        row[f"IC {k}"] = v.get("ic_mean")
    rows.append(row)
st.dataframe(pl.DataFrame(rows), hide_index=True, width="stretch")
st.caption(
    "Every run is listed, including failures and sanity runs. Feature-family ablations are runs "
    "trained with `--drop-family <family>`; compare their IC with the full-feature run."
)

trials = store.root / "tracking" / "trials.jsonl"
if trials.exists():
    t = pl.read_ndjson(trials, infer_schema_length=None)
    st.subheader("Trial log")
    st.metric(
        "Strategy configurations backtested (used by DSR)",
        t.filter((pl.col("kind") == "strategy") & ~pl.col("is_baseline").fill_null(False)).height,
    )
    st.dataframe(t.group_by("kind").len(), hide_index=True)
footer()
