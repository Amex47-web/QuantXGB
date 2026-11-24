"""From ordinal scores to expected returns (PRD §10.2, FR-PORT-02) — critical module.

Grinold's rule:  μ_i = IC̄ · σ_i,h · z_i

* z_i  : within-date z-score of the model score;
* σ_i,h: forecast volatility over the horizon, ``vol_ann · √(h/252)``;
* IC̄   : mean rank IC of this model's PAST out-of-sample predictions whose labels were
         fully realised by the decision date (label_end ≤ t). Negative IC̄ is clipped to
         0, and with fewer than ``min_history`` realised dates μ = 0 (the optimizer then
         simply minimises risk + turnover). No information from after t is used.
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import polars as pl


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    ok = ~np.isnan(x)
    out = np.zeros_like(x)
    if ok.sum() < 2:
        return out
    sd = x[ok].std(ddof=0)
    if sd > 0:
        out[ok] = (x[ok] - x[ok].mean()) / sd
    return out


def trailing_ic(
    ic_by_date: pl.DataFrame, label_end_by_date: pl.DataFrame, t: dt.date, min_history: int
) -> float:
    """Mean IC over OOS dates whose labels were realised by ``t``; NaN if too few."""
    known = label_end_by_date.filter(pl.col("label_end") <= t).select("date")
    ic = ic_by_date.join(known, on="date", how="inner")
    if ic.height < min_history:
        return float("nan")
    return float(ic["ic"].mean())  # type: ignore[arg-type]


def grinold_mu(scores: np.ndarray, vol_ann: np.ndarray, ic_bar: float, horizon: int) -> np.ndarray:
    if math.isnan(ic_bar) or ic_bar <= 0:
        return np.zeros(len(scores))
    sigma_h = np.nan_to_num(np.asarray(vol_ann, dtype=float)) * math.sqrt(horizon / 252.0)
    return ic_bar * sigma_h * zscore(scores)
