"""Macro features (PRD §8.2) — vintage-consistent, as-of joined (see validation.point_in_time)."""

from __future__ import annotations

import datetime as dt

import polars as pl

from quantxgb.config import FeaturesConfig, MacroFeatureSpec
from quantxgb.features.registry import FeatureSpec
from quantxgb.validation.point_in_time import asof_join_macro, macro_feature_table

_FORMULAS = {
    "level": "x_t",
    "diff": "x_t − x_(t−{k})",
    "pct_change": "x_t / x_(t−{k}) − 1",
    "pct_rank": "share of last {k} obs ≤ x_t",
    "zscore": "(x_t − mean_{k}) / std_{k}",
}


def _formula(s: MacroFeatureSpec) -> str:
    return f"{s.series}: " + _FORMULAS[s.transform].format(k=s.periods)


def specs(cfg: FeaturesConfig) -> list[FeatureSpec]:
    return [
        FeatureSpec(
            s.name,
            "macro",
            _formula(s) + " (obs = series observations, vintage-consistent)",
            s.periods or 0,
            f"Macro state from {s.series}",
            scope="macro",
            availability="as-of join on available_on (ALFRED vintage or obs + lag)",
        )
        for s in cfg.macro
    ]


def macro_frame(
    macro: pl.DataFrame | None,
    cfg: FeaturesConfig,
    decision_dates: list[dt.date],
    sessions: list[dt.date],
    revised_lag_sessions: int,
) -> pl.DataFrame:
    base = pl.DataFrame({"date": decision_dates}, schema={"date": pl.Date})
    if macro is None or macro.height == 0 or not cfg.macro:
        return base
    table = macro_feature_table(macro, cfg.macro, sessions, revised_lag_sessions)
    return asof_join_macro(decision_dates, table)
