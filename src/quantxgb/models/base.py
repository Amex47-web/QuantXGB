"""Shared model helpers."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import polars as pl


class Scorer(Protocol):
    name: str

    def fit(
        self, train: pl.DataFrame, features: list[str], valid: pl.DataFrame | None = None
    ) -> Scorer: ...

    def predict(self, df: pl.DataFrame) -> np.ndarray: ...


def to_matrix(df: pl.DataFrame, features: list[str]) -> np.ndarray:
    """Float64 design matrix with NaN for missing values (XGBoost's missing marker)."""
    return df.select([pl.col(c).cast(pl.Float64) for c in features]).fill_null(np.nan).to_numpy()


def query_ids(df: pl.DataFrame) -> np.ndarray:
    """Integer query id per row: one ranking group per decision date. ``df`` must be
    sorted by date."""
    d = df["date"].to_numpy().astype("datetime64[D]").astype(np.int64)
    if np.any(np.diff(d) < 0):
        raise ValueError("rows must be sorted by date for ranking groups")
    return d


def xs_zscore(df: pl.DataFrame, features: list[str]) -> pl.DataFrame:
    """Within-date z-score of each feature; missing → 0 (cross-sectional mean)."""
    exprs = []
    for c in features:
        x = pl.col(c).cast(pl.Float64)
        sd = x.std().over("date")
        z = pl.when(sd > 0).then((x - x.mean().over("date")) / sd).otherwise(0.0)
        exprs.append(z.fill_null(0.0).fill_nan(0.0).alias(c))
    return df.with_columns(exprs)
