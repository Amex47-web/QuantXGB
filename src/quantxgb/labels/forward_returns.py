"""Forward-return labels (PRD §4.4, §8.3) — a critical module for owner review.

With execution delay ``d`` (default 1: trade at the close of t+1) and horizon ``h``::

    fwd_ret_h(t)    = TR(t+d+h) / TR(t+d) − 1
    label_start(t)  = session t+d        (first close whose price enters the label)
    label_end(t)    = session t+d+h      (last close whose price enters the label)

Sessions are counted on each ticker's master-calendar grid. Labels are null whenever
either end falls beyond the data or on a session without a bar — they are never filled.

At decision dates, among eligible assets with a label:

    fwd_xs_ret = fwd_ret − mean(fwd_ret)                        (equal-weight cross-section)
    fwd_grade  = floor(n_grades · (rank − 1) / N)  ∈ {0..n_grades−1}, rank ascending, ties → ordinal
"""

from __future__ import annotations

import polars as pl


def daily_labels(prices: pl.DataFrame, horizon: int, delay: int) -> pl.DataFrame:
    """Per (ticker, date): fwd_ret, label_start, label_end."""
    tr = pl.col("tr_index")
    p = prices.select("ticker", "date", "tr_index").sort(["ticker", "date"])
    return p.with_columns(
        (tr.shift(-(delay + horizon)) / tr.shift(-delay) - 1.0).over("ticker").alias("fwd_ret"),
        pl.col("date").shift(-delay).over("ticker").alias("label_start"),
        pl.col("date").shift(-(delay + horizon)).over("ticker").alias("label_end"),
    ).drop("tr_index")


def attach_labels(matrix: pl.DataFrame, labels: pl.DataFrame, n_grades: int) -> pl.DataFrame:
    """Join labels onto the (eligible-only) feature matrix and add cross-sectional targets."""
    df = matrix.join(labels, on=["ticker", "date"], how="left")
    has = pl.col("fwd_ret").is_not_null()
    n = has.sum().over("date")
    xs_mean = pl.col("fwd_ret").mean().over("date")  # mean ignores nulls
    rank = pl.col("fwd_ret").rank("ordinal").over("date")
    df = df.with_columns(
        pl.when(has).then(pl.col("fwd_ret") - xs_mean).alias("fwd_xs_ret"),
        pl.when(has).then(((rank - 1) * n_grades / n).floor().cast(pl.Int32)).alias("fwd_grade"),
    )
    return df
