"""FR-PIT-05: labels use only prices from t+delay to t+delay+h, aligned to execution."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from quantxgb.labels.forward_returns import attach_labels, daily_labels


def _toy_prices() -> pl.DataFrame:
    dates = [dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(30)]
    tr = np.cumprod(np.r_[1.0, 1 + np.linspace(0.001, 0.03, 29)])
    return pl.DataFrame({"ticker": ["X"] * 30, "date": dates, "tr_index": tr})


@pytest.mark.parametrize(("h", "d"), [(5, 1), (3, 2), (1, 1)])
def test_label_window_alignment(h: int, d: int) -> None:
    p = _toy_prices()
    lab = daily_labels(p, horizon=h, delay=d)
    tr = p["tr_index"].to_numpy()
    dates = p["date"].to_list()
    for i in range(len(dates) - d - h):
        row = lab.row(i, named=True)
        assert row["label_start"] == dates[i + d]
        assert row["label_end"] == dates[i + d + h]
        assert row["label_start"] > row["date"], "label must start strictly after decision"
        assert np.isclose(row["fwd_ret"], tr[i + d + h] / tr[i + d] - 1)
    # rows whose window exceeds the data are null, never filled
    tail = lab.tail(d + h)
    assert tail["fwd_ret"].null_count() == d + h


def test_label_ignores_prices_up_to_execution() -> None:
    """Perturbing TR at <= t+delay-1 (i.e. before execution) must not change the label."""
    p = _toy_prices()
    base = daily_labels(p, horizon=5, delay=1)
    # multiply everything up to and including t=10 by a constant jump at t=10 only
    p2 = p.with_columns(
        pl.when(pl.int_range(pl.len()) == 10)
        .then(pl.col("tr_index") * 1.5)
        .otherwise(pl.col("tr_index"))
        .alias("tr_index")
    )
    lab2 = daily_labels(p2, horizon=5, delay=1)
    # label for decision t=10 uses TR[11]..TR[16] only → unchanged
    assert base.row(10, named=True)["fwd_ret"] == pytest.approx(lab2.row(10, named=True)["fwd_ret"])


def test_label_gap_is_null() -> None:
    p = _toy_prices().with_columns(
        pl.when(pl.int_range(pl.len()) == 12)
        .then(None)
        .otherwise(pl.col("tr_index"))
        .alias("tr_index")
    )
    lab = daily_labels(p, horizon=5, delay=1)
    # decision dates whose start or end hits the gap get null labels
    assert lab.row(11, named=True)["fwd_ret"] is None  # start = idx 12
    assert lab.row(6, named=True)["fwd_ret"] is None  # end = idx 12


def test_grades_and_excess() -> None:
    d = dt.date(2020, 1, 3)
    matrix = pl.DataFrame({"date": [d] * 10, "ticker": [f"T{i}" for i in range(10)]})
    labels = pl.DataFrame(
        {
            "date": [d] * 10,
            "ticker": [f"T{i}" for i in range(10)],
            "fwd_ret": [0.01 * i for i in range(9)] + [None],
            "label_start": [d] * 10,
            "label_end": [d] * 10,
        }
    )
    out = attach_labels(matrix, labels, n_grades=3).sort("ticker")
    xs = out["fwd_xs_ret"].to_list()
    assert xs[-1] is None
    assert np.isclose(sum(x for x in xs if x is not None), 0.0)
    grades = out["fwd_grade"].to_list()
    assert grades[:9] == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert grades[9] is None
