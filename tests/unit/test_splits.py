from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from quantxgb.config import DateRange, SplitsConfig
from quantxgb.utils.calendar import business_day_sessions, weekly_decision_dates
from quantxgb.validation.splits import (
    HoldoutViolation,
    assert_no_holdout,
    development_view,
    inner_split,
    walk_forward_folds,
)

SESS = business_day_sessions(dt.date(2010, 1, 1), dt.date(2020, 12, 31))
DATES = weekly_decision_dates(SESS)
H = 20


def _frame() -> pl.DataFrame:
    idx = {d: i for i, d in enumerate(SESS)}
    rows = []
    for d in DATES:
        i = idx[d]
        if i + 1 + H < len(SESS):
            rows.append((d, "X", 0.01, 0.0, 1, SESS[i + 1], SESS[i + 1 + H]))
    return pl.DataFrame(
        rows,
        schema=["date", "ticker", "fwd_ret", "fwd_xs_ret", "fwd_grade", "label_start", "label_end"],
        orient="row",
    )


SPLITS = SplitsConfig(
    warmup_end=dt.date(2010, 12, 31),
    development=DateRange(start=dt.date(2011, 1, 1), end=dt.date(2018, 12, 31)),
    holdout=DateRange(start=dt.date(2019, 2, 1), end=dt.date(2020, 12, 31)),
)


@pytest.mark.parametrize("embargo", [0, 5, 10])
def test_purge_and_embargo(embargo: int) -> None:
    df = development_view(_frame(), SPLITS)
    folds = walk_forward_folds(
        df["date"].to_list(), SESS, SPLITS.development.start, SPLITS.development.end, 3, 6, embargo
    )
    assert len(folds) >= 8
    for f in folds:
        train = df.filter(f.train_mask())
        test = df.filter(f.test_mask())
        assert train["date"].max() < test["date"].min()
        last_label = train["label_end"].max()
        gap = SESS.index(f.test_start) - SESS.index(last_label)
        assert gap > embargo, "label window must end > embargo sessions before the test block"
        assert f.test_end <= SPLITS.development.end
    # expanding window
    sizes = [df.filter(f.train_mask()).height for f in folds]
    assert sizes == sorted(sizes)


def test_development_view_nulls_labels_reaching_holdout() -> None:
    df = development_view(_frame(), SPLITS)
    assert df["date"].max() <= SPLITS.development.end
    lab = df.filter(pl.col("fwd_ret").is_not_null())
    assert lab["label_end"].max() < SPLITS.holdout.start


def test_holdout_guard_raises() -> None:
    with pytest.raises(HoldoutViolation):
        assert_no_holdout(_frame(), SPLITS)


def test_inner_split_is_purged() -> None:
    df = development_view(_frame(), SPLITS).filter(pl.col("date") < dt.date(2014, 1, 1))
    tr, va = inner_split(df, 0.2, SESS, 5)
    assert tr["label_end"].max() < va["date"].min()
    assert SESS.index(va["date"].min()) - SESS.index(tr["label_end"].max()) > 5
