"""Walk-forward splits with purging and embargo + the holdout guard (PRD §9.4, FR-VAL-01,
FR-PIT-06) — a critical module for owner review.

Expanding window. Test blocks of ``test_months`` start ``min_train_years`` after the start
of the development period. A row (decision date t) may be used for training in a fold
whose test block starts at T only if

    label_end(t) < session[idx(T) − embargo]

i.e. its whole label window ends strictly before the test block, with at least
``embargo`` additional sessions in between. Because training data always precedes the
test block in an expanding walk-forward, this single condition implements both purging
(label overlap) and the embargo gap.
"""

from __future__ import annotations

import bisect
import calendar
import datetime as dt
from dataclasses import dataclass

import polars as pl

from quantxgb.config import SplitsConfig


class HoldoutViolation(AssertionError):
    """Raised when development / model-selection code touches the holdout period."""


def development_view(dataset: pl.DataFrame, splits: SplitsConfig) -> pl.DataFrame:
    """Rows usable for development: decision date in the development period and, if a
    label exists, a label window that ends before the holdout starts. Labels that would
    reach into the holdout are nulled (the row can still be scored, never trained on)."""
    dev = dataset.filter(
        pl.col("date").is_between(splits.development.start, splits.development.end, closed="both")
    )
    if "label_end" in dev.columns:
        leaks = pl.col("label_end").is_not_null() & (pl.col("label_end") >= splits.holdout.start)
        label_cols = [c for c in ("fwd_ret", "fwd_xs_ret", "fwd_grade") if c in dev.columns]
        dev = dev.with_columns(
            [pl.when(leaks).then(None).otherwise(pl.col(c)).alias(c) for c in label_cols]
        )
    assert_no_holdout(dev, splits)
    return dev


def assert_no_holdout(df: pl.DataFrame, splits: SplitsConfig) -> None:
    h0 = splits.holdout.start
    if df.filter(pl.col("date") >= h0).height:
        raise HoldoutViolation("development data contains holdout decision dates")
    if "label_end" in df.columns and "fwd_ret" in df.columns:
        bad = df.filter(pl.col("fwd_ret").is_not_null() & (pl.col("label_end") >= h0))
        if bad.height:
            raise HoldoutViolation("development labels use prices from the holdout period")


@dataclass(frozen=True)
class Fold:
    index: int
    train_start: dt.date
    test_start: dt.date
    test_end: dt.date
    label_cutoff: dt.date  # training rows need label_end < label_cutoff

    def train_mask(self) -> pl.Expr:
        return (
            (pl.col("date") >= self.train_start)
            & (pl.col("date") < self.test_start)
            & pl.col("fwd_ret").is_not_null()
            & (pl.col("label_end") < self.label_cutoff)
        )

    def test_mask(self) -> pl.Expr:
        return pl.col("date").is_between(self.test_start, self.test_end, closed="both")


def _add_months(d: dt.date, months: int) -> dt.date:
    y, m = divmod(d.month - 1 + months, 12)
    year, month = d.year + y, m + 1
    return dt.date(year, month, min(d.day, calendar.monthrange(year, month)[1]))


def label_cutoff(test_start: dt.date, sessions: list[dt.date], embargo: int) -> dt.date:
    i = bisect.bisect_left(sessions, test_start)
    j = i - embargo
    if j <= 0:
        raise ValueError("not enough sessions before the test block for the embargo")
    return sessions[j]


def walk_forward_folds(
    decision_dates: list[dt.date],
    sessions: list[dt.date],
    dev_start: dt.date,
    dev_end: dt.date,
    min_train_years: int,
    test_months: int,
    embargo_sessions: int,
) -> list[Fold]:
    dates = sorted(d for d in decision_dates if dev_start <= d <= dev_end)
    if not dates:
        return []
    folds: list[Fold] = []
    block_start = _add_months(dev_start, 12 * min_train_years)
    k = 0
    while block_start <= dev_end:
        block_end = min(_add_months(block_start, test_months) - dt.timedelta(days=1), dev_end)
        in_block = [d for d in dates if block_start <= d <= block_end]
        if in_block:
            folds.append(
                Fold(
                    index=k,
                    train_start=dev_start,
                    test_start=in_block[0],
                    test_end=in_block[-1],
                    label_cutoff=label_cutoff(in_block[0], sessions, embargo_sessions),
                )
            )
            k += 1
        block_start = _add_months(block_start, test_months)
    return folds


def inner_split(
    train: pl.DataFrame, fraction: float, sessions: list[dt.date], embargo: int
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Purged inner split of a training fold: last ``fraction`` of decision dates validate."""
    dates = sorted(train["date"].unique().to_list())
    n_val = max(1, round(len(dates) * fraction))
    if len(dates) - n_val < 2:
        raise ValueError("training fold too short for an inner validation split")
    v0 = dates[-n_val]
    cutoff = label_cutoff(v0, sessions, embargo)
    inner_train = train.filter((pl.col("date") < v0) & (pl.col("label_end") < cutoff))
    inner_val = train.filter(pl.col("date") >= v0)
    return inner_train, inner_val
