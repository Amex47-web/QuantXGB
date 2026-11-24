"""Trading-calendar helpers. All session arithmetic in the project goes through here."""

from __future__ import annotations

import datetime as dt
from functools import lru_cache

import numpy as np
import polars as pl


class CalendarRangeError(ValueError):
    pass


@lru_cache(maxsize=16)
def _sessions(calendar: str, start: dt.date, end: dt.date) -> tuple[dt.date, ...]:
    import exchange_calendars as xcals

    # exchange_calendars defaults to a ~20-year window; request the range explicitly and
    # never clamp silently (that once dropped every bar before 2006).
    try:
        cal = xcals.get_calendar(calendar, start=start.isoformat(), end=end.isoformat())
    except ValueError as exc:
        raise CalendarRangeError(
            f"calendar {calendar} cannot cover {start} → {end}: {exc}"
        ) from None
    if cal.first_session.date() > start + dt.timedelta(days=7) or (
        cal.last_session.date() < end - dt.timedelta(days=7)
    ):
        raise CalendarRangeError(
            f"calendar {calendar} covers {cal.first_session.date()} → "
            f"{cal.last_session.date()}, requested {start} → {end}"
        )
    idx = cal.sessions_in_range(cal.first_session, cal.last_session)
    return tuple(ts.date() for ts in idx)


def trading_sessions(start: dt.date, end: dt.date, calendar: str = "XNYS") -> list[dt.date]:
    """Sessions of ``calendar`` in ``[start, end]`` (inclusive).

    ``WEEKDAYS`` is a holiday-free Monday-Friday calendar used by synthetic data.
    """
    if calendar == "WEEKDAYS":
        return business_day_sessions(start, end)
    return list(_sessions(calendar, start, end))


def business_day_sessions(start: dt.date, end: dt.date) -> list[dt.date]:
    """Weekday calendar (used for synthetic data / tests that must not depend on holidays)."""
    days = np.arange(
        np.datetime64(start, "D"),
        np.datetime64(end, "D") + np.timedelta64(1, "D"),
        dtype="datetime64[D]",
    )
    weekdays = days[np.is_busday(days)]
    return [d.astype(dt.date) for d in weekdays]


def shift_to_session(
    dates: pl.Series, sessions: list[dt.date], lag: int, strictly_after: bool
) -> pl.Series:
    """Map each date to a session ``lag`` sessions later.

    With ``strictly_after=True`` a date ``d`` maps to the ``lag``-th session strictly after
    ``d``; with ``False`` the 0th session is the first session on or after ``d``.
    Returns null where the target session is beyond the calendar.
    """
    sess = np.array(sessions, dtype="datetime64[D]")
    vals = dates.cast(pl.Date).to_numpy().astype("datetime64[D]")
    base = np.searchsorted(sess, vals, side="right" if strictly_after else "left")
    target = base + lag - (1 if strictly_after else 0)
    out: list[dt.date | None] = []
    for t in target:
        out.append(sess[t].astype(dt.date) if 0 <= t < len(sess) else None)
    return pl.Series(dates.name, out, dtype=pl.Date)


def weekly_decision_dates(sessions: list[dt.date]) -> list[dt.date]:
    """Last session of each ISO week."""
    if not sessions:
        return []
    df = pl.DataFrame({"date": sessions}).with_columns(
        pl.col("date").dt.iso_year().alias("y"), pl.col("date").dt.week().alias("w")
    )
    out = df.group_by(["y", "w"]).agg(pl.col("date").max()).sort("date")
    return out["date"].to_list()
