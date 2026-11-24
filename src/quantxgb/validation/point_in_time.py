"""Point-in-time utilities (PRD FR-PIT-01/02, §7.3).

Macro features are computed *vintage-consistently*:

* Non-revised series: the transform is applied along the observation sequence; each
  transformed value inherits the ``available_on`` of the latest observation it uses
  (earlier observations are always available earlier).
* Revised series (ALFRED): for every vintage date ``v`` we reconstruct the series exactly
  as published on ``v`` (rows with ``realtime_start <= v <= realtime_end``), apply the
  transform on that vintage only, and keep the value for its latest observation. The
  result is usable from ``available_on(v)``.

Decision dates are then as-of joined (backward) on ``available_on`` so that a decision
at the close of ``t`` only sees values with ``available_on <= t``.
"""

from __future__ import annotations

import datetime as dt

import polars as pl

from quantxgb.config import MacroFeatureSpec
from quantxgb.utils.calendar import shift_to_session


class LeakageError(AssertionError):
    """Raised when data dated after the decision time would be used."""


def _transform(values: pl.Expr, spec: MacroFeatureSpec) -> pl.Expr:
    k = spec.periods or 1
    if spec.transform == "level":
        return values
    if spec.transform == "diff":
        return values - values.shift(k)
    if spec.transform == "pct_change":
        return values / values.shift(k) - 1.0
    if spec.transform == "zscore":
        return (values - values.rolling_mean(k)) / values.rolling_std(k)
    if spec.transform == "pct_rank":
        # share of the last k observations (incl. current) that are <= current value
        return values.rolling_map(lambda s: float((s <= s[-1]).sum()) / len(s), window_size=k)
    raise ValueError(spec.transform)


def macro_feature_table(
    macro: pl.DataFrame,
    specs: list[MacroFeatureSpec],
    sessions: list[dt.date],
    revised_lag_sessions: int,
) -> pl.DataFrame:
    """Long table ``feature, available_on, observed_on, value`` for every macro feature."""
    out: list[pl.DataFrame] = []
    for spec in specs:
        s = macro.filter(pl.col("series_id") == spec.series)
        if s.height == 0:
            continue
        revised = bool(s["revised"][0])
        if not revised:
            t = (
                s.sort("observed_on")
                .with_columns(_transform(pl.col("value"), spec).alias("fv"))
                .select(
                    pl.lit(spec.name).alias("feature"),
                    "available_on",
                    "observed_on",
                    pl.col("fv").alias("value"),
                )
            )
        else:
            t = _vintage_consistent(s, spec, sessions, revised_lag_sessions)
        out.append(t.filter(pl.col("value").is_not_null() & pl.col("value").is_finite()))
    if not out:
        return pl.DataFrame(
            schema={
                "feature": pl.String,
                "available_on": pl.Date,
                "observed_on": pl.Date,
                "value": pl.Float64,
            }
        )
    return pl.concat(out).sort(["feature", "available_on", "observed_on"])


def _vintage_consistent(
    s: pl.DataFrame, spec: MacroFeatureSpec, sessions: list[dt.date], lag: int
) -> pl.DataFrame:
    vintages = s["realtime_start"].unique().sort().to_list()
    rows: list[tuple[dt.date, dt.date, float | None]] = []
    for v in vintages:
        snap = s.filter((pl.col("realtime_start") <= v) & (pl.col("realtime_end") >= v)).sort(
            "observed_on"
        )
        if snap.height == 0:
            continue
        val = snap.select(_transform(pl.col("value"), spec).last())[0, 0]
        rows.append((v, snap["observed_on"][-1], val))
    t = pl.DataFrame(
        {
            "vintage": [r[0] for r in rows],
            "observed_on": [r[1] for r in rows],
            "value": [r[2] for r in rows],
        },
        schema={"vintage": pl.Date, "observed_on": pl.Date, "value": pl.Float64},
    )
    avail = shift_to_session(t["vintage"], sessions, lag, strictly_after=lag > 0)
    return t.with_columns(avail.alias("available_on"), pl.lit(spec.name).alias("feature")).select(
        "feature", "available_on", "observed_on", "value"
    )


def asof_join_macro(decision_dates: list[dt.date], table: pl.DataFrame) -> pl.DataFrame:
    """Wide frame ``date, <feature>...`` with the latest value available at each date."""
    base = pl.DataFrame({"date": decision_dates}, schema={"date": pl.Date}).sort("date")
    if table.height == 0:
        return base
    out = base
    for (feat,), g in table.group_by(["feature"], maintain_order=True):
        right = (
            g.filter(pl.col("available_on").is_not_null())
            .sort(["available_on", "observed_on"])
            .unique(subset="available_on", keep="last", maintain_order=True)
            .select(pl.col("available_on"), pl.col("value").alias(str(feat)))
        )
        joined = base.join_asof(right, left_on="date", right_on="available_on", strategy="backward")
        bad = joined.filter(pl.col("available_on") > pl.col("date"))
        if bad.height:
            raise LeakageError(f"macro feature {feat}: value available after decision date")
        out = out.join(joined.select("date", str(feat)), on="date", how="left")
    return out


def assert_available(df: pl.DataFrame, decision_col: str, available_col: str) -> None:
    """Guard (FR-PIT-02): reject rows whose ``available_col`` is after ``decision_col``."""
    bad = df.filter(pl.col(available_col) > pl.col(decision_col))
    if bad.height:
        raise LeakageError(
            f"{bad.height} rows use data available after decision time "
            f"({available_col} > {decision_col}); first: {bad.head(1).to_dicts()}"
        )


def restrict_dates(
    df: pl.DataFrame, start: dt.date, end: dt.date, date_col: str = "date"
) -> pl.DataFrame:
    return df.filter(pl.col(date_col).is_between(start, end, closed="both"))
