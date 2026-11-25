"""Data-quality checks on curated tables (PRD FR-DATA-08, §7.6).

Each check returns ``Issue`` records. ``fail`` issues stop the pipeline; ``warn`` and
``flag`` issues go into the report for the owner to review manually.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import polars as pl

Severity = Literal["fail", "warn", "flag"]


@dataclass(frozen=True)
class Issue:
    check: str
    severity: Severity
    entity: str
    message: str
    count: int = 1
    examples: list[str] = field(default_factory=list)


@dataclass
class DQReport:
    issues: list[Issue]
    summary: dict[str, object]

    @property
    def passed(self) -> bool:
        return not any(i.severity == "fail" for i in self.issues)

    def to_frame(self) -> pl.DataFrame:
        if not self.issues:
            return pl.DataFrame(
                schema={
                    "check": pl.String,
                    "severity": pl.String,
                    "entity": pl.String,
                    "message": pl.String,
                    "count": pl.Int64,
                    "examples": pl.String,
                }
            )
        rows = [asdict(i) | {"examples": ", ".join(i.examples[:5])} for i in self.issues]
        return pl.DataFrame(rows)

    def to_markdown(self) -> str:
        lines = [
            "# Data-quality report",
            "",
            f"Generated: {dt.datetime.now(dt.UTC).isoformat(timespec='seconds')}",
            "",
            f"**Result: {'PASS' if self.passed else 'FAIL'}**",
            "",
            "## Summary",
            "",
        ]
        for k, v in self.summary.items():
            lines.append(f"- **{k}**: {v}")
        lines += ["", "## Issues", ""]
        if not self.issues:
            lines.append("No issues found.")
        else:
            lines.append("| Severity | Check | Entity | Count | Message | Examples |")
            lines.append("|---|---|---|---|---|---|")
            order = {"fail": 0, "flag": 1, "warn": 2}
            for i in sorted(self.issues, key=lambda x: (order[x.severity], x.check, x.entity)):
                ex = ", ".join(i.examples[:5])
                lines.append(
                    f"| {i.severity} | {i.check} | {i.entity} | {i.count} | {i.message} | {ex} |"
                )
        lines += [
            "",
            "> Owner action: review every `flag` against the source before using the data.",
            "",
        ]
        return "\n".join(lines)

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_markdown(), encoding="utf-8")
        return path


def _examples(df: pl.DataFrame, col: str = "date", n: int = 5) -> list[str]:
    return [str(v) for v in df[col].head(n).to_list()]


def check_prices(
    prices: pl.DataFrame,
    *,
    max_missing_frac: float = 0.005,
    outlier_sigma: float = 8.0,
    outlier_abs: float = 0.25,
    stale_run: int = 3,
    reconcile_tol: float = 0.005,
) -> list[Issue]:
    issues: list[Issue] = []
    dup = prices.group_by(["ticker", "date"]).len().filter(pl.col("len") > 1)
    if dup.height:
        issues.append(Issue("duplicates", "fail", "prices", "duplicate (ticker, date)", dup.height))

    for (ticker,), g in prices.group_by(["ticker"], maintain_order=True):
        t = str(ticker)
        g = g.sort("date")
        n = g.height
        missing = g.filter(pl.col("close").is_null())
        if missing.height:
            frac = missing.height / n
            sev: Severity = "fail" if frac > max_missing_frac else "warn"
            issues.append(
                Issue(
                    "calendar_completeness",
                    sev,
                    t,
                    f"{missing.height} sessions without a bar ({frac:.3%})",
                    missing.height,
                    _examples(missing),
                )
            )
        bars = g.filter(pl.col("close").is_not_null())
        nonpos = bars.filter((pl.col("close") <= 0) | (pl.col("open") <= 0) | (pl.col("high") <= 0))
        if nonpos.height:
            issues.append(
                Issue(
                    "non_positive_price",
                    "fail",
                    t,
                    "price <= 0",
                    nonpos.height,
                    _examples(nonpos),
                )
            )
        tol = 1e-6
        ohlc = bars.filter(
            (pl.col("low") > pl.col("high") * (1 + tol))
            | (pl.col("open") > pl.col("high") * (1 + tol))
            | (pl.col("close") > pl.col("high") * (1 + tol))
            | (pl.col("open") < pl.col("low") * (1 - tol))
            | (pl.col("close") < pl.col("low") * (1 - tol))
        )
        if ohlc.height:
            issues.append(
                Issue(
                    "ohlc_consistency",
                    "fail",
                    t,
                    "OHLC bounds violated",
                    ohlc.height,
                    _examples(ohlc),
                )
            )
        neg_vol = bars.filter(pl.col("volume") < 0)
        if neg_vol.height:
            issues.append(Issue("negative_volume", "fail", t, "volume < 0", neg_vol.height))

        sigma = pl.col("ret").rolling_std(252, min_samples=60).shift(1)
        out = bars.with_columns(sigma.alias("sig")).filter(
            (pl.col("ret").abs() > outlier_abs)
            | (pl.col("sig").is_not_null() & (pl.col("ret").abs() > outlier_sigma * pl.col("sig")))
        )
        if out.height:
            issues.append(
                Issue(
                    "outlier_return",
                    "flag",
                    t,
                    f"|ret| > {outlier_abs:.0%} or > {outlier_sigma}σ(252d)",
                    out.height,
                    _examples(out),
                )
            )
        same = (pl.col("close") == pl.col("close").shift(1)) & (pl.col("volume") > 0)
        runs = bars.with_columns(same.fill_null(False).alias("same")).with_columns(
            (pl.col("same") != pl.col("same").shift(1)).fill_null(True).cum_sum().alias("grp")
        )
        stale = (
            runs.filter(pl.col("same"))
            .group_by("grp")
            .agg(pl.len().alias("n"), pl.col("date").first().alias("date"))
            .filter(pl.col("n") >= stale_run - 1)
            .sort("date")
        )
        if stale.height:
            issues.append(
                Issue(
                    "stale_price",
                    "flag",
                    t,
                    f">= {stale_run} identical closes with volume > 0",
                    stale.height,
                    _examples(stale),
                )
            )
        rec = bars.filter(
            pl.col("adj_ret").is_not_null()
            & pl.col("ret").is_not_null()
            & ((pl.col("ret") - pl.col("adj_ret")).abs() > reconcile_tol)
        )
        if rec.height:
            issues.append(
                Issue(
                    "adjustment_reconciliation",
                    "flag",
                    t,
                    "in-house total return differs from Yahoo adj-close return by "
                    f"> {reconcile_tol:.1%}",
                    rec.height,
                    _examples(rec),
                )
            )
        splits = bars.filter((pl.col("splits") > 0) & (pl.col("splits") != 1.0))
        if splits.height:
            big = splits.filter(pl.col("ret").abs() > 0.2)
            issues.append(
                Issue(
                    "split_events",
                    "flag" if big.height else "warn",
                    t,
                    "split events present (close is split-adjusted; verify no jump)"
                    + (f"; {big.height} with |ret| > 20%" if big.height else ""),
                    splits.height,
                    _examples(splits),
                )
            )
        avail = bars.filter(pl.col("available_on") < pl.col("date"))
        if avail.height:
            issues.append(
                Issue("availability_sanity", "fail", t, "available_on < date", avail.height)
            )
    return issues


def check_macro(macro: pl.DataFrame) -> list[Issue]:
    issues: list[Issue] = []
    key = ["series_id", "observed_on", "realtime_start"]
    dup = macro.group_by(key).len().filter(pl.col("len") > 1)
    if dup.height:
        issues.append(
            Issue("duplicates", "fail", "macro", "duplicate (series, obs, vintage)", dup.height)
        )
    bad = macro.filter(pl.col("available_on") < pl.col("observed_on"))
    if bad.height:
        issues.append(
            Issue("availability_sanity", "fail", "macro", "available_on < observed_on", bad.height)
        )
    rev = macro.filter(pl.col("revised"))
    if rev.height:
        missing_vintage = rev.filter(pl.col("realtime_start").is_null())
        if missing_vintage.height:
            issues.append(
                Issue(
                    "vintage_integrity",
                    "fail",
                    "macro",
                    "revised series rows without realtime_start",
                    missing_vintage.height,
                )
            )
        early = rev.filter(pl.col("realtime_start") < pl.col("observed_on"))
        if early.height:
            issues.append(
                Issue(
                    "vintage_integrity",
                    "fail",
                    "macro",
                    "vintage starts before the observation period",
                    early.height,
                )
            )
        for (sid,), g in rev.group_by(["series_id"]):
            first_vintage = g["realtime_start"].min()
            issues.append(
                Issue(
                    "first_vintage",
                    "warn",
                    str(sid),
                    f"first ALFRED vintage {first_vintage!s}; earlier observations become "
                    "available only from that date",
                )
            )
    return issues


def run_data_quality(prices: pl.DataFrame, macro: pl.DataFrame | None) -> DQReport:
    issues = check_prices(prices)
    if macro is not None and macro.height:
        issues += check_macro(macro)
    first = (
        prices.filter(pl.col("close").is_not_null())
        .group_by("ticker")
        .agg(pl.col("date").min().alias("first"), pl.col("date").max().alias("last"))
    )
    summary: dict[str, object] = {
        "tickers": prices["ticker"].n_unique(),
        "price rows": prices.height,
        "date range": f"{prices['date'].min()!s} → {prices['date'].max()!s}",
        "first bar per ticker (verify vs. inception dates)": ", ".join(
            f"{r['ticker']}={r['first']}" for r in first.sort("ticker").iter_rows(named=True)
        ),
    }
    if macro is not None:
        summary["macro series"] = macro["series_id"].n_unique()
        summary["macro rows"] = macro.height
    return DQReport(issues=issues, summary=summary)
