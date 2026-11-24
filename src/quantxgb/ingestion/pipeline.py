"""Orchestrates download → raw snapshots → curated tables → data-quality report."""

from __future__ import annotations

import datetime as dt

import polars as pl

from quantxgb.config import AppConfig, Secrets
from quantxgb.ingestion.base import MacroProvider, PriceProvider, RawStore
from quantxgb.ingestion.normalize import normalize_macro, normalize_prices
from quantxgb.ingestion.store import CuratedStore
from quantxgb.utils.calendar import trading_sessions
from quantxgb.utils.io import get_logger, write_json
from quantxgb.validation.data_quality import DQReport, Issue, run_data_quality

log = get_logger(__name__)


def calendar_sessions(cfg: AppConfig, start: dt.date | None = None) -> list[dt.date]:
    """Master calendar from the earliest relevant date to ~3 months ahead (for lags)."""
    first = start or min(cfg.data.price_start, cfg.data.fred.observation_start)
    last = dt.date.today() + dt.timedelta(days=92)
    return trading_sessions(first, last, cfg.data.calendar)


def ingest_prices(cfg: AppConfig, provider: PriceProvider, tickers: list[str] | None = None) -> int:
    store = RawStore(cfg.raw_dir)
    n = 0
    for ticker in tickers or cfg.data.tickers:
        log.info("downloading %s from %s", ticker, provider.name)
        df = provider.fetch_daily(ticker, cfg.data.price_start, cfg.data.price_end)
        params = {"start": cfg.data.price_start, "end": cfg.data.price_end}
        rec = store.write(provider.name, ticker, df, params)
        log.info("  %s rows=%d sha=%s", ticker, rec.rows, rec.sha256[:10])
        n += 1
    return n


def ingest_macro(cfg: AppConfig, provider: MacroProvider) -> int:
    store = RawStore(cfg.raw_dir)
    meta: dict[str, object] = {}
    for spec in cfg.data.macro_series:
        log.info("downloading %s (all_vintages=%s)", spec.id, spec.revised)
        df = provider.fetch_series(spec.id, all_vintages=spec.revised)
        store.write(provider.name, spec.id, df, {"all_vintages": spec.revised})
        meta[spec.id] = provider.fetch_metadata(spec.id)
    write_json(meta, cfg.raw_dir / provider.name / "series_metadata.json")
    return len(cfg.data.macro_series)


def default_providers(cfg: AppConfig) -> tuple[PriceProvider, MacroProvider | None]:
    from quantxgb.ingestion.fred_provider import FredProvider
    from quantxgb.ingestion.yfinance_provider import YFinanceProvider

    secrets = Secrets()
    macro: MacroProvider | None = None
    if secrets.fred_api_key:
        macro = FredProvider(
            api_key=secrets.fred_api_key,
            base_url=cfg.data.fred.base_url,
            observation_start=cfg.data.fred.observation_start,
            max_per_minute=cfg.data.fred.max_requests_per_minute,
            page_limit=cfg.data.fred.page_limit,
        )
    return YFinanceProvider(), macro


def build_curated(
    cfg: AppConfig, price_provider: str = "yfinance", macro_provider: str = "fred"
) -> DQReport:
    """Normalise the latest raw snapshot of every configured series and run DQ checks."""
    raw = RawStore(cfg.raw_dir)
    sessions = calendar_sessions(cfg)
    frames: list[pl.DataFrame] = []
    extra: list[Issue] = []
    for asset in cfg.data.universe:
        rec = raw.latest(price_provider, asset.ticker)
        if rec is None:
            extra.append(Issue("missing_raw", "fail", asset.ticker, "no raw snapshot"))
            continue
        df, off = normalize_prices(
            raw.read(rec),
            asset.ticker,
            asset.asset_class,
            sessions,
            source=price_provider,
            snapshot_date=rec.snapshot_date,
            snapshot_sha256=rec.sha256,
            price_lag_sessions=cfg.data.availability.price_lag_sessions,
        )
        if off:
            extra.append(
                Issue(
                    "off_calendar_bars",
                    # a handful can be provider artefacts; many means a calendar problem
                    "fail" if len(off) > 0.005 * max(df.height, 1) else "flag",
                    asset.ticker,
                    f"bars on non-{cfg.data.calendar} dates dropped",
                    len(off),
                    [str(d) for d in off[:5]],
                )
            )
        frames.append(df)
    if not frames:
        raise RuntimeError("no price data — run `quantxgb data ingest` first")
    prices = pl.concat(frames, how="vertical").sort(["ticker", "date"])

    macro_frames: list[pl.DataFrame] = []
    for spec in cfg.data.macro_series:
        rec = raw.latest(macro_provider, spec.id)
        if rec is None:
            extra.append(Issue("missing_raw", "warn", spec.id, "no raw snapshot (macro skipped)"))
            continue
        macro_frames.append(
            normalize_macro(
                raw.read(rec),
                spec.id,
                spec.revised,
                sessions,
                daily_lag_sessions=cfg.data.availability.daily_macro_lag_sessions,
                revised_lag_sessions=cfg.data.availability.revised_macro_lag_sessions,
                source=macro_provider,
                snapshot_date=rec.snapshot_date,
                snapshot_sha256=rec.sha256,
            )
        )
    macro = pl.concat(macro_frames, how="vertical") if macro_frames else None

    report = run_data_quality(prices, macro)
    report.issues.extend(extra)
    curated = CuratedStore(cfg.curated_dir)
    hashes = curated.write(prices, macro)
    curated.build_duckdb(cfg.path(cfg.data.paths.curated).parent / "quantxgb.duckdb")
    report.summary["curated hashes"] = hashes
    report.write(cfg.artifacts_dir / "reports" / "data_quality.md")
    return report
