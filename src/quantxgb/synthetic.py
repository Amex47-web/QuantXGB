"""Synthetic market generator for tests, CI and the offline demo.

Synthetic data is **not** research evidence. It exists to (a) exercise the full pipeline
without API keys and (b) give tests a ground truth:

* ``signal > 0`` injects a persistent, predictable return component, so a correct
  pipeline must find positive out-of-sample IC;
* ``signal = 0`` is pure noise, so a correct pipeline must find IC ≈ 0. A positive IC
  on noise is a strong indication of leakage.

Daily total log-return of asset i::

    r_it = μ_i + β_i f_t + a_it + σ_i ε_it,     a_it = φ a_i,t−1 + signal · s_i · η_it

Prices follow ``close_t = close_{t−1}(1 + R_t) − D_t`` with occasional cash distributions
``D_t``, so the in-house total-return logic is exercised as well.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import polars as pl

from quantxgb.config import AppConfig, AssetSpec
from quantxgb.ingestion.normalize import normalize_macro, normalize_prices

CLASSES = ["equity_us", "equity_intl", "equity_sector", "rates", "credit", "commodity"]


@dataclass(frozen=True)
class SyntheticSpec:
    n_assets: int = 12
    start: dt.date = dt.date(2004, 1, 1)
    end: dt.date = dt.date(2016, 12, 31)
    signal: float = 1.0
    phi: float = 0.985
    seed: int = 0
    late_listing: int = 2  # number of assets that list ~2 years after start


def synthetic_universe(spec: SyntheticSpec) -> list[AssetSpec]:
    assets = [AssetSpec(ticker="MKT", asset_class="equity_us", description="synthetic market")]
    for i in range(1, spec.n_assets):
        assets.append(
            AssetSpec(ticker=f"A{i:02d}", asset_class=CLASSES[i % len(CLASSES)], description="")
        )
    # a rates proxy for corr_rates features
    assets[3 % len(assets)] = AssetSpec(ticker="RATES", asset_class="rates", description="")
    return assets


def generate_raw_prices(
    spec: SyntheticSpec, sessions: list[dt.date], universe: list[AssetSpec]
) -> dict[str, pl.DataFrame]:
    rng = np.random.default_rng(spec.seed)
    t_len, n = len(sessions), len(universe)
    f = rng.normal(0.0002, 0.011, t_len)  # market factor
    beta = rng.uniform(0.2, 1.3, n)
    beta[0] = 1.0
    sigma = rng.uniform(0.004, 0.012, n)
    mu = rng.normal(0.0001, 0.0001, n)
    s_i = sigma * 0.06
    a = np.zeros((t_len, n))
    eta = rng.normal(size=(t_len, n))
    for t in range(1, t_len):
        a[t] = spec.phi * a[t - 1] + spec.signal * s_i * eta[t]
    # the *previous* day's state drives today's return: predictable from the past only
    a_lag = np.vstack([np.zeros((1, n)), a[:-1]])
    r = mu + np.outer(f, beta) + a_lag + sigma * rng.normal(size=(t_len, n))
    r[:, 0] = mu[0] + f + sigma[0] * 0.2 * rng.normal(size=t_len)  # market ≈ factor

    out: dict[str, pl.DataFrame] = {}
    for j, asset in enumerate(universe):
        start_idx = 0
        if 0 < j <= spec.late_listing:
            start_idx = min(t_len - 1, 504 + 21 * j)
        R = np.expm1(r[start_idx:, j])
        m = R.shape[0]
        div = np.zeros(m)
        if asset.asset_class in ("rates", "credit", "equity_us"):
            div[np.arange(63, m, 63)] = 1.0  # quarterly flag
        close = np.empty(m)
        dist = np.zeros(m)
        close[0] = 50.0 + 10 * j
        for t in range(1, m):
            dist[t] = 0.006 * close[t - 1] if div[t] else 0.0
            close[t] = close[t - 1] * (1 + R[t]) - dist[t]
        tr = np.cumprod(np.r_[1.0, 1 + R[1:]])
        adj = tr / tr[-1] * close[-1]
        spread = np.abs(rng.normal(0, sigma[j] * 0.6, m)) + 1e-4
        high = close * np.exp(spread)
        low = close * np.exp(-spread)
        open_ = np.clip(close * np.exp(rng.normal(0, sigma[j] * 0.3, m)), low, high)
        volume = np.round(np.exp(rng.normal(14 - 0.2 * j, 0.4, m)))
        out[asset.ticker] = pl.DataFrame(
            {
                "date": sessions[start_idx:],
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "adj_close": adj,
                "volume": volume,
                "dividends": dist,
                "capital_gains": np.zeros(m),
                "splits": np.zeros(m),
            }
        ).with_columns(pl.col("date").cast(pl.Date))
    return out


def generate_raw_macro(spec: SyntheticSpec, sessions: list[dt.date]) -> dict[str, pl.DataFrame]:
    """A daily non-revised series and a monthly revised (vintaged) series."""
    rng = np.random.default_rng(spec.seed + 1)
    d10 = 4.0 + np.cumsum(rng.normal(0, 0.05, len(sessions)))
    vix = np.clip(18 + np.cumsum(rng.normal(0, 0.6, len(sessions))) * 0.3, 9, 80)
    far = "9999-12-31"
    today = dt.date.today().isoformat()

    def daily(vals: np.ndarray) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "realtime_start": [today] * len(sessions),
                "realtime_end": [today] * len(sessions),
                "date": [s.isoformat() for s in sessions],
                "value": [f"{v:.4f}" for v in vals],
            }
        )

    # monthly CPI-like index: first release ~14 days after month end, revised a month later
    months = pl.date_range(spec.start.replace(day=1), spec.end, "1mo", eager=True).to_list()
    level = 100.0 * np.cumprod(1 + rng.normal(0.002, 0.002, len(months)))
    rows: list[tuple[str, str, str, str]] = []
    for i, m in enumerate(months):
        nxt = (m.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
        first = nxt + dt.timedelta(days=13)
        second = (nxt.replace(day=28) + dt.timedelta(days=4)).replace(day=1) + dt.timedelta(days=13)
        rev = level[i] * (1 + rng.normal(0, 0.001))
        rows.append(
            (
                first.isoformat(),
                (second - dt.timedelta(days=1)).isoformat(),
                m.isoformat(),
                f"{level[i]:.3f}",
            )
        )
        rows.append((second.isoformat(), far, m.isoformat(), f"{rev:.3f}"))
    cpi = pl.DataFrame(
        {
            "realtime_start": [r[0] for r in rows],
            "realtime_end": [r[1] for r in rows],
            "date": [r[2] for r in rows],
            "value": [r[3] for r in rows],
        }
    )
    return {"DGS10": daily(d10), "VIXCLS": daily(vix), "CPIAUCSL": cpi}


def synthetic_config(base: AppConfig, spec: SyntheticSpec) -> AppConfig:
    """Adapt a config to the synthetic universe (dates, proxies, macro series)."""
    from quantxgb.config import DateRange, MacroFeatureSpec, MacroSeriesSpec, SplitsConfig

    universe = synthetic_universe(spec)
    data = base.data.model_copy(
        update={
            "universe": universe,
            "market_proxy": "MKT",
            "rates_proxy": "RATES",
            "calendar": "WEEKDAYS",
            "price_start": spec.start,
            "macro_series": [
                MacroSeriesSpec(id="DGS10", revised=False),
                MacroSeriesSpec(id="VIXCLS", revised=False),
                MacroSeriesSpec(id="CPIAUCSL", revised=True),
            ],
        }
    )
    features = base.features.model_copy(
        update={
            "macro": [
                MacroFeatureSpec(name="y10", series="DGS10", transform="level"),
                MacroFeatureSpec(name="y10_chg_20", series="DGS10", transform="diff", periods=20),
                MacroFeatureSpec(name="vix", series="VIXCLS", transform="level"),
                MacroFeatureSpec(
                    name="vix_pct_252", series="VIXCLS", transform="pct_rank", periods=252
                ),
                MacroFeatureSpec(
                    name="cpi_yoy", series="CPIAUCSL", transform="pct_change", periods=12
                ),
            ]
        }
    )
    years = (spec.end - spec.start).days / 365.25
    dev_start = spec.start + dt.timedelta(days=int(365.25 * 1.5))
    hold_start = spec.start + dt.timedelta(days=int(365.25 * (years - 1.5)))
    splits = SplitsConfig(
        warmup_end=dev_start - dt.timedelta(days=1),
        development=DateRange(start=dev_start, end=hold_start - dt.timedelta(days=45)),
        holdout=DateRange(start=hold_start, end=spec.end),
    )
    backtest = base.backtest.model_copy(
        update={
            "splits": splits,
            "benchmarks": base.backtest.benchmarks.model_copy(
                update={"sixty_forty": {"MKT": 0.6, "RATES": 0.4}, "market": "MKT"}
            ),
            "risk_free_series": None,
        }
    )
    validation = base.validation.model_copy(update={"min_train_years": 3, "test_months": 6})
    return base.model_copy(
        update={"data": data, "features": features, "backtest": backtest, "validation": validation}
    )


@dataclass
class SyntheticData:
    prices: pl.DataFrame
    macro: pl.DataFrame
    sessions: list[dt.date]
    raw_prices: dict[str, pl.DataFrame]
    raw_macro: dict[str, pl.DataFrame]


def make_synthetic(cfg: AppConfig, spec: SyntheticSpec) -> SyntheticData:
    from quantxgb.utils.calendar import business_day_sessions

    sessions = business_day_sessions(spec.start, spec.end + dt.timedelta(days=120))
    data_sessions = [s for s in sessions if s <= spec.end]
    raw_p = generate_raw_prices(spec, data_sessions, cfg.data.universe)
    raw_m = generate_raw_macro(spec, data_sessions)
    frames = []
    for asset in cfg.data.universe:
        df, _ = normalize_prices(
            raw_p[asset.ticker],
            asset.ticker,
            asset.asset_class,
            sessions,
            source="synthetic",
            snapshot_date="synthetic",
            snapshot_sha256="synthetic",
        )
        frames.append(df)
    prices = pl.concat(frames).sort(["ticker", "date"])
    revised = {s.id: s.revised for s in cfg.data.macro_series}
    macro = pl.concat(
        [
            normalize_macro(
                raw,
                sid,
                revised.get(sid, False),
                sessions,
                daily_lag_sessions=cfg.data.availability.daily_macro_lag_sessions,
                revised_lag_sessions=cfg.data.availability.revised_macro_lag_sessions,
                source="synthetic",
                snapshot_date="synthetic",
                snapshot_sha256="synthetic",
            )
            for sid, raw in raw_m.items()
        ]
    )
    return SyntheticData(prices, macro, sessions, raw_p, raw_m)
