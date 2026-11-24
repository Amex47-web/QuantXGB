"""Strategies: score-based optimised portfolios and weight baselines (PRD §4.5, §10).

Every strategy returns target weights for a signal date t using only information
available at the close of t. All strategies share the same rebalance schedule,
execution delay, constraints (optimised ones) and cost assumptions.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import polars as pl
from scipy.stats import norm

from quantxgb.config import AppConfig
from quantxgb.evaluation.signal_metrics import rank_ic_by_date
from quantxgb.portfolio.covariance import estimate_covariance
from quantxgb.portfolio.expected_returns import grinold_mu, trailing_ic
from quantxgb.portfolio.optimizer import optimize, two_piece_normal_scenarios


class Strategy(Protocol):
    name: str
    is_baseline: bool

    def target(self, t: dt.date, current: dict[str, float]) -> dict[str, float]: ...


@dataclass
class Context:
    cfg: AppConfig
    features: pl.DataFrame  # decision-date rows of eligible assets (date, ticker, asset_class, ...)
    returns: pl.DataFrame  # wide daily simple returns
    quantiles: pl.DataFrame | None = None  # date, ticker, q05, q50, q95

    def universe_at(self, t: dt.date) -> pl.DataFrame:
        return self.features.filter(pl.col("date") == t).sort("ticker")

    def returns_window(self, t: dt.date, tickers: list[str], length: int) -> np.ndarray:
        w = self.returns.filter(pl.col("date") <= t).tail(length).select(tickers)
        return w.cast(pl.Float64).fill_null(np.nan).to_numpy()


def _normalise(w: dict[str, float]) -> dict[str, float]:
    s = sum(w.values())
    return {k: v / s for k, v in w.items()} if s > 0 else {}


@dataclass
class EqualWeight:
    ctx: Context
    name: str = "equal_weight"
    is_baseline: bool = True

    def target(self, t: dt.date, current: dict[str, float]) -> dict[str, float]:
        u = self.ctx.universe_at(t)["ticker"].to_list()
        return {k: 1.0 / len(u) for k in u} if u else {}


@dataclass
class FixedWeights:
    ctx: Context
    weights: dict[str, float]
    name: str = "fixed"
    is_baseline: bool = True

    def target(self, t: dt.date, current: dict[str, float]) -> dict[str, float]:
        avail = set(self.ctx.universe_at(t)["ticker"].to_list())
        return _normalise({k: v for k, v in self.weights.items() if k in avail})


@dataclass
class InverseVol:
    ctx: Context
    feature: str = "vol_60"
    name: str = "inverse_vol"
    is_baseline: bool = True

    def target(self, t: dt.date, current: dict[str, float]) -> dict[str, float]:
        u = self.ctx.universe_at(t).filter(
            pl.col(self.feature).is_not_null() & (pl.col(self.feature) > 0)
        )
        return _normalise(
            dict(zip(u["ticker"].to_list(), (1.0 / u[self.feature]).to_list(), strict=True))
        )


@dataclass
class MomentumTopK:
    """B4 naive variant: equal weight in the top-k assets by 12-1 momentum."""

    ctx: Context
    k: int = 5
    feature: str = "mom_12_1"
    name: str = "momentum_topk"
    is_baseline: bool = True

    def target(self, t: dt.date, current: dict[str, float]) -> dict[str, float]:
        u = self.ctx.universe_at(t).filter(pl.col(self.feature).is_not_null())
        top = u.sort(self.feature, descending=True).head(self.k)["ticker"].to_list()
        return {k: 1.0 / len(top) for k in top} if top else {}


@dataclass
class OptimisedScores:
    """Model scores → Grinold μ → constrained optimiser (MV or CVaR)."""

    ctx: Context
    model: str
    predictions: (
        pl.DataFrame
    )  # all OOS predictions of this model (date, ticker, score, fwd_xs_ret, label_end)
    is_baseline: bool = False
    relax_log: list[str] = field(default_factory=list)
    _cache: dict[dt.date, tuple[dict[str, float], dict[str, float]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = f"opt_{self.model}"
        p = self.predictions
        self._ic = rank_ic_by_date(p)
        self._label_end = (
            p.group_by("date")
            .agg(pl.col("label_end").max())
            .filter(pl.col("label_end").is_not_null())
        )

    def target(self, t: dt.date, current: dict[str, float]) -> dict[str, float]:
        cached = self._cache.get(t)
        if cached is not None and _same(cached[0], current):
            return cached[1]
        out = self._compute(t, current)
        self._cache[t] = (dict(current), out)
        return out

    def _compute(self, t: dt.date, current: dict[str, float]) -> dict[str, float]:
        cfg = self.ctx.cfg
        bt = cfg.backtest
        u = self.ctx.universe_at(t)
        s = self.predictions.filter(pl.col("date") == t).select("ticker", "score")
        u = u.join(s, on="ticker", how="inner").filter(pl.col("score").is_not_null())
        if u.height < 2:
            return _normalise(current)
        tickers = u["ticker"].to_list()
        ic_bar = trailing_ic(self._ic, self._label_end, t, bt.expected_returns.ic_min_history)
        vol = u[bt.expected_returns.vol_feature].cast(pl.Float64).fill_null(np.nan).to_numpy()
        mu = grinold_mu(u["score"].to_numpy(), vol, ic_bar, cfg.labels.horizon)
        w_prev = np.array([current.get(k, 0.0) for k in tickers])
        forced = sum(v for k, v in current.items() if k not in set(tickers))
        classes = u["asset_class"].to_list()
        opt = bt.optimizer
        h = cfg.labels.horizon
        window = self.ctx.returns_window(t, tickers, opt.covariance_lookback)
        window = window[~np.isnan(window).any(axis=1)]
        sigma = scenarios = None
        if window.shape[0] >= max(60, h + 20):
            sigma = estimate_covariance(window, opt.covariance, opt.ewma_halflife) * h
        else:
            sigma = np.diag(np.nan_to_num(vol, nan=0.2) ** 2 * h / 252.0)
        if opt.type == "cvar":
            scenarios = self._scenarios(t, tickers, window, sigma)
        res = optimize(
            mu,
            w_prev,
            classes,
            bt.constraints,
            opt,
            sigma=sigma,
            scenarios=scenarios,
            forced_sell=forced,
        )
        if res.relaxed:
            self.relax_log.append(f"{t}: relaxed {res.relaxed} ({res.status})")
        return {k: float(x) for k, x in zip(tickers, res.weights, strict=True) if x > 1e-8}

    def _scenarios(
        self, t: dt.date, tickers: list[str], window: np.ndarray, sigma: np.ndarray
    ) -> np.ndarray:
        opt = self.ctx.cfg.backtest.optimizer
        h = self.ctx.cfg.labels.horizon
        if opt.cvar_scenarios == "historical" or self.ctx.quantiles is None:
            if window.shape[0] <= h:
                raise ValueError("not enough history for historical CVaR scenarios")
            growth = np.cumprod(1.0 + window, axis=0)
            growth = np.vstack([np.ones((1, growth.shape[1])), growth])
            return growth[h:] / growth[:-h] - 1.0
        q = self.ctx.quantiles.filter(pl.col("date") == t)
        q = pl.DataFrame({"ticker": tickers}).join(q, on="ticker", how="left")
        alphas = self.ctx.cfg.model.quantile.alphas
        cols = [c for c in q.columns if c.startswith("q") and c[1:].isdigit()]
        lo, mid, hi = (
            q[c].fill_null(0.0).to_numpy() for c in (cols[0], cols[len(cols) // 2], cols[-1])
        )
        d = np.sqrt(np.clip(np.diag(sigma), 1e-12, None))
        corr = sigma / np.outer(d, d)
        return two_piece_normal_scenarios(
            lo,
            mid,
            hi,
            corr,
            opt.n_copula_scenarios,
            float(norm.ppf(alphas[0])),
            float(norm.ppf(alphas[-1])),
            seed=self.ctx.cfg.backtest.seed,
        )


def _same(a: dict[str, float], b: dict[str, float]) -> bool:
    keys = set(a) | set(b)
    return all(math.isclose(a.get(k, 0.0), b.get(k, 0.0), abs_tol=1e-12) for k in keys)
