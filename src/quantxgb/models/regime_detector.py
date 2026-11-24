"""Regime models (PRD §9.3, FR-MODEL-05/06) — a critical module for owner review.

* ``RuleRegime``: volatility threshold set from the TRAINING window only + market trend.
* ``HMMRegime``: Gaussian HMM fitted on training-window daily market inputs. Inference
  uses the forward (filtering) recursion only, so the state probability at t depends on
  observations ≤ t. hmmlearn's ``predict_proba`` / Viterbi use forward-backward
  (smoothing) over the whole sequence and must NOT be used for features — that would leak.

States are ordered by the mean of the LAST configured input (``market_vol_20`` by
default) so that state 0 is the calmest; naming/interpreting states is the owner's job.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from quantxgb.config import HMMConfig, RegimeRulesConfig


class RuleRegime:
    def __init__(self, cfg: RegimeRulesConfig) -> None:
        self.cfg = cfg
        self.vol_feature: str | None = None
        self.threshold: float = float("nan")

    def fit(self, train_dates: pl.DataFrame) -> RuleRegime:
        """``train_dates``: one row per training decision date with market-level columns."""
        for f in (self.cfg.vol_feature, self.cfg.vol_fallback_feature):
            if f in train_dates.columns and train_dates[f].drop_nulls().len() > 20:
                self.vol_feature = f
                break
        if self.vol_feature is None:
            raise ValueError("no usable volatility feature for regime rules")
        vals = train_dates[self.vol_feature].drop_nulls().to_numpy()
        self.threshold = float(np.quantile(vals, self.cfg.high_vol_quantile))
        return self

    def predict(self, dates: pl.DataFrame) -> pl.DataFrame:
        assert self.vol_feature is not None
        vol = pl.col(self.vol_feature)
        trend = pl.col(self.cfg.trend_feature)
        vol_state = (
            pl.when(vol.is_null())
            .then(None)
            .when(vol > self.threshold)
            .then(pl.lit("high_vol"))
            .otherwise(pl.lit("normal_vol"))
        )
        trend_state = (
            pl.when(trend.is_null())
            .then(None)
            .when(trend > 0.5)
            .then(pl.lit("uptrend"))
            .otherwise(pl.lit("downtrend"))
        )
        out = dates.select("date", vol_state.alias("vol_state"), trend_state.alias("trend_state"))
        return out.with_columns(
            pl.concat_str([pl.col("trend_state"), pl.col("vol_state")], separator="/").alias(
                "regime"
            )
        )


class HMMRegime:
    def __init__(self, cfg: HMMConfig, seed: int = 42) -> None:
        self.cfg = cfg
        self.seed = seed
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.startprob_: np.ndarray | None = None
        self.transmat_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.covars_: np.ndarray | None = None
        self.order_: np.ndarray | None = None

    def _inputs(self, daily: pl.DataFrame) -> tuple[list[dt.date], np.ndarray]:
        d = daily.select(["date", *self.cfg.inputs]).drop_nulls().sort("date")
        return d["date"].to_list(), d.select(self.cfg.inputs).to_numpy().astype(float)

    def fit(self, daily_train: pl.DataFrame) -> HMMRegime:
        from hmmlearn.hmm import GaussianHMM

        _, X = self._inputs(daily_train)
        if X.shape[0] < 50 * self.cfg.n_states:
            raise ValueError("not enough daily observations to fit the HMM")
        self.mean_, self.std_ = X.mean(axis=0), X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        Z = (X - self.mean_) / self.std_
        m = GaussianHMM(
            n_components=self.cfg.n_states,
            covariance_type=self.cfg.covariance_type,
            n_iter=self.cfg.n_iter,
            random_state=self.seed,
        )
        m.fit(Z)
        # order states by the mean of the last input (volatility) — calmest first
        order = np.argsort(m.means_[:, -1])
        self.order_ = order
        self.startprob_ = m.startprob_[order]
        self.transmat_ = m.transmat_[np.ix_(order, order)]
        self.means_ = m.means_[order]
        self.covars_ = np.asarray(m.covars_)[order]  # always full matrices
        return self

    def filtered_probabilities(self, daily: pl.DataFrame) -> pl.DataFrame:
        """Forward-filtered P(state_t | x_1..x_t) for each daily observation."""
        assert self.transmat_ is not None and self.means_ is not None
        assert self.covars_ is not None and self.startprob_ is not None
        assert self.mean_ is not None and self.std_ is not None
        dates, X = self._inputs(daily)
        k = self.cfg.n_states
        cols = [f"hmm_p_state_{i}" for i in range(k)]
        if not dates:
            return pl.DataFrame(schema={"date": pl.Date, **{c: pl.Float64 for c in cols}})
        Z = (X - self.mean_) / self.std_
        loglik = np.column_stack(
            [
                multivariate_normal(self.means_[i], self.covars_[i], allow_singular=True).logpdf(Z)
                for i in range(k)
            ]
        ).reshape(len(dates), k)
        log_a = np.log(self.transmat_ + 1e-300)
        alpha = np.log(self.startprob_ + 1e-300) + loglik[0]
        alpha -= logsumexp(alpha)
        probs = np.empty((len(dates), k))
        probs[0] = np.exp(alpha)
        for t in range(1, len(dates)):
            alpha = logsumexp(alpha[:, None] + log_a, axis=0) + loglik[t]
            alpha -= logsumexp(alpha)
            probs[t] = np.exp(alpha)
        return pl.DataFrame({"date": dates, **{c: probs[:, i] for i, c in enumerate(cols)}})


def asof_dates(daily_probs: pl.DataFrame, decision_dates: list[dt.date]) -> pl.DataFrame:
    """Attach the latest daily filtered probability at or before each decision date."""
    left = pl.DataFrame({"date": decision_dates}, schema={"date": pl.Date}).sort("date")
    return left.join_asof(daily_probs.sort("date"), on="date", strategy="backward")
