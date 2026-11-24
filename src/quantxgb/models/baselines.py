"""Score baselines (PRD §4.5: B4 momentum, B6 linear, B7 random forest).

Weight-based baselines (equal weight, 60/40, market, inverse vol) live in
``quantxgb.backtest.strategies`` because they do not produce scores.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge

from quantxgb.config import BaselinesConfig
from quantxgb.models.base import to_matrix, xs_zscore


class MomentumScorer:
    """B4: score = a single momentum feature (default 12-1 momentum). No fitting."""

    name = "momentum"

    def __init__(self, feature: str = "mom_12_1") -> None:
        self.feature = feature

    def fit(
        self, train: pl.DataFrame, features: list[str], valid: pl.DataFrame | None = None
    ) -> MomentumScorer:
        return self

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        return df[self.feature].cast(pl.Float64).fill_null(np.nan).to_numpy()


class RidgeRanker:
    """B6: ridge regression of excess return on within-date z-scored features."""

    name = "ridge"

    def __init__(self, alpha: float = 10.0) -> None:
        self.alpha = alpha
        self.features: list[str] = []
        self.model = Ridge(alpha=alpha)

    def fit(
        self,
        train: pl.DataFrame,
        features: list[str],
        valid: pl.DataFrame | None = None,
        label: str = "fwd_xs_ret",
    ) -> RidgeRanker:
        self.features = list(features)
        t = xs_zscore(train.filter(pl.col(label).is_not_null()), self.features)
        self.model.fit(to_matrix(t, self.features), t[label].to_numpy())
        return self

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        return np.asarray(
            self.model.predict(to_matrix(xs_zscore(df, self.features), self.features))
        )


class RandomForestRanker:
    """B7: random forest on the same features and excess-return target."""

    name = "random_forest"

    def __init__(self, cfg: BaselinesConfig, seed: int = 42) -> None:
        rf = cfg.random_forest
        self.model = RandomForestRegressor(
            n_estimators=rf.n_estimators,
            max_depth=rf.max_depth,
            min_samples_leaf=rf.min_samples_leaf,
            max_features=rf.max_features,
            random_state=seed,
            n_jobs=-1,
        )
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        self.features: list[str] = []

    def fit(
        self,
        train: pl.DataFrame,
        features: list[str],
        valid: pl.DataFrame | None = None,
        label: str = "fwd_xs_ret",
    ) -> RandomForestRanker:
        self.features = list(features)
        t = train.filter(pl.col(label).is_not_null())
        X = self.imputer.fit_transform(to_matrix(t, self.features))
        self.model.fit(X, t[label].to_numpy())
        return self

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        X = self.imputer.transform(to_matrix(df, self.features))
        return np.asarray(self.model.predict(X))
