"""Model A — XGBoost learning-to-rank (PRD §9.1, FR-MODEL-02/03).

One query group per decision date; relevance = within-date grade (``fwd_grade``).
Scores are ordinal: they order assets within a date and are not return forecasts.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import xgboost as xgb

from quantxgb.config import RankerConfig
from quantxgb.models.base import query_ids, to_matrix


class XGBAlphaRanker:
    def __init__(self, cfg: RankerConfig, params: dict[str, Any] | None = None, seed: int = 42):
        self.cfg = cfg
        self.params = {**cfg.params, **(params or {})}
        self.seed = seed
        self.name = f"xgb_{cfg.objective.split(':')[1]}"
        self.model: xgb.XGBRanker | None = None
        self.features: list[str] = []
        self.best_iteration: int | None = None

    def fit(
        self,
        train: pl.DataFrame,
        features: list[str],
        valid: pl.DataFrame | None = None,
        label: str = "fwd_grade",
    ) -> XGBAlphaRanker:
        train = train.filter(pl.col(label).is_not_null()).sort(["date", "ticker"])
        self.features = list(features)
        kw: dict[str, Any] = {
            "objective": self.cfg.objective,
            "eval_metric": self.cfg.eval_metric,
            "tree_method": "hist",
            "random_state": self.seed,
            "n_jobs": -1,
            **self.params,
        }
        if "n_estimators" in kw:
            kw["n_estimators"] = int(kw["n_estimators"])
        if "max_depth" in kw:
            kw["max_depth"] = int(kw["max_depth"])
        fit_kw: dict[str, Any] = {"qid": query_ids(train), "verbose": False}
        if valid is not None:
            valid = valid.filter(pl.col(label).is_not_null()).sort(["date", "ticker"])
        if valid is not None and valid.height > 0:
            kw["early_stopping_rounds"] = self.cfg.early_stopping_rounds
            fit_kw["eval_set"] = [(to_matrix(valid, self.features), valid[label].to_numpy())]
            fit_kw["eval_qid"] = [query_ids(valid)]
        self.model = xgb.XGBRanker(**kw)
        self.model.fit(to_matrix(train, self.features), train[label].to_numpy(), **fit_kw)
        if "early_stopping_rounds" in kw:
            self.best_iteration = int(self.model.best_iteration)
        return self

    def with_n_estimators(self, n: int) -> XGBAlphaRanker:
        """Fresh, unfitted copy with a fixed number of trees (no early stopping)."""
        return XGBAlphaRanker(self.cfg, params={**self.params, "n_estimators": n}, seed=self.seed)

    def _iteration_range(self) -> tuple[int, int]:
        assert self.model is not None
        if self.best_iteration is not None:
            return (0, self.best_iteration + 1)
        return (0, int(self.model.get_booster().num_boosted_rounds()))

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("model not fitted")
        return np.asarray(
            self.model.predict(
                to_matrix(df, self.features), iteration_range=self._iteration_range()
            )
        )

    def contributions(self, df: pl.DataFrame) -> np.ndarray:
        """Exact TreeSHAP contributions (n, n_features + 1); last column = bias."""
        if self.model is None:
            raise RuntimeError("model not fitted")
        dm = xgb.DMatrix(to_matrix(df, self.features), feature_names=self.features)
        out = self.model.get_booster().predict(
            dm, pred_contribs=True, iteration_range=self._iteration_range()
        )
        return np.asarray(out)

    def save(self, path: str) -> None:
        assert self.model is not None
        self.model.save_model(path)
