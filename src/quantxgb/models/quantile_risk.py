"""Model B — XGBoost multi-quantile regression (PRD §9.2, FR-MODEL-04).

Target: absolute forward return ``fwd_ret``. One model, ``quantile_alpha`` = all alphas.
Quantile crossing is measured on raw outputs and then removed by sorting per row.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import xgboost as xgb

from quantxgb.config import QuantileConfig
from quantxgb.models.base import to_matrix


class XGBQuantileRisk:
    name = "xgb_quantile"

    def __init__(self, cfg: QuantileConfig, seed: int = 42) -> None:
        self.cfg = cfg
        self.alphas = np.array(cfg.alphas, dtype=float)
        self.seed = seed
        self.model: xgb.XGBRegressor | None = None
        self.features: list[str] = []
        self.best_iteration: int | None = None
        self.last_crossing_rate: float = float("nan")

    def with_n_estimators(self, n: int) -> XGBQuantileRisk:
        """Fresh, unfitted copy with a fixed number of trees (no early stopping)."""
        cfg = self.cfg.model_copy(update={"params": {**self.cfg.params, "n_estimators": n}})
        return XGBQuantileRisk(cfg, seed=self.seed)

    def fit(
        self,
        train: pl.DataFrame,
        features: list[str],
        valid: pl.DataFrame | None = None,
        label: str = "fwd_ret",
    ) -> XGBQuantileRisk:
        train = train.filter(pl.col(label).is_not_null())
        self.features = list(features)
        params: dict[str, Any] = dict(self.cfg.params)
        for k in ("n_estimators", "max_depth"):
            if k in params:
                params[k] = int(params[k])
        kw: dict[str, Any] = {
            "objective": "reg:quantileerror",
            "quantile_alpha": self.alphas,
            "tree_method": "hist",
            "random_state": self.seed,
            "n_jobs": -1,
            **params,
        }
        fit_kw: dict[str, Any] = {"verbose": False}
        if valid is not None:
            valid = valid.filter(pl.col(label).is_not_null())
        early = valid is not None and valid.height > 0
        if early:
            assert valid is not None
            kw["early_stopping_rounds"] = self.cfg.early_stopping_rounds
            fit_kw["eval_set"] = [(to_matrix(valid, self.features), valid[label].to_numpy())]
        self.model = xgb.XGBRegressor(**kw)
        self.model.fit(to_matrix(train, self.features), train[label].to_numpy(), **fit_kw)
        self.best_iteration = int(self.model.best_iteration) if early else None
        return self

    def _iteration_range(self) -> tuple[int, int]:
        assert self.model is not None
        if self.best_iteration is not None:
            return (0, self.best_iteration + 1)
        return (0, int(self.model.get_booster().num_boosted_rounds()))

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        """(n, n_alphas) array, non-crossing (sorted per row)."""
        if self.model is None:
            raise RuntimeError("model not fitted")
        raw = np.asarray(
            self.model.predict(
                to_matrix(df, self.features), iteration_range=self._iteration_range()
            )
        )
        if raw.ndim == 1:
            raw = raw[:, None]
        crossed = (np.diff(raw, axis=1) < 0).any(axis=1)
        self.last_crossing_rate = float(crossed.mean()) if raw.shape[0] else float("nan")
        return np.sort(raw, axis=1)
