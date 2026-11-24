"""Nested, budgeted Optuna tuning of the ranker (PRD §9.4, FR-MODEL-07).

Runs strictly inside a training fold: inner-train / inner-validation are a purged split of
the fold's training rows, so the outer test block is never seen. Objective = mean rank IC
on the inner validation slice. Every trial is appended to the global trial log.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import optuna
import polars as pl

from quantxgb.config import ModelConfig
from quantxgb.evaluation.signal_metrics import rank_ic_by_date
from quantxgb.models.alpha_ranker import XGBAlphaRanker
from quantxgb.tracking.trial_log import TrialLog
from quantxgb.validation.splits import inner_split

_INT_PARAMS = {"max_depth", "min_child_weight", "n_estimators"}
_LOG_PARAMS = {"learning_rate", "reg_lambda", "reg_alpha"}


def suggest(trial: optuna.Trial, space: dict[str, list[float]]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for name, (lo, hi) in space.items():
        if name in _INT_PARAMS:
            params[name] = trial.suggest_int(name, int(lo), int(hi))
        else:
            params[name] = trial.suggest_float(name, float(lo), float(hi), log=name in _LOG_PARAMS)
    return params


def tune_ranker(
    train: pl.DataFrame,
    features: list[str],
    cfg: ModelConfig,
    sessions: list[dt.date],
    embargo: int,
    trial_log: TrialLog | None = None,
    run_id: str = "",
    fold: int = -1,
) -> dict[str, Any]:
    inner_train, inner_val = inner_split(
        train, cfg.ranker.inner_validation_fraction, sessions, embargo
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial, cfg.tuning.search_space)
        model = XGBAlphaRanker(cfg.ranker, params=params, seed=cfg.seed)
        # early stopping on the inner validation slice; IC measured on the same slice
        model.fit(inner_train, features, valid=inner_val)
        scored = inner_val.with_columns(pl.Series("score", model.predict(inner_val)))
        ic = rank_ic_by_date(scored)["ic"]
        value = float(ic.mean()) if ic.len() else -1.0  # type: ignore[arg-type]
        if trial_log is not None:
            trial_log.log(
                "optuna", run_id, fold=fold, trial=trial.number, value=value, params=params
            )
        return value

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg.seed)
    )
    study.optimize(objective, n_trials=cfg.tuning.n_trials, show_progress_bar=False)
    return dict(study.best_params)
