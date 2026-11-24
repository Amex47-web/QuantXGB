"""Walk-forward model research (PRD Stage 3, §9).

For every fold (expanding window, purged + embargoed, development period only):

1. fit regime models on the training window (rule thresholds; HMM on daily inputs before
   the test block) and attach filtered regime probabilities;
2. fit baselines and XGBoost models (early stopping on a purged inner split, then refit on
   the whole training fold with the selected number of trees);
3. score the test block out of sample.

Outputs (``artifacts/runs/<run_id>/``): out-of-sample predictions, quantile forecasts,
regimes, per-fold and aggregate metrics, SHAP importance and the final models trained on
the full development period (used for live scoring — never for evaluation).
"""

from __future__ import annotations

import datetime as dt
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from quantxgb.config import AppConfig
from quantxgb.evaluation.signal_metrics import (
    historical_quantile_baseline,
    ic_summary,
    ndcg_by_date,
    quantile_metrics,
    rank_ic_by_date,
    top_k_stats,
)
from quantxgb.evaluation.stats import bootstrap_ci
from quantxgb.models.alpha_ranker import XGBAlphaRanker
from quantxgb.models.baselines import MomentumScorer, RandomForestRanker, RidgeRanker
from quantxgb.models.quantile_risk import XGBQuantileRisk
from quantxgb.models.regime_detector import HMMRegime, RuleRegime, asof_dates
from quantxgb.models.tuning import tune_ranker
from quantxgb.research.dataset import Dataset
from quantxgb.tracking.mlflow_utils import Tracker
from quantxgb.tracking.trial_log import TrialLog
from quantxgb.utils.io import get_logger, write_json, write_parquet_atomic
from quantxgb.validation.splits import (
    Fold,
    assert_no_holdout,
    development_view,
    inner_split,
    walk_forward_folds,
)

log = get_logger(__name__)

LABEL_COLS = ["fwd_ret", "fwd_xs_ret", "fwd_grade", "label_start", "label_end"]
MARKET_PREFIXES = ("market_", "avg_corr_")


@dataclass
class RunResult:
    run_id: str
    run_dir: Path
    metrics: dict[str, Any]
    predictions: pl.DataFrame
    quantiles: pl.DataFrame
    regimes: pl.DataFrame
    fold_metrics: list[dict[str, Any]] = field(default_factory=list)


def quantile_col(alpha: float) -> str:
    """Column name for a quantile forecast, e.g. 0.05 → q05, 0.5 → q50."""
    return f"q{round(alpha * 100):02d}"


def sessions_per_signal(dates: list[dt.date], sessions: list[dt.date]) -> float:
    if len(dates) < 2:
        return 1.0
    idx = np.searchsorted(
        np.array(sessions, dtype="datetime64[D]"), np.array(dates, dtype="datetime64[D]")
    )
    return float(np.median(np.diff(idx)))


def date_level(frame: pl.DataFrame, macro_cols: list[str]) -> pl.DataFrame:
    """One row per decision date with market-wide columns (identical across assets)."""
    cols = [c for c in frame.columns if c.startswith(MARKET_PREFIXES) or c in macro_cols]
    return frame.group_by("date").agg([pl.col(c).first() for c in cols]).sort("date")


def _shuffle_labels(df: pl.DataFrame, seed: int) -> pl.DataFrame:
    """Permute labels across assets within each date (sanity check: destroys signal)."""
    rng = np.random.default_rng(seed)
    parts = []
    for (_d,), g in df.group_by(["date"], maintain_order=True):
        perm = rng.permutation(g.height)
        parts.append(
            g.with_columns(
                [
                    pl.Series(c, g[c].to_numpy()[perm])
                    for c in ("fwd_ret", "fwd_xs_ret", "fwd_grade")
                ]
            )
        )
    return pl.concat(parts).sort(["date", "ticker"])


def _fit_two_step(
    model: XGBAlphaRanker | XGBQuantileRisk,
    train: pl.DataFrame,
    feats: list[str],
    sessions: list[dt.date],
    cfg: AppConfig,
) -> XGBAlphaRanker | XGBQuantileRisk:
    """Early-stop on a purged inner split, then refit on the whole fold with the selected
    number of trees (the inner validation slice is the most recent training data)."""
    try:
        inner_tr, inner_va = inner_split(
            train,
            cfg.model.ranker.inner_validation_fraction,
            sessions,
            cfg.validation.embargo_sessions,
        )
    except ValueError:
        return model.fit(train, feats)
    model.fit(inner_tr, feats, valid=inner_va)
    if model.best_iteration is None:
        return model
    return model.with_n_estimators(model.best_iteration + 1).fit(train, feats)


class FoldModels:
    """All models of one fold (or the final development-period fit)."""

    def __init__(self, cfg: AppConfig, ranker_params: dict[str, Any] | None = None) -> None:
        mc = cfg.model
        self.cfg = cfg
        self.ranker = XGBAlphaRanker(mc.ranker, params=ranker_params, seed=mc.seed)
        self.quantile = XGBQuantileRisk(mc.quantile, seed=mc.seed)
        self.scorers: dict[str, Any] = {
            "momentum": MomentumScorer(mc.baselines.momentum_feature),
            "ridge": RidgeRanker(mc.baselines.ridge_alpha),
            "random_forest": RandomForestRanker(mc.baselines, seed=mc.seed),
        }
        self.rule: RuleRegime | None = None
        self.hmm: HMMRegime | None = None
        self.features: list[str] = []

    def fit_regimes(self, train_dates: pl.DataFrame, daily: pl.DataFrame, before: dt.date) -> None:
        rc = self.cfg.model.regime
        self.rule = RuleRegime(rc.rules).fit(train_dates)
        self.hmm = None
        if rc.hmm.enabled:
            try:
                self.hmm = HMMRegime(rc.hmm, seed=self.cfg.model.seed).fit(
                    daily.filter(pl.col("date") < before)
                )
            except ValueError as exc:
                log.warning("HMM not fitted: %s", exc)

    def regime_frame(self, dates_df: pl.DataFrame, daily: pl.DataFrame) -> pl.DataFrame:
        assert self.rule is not None
        out = self.rule.predict(dates_df)
        if self.hmm is not None:
            probs = asof_dates(self.hmm.filtered_probabilities(daily), out["date"].to_list())
            out = out.join(probs, on="date", how="left")
        return out

    def hmm_feature_names(self) -> list[str]:
        if self.hmm is None or not self.cfg.model.regime.add_regime_features:
            return []
        return [f"hmm_p_state_{i}" for i in range(self.cfg.model.regime.hmm.n_states)]

    def fit(self, train: pl.DataFrame, features: list[str], sessions: list[dt.date]) -> None:
        self.features = features
        ranker = _fit_two_step(self.ranker, train, features, sessions, self.cfg)
        quantile = _fit_two_step(self.quantile, train, features, sessions, self.cfg)
        assert isinstance(ranker, XGBAlphaRanker) and isinstance(quantile, XGBQuantileRisk)
        self.ranker, self.quantile = ranker, quantile
        for s in self.scorers.values():
            s.fit(train, features)

    def score(self, df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
        """(long scores: date,ticker,model,score ; quantiles: date,ticker,q_*)."""
        ids = df.select("date", "ticker")

        def col(name: str, v: np.ndarray) -> pl.Series:
            return pl.Series(name, np.asarray(v, dtype=np.float64))

        parts = [
            ids.with_columns(
                pl.lit(self.ranker.name).alias("model"), col("score", self.ranker.predict(df))
            )
        ]
        for name, s in self.scorers.items():
            parts.append(ids.with_columns(pl.lit(name).alias("model"), col("score", s.predict(df))))
        q = self.quantile.predict(df).astype(np.float64)
        qcols = {quantile_col(a): q[:, j] for j, a in enumerate(self.cfg.model.quantile.alphas)}
        quant = ids.with_columns([pl.Series(k, v) for k, v in qcols.items()])
        return pl.concat(parts), quant


def run_walk_forward(
    ds: Dataset,
    cfg: AppConfig,
    run_id: str,
    run_dir: Path,
    tracker: Tracker | None = None,
    trial_log: TrialLog | None = None,
    shuffle_labels: bool = False,
    max_folds: int | None = None,
) -> RunResult:
    splits = cfg.backtest.splits
    dev = development_view(ds.frame, splits)
    if shuffle_labels:
        dev = _shuffle_labels(dev, cfg.model.seed)
    macro_cols = [m.name for m in cfg.features.macro if m.name in dev.columns]
    dates_all = date_level(dev, macro_cols)
    folds = walk_forward_folds(
        dev["date"].unique().to_list(),
        ds.sessions,
        splits.development.start,
        splits.development.end,
        cfg.validation.min_train_years,
        cfg.validation.test_months,
        cfg.validation.embargo_sessions,
    )
    if max_folds is not None:
        folds = folds[:max_folds]
    if not folds:
        raise RuntimeError("no walk-forward folds — check development period vs. data range")
    log.info("run %s: %d folds, %d features", run_id, len(folds), len(ds.feature_names))

    preds, quants, regimes, shap_rows, local_rows, fold_metrics = [], [], [], [], [], []
    for fold in folds:
        p, q, r, imp, local, fm = _run_fold(fold, dev, dates_all, ds, cfg, run_id, trial_log)
        preds.append(p)
        quants.append(q)
        regimes.append(r)
        shap_rows.append(imp)
        local_rows.append(local)
        fold_metrics.append(fm)
        log.info(
            "fold %d [%s → %s] xgb IC=%.4f n_train=%d",
            fold.index,
            fold.test_start,
            fold.test_end,
            fm.get("ic_mean_xgb", float("nan")),
            fm["n_train"],
        )
        if tracker:
            tracker.metrics({k: v for k, v in fm.items() if isinstance(v, float)}, step=fold.index)

    labels = dev.select("date", "ticker", "asset_class", *LABEL_COLS)
    predictions = pl.concat(preds).join(labels, on=["date", "ticker"], how="left")
    quantiles = pl.concat(quants).join(labels, on=["date", "ticker"], how="left")
    regime_df = pl.concat(regimes).unique(subset="date", keep="first").sort("date")
    importance = pl.concat(shap_rows)

    metrics = aggregate_metrics(predictions, quantiles, regime_df, dev, ds, cfg, folds)
    metrics["n_folds"] = len(folds)
    metrics["shuffle_labels"] = shuffle_labels

    run_dir.mkdir(parents=True, exist_ok=True)
    write_parquet_atomic(predictions, run_dir / "predictions.parquet")
    write_parquet_atomic(quantiles, run_dir / "quantiles.parquet")
    write_parquet_atomic(regime_df, run_dir / "regimes.parquet")
    write_parquet_atomic(importance, run_dir / "feature_importance.parquet")
    # local TreeSHAP of the fold model that produced each OOS prediction
    write_parquet_atomic(pl.concat(local_rows, how="diagonal"), run_dir / "shap_oos.parquet")
    write_json(fold_metrics, run_dir / "fold_metrics.json")
    write_json(metrics, run_dir / "metrics.json")
    write_json([f.__dict__ for f in folds], run_dir / "folds.json")

    final = fit_final_models(dev, dates_all, ds, cfg)
    (run_dir / "models").mkdir(exist_ok=True)
    with (run_dir / "models" / "final.pkl").open("wb") as fh:
        pickle.dump(final, fh)

    if trial_log is not None:
        trial_log.log(
            "walk_forward",
            run_id,
            config_hash=cfg.section_hash("features", "labels", "model", "validation"),
            ic_mean_xgb=metrics["models"].get(final.ranker.name, {}).get("ic_mean"),
            shuffle_labels=shuffle_labels,
        )
    if tracker:
        flat = {
            f"{m}.{k}": v
            for m, d in metrics["models"].items()
            for k, v in d.items()
            if isinstance(v, float)
        }
        tracker.metrics(flat)
        tracker.artifacts(run_dir / "metrics.json")
    return RunResult(run_id, run_dir, metrics, predictions, quantiles, regime_df, fold_metrics)


def _run_fold(
    fold: Fold,
    dev: pl.DataFrame,
    dates_all: pl.DataFrame,
    ds: Dataset,
    cfg: AppConfig,
    run_id: str,
    trial_log: TrialLog | None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    train = dev.filter(fold.train_mask()).sort(["date", "ticker"])
    test = dev.filter(fold.test_mask()).sort(["date", "ticker"])
    assert train["label_end"].max() < fold.test_start, "purging failed"  # type: ignore[operator]

    fm_obj = FoldModels(cfg)
    train_dates = dates_all.filter(pl.col("date").is_in(train["date"].unique().implode()))
    fm_obj.fit_regimes(train_dates, ds.market_daily, before=fold.test_start)
    fold_dates = dates_all.filter(pl.col("date") <= fold.test_end)
    reg = fm_obj.regime_frame(fold_dates, ds.market_daily.filter(pl.col("date") <= fold.test_end))
    hmm_cols = fm_obj.hmm_feature_names()
    if hmm_cols:
        train = train.join(reg.select("date", *hmm_cols), on="date", how="left")
        test = test.join(reg.select("date", *hmm_cols), on="date", how="left")
    features = ds.feature_names + hmm_cols

    params = None
    if cfg.model.tuning.enabled:
        params = tune_ranker(
            train,
            features,
            cfg.model,
            ds.sessions,
            cfg.validation.embargo_sessions,
            trial_log,
            run_id,
            fold.index,
        )
        fm_obj.ranker = XGBAlphaRanker(cfg.model.ranker, params=params, seed=cfg.model.seed)
    fm_obj.fit(train, features, ds.sessions)
    scores, quant = fm_obj.score(test)
    scores = scores.with_columns(pl.lit(fold.index).alias("fold"))
    quant = quant.with_columns(pl.lit(fold.index).alias("fold"))

    contrib = fm_obj.ranker.contributions(test)
    imp = pl.DataFrame(
        {
            "fold": [fold.index] * len(features),
            "feature": features,
            "mean_abs_shap": np.abs(contrib[:, :-1]).mean(axis=0),
        }
    )
    local = test.select("date", "ticker").with_columns(
        [pl.Series(f, contrib[:, j].astype(np.float64)) for j, f in enumerate(features)]
        + [pl.Series("_bias", contrib[:, -1].astype(np.float64)), pl.lit(fold.index).alias("fold")]
    )
    test_reg = reg.filter(pl.col("date").is_between(fold.test_start, fold.test_end))

    fm: dict[str, Any] = {
        "fold": fold.index,
        "test_start": str(fold.test_start),
        "test_end": str(fold.test_end),
        "label_cutoff": str(fold.label_cutoff),
        "n_train": train.height,
        "n_test": test.height,
        "ranker_trees": fm_obj.ranker._iteration_range()[1],
        "tuned_params": params,
        "quantile_crossing_rate": fm_obj.quantile.last_crossing_rate,
        "rule_vol_threshold": fm_obj.rule.threshold if fm_obj.rule else None,
    }
    labelled = scores.join(test.select("date", "ticker", "fwd_xs_ret"), on=["date", "ticker"])
    for model, g in labelled.group_by("model"):
        ic = rank_ic_by_date(g)
        key = "xgb" if str(model[0]).startswith("xgb") else str(model[0])
        fm[f"ic_mean_{key}"] = float(ic["ic"].mean()) if ic.height else float("nan")  # type: ignore[arg-type]
    return scores, quant, test_reg, imp, local, fm


def fit_final_models(
    dev: pl.DataFrame, dates_all: pl.DataFrame, ds: Dataset, cfg: AppConfig
) -> FoldModels:
    """Fit on every labelled development row (labels end before the holdout)."""
    train = dev.filter(pl.col("fwd_ret").is_not_null()).sort(["date", "ticker"])
    assert_no_holdout(train, cfg.backtest.splits)
    last = cfg.backtest.splits.development.end
    fm = FoldModels(cfg)
    fm.fit_regimes(
        dates_all.filter(pl.col("date").is_in(train["date"].unique().implode())),
        ds.market_daily,
        before=last + dt.timedelta(days=1),
    )
    hmm_cols = fm.hmm_feature_names()
    if hmm_cols:
        reg = fm.regime_frame(dates_all, ds.market_daily.filter(pl.col("date") <= last))
        train = train.join(reg.select("date", *hmm_cols), on="date", how="left")
    fm.fit(train, ds.feature_names + hmm_cols, ds.sessions)
    return fm


def aggregate_metrics(
    predictions: pl.DataFrame,
    quantiles: pl.DataFrame,
    regimes: pl.DataFrame,
    dev: pl.DataFrame,
    ds: Dataset,
    cfg: AppConfig,
    folds: list[Fold],
) -> dict[str, Any]:
    sps = sessions_per_signal(ds.dates, ds.sessions)
    nw = cfg.validation.ic_newey_west_lags or max(1, math.ceil(cfg.labels.horizon / sps))
    bc = cfg.validation.bootstrap
    block_dates = max(1.0, bc.mean_block_length / sps)
    out: dict[str, Any] = {
        "newey_west_lags": nw,
        "sessions_per_signal": sps,
        "models": {},
        "by_regime": {},
    }
    for (model,), g in predictions.group_by(["model"], maintain_order=True):
        ic = rank_ic_by_date(g)
        summary = ic_summary(ic, nw)
        _, lo, hi = bootstrap_ci(
            ic["ic"].to_numpy(),
            np.mean,
            block_dates,
            bc.n_resamples,
            bc.confidence,
            seed=cfg.model.seed,
        )
        summary.update({"ic_mean_ci_low": lo, "ic_mean_ci_high": hi})
        nd = ndcg_by_date(g)
        summary["ndcg_at_5"] = float(nd["ndcg"].mean()) if nd.height else float("nan")  # type: ignore[arg-type]
        summary.update(top_k_stats(g, k=cfg.backtest.benchmarks.momentum_top_k))
        out["models"][str(model)] = summary
        by_reg = ic.join(regimes.select("date", "regime"), on="date", how="left")
        out["by_regime"][str(model)] = {
            str(r[0]): {"ic_mean": float(x["ic"].mean()), "n_dates": x.height}  # type: ignore[arg-type]
            for r, x in by_reg.group_by(["regime"])
            if r[0] is not None
        }
    alphas = cfg.model.quantile.alphas
    qcols = [quantile_col(a) for a in alphas]
    lab = quantiles.filter(pl.col("fwd_ret").is_not_null())
    if lab.height:
        y = lab["fwd_ret"].to_numpy()
        qm = quantile_metrics(y, lab.select(qcols).to_numpy(), alphas)
        # benchmark: unconditional quantiles of each fold's training labels
        base_parts = []
        for f in folds:
            tr_y = dev.filter(f.train_mask())["fwd_ret"].to_numpy()
            te = lab.filter(pl.col("fold") == f.index)
            if te.height:
                base_parts.append(
                    (
                        te["fwd_ret"].to_numpy(),
                        historical_quantile_baseline(tr_y, te.height, alphas),
                    )
                )
        if base_parts:
            by = np.concatenate([b[0] for b in base_parts])
            bq = np.vstack([b[1] for b in base_parts])
            qm.update({f"baseline_{k}": v for k, v in quantile_metrics(by, bq, alphas).items()})
        cov_reg = lab.join(regimes.select("date", "regime"), on="date", how="left").with_columns(
            (
                (pl.col("fwd_ret") >= pl.col(qcols[0])) & (pl.col("fwd_ret") <= pl.col(qcols[-1]))
            ).alias("in")
        )
        out["quantile"] = {
            **qm,
            "coverage_by_regime": {
                str(r[0]): float(g["in"].cast(pl.Float64).to_numpy().mean())
                for r, g in cov_reg.group_by(["regime"])
                if r[0] is not None
            },
        }
    return out
