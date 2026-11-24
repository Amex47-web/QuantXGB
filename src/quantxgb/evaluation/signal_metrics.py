"""Signal-quality and distribution-forecast metrics (PRD §9.5)."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
from sklearn.metrics import ndcg_score


def rank_ic_by_date(
    df: pl.DataFrame, score: str = "score", target: str = "fwd_xs_ret", min_assets: int = 3
) -> pl.DataFrame:
    """Spearman correlation between score and realised target, per decision date."""
    d = df.filter(pl.col(score).is_not_null() & pl.col(target).is_not_null())
    d = d.filter(pl.col(score).is_not_nan())
    out = (
        d.group_by("date")
        .agg(
            pl.corr(pl.col(score), pl.col(target), method="spearman").alias("ic"),
            pl.len().alias("n"),
        )
        .filter(pl.col("n") >= min_assets)
        .sort("date")
    )
    return out.filter(pl.col("ic").is_not_null() & pl.col("ic").is_not_nan())


def newey_west_tstat(x: np.ndarray, lags: int) -> float:
    """t-statistic of the mean of ``x`` with Newey–West (Bartlett) standard errors."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 3:
        return float("nan")
    e = x - x.mean()
    s = float(e @ e) / n
    for lag in range(1, min(lags, n - 1) + 1):
        w = 1.0 - lag / (lags + 1.0)
        s += 2.0 * w * float(e[lag:] @ e[:-lag]) / n
    se = math.sqrt(max(s, 0.0) / n)
    return float(x.mean() / se) if se > 0 else float("nan")


def ic_summary(ic: pl.DataFrame, nw_lags: int) -> dict[str, float]:
    x = ic["ic"].to_numpy()
    if x.size == 0:
        return {"ic_mean": float("nan"), "n_dates": 0}
    sd = float(x.std(ddof=1)) if x.size > 1 else float("nan")
    return {
        "ic_mean": float(x.mean()),
        "ic_median": float(np.median(x)),
        "ic_std": sd,
        "icir": float(x.mean() / sd) if sd and sd > 0 else float("nan"),
        "ic_tstat_nw": newey_west_tstat(x, nw_lags),
        "ic_pct_positive": float((x > 0).mean()),
        "n_dates": int(x.size),
    }


def ndcg_by_date(
    df: pl.DataFrame, score: str = "score", grade: str = "fwd_grade", k: int = 5
) -> pl.DataFrame:
    """NDCG@k per date (linear gains on the 0..G-1 grades, sklearn definition)."""
    d = df.filter(pl.col(score).is_not_null() & pl.col(grade).is_not_null())
    rows = []
    for (date,), g in d.group_by(["date"], maintain_order=True):
        if g.height < 2:
            continue
        y = g[grade].to_numpy()[None, :].astype(float)
        s = g[score].to_numpy()[None, :].astype(float)
        if np.isnan(s).any():
            continue
        rows.append((date, float(ndcg_score(y, s, k=min(k, g.height)))))
    return pl.DataFrame(
        {"date": [r[0] for r in rows], "ndcg": [r[1] for r in rows]},
        schema={"date": pl.Date, "ndcg": pl.Float64},
    ).sort("date")


def top_k_stats(
    df: pl.DataFrame, score: str = "score", k: int = 5, n_quantiles: int = 5
) -> dict[str, float]:
    """Top-k hit rate (share of top-k with positive excess return) and top-minus-bottom
    quantile spread of raw forward returns, averaged over dates."""
    d = df.filter(pl.col(score).is_not_null() & pl.col("fwd_ret").is_not_null())
    d = d.with_columns(
        pl.col(score).rank("ordinal", descending=True).over("date").alias("_r"),
        pl.len().over("date").alias("_n"),
    )
    top = d.filter(pl.col("_r") <= k)
    hit = top.group_by("date").agg((pl.col("fwd_xs_ret") > 0).mean().alias("h"))["h"]
    q = d.with_columns(((pl.col("_r") - 1) * n_quantiles // pl.col("_n")).alias("_q"))
    spread = (
        q.group_by("date")
        .agg(
            pl.col("fwd_ret").filter(pl.col("_q") == 0).mean().alias("top"),
            pl.col("fwd_ret").filter(pl.col("_q") == n_quantiles - 1).mean().alias("bot"),
        )
        .with_columns((pl.col("top") - pl.col("bot")).alias("s"))["s"]
        .drop_nulls()
    )
    return {
        "top_k_hit_rate": float(hit.mean()) if hit.len() else float("nan"),  # type: ignore[arg-type]
        "quantile_spread": float(spread.mean()) if spread.len() else float("nan"),  # type: ignore[arg-type]
    }


def pinball_loss(y: np.ndarray, q: np.ndarray, alpha: float) -> float:
    diff = y - q
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1) * diff)))


def quantile_metrics(y: np.ndarray, preds: np.ndarray, alphas: list[float]) -> dict[str, float]:
    """Pinball loss per alpha, central-interval coverage and width."""
    ok = ~np.isnan(y)
    y, preds = y[ok], preds[ok]
    out: dict[str, float] = {}
    for j, a in enumerate(alphas):
        out[f"pinball_{a:g}"] = pinball_loss(y, preds[:, j], a)
    if len(alphas) >= 2:
        lo, hi = preds[:, 0], preds[:, -1]
        out["coverage"] = float(((y >= lo) & (y <= hi)).mean()) if y.size else float("nan")
        out["nominal_coverage"] = float(alphas[-1] - alphas[0])
        out["interval_width"] = float(np.mean(hi - lo)) if y.size else float("nan")
    return out


def historical_quantile_baseline(train_y: np.ndarray, n: int, alphas: list[float]) -> np.ndarray:
    """Unconditional quantiles of training labels — the benchmark for the quantile model."""
    q = np.quantile(train_y[~np.isnan(train_y)], alphas)
    return np.tile(q, (n, 1))
