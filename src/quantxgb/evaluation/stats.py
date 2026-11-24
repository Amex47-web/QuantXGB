"""Statistical tests for backtest overfitting (PRD §9.5, FR-VAL-05/06).

* Stationary bootstrap (Politis & Romano, 1994) confidence intervals.
* Probabilistic / Deflated Sharpe Ratio (Bailey & López de Prado, 2012/2014).
* Probability of Backtest Overfitting via CSCV (Bailey, Borwein, López de Prado, Zhu).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable

import numpy as np
from scipy.stats import kurtosis, norm, skew

EULER_GAMMA = 0.5772156649015329


def stationary_bootstrap_indices(
    n: int, mean_block: float, n_resamples: int, rng: np.random.Generator
) -> np.ndarray:
    """(n_resamples, n) index matrix; blocks have geometric lengths with mean ``mean_block``."""
    p = 1.0 / max(mean_block, 1.0)
    idx = np.empty((n_resamples, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, n_resamples)
    new_block = rng.random((n_resamples, n)) < p
    starts = rng.integers(0, n, (n_resamples, n))
    for t in range(1, n):
        idx[:, t] = np.where(new_block[:, t], starts[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def bootstrap_ci(
    x: np.ndarray,
    stat: Callable[[np.ndarray], float],
    mean_block: float = 20,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """(point estimate, lower, upper) percentile CI under the stationary bootstrap."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size < 10:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(x.size, mean_block, n_resamples, rng)
    stats = np.array([stat(x[i]) for i in idx])
    a = (1 - confidence) / 2
    return float(stat(x)), float(np.nanquantile(stats, a)), float(np.nanquantile(stats, 1 - a))


def sharpe_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if r.size < 2:
        return float("nan")
    sd = r.std(ddof=1)
    return float(r.mean() / sd * math.sqrt(periods_per_year)) if sd > 0 else float("nan")


def probabilistic_sharpe_ratio(returns: np.ndarray, sr_benchmark: float = 0.0) -> float:
    """PSR: P(true per-period SR > benchmark) given sample length, skew and kurtosis.
    ``sr_benchmark`` is per-period (not annualised)."""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    n = r.size
    if n < 10 or r.std(ddof=1) == 0:
        return float("nan")
    sr = r.mean() / r.std(ddof=1)
    g3 = skew(r)
    g4 = kurtosis(r, fisher=False)  # non-excess
    denom = math.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr**2, 1e-12))
    return float(norm.cdf((sr - sr_benchmark) * math.sqrt(n - 1) / denom))


def expected_max_sharpe(trial_srs: np.ndarray) -> float:
    """Expected maximum per-period SR among N independent trials with the observed
    cross-trial variance (False Strategy Theorem)."""
    srs = np.asarray(trial_srs, dtype=float)
    srs = srs[~np.isnan(srs)]
    n = srs.size
    if n < 2:
        return 0.0
    v = srs.var(ddof=1)
    return float(
        math.sqrt(v)
        * ((1 - EULER_GAMMA) * norm.ppf(1 - 1 / n) + EULER_GAMMA * norm.ppf(1 - 1 / (n * math.e)))
    )


def deflated_sharpe_ratio(returns: np.ndarray, trial_srs: np.ndarray) -> dict[str, float]:
    """DSR = PSR against the expected maximum SR of all trials tried.

    ``trial_srs`` must contain the per-period Sharpe ratio of EVERY configuration that was
    evaluated out of sample (including the selected one) — see ``tracking.trial_log``.
    """
    sr0 = expected_max_sharpe(trial_srs)
    return {
        "dsr": probabilistic_sharpe_ratio(returns, sr0),
        "sr0_per_period": sr0,
        "n_trials": int(np.sum(~np.isnan(np.asarray(trial_srs, dtype=float)))),
    }


def pbo_cscv(returns: np.ndarray, n_blocks: int = 16) -> dict[str, float]:
    """Probability of Backtest Overfitting by combinatorially symmetric cross-validation.

    ``returns``: (T, N) matrix of per-period returns of N candidate configurations over
    the same dates. For each split of the T rows into equal halves of ``n_blocks`` blocks
    (C(S, S/2) splits), the in-sample best configuration (by Sharpe) is ranked out of
    sample; PBO is the share of splits where it falls at or below the OOS median.
    """
    m = np.asarray(returns, dtype=float)
    m = m[~np.isnan(m).any(axis=1)]
    t_len, n = m.shape
    if n < 2 or t_len < n_blocks * 2 or n_blocks % 2:
        return {"pbo": float("nan"), "n_splits": 0}
    blocks = np.array_split(np.arange(t_len), n_blocks)

    def sr(rows: np.ndarray) -> np.ndarray:
        sub = m[rows]
        sd = sub.std(axis=0, ddof=1)
        return np.where(sd > 0, sub.mean(axis=0) / np.where(sd > 0, sd, 1), -np.inf)

    logits = []
    for combo in itertools.combinations(range(n_blocks), n_blocks // 2):
        is_rows = np.concatenate([blocks[i] for i in combo])
        oos_rows = np.concatenate([blocks[i] for i in range(n_blocks) if i not in combo])
        best = int(np.argmax(sr(is_rows)))
        oos = sr(oos_rows)
        rank = (oos < oos[best]).sum() + 0.5 * ((oos == oos[best]).sum() - 1) + 1
        w = rank / (n + 1)
        logits.append(math.log(w / (1 - w)))
    arr = np.array(logits)
    return {
        "pbo": float((arr <= 0).mean()),
        "n_splits": int(arr.size),
        "logit_median": float(np.median(arr)),
    }
