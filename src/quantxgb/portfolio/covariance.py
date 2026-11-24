"""Covariance estimation (PRD §10.3, FR-PORT-01). Inputs are daily simple returns."""

from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf


def estimate_covariance(
    returns: np.ndarray, method: str = "ledoit_wolf", halflife: int = 60
) -> np.ndarray:
    """Daily covariance from a (T, N) returns window without NaNs."""
    r = np.asarray(returns, dtype=float)
    if r.ndim != 2 or r.shape[0] < 2:
        raise ValueError("need a (T>=2, N) returns matrix")
    if np.isnan(r).any():
        raise ValueError("returns window contains NaN")
    if method == "sample":
        cov = np.cov(r, rowvar=False, ddof=1)
    elif method == "ledoit_wolf":
        cov = LedoitWolf().fit(r).covariance_
    elif method == "ewma":
        t = r.shape[0]
        lam = 0.5 ** (1.0 / halflife)
        w = lam ** np.arange(t - 1, -1, -1)
        w /= w.sum()
        mu = w @ r
        x = r - mu
        cov = (x * w[:, None]).T @ x
    else:
        raise ValueError(f"unknown covariance method {method}")
    cov = np.atleast_2d(cov)
    return 0.5 * (cov + cov.T)


def nearest_psd(cov: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    vals, vecs = np.linalg.eigh(0.5 * (cov + cov.T))
    vals = np.clip(vals, eps, None)
    return (vecs * vals) @ vecs.T
