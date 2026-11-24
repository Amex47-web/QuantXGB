"""Constrained portfolio optimisation (PRD §10.3, FR-PORT-03/04).

Mean-variance:
    max  μᵀw − λ wᵀΣw − γ‖w − w₀‖₁
    s.t. 1ᵀw = 1, 0 ≤ w ≤ w_max, Σ_{i∈c} w_i ≤ c_max, ‖w − w₀‖₁ + s ≤ τ_max

CVaR (Rockafellar–Uryasev, losses L_s = −r_sᵀw):
    max  μᵀw − κ (ζ + 1/((1−β)S) Σ u_s) − γ‖w − w₀‖₁
    s.t. u_s ≥ −r_sᵀw − ζ, u_s ≥ 0, plus the constraints above

``s`` is the weight that must be sold anyway in assets that left the eligible set.
The turnover cap is not applied to the very first allocation (from cash). If a problem is
infeasible, constraints are relaxed in a fixed, logged order: turnover cap → asset-class
caps → hold previous weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np

from quantxgb.config import ConstraintsConfig, OptimizerConfig
from quantxgb.portfolio.covariance import nearest_psd


@dataclass
class OptimResult:
    weights: np.ndarray
    status: str
    relaxed: list[str] = field(default_factory=list)


def _class_matrix(classes: list[str]) -> tuple[list[str], np.ndarray]:
    uniq = sorted(set(classes))
    m = np.array([[1.0 if c == u else 0.0 for c in classes] for u in uniq])
    return uniq, m


def _solve(problem: cp.Problem) -> str:
    for solver in ("CLARABEL", "ECOS", "SCS"):
        if solver not in cp.installed_solvers():
            continue
        try:
            problem.solve(solver=solver)
        except cp.error.SolverError:
            continue
        if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return str(problem.status)
        if problem.status in (cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE):
            return "infeasible"
    return str(problem.status)


def optimize(
    mu: np.ndarray,
    w_prev: np.ndarray,
    classes: list[str],
    cons: ConstraintsConfig,
    opt: OptimizerConfig,
    sigma: np.ndarray | None = None,
    scenarios: np.ndarray | None = None,
    forced_sell: float = 0.0,
) -> OptimResult:
    n = len(mu)
    if n == 0:
        return OptimResult(np.zeros(0), "empty")
    w_prev = np.asarray(w_prev, dtype=float)
    initial = w_prev.sum() + forced_sell < 1e-9
    relax_steps: list[list[str]] = [[], ["turnover_cap"], ["turnover_cap", "class_caps"]]
    for relaxed in relax_steps:
        w = cp.Variable(n)
        constraints = [cp.sum(w) == 1, w <= cons.w_max]
        if cons.long_only:
            constraints.append(w >= 0)
        if "class_caps" not in relaxed:
            _, cm = _class_matrix(classes)
            constraints.append(cm @ w <= cons.asset_class_max)
        trade = cp.norm1(w - w_prev)
        if not initial and "turnover_cap" not in relaxed:
            constraints.append(trade + forced_sell <= cons.turnover_max)
        objective = mu @ w - opt.turnover_penalty * trade
        if opt.type == "mean_variance":
            if sigma is None:
                raise ValueError("mean-variance needs a covariance matrix")
            s = cp.psd_wrap(nearest_psd(sigma))
            objective = objective - opt.risk_aversion * cp.quad_form(w, s)
        else:
            if scenarios is None or scenarios.shape[1] != n:
                raise ValueError("CVaR needs a (S, N) scenario matrix")
            n_s = scenarios.shape[0]
            zeta = cp.Variable()
            u = cp.Variable(n_s, nonneg=True)
            constraints.append(u >= -scenarios @ w - zeta)
            cvar = zeta + cp.sum(u) / ((1 - opt.cvar_beta) * n_s)
            objective = objective - opt.cvar_kappa * cvar
        status = _solve(cp.Problem(cp.Maximize(objective), constraints))
        if status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and w.value is not None:
            x = np.clip(np.asarray(w.value, dtype=float), 0.0 if cons.long_only else -np.inf, None)
            if cons.long_only and x.sum() > 0:
                x = x / x.sum()
            return OptimResult(x, status, relaxed)
    held = w_prev / w_prev.sum() if w_prev.sum() > 0 else np.full(n, 1.0 / n)
    return OptimResult(held, "fallback_hold", ["all"])


def two_piece_normal_scenarios(
    q_lo: np.ndarray,
    q_mid: np.ndarray,
    q_hi: np.ndarray,
    corr: np.ndarray,
    n_scenarios: int,
    z_lo: float,
    z_hi: float,
    seed: int,
) -> np.ndarray:
    """Gaussian-copula scenarios with two-piece-normal marginals matched to three quantiles.

    For z ≥ 0: x = q_mid + z·(q_hi − q_mid)/z_hi;  for z < 0: x = q_mid + z·(q_mid − q_lo)/|z_lo|.
    """
    rng = np.random.default_rng(seed)
    n = len(q_mid)
    c = nearest_psd(corr)
    d = np.sqrt(np.clip(np.diag(c), 1e-12, None))
    c = c / np.outer(d, d)
    z = rng.multivariate_normal(np.zeros(n), c, size=n_scenarios, method="eigh")
    up = (q_hi - q_mid) / z_hi
    dn = (q_mid - q_lo) / abs(z_lo)
    return np.where(z >= 0, q_mid + z * up, q_mid + z * dn)
