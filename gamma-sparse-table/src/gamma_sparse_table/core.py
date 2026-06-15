"""Core routines for log-gamma estimation in sparse contingency tables.

The code uses a finite binary table, log-linear features with main effects and
pairwise interactions, and a log-gamma empirical criterion.  The functions are
written so that scripts and notebooks can share the same implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp


Array = np.ndarray


@dataclass
class SparseTableProblem:
    """A binary sparse-table log-linear problem."""

    q: int
    n: int
    X: Array
    F: Array
    feature_names: list[str]
    theta0: Array
    active: Array
    p_clean: Array
    eta_clean: Array

    @property
    def K(self) -> int:
        return int(self.F.shape[0])

    @property
    def d(self) -> int:
        return int(self.F.shape[1])


@dataclass
class FitOptions:
    """Numerical options for L-BFGS fitting."""

    ridge: float = 1e-4
    maxiter: int = 500
    gtol: float = 1e-6
    ftol: float = 1e-10
    maxls: int = 50


def default_gamma_grid(max_gamma: float = 2.0, extended: bool = False) -> Array:
    """Return the gamma grid used in the simulation notebooks.

    The grid is dense on [0, 1] and coarser above 1.  Setting ``extended=True``
    adds points up to 3, useful for exploratory scans.
    """

    pieces: list[Array] = [
        np.linspace(0.0, 1.0, 21),
        np.linspace(1.1, min(2.0, max_gamma), 10),
    ]
    if extended and max_gamma > 2.0:
        pieces.append(np.linspace(2.2, max_gamma, 5))
    grid = np.unique(np.round(np.r_[tuple(pieces)], 2))
    return grid[grid <= max_gamma]


def make_binary_table(q: int) -> Array:
    """Return all binary cells x in {0, 1}^q as a 2^q by q array."""

    K = 2**q
    X = np.zeros((K, q), dtype=float)
    for k in range(K):
        X[k, :] = [(k >> j) & 1 for j in range(q)]
    return X


def make_features(X: Array, include_pairs: bool = True) -> tuple[Array, list[str]]:
    """Construct centered main effects and pairwise interactions.

    No intercept is included because probabilities are normalized by softmax.
    The feature values are in {-1, 1} for main effects and products for pairwise
    interactions.
    """

    Z = 2.0 * X - 1.0
    feats: list[Array] = []
    names: list[str] = []

    for j in range(Z.shape[1]):
        feats.append(Z[:, j])
        names.append(f"main_{j + 1}")

    if include_pairs:
        for j, k in combinations(range(Z.shape[1]), 2):
            feats.append(Z[:, j] * Z[:, k])
            names.append(f"int_{j + 1}_{k + 1}")

    return np.column_stack(feats), names


def default_theta(feature_names: list[str]) -> Array:
    """Default clean sparse parameter used in the experiments.

    The values reproduce the notebook setting for q=10 when all listed features
    are present.  If q is smaller, unavailable features are skipped.
    """

    theta = np.zeros(len(feature_names), dtype=float)
    values = {
        "main_1": 0.45,
        "main_2": -0.35,
        "main_3": 0.30,
        "main_4": -0.25,
        "main_5": 0.20,
        "main_6": -0.15,
        "int_1_2": 0.30,
        "int_1_5": -0.25,
        "int_2_4": 0.22,
        "int_3_7": -0.20,
    }
    index = {name: i for i, name in enumerate(feature_names)}
    for name, value in values.items():
        if name in index:
            theta[index[name]] = value
    return theta


def softmax_from_theta(theta: Array, F: Array) -> Array:
    """Compute p_theta(x) proportional to exp(theta^T f(x))."""

    eta = F @ theta
    eta = eta - np.max(eta)
    p = np.exp(eta)
    return p / p.sum()


def create_problem(q: int = 10, n: int = 3000) -> SparseTableProblem:
    """Create the default sparse binary table problem."""

    X = make_binary_table(q)
    F, feature_names = make_features(X, include_pairs=True)
    theta0 = default_theta(feature_names)
    active = np.flatnonzero(theta0 != 0)
    p_clean = softmax_from_theta(theta0, F)
    eta_clean = F @ theta0
    return SparseTableProblem(
        q=q,
        n=n,
        X=X,
        F=F,
        feature_names=feature_names,
        theta0=theta0,
        active=active,
        p_clean=p_clean,
        eta_clean=eta_clean,
    )


def make_diffuse_sparse_noise_distribution(
    p_clean: Array,
    n: int,
    m_sparse: int,
    lambda_sparse: float,
    rng: np.random.Generator,
    lower_probability_quantile: float = 0.40,
) -> tuple[Array, Array, float]:
    """Add diffuse sparse-cell noise to low-probability clean cells."""

    cutoff = np.quantile(p_clean, lower_probability_quantile)
    candidates = np.flatnonzero(p_clean <= cutoff)
    if len(candidates) < m_sparse:
        candidates = np.arange(len(p_clean))

    noise_cells = rng.choice(candidates, size=m_sparse, replace=False)
    eps_total = m_sparse * lambda_sparse / n
    if eps_total >= 0.30:
        raise ValueError("Total sparse-noise mass is too large.")

    p_data = (1.0 - eps_total) * p_clean.copy()
    p_data[noise_cells] += lambda_sparse / n
    p_data = p_data / p_data.sum()
    return p_data, noise_cells, eps_total


def make_uniform_noisy_distribution(p_clean: Array, eps_uniform: float) -> Array:
    """Return (1 - eps) p_clean + eps * uniform."""

    u = np.ones_like(p_clean) / len(p_clean)
    p_data = (1.0 - eps_uniform) * p_clean + eps_uniform * u
    return p_data / p_data.sum()


def make_high_leverage_sparse_noise_distribution(
    problem: SparseTableProblem,
    m_stress: int,
    lambda_stress: float,
    rng: np.random.Generator,
    low_eta_quantile: float = 0.15,
) -> tuple[Array, Array, float]:
    """Add sparse noise to high-leverage cells rare under the clean model.

    Stress cells are selected from the lower tail of the clean linear predictor
    and then preferentially from the smallest clean probabilities.
    """

    cutoff = np.quantile(problem.eta_clean, low_eta_quantile)
    candidates = np.flatnonzero(problem.eta_clean <= cutoff)
    if len(candidates) < m_stress:
        raise ValueError(
            "Not enough low-eta candidates. Increase low_eta_quantile or reduce m_stress."
        )

    candidates_sorted = candidates[np.argsort(problem.p_clean[candidates])]
    top_pool_size = min(len(candidates_sorted), max(2 * m_stress, m_stress))
    pool = candidates_sorted[:top_pool_size]
    stress_cells = rng.choice(pool, size=m_stress, replace=False)

    eps_total = m_stress * lambda_stress / problem.n
    if eps_total >= 0.30:
        raise ValueError("Stress-noise mass is too large.")

    p_data = (1.0 - eps_total) * problem.p_clean.copy()
    p_data[stress_cells] += lambda_stress / problem.n
    p_data = p_data / p_data.sum()
    return p_data, stress_cells, eps_total


def loss_grad_loggamma(
    theta: Array,
    F: Array,
    phat: Array,
    gamma: float,
    ridge: float = 0.0,
) -> tuple[float, Array]:
    """Return log-gamma loss and gradient.

    For gamma=0 the criterion is the negative multinomial log-likelihood per
    observation.  For gamma>0 it is the log-gamma empirical loss up to terms
    independent of theta.
    """

    eta = F @ theta
    logZ = logsumexp(eta)
    logp = eta - logZ
    p = np.exp(logp)

    if gamma == 0.0:
        loss = -float(np.dot(phat, logp))
        grad = F.T @ (p - phat)
    else:
        mask = phat > 0
        logA_terms = np.full_like(logp, -np.inf)
        logA_terms[mask] = np.log(phat[mask]) + gamma * logp[mask]
        logA = logsumexp(logA_terms)

        wA = np.zeros_like(p)
        wA[mask] = np.exp(logA_terms[mask] - logA)

        logB_terms = (1.0 + gamma) * logp
        logB = logsumexp(logB_terms)
        wB = np.exp(logB_terms - logB)

        loss = -(1.0 / gamma) * float(logA) + (1.0 / (1.0 + gamma)) * float(logB)
        grad = -F.T @ wA + F.T @ wB

    if ridge > 0:
        loss += 0.5 * ridge * float(np.dot(theta, theta))
        grad = grad + ridge * theta

    return float(loss), np.asarray(grad)


def fit_gamma_estimator(
    phat: Array,
    F: Array,
    gamma: float,
    theta_init: Array | None = None,
    options: FitOptions | None = None,
) -> tuple[Array, float, bool, int]:
    """Fit a log-gamma estimator by L-BFGS-B."""

    if options is None:
        options = FitOptions()
    d = F.shape[1]
    if theta_init is None:
        theta_init = np.zeros(d)

    def objective(th: Array) -> tuple[float, Array]:
        return loss_grad_loggamma(th, F, phat, gamma, ridge=options.ridge)

    result = minimize(
        fun=lambda th: objective(th)[0],
        x0=theta_init,
        jac=lambda th: objective(th)[1],
        method="L-BFGS-B",
        options={
            "maxiter": options.maxiter,
            "gtol": options.gtol,
            "ftol": options.ftol,
            "maxls": options.maxls,
        },
    )
    return result.x, float(result.fun), bool(result.success), int(result.nit)


def fit_gamma_path_from_counts(
    counts: Array,
    problem: SparseTableProblem,
    gamma_grid: Iterable[float],
    options: FitOptions | None = None,
) -> tuple[Array, Array, Array, Array]:
    """Fit the full gamma path from multinomial counts."""

    phat = counts / counts.sum()
    theta_prev = np.zeros(problem.d)
    theta_list: list[Array] = []
    p_list: list[Array] = []
    success_list: list[bool] = []
    nit_list: list[int] = []

    for gamma in np.asarray(list(gamma_grid), dtype=float):
        theta_hat, _obj, success, nit = fit_gamma_estimator(
            phat=phat,
            F=problem.F,
            gamma=float(gamma),
            theta_init=theta_prev,
            options=options,
        )
        theta_prev = theta_hat.copy()
        theta_list.append(theta_hat.copy())
        p_list.append(softmax_from_theta(theta_hat, problem.F))
        success_list.append(success)
        nit_list.append(nit)

    return (
        np.vstack(theta_list),
        np.vstack(p_list),
        np.asarray(success_list, dtype=bool),
        np.asarray(nit_list, dtype=int),
    )


def evaluate_theta(theta_hat: Array, problem: SparseTableProblem) -> dict[str, float]:
    """Evaluate a fitted theta against the clean target."""

    inactive = np.setdiff1d(np.arange(problem.d), problem.active)
    p_hat = softmax_from_theta(theta_hat, problem.F)
    return {
        "rmse_all": float(np.sqrt(np.mean((theta_hat - problem.theta0) ** 2))),
        "rmse_active": float(
            np.sqrt(np.mean((theta_hat[problem.active] - problem.theta0[problem.active]) ** 2))
        ),
        "rmse_inactive": float(np.sqrt(np.mean(theta_hat[inactive] ** 2))),
        "prob_l2": float(np.sqrt(np.mean((p_hat - problem.p_clean) ** 2))),
    }


def wj_select_gamma_from_counts(
    counts: Array,
    problem: SparseTableProblem,
    gamma_grid: Iterable[float],
    rng: np.random.Generator,
    pilot_gamma: float = 1.0,
    b_resample: int = 10,
    subsample_frac: float = 0.70,
    options: FitOptions | None = None,
) -> tuple[float, pd.DataFrame, Array, Array, Array]:
    """Warwick--Jones type gamma selection on fitted-distribution scale.

    The pilot fit is the gamma value closest to ``pilot_gamma``.  Resampled
    count vectors are generated by independent binomial thinning of cell counts.
    """

    gamma_grid = np.asarray(list(gamma_grid), dtype=float)
    theta_path, p_path, success, _nit = fit_gamma_path_from_counts(
        counts, problem, gamma_grid, options=options
    )

    pilot_idx = int(np.argmin(np.abs(gamma_grid - pilot_gamma)))
    p_pilot = p_path[pilot_idx].copy()

    p_resample_paths: list[Array] = []
    for _ in range(b_resample):
        counts_sub = rng.binomial(counts, subsample_frac)
        if counts_sub.sum() == 0:
            continue
        _theta_sub, p_sub, _success_sub, _nit_sub = fit_gamma_path_from_counts(
            counts_sub, problem, gamma_grid, options=options
        )
        p_resample_paths.append(p_sub)

    if not p_resample_paths:
        raise RuntimeError("All resampled count vectors were empty.")

    p_resample = np.stack(p_resample_paths, axis=0)
    p_bar = p_resample.mean(axis=0)
    bias2_p = np.mean((p_bar - p_pilot[None, :]) ** 2, axis=1)
    var_p = np.mean(np.mean((p_resample - p_bar[None, :, :]) ** 2, axis=2), axis=0)
    mse_p = bias2_p + var_p

    idx_selected = int(np.argmin(mse_p))
    gamma_wj = float(gamma_grid[idx_selected])
    curve = pd.DataFrame(
        {
            "gamma": gamma_grid,
            "bias2_p": bias2_p,
            "var_p": var_p,
            "mse_p": mse_p,
            "gamma_wj": gamma_wj,
        }
    )
    return gamma_wj, curve, theta_path, p_path, success


def summarize_by_gamma(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize gamma-path Monte Carlo results."""

    return (
        df.groupby("gamma")
        .agg(
            rmse_all_mean=("rmse_all", "mean"),
            rmse_all_se=("rmse_all", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
            rmse_active_mean=("rmse_active", "mean"),
            rmse_active_se=("rmse_active", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
            rmse_inactive_mean=("rmse_inactive", "mean"),
            rmse_inactive_se=("rmse_inactive", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
            prob_l2_mean=("prob_l2", "mean"),
            prob_l2_se=("prob_l2", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
            success_rate=("success", "mean"),
        )
        .reset_index()
        .sort_values("gamma")
    )


def summarize_methods(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize method-comparison Monte Carlo results."""

    group_cols = ["scenario", "method"] if "scenario" in df.columns else ["method"]
    return (
        df.groupby(group_cols)
        .agg(
            gamma_mean=("gamma", "mean"),
            gamma_sd=("gamma", "std"),
            rmse_all_mean=("rmse_all", "mean"),
            rmse_all_se=("rmse_all", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
            rmse_active_mean=("rmse_active", "mean"),
            rmse_active_se=("rmse_active", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
            rmse_inactive_mean=("rmse_inactive", "mean"),
            rmse_inactive_se=("rmse_inactive", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
            prob_l2_mean=("prob_l2", "mean"),
            prob_l2_se=("prob_l2", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
            success_rate=("success", "mean"),
        )
        .reset_index()
    )
