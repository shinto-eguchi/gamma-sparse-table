import numpy as np

from gamma_sparse_table.core import (
    FitOptions,
    create_problem,
    default_gamma_grid,
    fit_gamma_estimator,
    loss_grad_loggamma,
)


def test_loss_gradient_matches_finite_difference():
    problem = create_problem(q=4, n=300)
    rng = np.random.default_rng(123)
    counts = rng.multinomial(problem.n, problem.p_clean)
    phat = counts / counts.sum()
    theta = rng.normal(scale=0.05, size=problem.d)
    gamma = 0.8
    loss, grad = loss_grad_loggamma(theta, problem.F, phat, gamma, ridge=1e-4)
    eps = 1e-6
    numeric = np.zeros_like(theta)
    for j in range(problem.d):
        e = np.zeros_like(theta)
        e[j] = eps
        lp, _ = loss_grad_loggamma(theta + e, problem.F, phat, gamma, ridge=1e-4)
        lm, _ = loss_grad_loggamma(theta - e, problem.F, phat, gamma, ridge=1e-4)
        numeric[j] = (lp - lm) / (2 * eps)
    assert np.allclose(grad, numeric, atol=1e-5)
    assert np.isfinite(loss)


def test_gamma_fit_smoke():
    problem = create_problem(q=4, n=300)
    rng = np.random.default_rng(456)
    counts = rng.multinomial(problem.n, problem.p_clean)
    phat = counts / counts.sum()
    theta_hat, value, success, nit = fit_gamma_estimator(
        phat, problem.F, gamma=0.5, options=FitOptions(maxiter=100)
    )
    assert theta_hat.shape == (problem.d,)
    assert np.isfinite(value)
    assert nit >= 0
    assert isinstance(success, bool)


def test_default_gamma_grid_contains_key_values():
    grid = default_gamma_grid(max_gamma=2.0)
    for val in [0.0, 0.5, 0.8, 1.0, 1.3]:
        assert np.any(np.isclose(grid, val))
