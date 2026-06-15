"""Log-gamma experiments for sparse contingency tables."""

from .core import (
    SparseTableProblem,
    create_problem,
    default_gamma_grid,
    evaluate_theta,
    fit_gamma_estimator,
    fit_gamma_path_from_counts,
    loss_grad_loggamma,
    make_binary_table,
    make_diffuse_sparse_noise_distribution,
    make_features,
    make_high_leverage_sparse_noise_distribution,
    make_uniform_noisy_distribution,
    softmax_from_theta,
    summarize_by_gamma,
    summarize_methods,
    wj_select_gamma_from_counts,
)

__all__ = [
    "SparseTableProblem",
    "create_problem",
    "default_gamma_grid",
    "evaluate_theta",
    "fit_gamma_estimator",
    "fit_gamma_path_from_counts",
    "loss_grad_loggamma",
    "make_binary_table",
    "make_diffuse_sparse_noise_distribution",
    "make_features",
    "make_high_leverage_sparse_noise_distribution",
    "make_uniform_noisy_distribution",
    "softmax_from_theta",
    "summarize_by_gamma",
    "summarize_methods",
    "wj_select_gamma_from_counts",
]
