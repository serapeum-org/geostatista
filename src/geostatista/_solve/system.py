"""Ordinary-kriging system assembly and solve (covariance form).

For a bounded variogram model with sill `C(0) = sill`, the covariance is `C(h) = sill - gamma(h)`. Ordinary
kriging solves, for weights `w` and Lagrange multiplier `mu`:

    [ C_ss  1 ] [ w  ]   [ C_st ]
    [ 1^T   0 ] [ mu ] = [ 1    ]

where `C_ss` is the sample-sample covariance, `C_st` the sample-target covariance. The kriging variance is
`sill - w . C_st - mu`.
"""

import numpy as np


def solve_ordinary(cov_ss: np.ndarray, cov_st: np.ndarray, sill: float) -> tuple[np.ndarray, float]:
    """Solve the ordinary-kriging system at one target.

    Args:
        cov_ss: Sample-sample covariance matrix, shape `(n, n)`.
        cov_st: Sample-target covariance vector, shape `(n,)`.
        sill: The model sill `C(0)` (a priori variance).

    Returns:
        A `(weights, variance)` tuple: the `n` kriging weights and the (non-negative) kriging variance.
    """
    n = cov_ss.shape[0]
    lhs = np.ones((n + 1, n + 1), dtype=float)
    lhs[:n, :n] = cov_ss
    lhs[n, n] = 0.0
    rhs = np.ones(n + 1, dtype=float)
    rhs[:n] = cov_st
    try:
        solution = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        solution = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    weights = solution[:n]
    lagrange = solution[n]
    variance = float(sill - weights @ cov_st - lagrange)
    result = (weights, max(variance, 0.0))
    return result
