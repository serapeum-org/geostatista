"""Least-squares fitting of a bounded variogram model to an empirical cloud.

Weighted by pair count (near, well-populated lags matter most), with non-negative bounds and method-of-moments
seeds so `curve_fit` starts near the answer. Non-convergence raises a clear error rather than returning garbage.
"""

from functools import partial

import numpy as np
from scipy.optimize import curve_fit

from .. import models


class VariogramFitError(RuntimeError):
    """Raised when the variogram model fit fails to converge."""


def _model_callable(model_name: str):
    """Return an `f(h, nugget, sill, range)` callable for `model_name` (matern uses nu=1.5)."""
    if model_name not in models.BOUNDED_MODELS:
        raise ValueError(
            f"fit: model {model_name!r} is not a fittable bounded model "
            f"(choose from {sorted(models.BOUNDED_MODELS)})"
        )
    func = models.BOUNDED_MODELS[model_name]
    if model_name == "matern":
        func = partial(models.matern, nu=1.5)
    return func


def _partial_sill_model(h: np.ndarray, nugget: float, partial_sill: float, rng: float, *, base) -> np.ndarray:
    """Evaluate `base` with `sill = nugget + partial_sill` — the reparameterization curve_fit optimizes over."""
    return base(h, nugget, nugget + partial_sill, rng)


def fit_model(
    lags: np.ndarray, semivariance: np.ndarray, counts: np.ndarray, model_name: str
) -> tuple[float, float, float]:
    """Fit `(nugget, sill, range)` for `model_name` to the empirical variogram.

    Args:
        lags: Lag distances, shape `(n_lags,)`.
        semivariance: Empirical semivariance per lag (may contain NaN for empty bins).
        counts: Pair count per lag (used as fit weights).
        model_name: One of the bounded models (`spherical`, `exponential`, `gaussian`, `matern`).

    Returns:
        The fitted `(nugget, sill, range)`.

    Raises:
        VariogramFitError: If the least-squares fit does not converge.
    """
    func = _model_callable(model_name)
    # Fit on (nugget, partial_sill, range) with partial_sill >= 0, so the returned sill = nugget + partial_sill is
    # always >= nugget. Fitting (nugget, sill) with independent box bounds would let curve_fit return nugget > sill
    # (a negative partial sill), which makes the model non-monotone and the kriging covariance non-PSD.
    fit_func = partial(_partial_sill_model, base=func)
    valid = np.isfinite(semivariance) & (counts > 0)
    if valid.sum() < 3:
        raise VariogramFitError("fit: fewer than 3 populated lag bins — cannot fit a 3-parameter model")
    h = np.asarray(lags, dtype=float)[valid]
    g = np.asarray(semivariance, dtype=float)[valid]
    weights = np.asarray(counts, dtype=float)[valid]

    emp_sill = float(np.max(g))
    emp_range = float(np.max(h))
    p0 = [0.0, emp_sill, 0.5 * emp_range]
    lower = [0.0, 0.0, 1e-9 * emp_range]
    upper = [emp_sill if emp_sill > 0 else np.inf, 2.0 * emp_sill if emp_sill > 0 else np.inf, 3.0 * emp_range]
    sigma = 1.0 / np.sqrt(weights)
    try:
        popt, _ = curve_fit(
            fit_func, h, g, p0=p0, bounds=(lower, upper), sigma=sigma, absolute_sigma=False, maxfev=10000
        )
    except (RuntimeError, ValueError) as exc:
        raise VariogramFitError(f"fit: {model_name} model did not converge — {exc}") from exc
    nugget, partial_sill, rng = float(popt[0]), float(popt[1]), float(popt[2])
    result = (nugget, nugget + partial_sill, rng)
    return result
