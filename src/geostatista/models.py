"""Theoretical variogram models — gamma(h; nugget, sill, range).

Each model is a pure `numpy` function returning the semivariance at lag distance `h`. All models satisfy
`gamma(0) = 0` and rise to (or toward) the `sill`. The `nugget` is the limiting value as `h -> 0+` (the
micro-scale discontinuity), and `sill` is the total sill (nugget + partial sill).

The bounded models use the *practical-range* convention: for `exponential` and `gaussian`, `gamma(range)` reaches
~95% of the partial sill (spherical reaches the sill exactly at `range`). `matern` is the exception — it uses the
Stein `sqrt(2*nu)*h/range` (1/e) scaling, so a fitted `matern` range is not directly comparable to the others' on
the same data.
"""

from collections.abc import Callable

import numpy as np
from scipy.special import gamma as _gamma_fn
from scipy.special import kv as _bessel_k


def spherical(h: np.ndarray, nugget: float, sill: float, rng: float) -> np.ndarray:
    """Spherical model — reaches the sill exactly at `h = range`."""
    h = np.asarray(h, dtype=float)
    ratio = np.clip(h / rng, 0.0, 1.0)
    shape = 1.5 * ratio - 0.5 * ratio**3
    gamma = nugget + (sill - nugget) * shape
    result = np.where(h <= 0.0, 0.0, gamma)
    return result


def exponential(h: np.ndarray, nugget: float, sill: float, rng: float) -> np.ndarray:
    """Exponential model — approaches the sill asymptotically (practical range at `h = range`)."""
    h = np.asarray(h, dtype=float)
    shape = 1.0 - np.exp(-3.0 * h / rng)
    gamma = nugget + (sill - nugget) * shape
    result = np.where(h <= 0.0, 0.0, gamma)
    return result


def gaussian(h: np.ndarray, nugget: float, sill: float, rng: float) -> np.ndarray:
    """Gaussian model — smooth, parabolic near the origin (practical range at `h = range`)."""
    h = np.asarray(h, dtype=float)
    shape = 1.0 - np.exp(-3.0 * (h / rng) ** 2)
    gamma = nugget + (sill - nugget) * shape
    result = np.where(h <= 0.0, 0.0, gamma)
    return result


def matern(h: np.ndarray, nugget: float, sill: float, rng: float, nu: float = 1.5) -> np.ndarray:
    """Matern model with smoothness `nu` (Stein 1/e range convention — see the module docstring)."""
    h = np.asarray(h, dtype=float)
    scaled = np.sqrt(2.0 * nu) * np.where(h > 0.0, h, 1.0) / rng
    coef = (2.0 ** (1.0 - nu)) / _gamma_fn(nu)
    rho = coef * (scaled**nu) * _bessel_k(nu, scaled)
    rho = np.where(h <= 0.0, 1.0, rho)
    shape = 1.0 - rho
    gamma = nugget + (sill - nugget) * shape
    result = np.where(h <= 0.0, 0.0, gamma)
    return result


def power(h: np.ndarray, nugget: float, scale: float, exponent: float) -> np.ndarray:
    """Power model — unbounded (non-stationary); `0 < exponent < 2`. No sill."""
    h = np.asarray(h, dtype=float)
    gamma = nugget + scale * np.power(np.where(h > 0.0, h, 0.0), exponent)
    result = np.where(h <= 0.0, 0.0, gamma)
    return result


def nugget(h: np.ndarray, nugget_value: float) -> np.ndarray:
    """Pure-nugget model — no spatial structure; `gamma(h) = nugget` for all `h > 0`."""
    h = np.asarray(h, dtype=float)
    result = np.where(h <= 0.0, 0.0, float(nugget_value))
    return result


# Registry of the bounded (nugget, sill, range) models used by `Variogram.fit`.
# `power` and `nugget` have different parameterizations and are exposed as functions only.
BOUNDED_MODELS: dict[str, Callable[..., np.ndarray]] = {
    "spherical": spherical,
    "exponential": exponential,
    "gaussian": gaussian,
    "matern": matern,
}

MODELS: dict[str, Callable[..., np.ndarray]] = {
    **BOUNDED_MODELS,
    "power": power,
    "nugget": nugget,
}
