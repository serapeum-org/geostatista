"""Empirical variogram and the fitted `Variogram` model.

The empirical variogram bins pairwise squared differences by lag distance; `Variogram.fit` fits a theoretical
model (`spherical`/`exponential`/`gaussian`/`matern`) to that cloud, yielding the `(nugget, sill, range)` triple
kriging needs.
"""

from functools import partial

import numpy as np
import pandas as pd

from . import models
from ._solve.fit import fit_model


def empirical_variogram(
    coords: np.ndarray,
    values: np.ndarray,
    n_lags: int = 15,
    max_dist: float | None = None,
    estimator: str = "matheron",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the empirical semivariance cloud.

    Args:
        coords: Point coordinates, shape `(n, 2)`.
        values: Observed values, shape `(n,)`.
        n_lags: Number of lag bins.
        max_dist: Maximum lag distance; defaults to half the bounding-box diagonal.
        estimator: `"matheron"` (classic) or `"cressie"` (outlier-robust).

    Returns:
        A `(lags, semivariance, counts)` tuple. Empty lag bins yield `NaN` semivariance and zero count.
    """
    from scipy.spatial.distance import pdist

    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float)
    dist = pdist(coords)
    vdiff = pdist(values.reshape(-1, 1))
    if max_dist is None:
        extent = coords.max(axis=0) - coords.min(axis=0)
        max_dist = 0.5 * float(np.sqrt((extent**2).sum()))
    edges = np.linspace(0.0, max_dist, n_lags + 1)
    lags = 0.5 * (edges[:-1] + edges[1:])
    semivariance = np.full(n_lags, np.nan)
    counts = np.zeros(n_lags, dtype=int)
    bin_index = np.digitize(dist, edges) - 1
    for k in range(n_lags):
        selected = bin_index == k
        count = int(selected.sum())
        counts[k] = count
        if count > 0:
            diffs = vdiff[selected]
            if estimator == "cressie":
                robust = np.mean(np.sqrt(diffs)) ** 4
                semivariance[k] = 0.5 * robust / (0.457 + 0.494 / count + 0.045 / count**2)
            else:
                semivariance[k] = 0.5 * float(np.mean(diffs**2))
    return lags, semivariance, counts


class Variogram:
    """An empirical variogram cloud, optionally fitted to a theoretical model.

    Construct via `Samples.variogram(...)`; call `.fit(model=...)` to estimate `(nugget, sill, range)`, then
    `.predict(h)` for the modeled semivariance at arbitrary lags.
    """

    def __init__(
        self,
        lags: np.ndarray,
        semivariance: np.ndarray,
        counts: np.ndarray,
        estimator: str = "matheron",
    ):
        self.lags = np.asarray(lags, dtype=float)
        self.semivariance = np.asarray(semivariance, dtype=float)
        self.counts = np.asarray(counts, dtype=int)
        self.estimator = estimator
        self.model: str | None = None
        self.nugget: float | None = None
        self.sill: float | None = None
        self.range_: float | None = None
        self._func = None

    def fit(self, model: str = "spherical") -> "Variogram":
        """Fit a theoretical `model` to the empirical cloud, setting `.nugget`, `.sill`, `.range_`.

        Returns `self` for chaining. Raises `VariogramFitError` on non-convergence.
        """
        nugget, sill, rng = fit_model(self.lags, self.semivariance, self.counts, model)
        self.model = model
        self.nugget, self.sill, self.range_ = nugget, sill, rng
        func = models.BOUNDED_MODELS[model]
        self._func = partial(func, nu=1.5) if model == "matern" else func
        return self

    def predict(self, h: np.ndarray) -> np.ndarray:
        """Modeled semivariance at lag(s) `h`. Requires a prior `.fit()`."""
        if self._func is None:
            raise RuntimeError("predict: call .fit(model=...) before predicting")
        result = self._func(h, self.nugget, self.sill, self.range_)
        return result

    def to_dataframe(self) -> pd.DataFrame:
        """Return the empirical cloud (`lag`, `semivariance`, `count`) as a `DataFrame`."""
        frame = pd.DataFrame(
            {"lag": self.lags, "semivariance": self.semivariance, "count": self.counts}
        )
        return frame

    def plot(self, ax=None, *, title: str = "Variogram"):
        """Plot the empirical cloud (+ fitted curve, once fitted) via cleopatra.

        Draws the empirical semivariance as a scatter and, if the variogram has been fitted, overlays the model
        curve on the same axis. Requires the `viz` extra (cleopatra). Returns the matplotlib `(fig, ax)`.

        Raises:
            ImportError: If cleopatra (the `viz` extra) is not installed.
        """
        try:
            from cleopatra.glyphs.primitives.line_glyph import LineGlyph
            from cleopatra.glyphs.primitives.scatter_glyph import ScatterGlyph
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Variogram.plot requires the 'viz' extra (cleopatra): install geostatista[viz]"
            ) from exc
        valid = np.isfinite(self.semivariance)
        fig, ax, _ = ScatterGlyph(self.lags[valid], self.semivariance[valid], ax=ax).plot(
            ax=ax, title=title, add_colorbar=False
        )
        if self._func is not None:
            lags = np.linspace(0.0, float(self.lags.max()), 100)
            LineGlyph(lags, self.predict(lags), ax=ax).line(ax=ax, label=self.model or "model")
        result = (fig, ax)
        return result

    def __repr__(self) -> str:
        if self.model is None:
            body = f"<Variogram unfitted, {len(self.lags)} lags, estimator={self.estimator!r}>"
        else:
            body = (
                f"<Variogram model={self.model!r} nugget={self.nugget:.4g} "
                f"sill={self.sill:.4g} range={self.range_:.4g}>"
            )
        return body
