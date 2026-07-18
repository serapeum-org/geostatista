"""Ordinary-kriging engine — from fitted variogram + samples to a `KrigedSurface`.

Loops target cells through the moving neighborhood (`_solve/neighborhood.py`) and the OK solve
(`_solve/system.py`). Coincident sample points are pre-averaged (the documented duplicate policy).
"""

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform

from ._solve.neighborhood import Neighborhood
from ._solve.system import solve_ordinary
from .surface import KrigedSurface


def average_duplicates(coords: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pre-average values at coincident coordinates (the documented degenerate-input policy)."""
    unique, inverse = np.unique(coords, axis=0, return_inverse=True)
    if len(unique) == len(coords):
        return coords, values
    inverse = inverse.ravel()
    summed = np.zeros(len(unique))
    counts = np.zeros(len(unique))
    np.add.at(summed, inverse, values)
    np.add.at(counts, inverse, 1.0)
    result = (unique, summed / counts)
    return result


class OrdinaryKriging:
    """Ordinary kriging with a fitted `Variogram` over scattered samples."""

    def __init__(self, coords: np.ndarray, values: np.ndarray, variogram, n_neighbors: int | None = 32):
        coords, values = average_duplicates(np.asarray(coords, dtype=float), np.asarray(values, dtype=float))
        if variogram.sill is None:
            raise ValueError("OrdinaryKriging: variogram must be fitted (call .fit) before kriging")
        if variogram.sill <= 0.0:
            raise ValueError("OrdinaryKriging: variogram sill is 0 (constant field) — kriging is undefined")
        self.coords = coords
        self.values = values
        self.variogram = variogram
        self.sill = float(variogram.sill)
        self.n_neighbors = n_neighbors
        self.neighborhood = Neighborhood(coords, n_neighbors=n_neighbors)

    def _covariance(self, dist: np.ndarray) -> np.ndarray:
        """Covariance `C(h) = sill - gamma(h)` from the fitted variogram."""
        return self.sill - self.variogram.predict(dist)

    def predict_point(self, target: np.ndarray) -> tuple[float, float]:
        """Kriged estimate + variance at a single `target` point (shape `(2,)`)."""
        target = np.asarray(target, dtype=float)
        idx = self.neighborhood.query(target)
        pts = self.coords[idx]
        vals = self.values[idx]
        cov_ss = self._covariance(squareform(pdist(pts)))
        cov_st = self._covariance(cdist(pts, target.reshape(1, 2)).ravel())
        weights, variance = solve_ordinary(cov_ss, cov_st, self.sill)
        result = (float(weights @ vals), variance)
        return result

    def predict(self, grid_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Kriged estimate + variance at each point in `grid_points` (shape `(m, 2)`)."""
        grid_points = np.asarray(grid_points, dtype=float)
        estimate = np.empty(len(grid_points))
        variance = np.empty(len(grid_points))
        for i in range(len(grid_points)):
            estimate[i], variance[i] = self.predict_point(grid_points[i])
        result = (estimate, variance)
        return result

    def predict_grid(
        self,
        *,
        cell_size: float | None = None,
        bounds: tuple[float, float, float, float] | None = None,
        template=None,
        epsg: int = 4326,
        nodata: float = -9999.0,
    ) -> KrigedSurface:
        """Krige onto a regular grid (from `cell_size` + `bounds`, or an existing `template` Dataset)."""
        if template is not None:
            geo = template.geotransform
            shape = np.asarray(template.read_array()).shape
            nrows, ncols = shape[-2], shape[-1]
            minx, cell_x, top_y, cell_y = geo[0], geo[1], geo[3], -geo[5]
            xs = minx + (np.arange(ncols) + 0.5) * cell_x
            ys = top_y - (np.arange(nrows) + 0.5) * cell_y
            epsg = int(template.epsg)
        else:
            if cell_size is None:
                raise ValueError("predict_grid: provide cell_size (or a template Dataset)")
            if bounds is None:
                minx, miny = self.coords.min(axis=0)
                maxx, maxy = self.coords.max(axis=0)
            else:
                minx, miny, maxx, maxy = bounds
            ncols = int(np.ceil((maxx - minx) / cell_size))
            nrows = int(np.ceil((maxy - miny) / cell_size))
            xs = minx + (np.arange(ncols) + 0.5) * cell_size
            ys = maxy - (np.arange(nrows) + 0.5) * cell_size
            geo = (minx, cell_size, 0.0, maxy, 0.0, -cell_size)
        mesh_x, mesh_y = np.meshgrid(xs, ys)
        points = np.column_stack([mesh_x.ravel(), mesh_y.ravel()])
        estimate, variance = self.predict(points)
        surface = KrigedSurface.from_arrays(
            estimate.reshape(nrows, ncols),
            variance.reshape(nrows, ncols),
            geo=geo,
            epsg=epsg,
            nodata=nodata,
            model=self.variogram.model,
            nugget=self.variogram.nugget,
            sill=self.variogram.sill,
            range_=self.variogram.range_,
            n_neighbors=self.n_neighbors,
        )
        return surface
