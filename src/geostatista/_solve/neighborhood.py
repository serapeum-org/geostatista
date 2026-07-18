"""cKDTree moving neighborhood — the escape hatch from the O(n^3) global kriging solve.

Selecting the nearest `n_neighbors` sample points around each target cell keeps kriging usable well past a few
thousand samples. This is deliberately NOT the Phase-3 spatial-weights matrix; do not merge the two.
"""

import numpy as np
from scipy.spatial import cKDTree


class Neighborhood:
    """Nearest-sample selector around an arbitrary target location.

    Args:
        coords: Sample coordinates, shape `(n, 2)`.
        n_neighbors: Number of nearest samples to return per target. `None` uses all samples (global solve).
        max_dist: Optional cap; samples beyond this distance are dropped even if within the nearest `n_neighbors`.
    """

    def __init__(self, coords: np.ndarray, n_neighbors: int | None = None, max_dist: float | None = None):
        self.coords = np.asarray(coords, dtype=float)
        self.n = len(self.coords)
        self.tree = cKDTree(self.coords)
        self.n_neighbors = n_neighbors
        self.max_dist = max_dist

    def query(self, point: np.ndarray) -> np.ndarray:
        """Return the sample indices in the neighborhood of `point` (shape `(2,)`)."""
        k = self.n if self.n_neighbors is None else min(self.n_neighbors, self.n)
        dist, idx = self.tree.query(np.asarray(point, dtype=float), k=k)
        idx = np.atleast_1d(idx)
        dist = np.atleast_1d(dist)
        if self.max_dist is not None:
            idx = idx[dist <= self.max_dist]
        return idx
