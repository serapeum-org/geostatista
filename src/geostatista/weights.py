"""`Weights` — a sparse spatial-weights matrix (contiguity / knn / distance-band).

Every autocorrelation statistic is a quadratic form in this matrix. Stored as `scipy.sparse`; row-standardizable.
This is a standalone domain object (it wraps `scipy.sparse`, not a pyramids type) and must never be confused with
the kriging moving neighborhood in `_solve/`.
"""

import numpy as np
from loguru import logger
from scipy import sparse
from scipy.spatial import cKDTree


class Weights:
    """A spatial-weights matrix over `n` features, backed by `scipy.sparse`."""

    def __init__(self, matrix, ids: np.ndarray | None = None, transform: str = "b"):
        self.sparse = sparse.csr_matrix(matrix).astype(float)
        self.n = self.sparse.shape[0]
        self.ids = np.arange(self.n) if ids is None else np.asarray(ids)
        self.transform_applied = transform
        islands = self.islands
        if len(islands):
            logger.warning(f"Weights: {len(islands)} island(s) with no neighbors — row-standardized rows stay zero")

    @property
    def cardinalities(self) -> np.ndarray:
        """Number of neighbors per feature."""
        return self.sparse.getnnz(axis=1)

    @property
    def islands(self) -> np.ndarray:
        """Ids of features with zero neighbors."""
        return self.ids[self.cardinalities == 0]

    @property
    def neighbors(self) -> dict:
        """Mapping `id -> array of neighbor ids`."""
        csr = self.sparse
        mapping = {int(self.ids[i]): self.ids[csr.indices[csr.indptr[i]:csr.indptr[i + 1]]] for i in range(self.n)}
        return mapping

    def transform(self, kind: str = "r") -> "Weights":
        """Return a re-weighted copy: `'r'` row-standardized (rows sum to 1), `'b'` binary."""
        if kind == "r":
            rowsum = np.asarray(self.sparse.sum(axis=1)).ravel()
            rowsum[rowsum == 0.0] = 1.0                              # islands keep a zero row (no div-by-zero)
            scaled = sparse.diags(1.0 / rowsum) @ self.sparse
            result = Weights(scaled, self.ids, transform="r")
        elif kind == "b":
            result = Weights((self.sparse > 0).astype(float), self.ids, transform="b")
        else:
            raise ValueError(f"transform: kind must be 'r' or 'b', got {kind!r}")
        return result

    @classmethod
    def _contiguity(cls, fc, rook: bool) -> "Weights":
        from shapely import STRtree

        geoms = list(fc.geometry.values)
        if not all(g.geom_type in ("Polygon", "MultiPolygon") for g in geoms):
            raise ValueError("contiguity weights require polygon geometries")
        tree = STRtree(geoms)
        n = len(geoms)
        rows, cols = [], []
        for i, g in enumerate(geoms):
            for j in tree.query(g):
                j = int(j)
                if j == i:
                    continue
                inter = g.intersection(geoms[j])
                if inter.is_empty or getattr(inter, "area", 0.0) > 1e-12:
                    continue
                shared_edge = getattr(inter, "length", 0.0) > 0.0
                if shared_edge or not rook:                         # rook: edge only; queen: vertex or edge
                    rows.append(i)
                    cols.append(j)
        matrix = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
        return cls(matrix)

    @classmethod
    def queen(cls, fc) -> "Weights":
        """Queen contiguity — features sharing a vertex or an edge are neighbors."""
        return cls._contiguity(fc, rook=False)

    @classmethod
    def rook(cls, fc) -> "Weights":
        """Rook contiguity — only features sharing an edge (positive-length boundary) are neighbors."""
        return cls._contiguity(fc, rook=True)

    @classmethod
    def knn(cls, fc, k: int) -> "Weights":
        """k-nearest-neighbor weights by centroid distance (asymmetric)."""
        coords = np.column_stack([fc.geometry.centroid.x.to_numpy(), fc.geometry.centroid.y.to_numpy()])
        n = len(coords)
        tree = cKDTree(coords)
        _, idx = tree.query(coords, k=k + 1)                        # +1 to drop self
        rows = np.repeat(np.arange(n), k)
        cols = idx[:, 1:].ravel()
        matrix = sparse.csr_matrix((np.ones(n * k), (rows, cols)), shape=(n, n))
        return cls(matrix)

    @classmethod
    def distance_band(cls, fc, threshold: float, *, binary: bool = True) -> "Weights":
        """Distance-band weights — neighbors within `threshold`; inverse-distance weighted when `binary=False`."""
        coords = np.column_stack([fc.geometry.centroid.x.to_numpy(), fc.geometry.centroid.y.to_numpy()])
        n = len(coords)
        pairs = np.asarray(list(cKDTree(coords).query_pairs(threshold)))
        if len(pairs) == 0:
            return cls(sparse.csr_matrix((n, n)))
        rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
        cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
        if binary:
            data = np.ones(len(rows))
        else:
            dist = np.linalg.norm(coords[rows] - coords[cols], axis=1)
            data = 1.0 / dist
        matrix = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
        return cls(matrix)
