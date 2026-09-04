"""Spatial autocorrelation — global (Moran's I, Geary's C) and local (LISA, Getis-Ord Gi*) statistics.

Free functions taking a `FeatureCollection` + a column + a `Weights`; global stats return a typed result, local
stats return the input features annotated with per-feature inference (z / p / cluster / hotspot columns).
"""

from dataclasses import dataclass

import numpy as np
from loguru import logger
from scipy import sparse
from scipy.stats import norm

from .weights import Weights

_QUADRANT_LABEL = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}


def _sim_z(value: float, sims: np.ndarray) -> float:
    """Standardize `value` against the permutation distribution `sims`, guarding a zero spread (constant field)."""
    std = float(sims.std())
    return 0.0 if std == 0.0 else (value - float(sims.mean())) / std


def _s1_s2(matrix) -> tuple[float, float]:
    """Weight sums S1 and S2 used by the analytic Moran/Geary variances."""
    symm = matrix + matrix.T
    s1 = 0.5 * float(symm.multiply(symm).sum())
    row = np.asarray(matrix.sum(axis=1)).ravel()
    col = np.asarray(matrix.sum(axis=0)).ravel()
    s2 = float(((row + col) ** 2).sum())
    return s1, s2


def _folded_p(sims: np.ndarray, observed: float, permutations: int) -> float:
    """esda-style folded permutation p-value."""
    above = int((sims >= observed).sum())
    larger = min(above, permutations - above)
    return (larger + 1.0) / (permutations + 1.0)


@dataclass
class MoranResult:
    """Global Moran's I with analytic (normality) and permutation inference."""

    I: float
    EI: float
    z_norm: float
    p_norm: float
    z_sim: float
    p_sim: float
    n: int
    permutations: int
    transform: str
    weights_kind: str = "custom"

    def summary(self) -> dict:
        return {"I": self.I, "EI": self.EI, "z": self.z_norm, "p_norm": self.p_norm, "p_sim": self.p_sim}

    def __repr__(self) -> str:
        return f"<MoranResult I={self.I:.4f} EI={self.EI:.4f} z={self.z_norm:.3f} p_sim={self.p_sim:.4f}>"


@dataclass
class GearyResult:
    """Global Geary's C with permutation inference."""

    C: float
    EC: float
    z_sim: float
    p_sim: float
    n: int
    permutations: int

    def summary(self) -> dict:
        return {"C": self.C, "EC": self.EC, "z": self.z_sim, "p_sim": self.p_sim}

    def __repr__(self) -> str:
        return f"<GearyResult C={self.C:.4f} EC=1 z={self.z_sim:.3f} p_sim={self.p_sim:.4f}>"


def morans_i(fc, column: str, w: Weights, *, permutations: int = 999, transform: str = "r", seed: int | None = None) -> MoranResult:
    """Global Moran's I of `column` under weights `w`."""
    weights = w.transform(transform)
    matrix = weights.sparse
    x = fc[column].to_numpy(dtype=float)
    z = x - x.mean()
    n = len(x)
    s0 = float(matrix.sum())
    z2 = float((z**2).sum())
    expected = -1.0 / (n - 1)
    if z2 == 0.0:
        logger.warning(f"morans_i: column {column!r} is constant — Moran's I is undefined")
        result = MoranResult(np.nan, expected, np.nan, np.nan, np.nan, np.nan, n, permutations, transform, w.transform_applied)
    else:
        moran = (n / s0) * float(z @ (matrix @ z)) / z2
        s1, s2 = _s1_s2(matrix)
        var = (n * n * s1 - n * s2 + 3 * s0 * s0) / (s0 * s0 * (n * n - 1)) - expected * expected
        z_norm = (moran - expected) / np.sqrt(var)
        p_norm = 2.0 * (1.0 - norm.cdf(abs(z_norm)))
        rng = np.random.default_rng(seed)
        sims = np.empty(permutations)
        for p in range(permutations):
            zp = rng.permutation(z)
            sims[p] = (n / s0) * float(zp @ (matrix @ zp)) / z2
        p_sim = _folded_p(sims, moran, permutations)
        result = MoranResult(
            moran, expected, float(z_norm), float(p_norm), _sim_z(moran, sims), p_sim, n, permutations, transform, w.transform_applied
        )
    return result


def _geary_c(values: np.ndarray, row: np.ndarray, col: np.ndarray, data: np.ndarray, n: int, s0: float, z2: float) -> float:
    """Geary's C for `values` over the sparse-weight `(row, col, data)` triplet."""
    numer = float((data * (values[row] - values[col]) ** 2).sum())
    return (n - 1) * numer / (2.0 * s0 * z2)


def gearys_c(fc, column: str, w: Weights, *, permutations: int = 999, transform: str = "r", seed: int | None = None) -> GearyResult:
    """Global Geary's C of `column` under weights `w` (row-standardized by default, consistent with `morans_i`)."""
    matrix = w.transform(transform).sparse
    x = fc[column].to_numpy(dtype=float)
    n = len(x)
    z2 = float(((x - x.mean()) ** 2).sum())
    s0 = float(matrix.sum())
    coo = matrix.tocoo()
    if z2 == 0.0:
        logger.warning(f"gearys_c: column {column!r} is constant — Geary's C is undefined")
        result = GearyResult(np.nan, 1.0, np.nan, np.nan, n, permutations)
    else:
        geary = _geary_c(x, coo.row, coo.col, coo.data, n, s0, z2)
        rng = np.random.default_rng(seed)
        sims = np.array(
            [_geary_c(rng.permutation(x), coo.row, coo.col, coo.data, n, s0, z2) for _ in range(permutations)]
        )
        p_sim = _folded_p(sims, geary, permutations)
        result = GearyResult(geary, 1.0, _sim_z(geary, sims), p_sim, n, permutations)
    return result


def local_morans(fc, column: str, w: Weights, *, permutations: int = 999, alpha: float = 0.05, seed: int | None = None):
    """Local Moran (LISA) — annotate `fc` with `local_i`, `z_sim`, `p_sim`, `quadrant`, `cluster`."""
    weights = w.transform("r")
    matrix = weights.sparse
    x = fc[column].to_numpy(dtype=float)
    n = len(x)
    z = x - x.mean()
    m2 = float((z**2).sum()) / (n - 1)             # esda/Anselin normalization, so local_i matches the reference
    quadrant = np.zeros(n, dtype=int)
    local = np.full(n, np.nan)
    z_sim = np.full(n, np.nan)
    p_sim = np.full(n, np.nan)
    cluster = np.array(["ns"] * n, dtype=object)
    if m2 == 0.0:
        logger.warning(f"local_morans: column {column!r} is constant — LISA is undefined")
    else:
        lag = np.asarray(matrix @ z).ravel()
        local = (z / m2) * lag
        quadrant[(z > 0) & (lag > 0)] = 1
        quadrant[(z < 0) & (lag > 0)] = 2
        quadrant[(z < 0) & (lag < 0)] = 3
        quadrant[(z > 0) & (lag < 0)] = 4
        rng = np.random.default_rng(seed)
        for i in range(n):
            start, end = matrix.indptr[i], matrix.indptr[i + 1]
            neigh_w = matrix.data[start:end]
            k = len(neigh_w)
            if k == 0:
                continue
            others = np.delete(z, i)
            # conditional permutation: reassign the other n-1 values to the k neighbour slots WITHOUT replacement
            # (matches esda/pysal; with-replacement would inflate the null spread by sqrt((n-1)/(n-1-k))).
            picks = np.argsort(rng.random((permutations, len(others))), axis=1)[:, :k]
            sims = (z[i] / m2) * (others[picks] @ neigh_w)
            p_sim[i] = _folded_p(sims, local[i], permutations)
            z_sim[i] = _sim_z(local[i], sims)
        significant = p_sim <= alpha
        for i in range(n):
            if significant[i] and quadrant[i]:
                cluster[i] = _QUADRANT_LABEL[quadrant[i]]

    out = fc.copy()
    out["local_i"] = local
    out["z_sim"] = z_sim
    out["p_sim"] = p_sim
    out["quadrant"] = quadrant
    out["cluster"] = cluster
    out.attrs["provenance"] = {"stat": "local_morans", "weights": w.transform_applied, "permutations": permutations}
    return out


def getis_ord_gi(fc, column: str, w: Weights, *, star: bool = True, alpha: float = 0.05):
    """Getis-Ord Gi/Gi* — annotate `fc` with `gi`, `z`, `p`, `hotspot` (hot/cold/ns).

    `gi` is the Getis-Ord statistic itself (`sum_j w_ij x_j / sum_j x_j`). With `star=True` the focal feature is
    included (Gi*) and the standardization uses the global mean/variance; with `star=False` it is excluded (Gi)
    and the standardization uses the leave-one-out moments over the other `n-1` observations. The weights are used
    binary (with a unit self-weight for Gi*), regardless of the input `Weights` transform.

    `gi` assumes a positive variable — for data summing to <= 0 the ratio is undefined and `gi` is nan, though the
    `z` / `hotspot` outputs remain valid for signed data.
    """
    x = fc[column].to_numpy(dtype=float)
    n = len(x)
    total = float(x.sum())
    total_sq = float((x**2).sum())
    binary = (w.sparse > 0).astype(float)
    matrix = (binary + sparse.identity(n, format="csr")) if star else binary
    wi = np.asarray(matrix.sum(axis=1)).ravel()
    wi2 = np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel()
    lag = np.asarray(matrix @ x).ravel()
    if star:
        mean = np.full(n, total / n)
        variance = total_sq / n - mean**2
        spread = np.sqrt(np.maximum((n * wi2 - wi**2) / (n - 1), 0.0))
        gi = np.divide(lag, total, out=np.full(n, np.nan), where=total != 0.0)
    else:
        mean = (total - x) / (n - 1)
        variance = (total_sq - x**2) / (n - 1) - mean**2
        spread = np.sqrt(np.maximum(((n - 1) * wi2 - wi**2) / (n - 2), 0.0))
        gi = np.divide(lag, total - x, out=np.full(n, np.nan), where=(total - x) != 0.0)
    std = np.sqrt(np.maximum(variance, 0.0))
    scale = std * spread
    # z is undefined where the denominator collapses — a constant field (std=0) or a degenerate topology
    # (spread=0, e.g. a feature adjacent to every other): emit nan (-> p=nan, hotspot=ns) rather than a 0 that
    # reads as "not significant".
    z = np.divide(lag - mean * wi, scale, out=np.full(n, np.nan), where=scale > 0.0)
    if not np.all(scale > 0.0):
        logger.warning(f"getis_ord_gi: {int((scale <= 0.0).sum())} feature(s) have a degenerate Gi denominator -> nan")
    p = 2.0 * (1.0 - norm.cdf(np.abs(z)))

    hotspot = np.array(["ns"] * n, dtype=object)
    hotspot[(z > 0) & (p <= alpha)] = "hot"
    hotspot[(z < 0) & (p <= alpha)] = "cold"

    out = fc.copy()
    out["gi"] = gi
    out["z"] = z
    out["p"] = p
    out["hotspot"] = hotspot
    out.attrs["provenance"] = {"stat": "getis_ord_gi", "star": star, "weights": "binary"}
    return out


def _build_weights(fc, weights: "Weights | str") -> Weights:
    """Resolve a `Weights` instance or a named default (`queen`/`rook`/`knn`/`distance_band`)."""
    if isinstance(weights, Weights):
        return weights
    builders = {"queen": Weights.queen, "rook": Weights.rook}
    if weights in builders:
        return builders[weights](fc)
    if weights == "knn":
        return Weights.knn(fc, 8)
    if weights == "distance_band":
        coords = np.column_stack([fc.geometry.centroid.x.to_numpy(), fc.geometry.centroid.y.to_numpy()])
        from scipy.spatial import cKDTree

        dist, _ = cKDTree(coords).query(coords, k=2)
        return Weights.distance_band(fc, float(dist[:, 1].mean()))
    raise ValueError(f"unknown weights spec {weights!r}")


def spatial_autocorrelation(fc, column: str, *, weights: "Weights | str" = "queen", seed: int | None = None) -> dict:
    """One-call global Moran's I (the #576 Item-5 facade). Returns `{I, EI, z, p, n, weights}`.

    `z` and `p` are the analytic (normality-based) pair, so the result is deterministic; pass `seed` to also make
    the permutation inference inside Moran's I reproducible.
    """
    w = _build_weights(fc, weights)
    result = morans_i(fc, column, w, seed=seed)
    return {
        "I": result.I,
        "EI": result.EI,
        "z": result.z_norm,
        "p": result.p_norm,
        "n": result.n,
        "weights": w.transform_applied,
    }


def hotspots(fc, column: str, *, weights: "Weights | str" = "queen"):
    """One-call Getis-Ord Gi* hotspot map (the #576 Item-5 facade). Returns an annotated FeatureCollection."""
    w = _build_weights(fc, weights)
    return getis_ord_gi(fc, column, w, star=True)


# --- significance-class maps (S6) — via cleopatra, conventional diverging palette ------------------------

# Ordinal codes so a diverging colormap renders the conventional look (high=red, low=blue, non-significant=neutral).
_HOTSPOT_ORDINAL = {"cold": -1.0, "ns": 0.0, "hot": 1.0}
_CLUSTER_ORDINAL = {"LL": -2.0, "LH": -1.0, "ns": 0.0, "HL": 1.0, "HH": 2.0}


def _polygon_exterior(geom) -> np.ndarray:
    """The exterior ring of a (multi)polygon as an `(n, 2)` coordinate array."""
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda part: part.area)
    return np.asarray(geom.exterior.coords)


def _class_choropleth(fc, codes: np.ndarray, title: str, vabs: float, ax):
    """Render a per-feature diverging choropleth of signed class `codes` via cleopatra's PolygonGlyph."""
    try:
        from cleopatra.glyphs.primitives.polygon_glyph import PolygonGlyph
    except ImportError as exc:
        raise ImportError(
            "plotting requires cleopatra >=0.31.0 (the 'viz' extra): install or upgrade geostatista[viz]"
        ) from exc
    polygons = [_polygon_exterior(g) for g in fc.geometry.values]
    fig, ax, _ = PolygonGlyph(
        polygons, values=np.asarray(codes, dtype=float), ax=ax, cmap="coolwarm", vmin=-vabs, vmax=vabs
    ).plot(ax=ax, title=title)
    result = (fig, ax)
    return result


def plot_lisa(fc, *, ax=None):
    """Map the Local Moran (LISA) `cluster` classes — HH red … LL blue, ns neutral. Returns `(fig, ax)`.

    The classes are placed on a diverging ordinal scale (HH=+2, HL=+1, LH=-1, LL=-2, ns=0), so the HL/LH spatial
    *outliers* share the warm/cool hue family with the HH/LL *clusters* rather than getting distinct categorical
    colours. This reads as a hot-to-cold gradient; use `fc["cluster"]` directly if you need the nominal classes.
    """
    codes = fc["cluster"].map(_CLUSTER_ORDINAL).to_numpy(dtype=float)
    return _class_choropleth(fc, codes, "LISA clusters", 2.0, ax)


def plot_hotspots(fc, *, ax=None):
    """Map the Getis-Ord Gi* `hotspot` classes — hot red, cold blue, ns neutral. Returns `(fig, ax)`."""
    codes = fc["hotspot"].map(_HOTSPOT_ORDINAL).to_numpy(dtype=float)
    return _class_choropleth(fc, codes, "Getis-Ord Gi* hotspots", 1.0, ax)
