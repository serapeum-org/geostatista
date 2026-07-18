"""Edge-case / internal-branch coverage: solver fallbacks, result reprs, private helpers, and error paths."""

import numpy as np
import geopandas as gpd
import pytest
from pyramids.feature import FeatureCollection
from shapely.geometry import MultiPolygon, Point, box

from geostatista import Samples, Weights, gearys_c, morans_i
from geostatista._solve.fit import fit_model
from geostatista._solve.neighborhood import Neighborhood
from geostatista._solve.system import solve_ordinary


def _samples(n: int = 40, seed: int = 0) -> Samples:
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0.0, 100.0, (n, 2))
    z = np.sin(xy[:, 0] / 25.0) * 10.0 + 20.0
    return Samples(gpd.GeoDataFrame({"z": z}, geometry=[Point(x, y) for x, y in xy], crs="EPSG:32633"))


def _polys(n: int) -> FeatureCollection:
    return FeatureCollection(
        gpd.GeoDataFrame({"v": np.arange(n, dtype=float)}, geometry=[box(c, 0, c + 1, 1) for c in range(n)], crs="EPSG:32633")
    )


def test_fit_rejects_unfittable_model():
    with pytest.raises(ValueError):
        fit_model(np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 2.0, 3.0, 3.5]), np.array([5, 5, 5, 5]), "power")


def test_neighborhood_max_dist_filters():
    nb = Neighborhood(np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]]), n_neighbors=3, max_dist=2.0)
    assert 2 not in nb.query(np.array([0.0, 0.0]))          # the point 10 away is dropped


def test_solve_ordinary_singular_uses_lstsq():
    weights, var = solve_ordinary(np.array([[1.0, 1.0], [1.0, 1.0]]), np.array([0.5, 0.5]), sill=1.0)
    assert np.isfinite(weights).all() and var >= 0.0        # rank-deficient LHS -> lstsq fallback


def test_result_repr_and_summary():
    fc = _polys(4)
    w = Weights.queen(fc)
    mr = morans_i(fc, "v", w, permutations=49, seed=0)
    assert "MoranResult" in repr(mr) and set(mr.summary()) == {"I", "EI", "z", "p_norm", "p_sim"}
    gr = gearys_c(fc, "v", w, permutations=49, seed=0)
    assert "GearyResult" in repr(gr) and "C" in gr.summary()


def test_build_weights_unknown_spec_raises():
    from geostatista.autocorrelation import _build_weights

    with pytest.raises(ValueError):
        _build_weights(_polys(3), "bogus")


def test_polygon_exterior_handles_multipolygon():
    from geostatista.autocorrelation import _polygon_exterior

    coords = _polygon_exterior(MultiPolygon([box(0, 0, 1, 1), box(5, 5, 7, 7)]))
    assert coords.shape[1] == 2                             # largest part's exterior ring


def test_krige_explicit_bounds():
    s = _samples(50)
    surface = s.krige("z", s.variogram("z").fit("spherical"), cell_size=10.0, bounds=(0.0, 0.0, 50.0, 50.0))
    assert np.asarray(surface.read_array()).shape == (2, 5, 5)


def test_samples_values_and_coords():
    s = _samples(20)
    assert s.values("z").shape == (20,)
    assert s.coords("z").shape == (20, 2)


def test_weights_transform_invalid_kind():
    with pytest.raises(ValueError):
        Weights.queen(_polys(3)).transform("bogus")


def test_variogram_repr_and_empty_bins():
    from geostatista.variogram import empirical_variogram

    vg = _samples(30).variogram("z", n_lags=10)
    assert "unfitted" in repr(vg)
    vg.fit("spherical")
    assert "model='spherical'" in repr(vg)
    # explicit max_dist + a far empty bin (four clustered points, wide range) -> zero count, NaN semivariance
    coords = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1]])
    lags, semiv, counts = empirical_variogram(coords, np.array([1.0, 2.0, 3.0, 4.0]), n_lags=10, max_dist=5.0)
    assert (counts == 0).any() and np.all(np.isnan(semiv[counts == 0]))
