"""Numerical-parity checks against esda / libpysal / pykrige.

Marked `oracle` and gated with `importorskip`: they run only when the reference library is installed (dev-only —
the oracles are never a geostatista dependency) and are cleanly skipped otherwise.
"""

import numpy as np
import geopandas as gpd
import pytest
from pyramids.feature import FeatureCollection
from shapely.geometry import Point, box

from geostatista import Variogram, Weights, getis_ord_gi, local_morans, morans_i
from geostatista.kriging import OrdinaryKriging

pytestmark = pytest.mark.oracle


def _lattice(n: int = 6) -> gpd.GeoDataFrame:
    polys, value = [], []
    for r in range(n):
        for c in range(n):
            polys.append(box(c, r, c + 1, r + 1))
            value.append(float(r * n + c))
    return gpd.GeoDataFrame({"v": value}, geometry=polys, crs="EPSG:32633")


def test_morans_i_matches_esda():
    esda = pytest.importorskip("esda")
    libpysal = pytest.importorskip("libpysal")
    gdf = _lattice()
    fc = FeatureCollection(gdf)

    w_lp = libpysal.weights.Queen.from_dataframe(gdf, use_index=False)
    w_lp.transform = "r"
    reference = esda.Moran(gdf["v"].to_numpy(), w_lp, permutations=999)

    mine = morans_i(fc, "v", Weights.queen(fc), permutations=999, seed=0)
    assert mine.I == pytest.approx(reference.I, rel=1e-6)
    assert mine.EI == pytest.approx(reference.EI, rel=1e-9)
    assert mine.z_norm == pytest.approx(reference.z_norm, rel=1e-3)


def test_local_morans_correlates_with_esda():
    esda = pytest.importorskip("esda")
    libpysal = pytest.importorskip("libpysal")
    gdf = _lattice()
    fc = FeatureCollection(gdf)

    w_lp = libpysal.weights.Queen.from_dataframe(gdf, use_index=False)
    w_lp.transform = "r"
    reference = esda.Moran_Local(gdf["v"].to_numpy(), w_lp, permutations=999, seed=0)

    mine = local_morans(fc, "v", Weights.queen(fc), permutations=999, seed=0)
    corr = np.corrcoef(mine["local_i"].to_numpy(), reference.Is)[0, 1]
    assert corr > 0.999
    # quadrants use the same 1=HH, 2=LH, 3=LL, 4=HL convention
    assert np.mean(mine["quadrant"].to_numpy() == reference.q) > 0.95


def test_getis_ord_matches_esda():
    esda = pytest.importorskip("esda")
    libpysal = pytest.importorskip("libpysal")
    gdf = _lattice()
    fc = FeatureCollection(gdf)

    w_lp = libpysal.weights.Queen.from_dataframe(gdf, use_index=False)
    reference = esda.G_Local(gdf["v"].to_numpy(), w_lp, star=True, permutations=0)

    mine = getis_ord_gi(fc, "v", Weights.queen(fc), star=True)
    assert np.corrcoef(mine["z"].to_numpy(), reference.Zs)[0, 1] > 0.99


def test_ordinary_kriging_matches_pykrige():
    pykrige = pytest.importorskip("pykrige")
    rng = np.random.default_rng(0)
    xy = rng.uniform(0.0, 100.0, (40, 2))
    z = np.sin(xy[:, 0] / 20.0) * 10.0 + 20.0

    nugget, partial_sill, rng_ = 0.0, 5.0, 40.0
    vg = Variogram(np.array([1.0]), np.array([1.0]), np.array([1]))
    vg.model = "spherical"
    vg.nugget, vg.sill, vg.range_ = nugget, nugget + partial_sill, rng_
    from geostatista import models

    vg._func = models.spherical

    targets = rng.uniform(10.0, 90.0, (5, 2))
    mine = OrdinaryKriging(xy, z, vg, n_neighbors=None)
    mine_est = np.array([mine.predict_point(t)[0] for t in targets])

    reference = pykrige.ok.OrdinaryKriging(
        xy[:, 0], xy[:, 1], z, variogram_model="spherical",
        variogram_parameters=[partial_sill, rng_, nugget],
    )
    ref_est, _ = reference.execute("points", targets[:, 0], targets[:, 1])
    np.testing.assert_allclose(mine_est, np.asarray(ref_est).ravel(), rtol=1e-3, atol=1e-3)
