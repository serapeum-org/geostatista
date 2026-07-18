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


def test_local_morans_matches_esda():
    esda = pytest.importorskip("esda")
    libpysal = pytest.importorskip("libpysal")
    gdf = _lattice()
    fc = FeatureCollection(gdf)

    w_lp = libpysal.weights.Queen.from_dataframe(gdf, use_index=False)
    w_lp.transform = "r"
    reference = esda.Moran_Local(gdf["v"].to_numpy(), w_lp, permutations=999, seed=0)

    mine = local_morans(fc, "v", Weights.queen(fc), permutations=999, seed=0)
    # local_i is deterministic -> exact parity (catches any n vs n-1 scaling error, not just proportionality)
    np.testing.assert_allclose(mine["local_i"].to_numpy(), reference.Is, rtol=1e-9)
    assert np.mean(mine["quadrant"].to_numpy() == reference.q) > 0.95      # same 1=HH,2=LH,3=LL,4=HL convention
    # both use the without-replacement conditional-permutation null, so z_sim magnitudes agree (no WR bias)
    ratio = np.sqrt(np.mean(mine["z_sim"].to_numpy() ** 2) / np.mean(reference.z_sim**2))
    assert 0.9 < ratio < 1.1


def test_getis_ord_matches_esda():
    esda = pytest.importorskip("esda")
    libpysal = pytest.importorskip("libpysal")
    gdf = _lattice()
    fc = FeatureCollection(gdf)

    w_lp = libpysal.weights.Queen.from_dataframe(gdf, use_index=False)
    # match our convention exactly: binary weights with a unit self-weight (classic binary Gi*).
    reference = esda.G_Local(gdf["v"].to_numpy(), w_lp, transform="B", star=1.0, permutations=0)

    mine = getis_ord_gi(fc, "v", Weights.queen(fc), star=True)
    np.testing.assert_allclose(mine["z"].to_numpy(), reference.Zs, rtol=1e-6, atol=1e-6)


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
