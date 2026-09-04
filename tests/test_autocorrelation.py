"""Tests for the spatial-weights subsystem and autocorrelation statistics."""

import numpy as np
import geopandas as gpd
import pytest
from pyramids.feature import FeatureCollection
from shapely.geometry import box

from geostatista import (
    Weights,
    gearys_c,
    getis_ord_gi,
    hotspots,
    local_morans,
    morans_i,
    spatial_autocorrelation,
)


def lattice(n: int = 5) -> FeatureCollection:
    """An n x n lattice of unit-square polygons; `v` increases by row, `block` is a planted 2x2 hot cluster."""
    polys, row_value = [], []
    for r in range(n):
        for c in range(n):
            polys.append(box(c, r, c + 1, r + 1))
            row_value.append(float(r))
    block = np.ones(n * n)
    grid = np.arange(n * n).reshape(n, n)
    for idx in grid[:2, :2].ravel():
        block[idx] = 10.0                          # planted hot cluster (top-left)
    for idx in grid[-2:, -2:].ravel():
        block[idx] = -10.0                         # planted cold cluster (bottom-right)
    rand = np.random.default_rng(0).normal(size=n * n)
    gdf = gpd.GeoDataFrame({"v": row_value, "block": block, "rand": rand}, geometry=polys, crs="EPSG:32633")
    return FeatureCollection(gdf)


# --- Weights (S1) --------------------------------------------------------------

def test_queen_rook_cardinalities():
    fc = lattice(5)
    queen = Weights.queen(fc)
    rook = Weights.rook(fc)
    assert queen.cardinalities.sum() == 144        # 4*3 + 12*5 + 9*8
    assert queen.cardinalities.max() == 8 and queen.cardinalities.min() == 3
    assert rook.cardinalities.sum() == 80          # 4*2 + 12*3 + 9*4
    assert rook.cardinalities.max() == 4 and rook.cardinalities.min() == 2


def test_knn_and_row_standardize():
    fc = lattice(5)
    w = Weights.knn(fc, 4)
    assert np.all(w.cardinalities == 4)
    rowsum = np.asarray(w.transform("r").sparse.sum(axis=1)).ravel()
    np.testing.assert_allclose(rowsum, 1.0)


def test_knn_rejects_k_ge_n():
    fc = lattice(3)                                 # 9 features
    with pytest.raises(ValueError):
        Weights.knn(fc, 9)


def test_neighbors_property_and_inverse_distance_band():
    fc = lattice(4)
    w = Weights.queen(fc)
    assert isinstance(w.neighbors, dict) and len(w.neighbors) == 16
    inv = Weights.distance_band(fc, 1.5, binary=False)
    assert (inv.sparse.data > 0).all() and (inv.sparse.data != 1.0).any()   # inverse-distance weights, not binary


def test_facade_string_weight_specs():
    fc = lattice(4)
    for spec in ("queen", "rook", "knn", "distance_band"):
        summary = spatial_autocorrelation(fc, "v", weights=spec)
        assert summary["weights"] == "b" and summary["n"] == 16


def test_local_morans_island_is_ns():
    from shapely.geometry import box

    polys = [box(0, 0, 1, 1), box(1, 0, 2, 1), box(50, 50, 51, 51)]   # two adjacent + one far island
    fc = FeatureCollection(gpd.GeoDataFrame({"v": [1.0, 2.0, 3.0]}, geometry=polys, crs="EPSG:32633"))
    out = local_morans(fc, "v", Weights.queen(fc), permutations=49, seed=0)
    assert out["cluster"].iloc[2] == "ns" and np.isnan(out["p_sim"].iloc[2])   # island: no neighbors


def test_distance_band_symmetric_and_islands():
    fc = lattice(5)
    w = Weights.distance_band(fc, 1.5)             # 1.5 > diagonal (~1.414) -> queen-like band (degree 8 interior)
    diff = (w.sparse - w.sparse.T)
    assert abs(diff).sum() == 0.0                  # symmetric
    tiny = Weights.distance_band(fc, 0.1)          # nobody within 0.1 -> all islands
    assert len(tiny.islands) == 25


def test_contiguity_requires_polygons():
    from shapely.geometry import Point

    pts = FeatureCollection(gpd.GeoDataFrame({"v": [1, 2, 3]}, geometry=[Point(0, 0), Point(1, 1), Point(2, 2)]))
    with pytest.raises(ValueError):
        Weights.queen(pts)


# --- global autocorrelation (S2) -----------------------------------------------

def test_morans_i_detects_gradient():
    fc = lattice(6)
    w = Weights.queen(fc)
    clustered = morans_i(fc, "v", w, permutations=199, seed=1)
    assert clustered.I > 0.3                        # strong positive autocorrelation
    assert clustered.EI == pytest.approx(-1.0 / (36 - 1))
    assert clustered.p_sim <= 0.05
    random = morans_i(fc, "rand", w, permutations=199, seed=1)
    assert random.I < clustered.I                   # random field far less autocorrelated


def test_gearys_c_gradient():
    fc = lattice(6)
    w = Weights.queen(fc)
    result = gearys_c(fc, "v", w, permutations=199, seed=1)
    assert result.EC == 1.0
    assert result.C < 1.0                            # positive autocorrelation -> C below 1


# --- local autocorrelation (S3) ------------------------------------------------

def test_local_morans_columns_and_clusters():
    fc = lattice(6)
    w = Weights.queen(fc)
    out = local_morans(fc, "v", w, permutations=199, seed=1)
    assert isinstance(out, FeatureCollection)
    for col in ("local_i", "z_sim", "p_sim", "quadrant", "cluster"):
        assert col in out.columns
    assert set(np.unique(out["quadrant"])).issubset({0, 1, 2, 3, 4})
    assert (out["cluster"] != "ns").any()           # the gradient produces some significant clusters


# --- Getis-Ord Gi* (S4) --------------------------------------------------------

def test_getis_ord_hotspot():
    fc = lattice(5)
    w = Weights.queen(fc)
    out = getis_ord_gi(fc, "block", w, star=True)
    for col in ("gi", "z", "p", "hotspot"):
        assert col in out.columns
    grid = np.arange(25).reshape(5, 5)
    block_idx = grid[:2, :2].ravel()
    assert (out["hotspot"].to_numpy()[block_idx] == "hot").any()   # the planted block is hot
    assert "cold" in set(out["hotspot"])                           # low-value region is cold


def test_getis_ord_gi_statistic_matches_definition():
    """`gi` is the Getis-Ord ratio statistic; star includes self (denom = total), non-star excludes it."""
    from shapely.geometry import box

    # a row of three unit boxes: queen neighbors A~B, B~C (A and C are not adjacent).
    polys = [box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)]
    fc = FeatureCollection(gpd.GeoDataFrame({"x": [1.0, 2.0, 3.0]}, geometry=polys, crs="EPSG:32633"))
    w = Weights.queen(fc)
    star = getis_ord_gi(fc, "x", w, star=True)
    # Gi* = sum_{j in N(i) U {i}} x_j / total(=6): A={A,B}=3/6, B={A,B,C}=6/6, C={B,C}=5/6
    np.testing.assert_allclose(star["gi"].to_numpy(), [3 / 6, 6 / 6, 5 / 6], rtol=1e-9)
    non_star = getis_ord_gi(fc, "x", w, star=False)
    # Gi = sum_{j in N(i)} x_j / (total - x_i): A={B}=2/5, B={A,C}=4/4, C={B}=2/3
    np.testing.assert_allclose(non_star["gi"].to_numpy(), [2 / 5, 4 / 4, 2 / 3], rtol=1e-9)
    # feature B is adjacent to every other feature -> degenerate Gi denominator -> z is nan (both branches)
    assert np.isnan(star["z"].to_numpy()[1]) and np.isnan(non_star["z"].to_numpy()[1])
    # the finite features (A, C) standardize differently between star and non-star
    finite = [0, 2]
    assert not np.allclose(star["z"].to_numpy()[finite], non_star["z"].to_numpy()[finite])


def test_constant_field_is_guarded():
    """A constant column has no autocorrelation — statistics are nan (not inf/crash)."""
    from shapely.geometry import box

    polys = [box(c, 0, c + 1, 1) for c in range(6)]
    fc = FeatureCollection(gpd.GeoDataFrame({"c": np.ones(6)}, geometry=polys, crs="EPSG:32633"))
    w = Weights.queen(fc)
    assert np.isnan(morans_i(fc, "c", w, permutations=49, seed=0).I)
    assert np.isnan(gearys_c(fc, "c", w, permutations=49, seed=0).C)
    lisa = local_morans(fc, "c", w, permutations=49, seed=0)
    assert np.all(np.isnan(lisa["local_i"].to_numpy())) and (lisa["cluster"] == "ns").all()
    gi = getis_ord_gi(fc, "c", w)
    assert np.all(np.isnan(gi["z"].to_numpy())) and (gi["hotspot"] == "ns").all()   # constant -> undefined (nan)


# --- facade (S5) ---------------------------------------------------------------

def test_facade_functions():
    fc = lattice(5)
    summary = spatial_autocorrelation(fc, "v", weights="queen")
    assert set(summary) == {"I", "EI", "z", "p", "n", "weights"}
    assert summary["n"] == 25
    assert summary == spatial_autocorrelation(fc, "v", weights="queen")   # deterministic (analytic z/p)
    hot = hotspots(fc, "block", weights="queen")
    assert isinstance(hot, FeatureCollection)
    assert "hotspot" in hot.columns


def test_facade_accepts_a_prebuilt_weights_instance():
    """The `Weights | str` facade argument takes a ready-made `Weights`, not only a named default."""
    fc = lattice(5)
    summary = spatial_autocorrelation(fc, "v", weights=Weights.knn(fc, 4))
    assert summary["n"] == 25
    assert summary["weights"] == "b"
    built = spatial_autocorrelation(fc, "v", weights=Weights.queen(fc))
    assert built == spatial_autocorrelation(fc, "v", weights="queen")   # same matrix, passed either way
    assert "hotspot" in hotspots(fc, "block", weights=Weights.queen(fc)).columns
