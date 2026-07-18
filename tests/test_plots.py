"""Plot-rendering tests via cleopatra (headless). Marked `plots`; the cleopatra-backed test skips when absent."""

import pathlib
import re

import numpy as np
import geopandas as gpd
import pytest
from shapely.geometry import Point

from geostatista import Samples

pytestmark = pytest.mark.plots


def _samples(n: int = 60, seed: int = 0) -> Samples:
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0.0, 100.0, (n, 2))
    z = np.sin(xy[:, 0] / 25.0) * np.cos(xy[:, 1] / 25.0) * 10.0 + 20.0
    return Samples(gpd.GeoDataFrame({"z": z}, geometry=[Point(x, y) for x, y in xy], crs="EPSG:32633"))


def test_no_direct_matplotlib_import_in_package():
    """geostatista draws only through cleopatra — never matplotlib/cartopy/contextily directly."""
    src = pathlib.Path(__file__).resolve().parent.parent / "src" / "geostatista"
    banned = re.compile(r"^\s*(import (matplotlib|cartopy|contextily)|from (matplotlib|cartopy|contextily))", re.M)
    offenders = [p.name for p in src.rglob("*.py") if banned.search(p.read_text(encoding="utf-8"))]
    assert not offenders, f"direct matplotlib/cartopy/contextily import in {offenders}"


def test_variogram_plot_via_cleopatra():
    pytest.importorskip("cleopatra")
    samples = _samples()
    vg = samples.variogram("z", n_lags=12)
    fig, ax = vg.plot()                                   # empirical scatter only
    assert fig is not None and ax is not None
    vg.fit("spherical")
    _, ax2 = vg.plot()                                    # scatter + fitted model curve
    assert len(ax2.get_lines()) >= 1                      # the model curve is a Line2D


def _tracts(n: int = 6):
    from pyramids.feature import FeatureCollection
    from shapely.geometry import box

    polys, value = [], []
    for r in range(n):
        for c in range(n):
            polys.append(box(c, r, c + 1, r + 1))
            value.append(float(r))
    return FeatureCollection(gpd.GeoDataFrame({"v": value}, geometry=polys, crs="EPSG:32633"))


def test_lisa_and_hotspot_maps_via_cleopatra():
    pytest.importorskip("cleopatra")
    from geostatista import Weights, getis_ord_gi, local_morans, plot_hotspots, plot_lisa

    tracts = _tracts()
    w = Weights.queen(tracts)
    lisa = local_morans(tracts, "v", w, permutations=99, seed=0)
    fig1, ax1 = plot_lisa(lisa)
    assert fig1 is not None and len(ax1.collections) >= 1     # the polygon collection was drawn

    hot = getis_ord_gi(tracts, "v", w, star=True)
    fig2, ax2 = plot_hotspots(hot)
    assert fig2 is not None and len(ax2.collections) >= 1
