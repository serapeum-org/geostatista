"""Tests for the kriging pipeline: models, variogram, Samples, ordinary kriging, cross-validation."""

import numpy as np
import geopandas as gpd
import pytest
from geostatista._crs import crs_equal
from shapely.geometry import Point

from geostatista import KrigedSurface, Samples, Variogram, models
from geostatista.kriging import OrdinaryKriging


def make_samples(n: int = 60, seed: int = 0, crs: str = "EPSG:32633") -> Samples:
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0.0, 100.0, (n, 2))
    z = np.sin(xy[:, 0] / 25.0) * np.cos(xy[:, 1] / 25.0) * 10.0 + 20.0
    gdf = gpd.GeoDataFrame({"z": z}, geometry=[Point(x, y) for x, y in xy], crs=crs)
    return Samples(gdf)


def fitted_variogram(samples: Samples, nugget: float | None = None) -> Variogram:
    vg = samples.variogram("z", n_lags=12).fit("spherical")
    if nugget is not None:
        vg.nugget = nugget
    return vg


# --- models (K3) ---------------------------------------------------------------

@pytest.mark.parametrize("name", ["spherical", "exponential", "gaussian", "matern"])
def test_models_zero_and_sill(name):
    func = models.BOUNDED_MODELS[name]
    assert func(0.0, 0.5, 2.0, 10.0) == pytest.approx(0.0)          # gamma(0) = 0
    far = float(func(1e6, 0.5, 2.0, 10.0))
    assert far == pytest.approx(2.0, rel=1e-3)                       # -> sill at large h
    # monotone non-decreasing on a grid
    h = np.linspace(0.1, 50.0, 200)
    g = func(h, 0.5, 2.0, 10.0)
    assert np.all(np.diff(g) >= -1e-9)


def test_power_and_nugget_models():
    assert models.power(0.0, 0.1, 0.5, 1.5) == pytest.approx(0.0)
    assert float(models.power(4.0, 0.0, 1.0, 2.0)) == pytest.approx(16.0)
    assert models.nugget(0.0, 3.0) == pytest.approx(0.0)
    assert float(models.nugget(5.0, 3.0)) == pytest.approx(3.0)


# --- variogram (K2/K3) ---------------------------------------------------------

def test_variogram_empirical_and_fit():
    s = make_samples(120)
    vg = s.variogram("z", n_lags=12)
    frame = vg.to_dataframe()
    assert set(frame.columns) == {"lag", "semivariance", "count"}
    assert 0 < vg.counts.sum() <= 120 * 119 // 2                    # pairs within max_dist, at most all pairs
    vg.fit("spherical")
    assert vg.nugget >= 0 and vg.sill > 0 and vg.range_ > 0
    assert float(vg.predict(0.0)) == pytest.approx(0.0)


@pytest.mark.parametrize("model", ["spherical", "exponential", "gaussian", "matern"])
def test_variogram_fit_sill_never_below_nugget(model):
    """The (nugget, partial_sill>=0) reparameterization guarantees a monotone, PSD-safe model."""
    s = make_samples(90)
    vg = s.variogram("z", n_lags=12).fit(model)
    assert vg.sill >= vg.nugget - 1e-9


def test_variogram_fit_nonconvergence_is_typed():
    from geostatista._solve.fit import VariogramFitError

    lags = np.array([1.0, 2.0])
    semiv = np.array([np.nan, np.nan])
    counts = np.array([0, 0])
    with pytest.raises(VariogramFitError):
        from geostatista._solve.fit import fit_model

        fit_model(lags, semiv, counts, "spherical")


# --- Samples validation (K1) ---------------------------------------------------

def test_samples_is_feature_collection_and_typed_slice():
    from pyramids.feature import FeatureCollection

    s = make_samples(10)
    assert isinstance(s, FeatureCollection)
    assert isinstance(s.iloc[:5], Samples)  # _constructor keeps the subclass


def test_samples_validation_errors():
    s = make_samples(10)
    with pytest.raises(ValueError):
        s.variogram("does_not_exist")
    # fewer than 3 points
    small = make_samples(2)
    with pytest.raises(ValueError):
        small.variogram("z")


def test_samples_drops_nan():
    s = make_samples(20)
    s.loc[s.index[:3], "z"] = np.nan
    coords, values = s._clean("t", "z")
    assert len(values) == 17


# --- ordinary kriging (K4/K5) --------------------------------------------------

def test_kriging_exact_at_sample_location():
    s = make_samples(50)
    vg = fitted_variogram(s, nugget=0.0)                            # no nugget -> exact interpolator
    coords, values = s._clean("t", "z")
    ok = OrdinaryKriging(coords, values, vg, n_neighbors=None)
    est, var = ok.predict_point(coords[7])
    assert est == pytest.approx(values[7], abs=1e-6)
    assert var == pytest.approx(0.0, abs=1e-6)


def test_duplicate_points_do_not_crash():
    s = make_samples(30)
    coords, values = s._clean("t", "z")
    coords = np.vstack([coords, coords[0]])                         # duplicate a point
    values = np.append(values, values[0] + 5.0)
    vg = fitted_variogram(s)
    ok = OrdinaryKriging(coords, values, vg, n_neighbors=None)
    est, var = ok.predict_point(np.array([50.0, 50.0]))
    assert np.isfinite(est) and var >= 0.0


def test_global_and_neighborhood_agree_when_n_ge_count():
    s = make_samples(25)
    vg = fitted_variogram(s)
    coords, values = s._clean("t", "z")
    target = np.array([40.0, 60.0])
    e_global, v_global = OrdinaryKriging(coords, values, vg, n_neighbors=None).predict_point(target)
    e_local, v_local = OrdinaryKriging(coords, values, vg, n_neighbors=100).predict_point(target)
    assert e_global == pytest.approx(e_local, rel=1e-9)
    assert v_global == pytest.approx(v_local, rel=1e-9)


# --- KrigedSurface (K5) --------------------------------------------------------

def test_krige_to_surface():
    s = make_samples(60)
    vg = fitted_variogram(s)
    surface = s.krige("z", vg, cell_size=5.0, n_neighbors=16)
    assert isinstance(surface, KrigedSurface)
    assert surface.band_count == 2
    assert surface.epsg == 32633
    arr = np.asarray(surface.read_array())
    variance = arr[1]
    assert np.all(variance >= -1e-9)                                # variance non-negative
    assert surface.variance.band_count == 1
    assert surface.model == "spherical"


def test_band_datasets_preserve_nodata():
    s = make_samples(30)
    vg = fitted_variogram(s)
    surface = s.krige("z", vg, cell_size=10.0, nodata=-1234.0)
    assert surface.variance.no_data_value[0] == -1234.0             # not the hardcoded -9999
    assert surface.estimate.no_data_value[0] == -1234.0


def test_idw_delegates_to_pyramids():
    from pyramids.dataset import Dataset

    s = make_samples(30)
    surface = s.interpolate_to_raster("z", method="idw", cell_size=10.0)  # non-kriging -> pyramids gdal.Grid
    assert isinstance(surface, Dataset)
    assert surface.band_count == 1                                   # IDW is single-band (no variance)


def test_krige_with_string_variogram_autofits():
    s = make_samples(50)
    surface = s.krige("z", "spherical", cell_size=10.0)             # string model name -> internal auto-fit
    assert isinstance(surface, KrigedSurface)
    assert surface.band_count == 2 and surface.model == "spherical"


def test_cressie_estimator_is_robust_to_outlier():
    s = make_samples(80)
    s.loc[s.index[0], "z"] = 500.0                                  # inject an outlier
    matheron = s.variogram("z", n_lags=10, estimator="matheron")
    cressie = s.variogram("z", n_lags=10, estimator="cressie")
    assert np.nanmean(cressie.semivariance) < np.nanmean(matheron.semivariance)  # robust to the outlier


def test_krige_onto_template_grid():
    from pyramids.dataset import Dataset, GeoReference

    s = make_samples(50)
    vg = fitted_variogram(s)
    template = Dataset.from_array(
        np.zeros((10, 12)),
        geo_ref=GeoReference(geo=(0.0, 8.0, 0.0, 100.0, 0.0, -8.0), epsg=32633),
    )
    surface = s.krige("z", vg, template=template)
    assert np.asarray(surface.read_array()).shape == (2, 10, 12)     # aligns cell-for-cell with the template
    assert surface.epsg == 32633


# --- CRS resolution (pyramids >=0.47: `Dataset.epsg` is None for "no CRS" AND for "no EPSG code") ---

def geostationary_wkt() -> str:
    """A real projection the EPSG register does not name, so `Dataset.epsg` reports None for it."""
    # `osgeo` is safe to import here even though geostatista takes no direct GDAL dependency: the
    # module-level `from geostatista import ...` above has already run `import pyramids`, which
    # activates the vendored bindings (see CLAUDE.md on the load-bearing import order).
    from osgeo import osr

    sr = osr.SpatialReference()
    sr.ImportFromProj4("+proj=geos +h=35785831 +lon_0=0 +datum=WGS84 +units=m +no_defs")
    return sr.ExportToWkt()


def template_with(epsg) -> "Dataset":
    """A 10x12 template raster georeferenced with `epsg` (a code, a WKT string, or None)."""
    from pyramids.dataset import Dataset, GeoReference

    return Dataset.from_array(
        np.zeros((10, 12)), geo_ref=GeoReference(geo=(0.0, 8.0, 0.0, 100.0, 0.0, -8.0), epsg=epsg)
    )


def test_crs_less_template_keeps_the_samples_crs():
    s = make_samples(40)
    surface = s.krige("z", fitted_variogram(s), template=template_with(None))
    assert surface.epsg == 32633                                     # template gives the grid, not the CRS


def test_crs_less_template_keeps_an_explicit_epsg():
    s = make_samples(40)
    engine = OrdinaryKriging(*s._clean("t", "z"), fitted_variogram(s))
    surface = engine.predict_grid(template=template_with(None), epsg=3857)
    assert surface.epsg == 3857                                      # the argument is not overwritten


def test_template_crs_wins_over_the_samples_crs():
    s = make_samples(40, crs="EPSG:3857")                            # template and layer agree
    surface = s.krige("z", fitted_variogram(s), template=template_with(3857))
    assert surface.epsg == 3857                                      # a template that names a CRS decides


def test_template_crs_disagreeing_with_the_layer_is_refused():
    s = make_samples(40)                                             # samples are EPSG:32633
    with pytest.raises(ValueError, match="differs from the layer"):
        s.krige("z", fitted_variogram(s), template=template_with(3857))


def test_crs_less_layer_accepts_a_template_that_names_a_crs():
    """A layer with no CRS has nothing to disagree with, so the template's CRS is adopted rather than refused."""
    s = make_samples(40, crs=None)
    surface = s.krige("z", fitted_variogram(s), template=template_with(32633))
    assert surface.epsg == 32633                                     # adopted, not rejected as a mismatch
    assert np.asarray(surface.read_array()).shape == (2, 10, 12)


def test_template_without_epsg_code_keeps_its_projection():
    wkt = geostationary_wkt()
    s = make_samples(40, crs=wkt)                                    # layer and template agree
    template = template_with(wkt)
    assert template.epsg is None and template.crs                    # a real CRS that has no EPSG code
    surface = s.krige("z", fitted_variogram(s), template=template)
    assert surface.epsg is None
    assert "geos" in surface.crs.lower()                             # WKT carried through, not discarded


def test_crs_less_samples_produce_an_unreferenced_surface():
    rng = np.random.default_rng(0)
    xy = rng.uniform(0.0, 100.0, (40, 2))
    z = np.sin(xy[:, 0] / 25.0) * np.cos(xy[:, 1] / 25.0) * 10.0 + 20.0
    s = Samples(gpd.GeoDataFrame({"z": z}, geometry=[Point(x, y) for x, y in xy], crs=None))
    assert s._epsg() is None                                         # no CRS is not EPSG:4326
    surface = s.krige("z", fitted_variogram(s), cell_size=10.0)
    assert surface.epsg is None
    assert not surface.crs                                           # WGS 84 is not invented


def test_samples_crs_without_an_epsg_code_reaches_the_surface():
    """A sample layer on a projection the EPSG register does not name keeps that projection, not WGS 84."""
    s = make_samples(40, crs=geostationary_wkt())
    assert s.crs.to_epsg() is None                                   # a real CRS that has no EPSG code
    assert isinstance(s._epsg(), str)                                # WKT specification, not the old 4326 fallback
    surface = s.krige("z", fitted_variogram(s), cell_size=10.0)
    assert surface.epsg is None
    assert "geos" in surface.crs.lower()                             # the samples' projection, carried through


def test_predict_grid_invents_no_crs():
    """Told nothing about the CRS, the engine leaves the surface unreferenced rather than guessing WGS 84."""
    s = make_samples(30)
    engine = OrdinaryKriging(*s._clean("t", "z"), fitted_variogram(s))
    surface = engine.predict_grid(cell_size=20.0)
    assert surface.epsg is None
    assert not surface.crs


def test_bands_carry_the_surface_crs():
    s = make_samples(40)
    surface = s.krige("z", fitted_variogram(s), template=template_with(32633))
    assert surface.epsg == 32633
    for band in (surface.estimate, surface.variance):
        assert crs_equal(band.epsg, surface.epsg)
        assert crs_equal(band.crs, surface.crs)


def test_bands_of_an_unreferenced_surface_stay_unreferenced():
    """The `_band` fix must carry an *absent* CRS through too, not just a present one."""
    s = make_samples(40, crs=None)
    surface = s.krige("z", fitted_variogram(s), cell_size=10.0)
    assert surface.epsg is None and not surface.crs
    for band in (surface.estimate, surface.variance):
        assert band.epsg is None
        assert not band.crs


def test_bands_carry_a_projection_that_has_no_epsg_code():
    wkt = geostationary_wkt()
    s = make_samples(40, crs=wkt)
    surface = s.krige("z", fitted_variogram(s), template=template_with(wkt))
    for band in (surface.estimate, surface.variance):
        assert crs_equal(band.crs, surface.crs)                      # not dropped by `_band`
        assert "geos" in band.crs.lower()


def test_template_is_rejected_on_the_idw_branch():
    s = make_samples(40)
    with pytest.raises(ValueError, match="only supported for method='kriging'"):
        s.interpolate_to_raster("z", method="idw", cell_size=10.0, template=template_with(32633))


def test_idw_honours_an_explicit_n_neighbors():
    s = make_samples(40)
    unlimited = s.interpolate_to_raster("z", method="idw", cell_size=10.0)
    limited = s.interpolate_to_raster("z", method="idw", cell_size=10.0, n_neighbors=4)
    a, b = np.asarray(unlimited.read_array()), np.asarray(limited.read_array())
    assert not np.allclose(a, b)                                     # invdistnn(4), not invdist over all points


def test_idw_without_n_neighbors_keeps_pyramids_own_default():
    """An unspoken `n_neighbors` is not forwarded, so IDW stays `invdist` instead of becoming `invdistnn(32)`."""
    from pyramids.feature import FeatureCollection

    s = make_samples(40)
    omitted = np.asarray(s.interpolate_to_raster("z", method="idw", cell_size=10.0).read_array())
    pyramids_default = np.asarray(
        FeatureCollection(s).interpolate_to_raster("z", method="idw", cell_size=10.0).read_array()
    )
    kriging_default = np.asarray(
        s.interpolate_to_raster("z", method="idw", cell_size=10.0, n_neighbors=32).read_array()
    )
    np.testing.assert_allclose(omitted, pyramids_default)            # the base method's own call, unaltered
    assert not np.allclose(omitted, kriging_default)                 # kriging's 32 must not leak into IDW


def test_idw_accepts_an_explicit_none_as_all_points():
    """`None` is a spoken value, not the unset sentinel: it reaches pyramids and means every point."""
    s = make_samples(40)
    omitted = np.asarray(s.interpolate_to_raster("z", method="idw", cell_size=10.0).read_array())
    explicit = np.asarray(
        s.interpolate_to_raster("z", method="idw", cell_size=10.0, n_neighbors=None).read_array()
    )
    np.testing.assert_allclose(omitted, explicit)


def test_kriging_without_n_neighbors_uses_the_moving_neighborhood_default():
    """The sentinel's kriging arm: saying nothing means 32 neighbours, recorded on the surface's provenance."""
    s = make_samples(40)
    surface = s.interpolate_to_raster("z", method="kriging", variogram=fitted_variogram(s), cell_size=20.0)
    assert surface.n_neighbors == 32
    assert surface.meta_data.get("GS_NEIGHBORS") == "32"


def test_predict_grid_requires_cell_size_or_template():
    s = make_samples(30)
    engine = OrdinaryKriging(*s._clean("t", "z"), fitted_variogram(s))
    with pytest.raises(ValueError):
        engine.predict_grid()                                        # neither cell_size nor template


def test_variogram_predict_before_fit_raises():
    s = make_samples(20)
    with pytest.raises(RuntimeError):
        s.variogram("z").predict(10.0)


def test_kriging_rejects_an_unfitted_variogram():
    """An empirical variogram that was never `.fit()` has no sill, so kriging refuses it up front."""
    s = make_samples(20)
    with pytest.raises(ValueError, match="must be fitted"):
        OrdinaryKriging(*s._clean("t", "z"), s.variogram("z"))


def test_surface_estimate_band():
    s = make_samples(40)
    surface = s.krige("z", fitted_variogram(s), cell_size=10.0)
    assert surface.estimate.band_count == 1


def test_krige_constant_field_raises():
    from geostatista import VariogramFitError

    xy = np.random.default_rng(0).uniform(0.0, 100.0, (20, 2))
    s = Samples(gpd.GeoDataFrame({"z": np.ones(20)}, geometry=[Point(x, y) for x, y in xy], crs="EPSG:32633"))
    with pytest.raises(VariogramFitError):                          # constant field -> zero sill, refuse to krige
        s.krige("z", "spherical", cell_size=10.0)


def test_surface_roundtrip(tmp_path):
    s = make_samples(40)
    vg = fitted_variogram(s)
    surface = s.krige("z", vg, cell_size=10.0)
    out = tmp_path / "krige.tif"
    surface.to_file(str(out))
    assert out.exists()
    from pyramids.dataset import Dataset

    reopened = Dataset.read_file(str(out))
    assert reopened.band_count == 2
    np.testing.assert_allclose(
        np.asarray(reopened.read_array()), np.asarray(surface.read_array()), rtol=1e-4
    )


def test_surface_provenance_tags_persist(tmp_path):
    s = make_samples(40)
    vg = fitted_variogram(s)
    surface = s.krige("z", vg, cell_size=10.0)
    assert surface.meta_data.get("GS_MODEL") == "spherical"          # tagged in from_arrays
    out = tmp_path / "prov.tif"
    surface.to_file(str(out))
    from pyramids.dataset import Dataset

    reopened = Dataset.read_file(str(out))
    assert reopened.meta_data.get("GS_CLASS") == "KrigedSurface"      # provenance survives the write
    assert reopened.meta_data.get("GS_MODEL") == "spherical"
    assert reopened.meta_data.get("GS_NEIGHBORS") == "32"


# --- cross-validation (K6) -----------------------------------------------------

def test_cross_validate():
    s = make_samples(50)
    vg = fitted_variogram(s)
    cv = s.cross_validate("z", vg, n_neighbors=None)
    assert len(cv) == 50
    assert set(cv.columns) == {
        "observed",
        "predicted",
        "error",
        "kriging_variance",
        "standardized_error",
    }
    summary = cv.attrs["summary"]
    assert set(summary) == {"ME", "RMSE", "mean_standardized_error", "RMSE_standardized", "correlation"}
    assert summary["RMSE"] >= 0.0
