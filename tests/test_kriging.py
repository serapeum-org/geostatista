"""Tests for the kriging pipeline: models, variogram, Samples, ordinary kriging, cross-validation."""

import numpy as np
import geopandas as gpd
import pytest
from shapely.geometry import Point

from geostatista import KrigedSurface, Samples, Variogram, models
from geostatista.kriging import OrdinaryKriging


def make_samples(n: int = 60, seed: int = 0) -> Samples:
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0.0, 100.0, (n, 2))
    z = np.sin(xy[:, 0] / 25.0) * np.cos(xy[:, 1] / 25.0) * 10.0 + 20.0
    gdf = gpd.GeoDataFrame({"z": z}, geometry=[Point(x, y) for x, y in xy], crs="EPSG:32633")
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
