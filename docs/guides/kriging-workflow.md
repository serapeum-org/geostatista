# Kriging workflow

The kriging pipeline is: **explore spatial structure → fit a variogram → krige onto a grid → validate honestly.**
Everything hangs off `Samples`, a `FeatureCollection` subclass, so `column` is always a method argument.

## 1. Load points into `Samples`

```python
from pyramids.feature import FeatureCollection
from geostatista import Samples

gauges = FeatureCollection.read_file("rain_gauges.geojson")
samples = Samples(gauges)                       # a Samples is-a FeatureCollection
```

## 2. Explore spatial structure — the variogram

Look before you krige. The empirical variogram bins pairwise squared differences by lag distance.

```python
vg = samples.variogram("rain", n_lags=15, max_dist=50_000)
vg.to_dataframe()                               # lag / semivariance / count
```

## 3. Fit a model

Kriging needs a valid positive-definite model, not the empirical cloud. Choose `spherical`, `exponential`,
`gaussian`, or `matern`.

```python
vg.fit(model="spherical")                       # sets vg.nugget / vg.sill / vg.range_
vg.predict(1000.0)                              # modeled semivariance at a 1 km lag
```

**Which model?** `spherical` reaches its sill exactly at the range; `exponential` and `gaussian` approach it
asymptotically (`gaussian` is smoothest near the origin); `matern` adds a smoothness parameter. Fit a few and let
cross-validation (step 5) pick the best. The `estimator` argument to `variogram()` chooses how the empirical cloud
is built — `"matheron"` (classic) or `"cressie"` (robust to outliers).

**Look at the fit.** With the `viz` extra installed, `Variogram.plot` overlays the fitted curve on the empirical
cloud (via cleopatra):

```python
fig, ax = vg.plot(title="Rainfall variogram")   # scatter of the cloud + the fitted model curve
```

## 4. Krige onto a grid

Kriging returns a `KrigedSurface` — a 2-band `Dataset` (band 0 estimate, band 1 variance).

```python
surface = samples.interpolate_to_raster(
    "rain", method="kriging", variogram=vg, cell_size=1000, n_neighbors=32
)
# equivalently: surface = samples.krige("rain", vg, cell_size=1000, n_neighbors=32)

surface.to_file("rain.tif")                     # 2-band GeoTIFF, self-describing (GS_* metadata tags)
estimate = surface.estimate                     # band 0 as a Dataset
variance = surface.variance                     # band 1 as a Dataset — the reason to prefer kriging over IDW

estimate.plot()                                 # map the interpolated surface (via cleopatra)
variance.plot()                                 # map where the estimate is least certain
```

You can pass a model **name** instead of a fitted `Variogram` to auto-fit it internally
(`variogram="spherical"`), or an existing `Dataset` as a `template=` to align the output grid cell-for-cell. For
large sample sets, `n_neighbors` bounds the moving neighborhood around each target cell (pass `n_neighbors=None`
for the exact global solve).

## 5. Validate honestly — leave-one-out

```python
cv = samples.cross_validate("rain", vg)
cv.attrs["summary"]                             # ME, RMSE, mean/standardized error, correlation
```

For a well-specified variogram the mean standardized error is ≈ 0 and the standardized RMSE ≈ 1.

## Where IDW lives

`Samples.interpolate_to_raster("rain", method="idw")` delegates to pyramids' `gdal.Grid` IDW — geostatista only owns
the kriging path (see [ADR 0001](../adr/0001-pyramids-geostatista-boundary.md)).
