---
title: Kriging in Python — how geostatista compares to PyKrige, GSTools, scikit-gstat and PySAL
description: An honest comparison of the Python geostatistics libraries — variograms, ordinary kriging and
  spatial autocorrelation — and where geostatista fits among them.
---

# Kriging in Python: which library should you use?

Python has several good geostatistics libraries and they do not overlap as much as their descriptions suggest.
This page says plainly what each is for and where `geostatista` fits, so you can pick the right one — which is
often not this one.

## The short answer

| You want to… | Use |
|---|---|
| Krige a point dataset, nothing else | [PyKrige](https://pypi.org/project/PyKrige/) |
| Model random fields, simulate, or do serious variography | [GSTools](https://pypi.org/project/gstools/) |
| Explore variograms, directionally or over space-time | [scikit-gstat](https://pypi.org/project/scikit-gstat/) |
| Moran's I, LISA, Getis-Ord on a GeoDataFrame | [esda][esda] with [libpysal][libpysal] |
| Krige **and** get a georeferenced raster out, in a GDAL workflow | **geostatista** |
| Kriging **and** autocorrelation behind one object model | **geostatista** |

If you already have a rasterio or xarray pipeline and only need kriging, PyKrige or GSTools will serve you better
than this library. There is no advantage in adopting a stack for one function.

## What geostatista is

The geostatistics tier of the [serapeum](https://github.com/serapeum-org) stack. Input is scattered point
observations; output is a continuous surface plus an uncertainty estimate, or per-feature autocorrelation
statistics.

It covers:

- **Variograms** — empirical clouds (Matheron and Cressie estimators) and fitted models: spherical, exponential,
  gaussian, matern, plus power and nugget functions.
- **Ordinary kriging** — onto a regular grid, returning a two-band `KrigedSurface` where band 0 is the estimate
  and band 1 the kriging variance, with a `cKDTree` moving neighbourhood for large sample sets.
- **Validation** — leave-one-out cross-validation with ME, RMSE and standardised-error diagnostics.
- **Spatial autocorrelation** — a sparse `Weights` matrix (queen and rook contiguity, k-nearest, distance band)
  feeding global Moran's I and Geary's C and local Moran/LISA and Getis-Ord Gi\*.

```python
from pyramids.feature import FeatureCollection
from geostatista import Samples

gauges = FeatureCollection.read_file("rain_gauges.geojson")
samples = Samples(gauges)                       # a Samples is-a FeatureCollection

vg = samples.variogram("rain", n_lags=15, max_dist=50_000)
vg.fit(model="spherical")                       # sets vg.nugget / vg.sill / vg.range_

surface = samples.krige("rain", vg, cell_size=1000, n_neighbors=32)
surface.to_file("rain_kriged.tif")              # band 0 estimate, band 1 variance
```

## Where it differs

**The output is a georeferenced raster, not an array.** `KrigedSurface` is a
[pyramids](https://github.com/serapeum-org/pyramids) `Dataset`, so it carries CRS, geotransform and nodata, and
writes to GeoTIFF, NetCDF or COG with `to_file`. PyKrige and GSTools return NumPy arrays; georeferencing them is
your problem. If the kriged surface is an intermediate step in a GIS pipeline rather than the end of the
analysis, that difference is the whole point.

**Kriging and autocorrelation share one object model.** Everything hangs off `Samples`, a `FeatureCollection`
subclass, so the same object that fits a variogram also computes Moran's I. Elsewhere that means PyKrige or
GSTools for one and esda plus libpysal for the other, with a conversion between them.

**The uncertainty comes back with the estimate.** The kriging variance is band 1 of the same raster, not a second
return value to keep track of.

## Where the others are stronger

Said plainly, because picking the wrong library wastes more of your time than any feature gap.

**PyKrige** is the most direct route to a kriged surface. It has universal and regression kriging and 3-D
kriging, none of which geostatista has. If kriging is all you need, start there.

**GSTools** is a far deeper geostatistical toolbox: simple, ordinary, universal and external-drift kriging,
random field generation including conditioned fields, a large library of covariance models you can also define
yourself, directional variogram estimation, and data normalisation. For research geostatistics rather than
applied interpolation it is the more serious tool, and it integrates with PyKrige.

**scikit-gstat** describes itself as geostatistics in the scipy style and is built for variogram exploration,
including directional and space-time variograms and model diagnostics that go beyond the fitting here.

**esda and libpysal** are the reference implementations for exploratory spatial data analysis. The PySAL
ecosystem has far more estimators, inference options and spatial-weights machinery than the focused subset here.
If spatial statistics is the work rather than a step in it, use PySAL.

**rasterio, rioxarray and GDAL** are not geostatistics libraries, but note that plain
[IDW](https://gdal.org/programs/gdal_grid.html) and GDAL's free interpolation algorithms live in pyramids, not
here — see [ADR 0001](adr/0001-pyramids-geostatista-boundary.md) for that boundary. If inverse-distance
weighting is enough, you do not need a geostatistics library at all.

## Honest limitations

- Ordinary kriging only. No universal, regression, indicator or co-kriging.
- Two dimensions only.
- No conditional simulation.
- The variogram model set is deliberately small.
- It pulls in pyramids and, with the `viz` extra, cleopatra. That is reasonable if you are already in this stack
  and heavy if you are not.

## Installing

```bash
pip install geostatista
conda install -c conda-forge geostatista
```

Plotting — variogram, LISA cluster and hotspot maps — is an extra, so a bare install carries no plotting stack:

```bash
pip install "geostatista[viz]"
```

## Where to go next

- [Kriging workflow](guides/kriging-workflow.md) — the full path from points to surface
- [Spatial autocorrelation](guides/spatial-autocorrelation.md) — weights, Moran's I, LISA and hotspots
- [Variogram models](examples/03_variogram_models.ipynb) — choosing and fitting a model

[esda]: https://pypi.org/project/esda/
[libpysal]: https://pypi.org/project/libpysal/
