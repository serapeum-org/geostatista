[![Documentation](https://img.shields.io/badge/Documentation-blue?logo=github&logoColor=white)](https://serapeum-org.github.io/geostatista/)
[![Python Versions](https://img.shields.io/pypi/pyversions/geostatista.png)](https://img.shields.io/pypi/pyversions/geostatista)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![PyPI version](https://badge.fury.io/py/geostatista.svg)](https://badge.fury.io/py/geostatista)

# geostatista

**geostatista** is the geostatistics tier of the serapeum stack: variograms, kriging,
and spatial autocorrelation. Input is scattered point observations; output is a
continuous surface plus an uncertainty estimate, or per-feature autocorrelation
statistics. It is built on top of
[pyramids](https://github.com/serapeum-org/pyramids) and its GDAL stack.

## Main Features

- **Variograms** — empirical variogram clouds (Matheron / Cressie estimators) and
  fitted theoretical models (`spherical`, `exponential`, `gaussian`, `matern`).
- **Ordinary kriging** — onto a regular grid, returning a 2-band `KrigedSurface`
  (band 0 = estimate, band 1 = kriging variance) with a moving neighborhood for
  large sample sets.
- **Leave-one-out cross-validation** of the kriged surface (ME / RMSE / standardized
  error).
- **Spatial autocorrelation** — a sparse `Weights` matrix (queen / rook / k-nearest /
  distance-band) feeding global (Moran's I, Geary's C) and local (Local Moran / LISA,
  Getis-Ord Gi*) statistics.
- **Plotting** — variogram, LISA, and hotspot maps through cleopatra (the `viz` extra).

## Installation

- pip (PyPI):

```bash
pip install geostatista
```

- conda (conda-forge):

```bash
conda install -c conda-forge geostatista
```

- pixi (development):

```bash
pixi add geostatista
```

See [Installation](installation.md) for the full guide.

## Quick start

Krige scattered point observations onto a grid, with an uncertainty band:

```python
from geostatista import Samples

samples = Samples.read_file("rain_gauges.geojson")   # a Samples is-a FeatureCollection

vg = samples.variogram("rain").fit(model="spherical")   # explore + fit spatial structure
surface = samples.krige("rain", vg, cell_size=1000)      # 2-band KrigedSurface (estimate + variance)
surface.to_file("rain.tif")                              # self-describing GeoTIFF (GS_* tags)
```

Measure spatial autocorrelation over polygon features:

```python
from geostatista import Weights, morans_i

w = Weights.queen(tracts)          # queen-contiguity weights
morans_i(tracts, "income", w)      # global Moran's I (I, EI, z, p)
```

## Next steps

- Follow the [kriging workflow guide](guides/kriging-workflow.md) or the
  [spatial-autocorrelation guide](guides/spatial-autocorrelation.md).
- Run the example notebooks: [kriging workflow](examples/01_kriging_workflow.ipynb),
  [spatial autocorrelation](examples/02_spatial_autocorrelation.ipynb).
- Browse the API reference: [Samples](reference/samples.md),
  [Variogram & models](reference/variogram.md), [Kriging](reference/kriging.md),
  [Weights](reference/weights.md),
  [Spatial autocorrelation](reference/autocorrelation.md).
- Read the [Change log](change-log.md).
