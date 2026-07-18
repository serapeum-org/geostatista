[![Documentation](https://img.shields.io/badge/Documentation-blue?logo=github&logoColor=white)](https://serapeum-org.github.io/geostatista/)
[![Python Versions](https://img.shields.io/pypi/pyversions/geostatista.png)](https://img.shields.io/pypi/pyversions/geostatista)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![PyPI version](https://badge.fury.io/py/geostatista.svg)](https://badge.fury.io/py/geostatista)
[![codecov](https://codecov.io/gh/serapeum-org/geostatista/branch/main/graph/badge.svg)](https://codecov.io/gh/serapeum-org/geostatista)

Current release info
====================

| Name | Downloads | Version | Platforms |
| --- | --- | --- | --- |
| [![Conda Recipe](https://img.shields.io/badge/recipe-geostatista-green.svg)](https://anaconda.org/conda-forge/geostatista) | [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/geostatista.svg)](https://anaconda.org/conda-forge/geostatista) [![Downloads](https://pepy.tech/badge/geostatista)](https://pepy.tech/project/geostatista) ![PyPI - Downloads](https://img.shields.io/pypi/dd/geostatista?color=blue&style=flat-square) | [![Conda Version](https://img.shields.io/conda/vn/conda-forge/geostatista.svg)](https://anaconda.org/conda-forge/geostatista) [![PyPI version](https://badge.fury.io/py/geostatista.svg)](https://badge.fury.io/py/geostatista) | [![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/geostatista.svg)](https://anaconda.org/conda-forge/geostatista) |

# geostatista

**geostatista** is the geostatistics tier of the [serapeum](https://github.com/serapeum-org)
stack: variograms, kriging, and spatial autocorrelation, built on top of
[pyramids](https://github.com/serapeum-org/pyramids) and its GDAL stack. Input is
scattered point observations; output is a continuous surface plus an uncertainty
estimate, or per-feature autocorrelation statistics.

## Main Features

- **Variograms** — empirical variogram clouds (Matheron / Cressie estimators) and
  fitted theoretical models (`spherical`, `exponential`, `gaussian`, `matern`,
  plus `power` / `nugget` functions).
- **Ordinary kriging** — from a fitted variogram onto a regular grid, returning a
  2-band `KrigedSurface` (band 0 = estimate, band 1 = kriging variance) with a
  `cKDTree` moving neighborhood for large sample sets.
- **Validation** — leave-one-out cross-validation with ME / RMSE / standardized-error
  diagnostics.
- **Spatial autocorrelation** — a sparse `Weights` matrix (queen / rook contiguity,
  k-nearest, distance-band) feeding global Moran's I and Geary's C, and local
  Local Moran (LISA) and Getis-Ord Gi* statistics, with one-call
  `spatial_autocorrelation` / `hotspots` facades.
- **Plotting** — variogram, LISA-cluster, and hotspot maps through
  [cleopatra](https://github.com/serapeum-org/cleopatra) (the `viz` extra).

Everything hangs off `Samples`, a `FeatureCollection` subclass, so a column name is
always a method argument. See `planning/architecture.md` for the full scope and
object model, and [ADR 0001](docs/adr/0001-pyramids-geostatista-boundary.md) for the
pyramids ↔ geostatista boundary (pyramids keeps IDW and the free `gdal.Grid`
algorithms; geostatista owns every variogram and kriging variant).

## Installation

### conda (conda-forge)

```bash
conda install -c conda-forge geostatista
```

List the versions available on your platform with:

```bash
conda search geostatista --channel conda-forge
```

### pip (PyPI)

```bash
pip install geostatista
```

### From GitHub (latest development)

```bash
pip install git+https://github.com/serapeum-org/geostatista
```

## Quick start

Krige scattered point observations onto a grid, with an uncertainty band:

```python
from geostatista import Samples

samples = Samples.read_file("rain_gauges.geojson")   # a Samples is-a FeatureCollection

vg = samples.variogram("rain").fit(model="spherical")   # explore + fit spatial structure
surface = samples.krige("rain", vg, cell_size=1000)      # 2-band KrigedSurface (estimate + variance)
surface.to_file("rain.tif")                              # self-describing GeoTIFF (GS_* tags)

cv = samples.cross_validate("rain", vg)                  # leave-one-out diagnostics
print(cv.attrs["summary"])                               # ME, RMSE, standardized error, correlation
```

Measure and map spatial autocorrelation over polygon features:

```python
from geostatista import Weights, morans_i, local_morans

w = Weights.queen(tracts)                 # queen-contiguity weights
morans_i(tracts, "income", w)             # global Moran's I (I, EI, z, p)
lisa = local_morans(tracts, "income", w)  # per-feature LISA clusters (HH / LL / HL / LH / ns)
```

See the [documentation](https://serapeum-org.github.io/geostatista/) for the full guide.
