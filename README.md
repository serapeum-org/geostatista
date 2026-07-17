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

- Spatial interpolation via inverse-distance weighting (IDW/ISDW).

## Roadmap

- Empirical variograms, model fitting, and ordinary kriging with a variance band.
- Leave-one-out cross-validation of the kriged surface.
- A spatial-weights subsystem for Moran's I / Getis-Ord Gi*.

See `planning/architecture.md` for the full scope and object model.

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

```python
import geostatista

print(geostatista.__version__)
```

See the [documentation](https://serapeum-org.github.io/geostatista/) for the full guide.
