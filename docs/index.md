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

- Spatial interpolation via inverse-distance weighting (IDW/ISDW).
- Roadmap: empirical variograms, model fitting, ordinary kriging with a variance
  band, leave-one-out cross-validation, and (later) a spatial-weights subsystem for
  Moran's I / Getis-Ord Gi*.

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

```python
import geostatista

print(geostatista.__version__)
```

## Next steps

- Browse the [API Reference](reference/tools.md).
- Read the [Change log](change-log.md).
