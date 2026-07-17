# Installation

**Package name:** `geostatista`
**Supported Python versions:** 3.11, 3.12, 3.13 (requires `>=3.11,<4`)

Please install `geostatista` in a virtual environment so its requirements don't
tamper with your system Python.

## With conda (recommended)

`geostatista` and its native dependency (GDAL, pulled in transitively through
`pyramids-gis`) are easiest to install via the
[conda-forge](https://conda-forge.org/) channel:

```console
conda install -c conda-forge geostatista
```

This installs `geostatista` together with all dependencies, including Python and
GDAL.

## With pip (PyPI)

```console
pip install geostatista
```

GDAL is not pulled in directly by pip (no Windows wheel exists on PyPI). It arrives
transitively through the `pyramids-gis` and `pyogrio` wheels, which vendor GDAL, so
a plain `pip install geostatista` works on every platform.

## With pixi (development)

This repository is configured with [pixi](https://pixi.sh). To set up a development
environment that manages GDAL via conda-forge:

```console
git clone https://github.com/serapeum-org/geostatista.git
cd geostatista
pixi install -e dev
pixi run -e dev main      # runs the test suite
```

Pixi environments available:

| Environment | Purpose |
|-------------|---------|
| `dev` | Default development env (test tooling) |
| `docs` | Documentation toolchain (mkdocs + plugins) |
| `notebook` | Jupyter notebook environment |
| `py311`, `py312`, `py313` | Single-Python-version test envs |

## Install directly from GitHub

Latest `main`:

```console
pip install "git+https://github.com/serapeum-org/geostatista.git"
```

A specific tagged release:

```console
pip install "git+https://github.com/serapeum-org/geostatista.git@<version>"
```

## Verify the install

```python
import geostatista
print(geostatista.__version__)
```
