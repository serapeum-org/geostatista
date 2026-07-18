"""geostatista - the geostatistics tier of the serapeum stack.

Variograms, kriging, and spatial autocorrelation on top of pyramids.
"""

# Import pyramids FIRST: it activates the vendored GDAL bindings before any
# `from osgeo import gdal` in a submodule runs. Kept optional so that tooling
# which only needs the package metadata (e.g. build backends) does not hard-fail
# when the GIS stack is absent.
try:
    import pyramids  # noqa: F401
except ImportError:  # pragma: no cover
    pass

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

from . import models
from .kriging import OrdinaryKriging
from .samples import Samples
from .surface import KrigedSurface
from .variogram import Variogram

__all__ = [
    "Samples",
    "Variogram",
    "KrigedSurface",
    "OrdinaryKriging",
    "models",
]
