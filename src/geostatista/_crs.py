"""One import point for pyramids' CRS helpers.

`crs_spec` and `crs_equal` live in `pyramids.base.crs`, a module path pyramids does not re-export
from its package root. Three modules here need them, and the 0.46 -> 0.59 window this package just
crossed reorganized exactly this area twice (`create_from_array` -> `from_array`, the `GeoReference`
value object). Routing the import through one module makes the next reshuffle a one-line change
instead of a three-file one.

`crs_spec(epsg, wkt)` resolves the usable CRS specification — the EPSG code when there is one, else
the WKT, else `None`. `crs_equal(a, b)` compares two such specifications without relying on WKT
string identity, which varies between GDAL/PROJ builds.
"""

from pyramids.base.crs import crs_equal, crs_spec

__all__ = ["crs_equal", "crs_spec"]
