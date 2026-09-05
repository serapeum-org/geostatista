"""`KrigedSurface` — a 2-band `Dataset` carrying the kriged estimate + variance and its provenance.

Band 0 is the kriging estimate, band 1 the kriging variance. `Dataset` is a thin `gdal.Dataset` wrapper, so
subclassing it is safe (no pandas `_constructor` machinery). Provenance (the variogram that produced the surface)
is stored as typed attributes and, on `persist_metadata`, as raster tags so a written GeoTIFF is self-describing.
"""

import numpy as np
from pyramids.dataset import Dataset, GeoReference

from ._crs import crs_spec


class KrigedSurface(Dataset):
    """A kriged surface: band 0 estimate, band 1 variance, tagged with the producing variogram."""

    def __init__(
        self,
        src,
        access: str = "read_only",
        *,
        model: str | None = None,
        nugget: float | None = None,
        sill: float | None = None,
        range_: float | None = None,
        n_neighbors: int | None = None,
    ):
        super().__init__(src, access)
        self.model = model
        self.nugget = nugget
        self.sill = sill
        self.range_ = range_
        self.n_neighbors = n_neighbors

    @classmethod
    def from_arrays(
        cls,
        estimate: np.ndarray,
        variance: np.ndarray,
        *,
        geo: tuple[float, float, float, float, float, float],
        epsg: int | str | None,
        nodata: float = -9999.0,
        model: str | None = None,
        nugget: float | None = None,
        sill: float | None = None,
        range_: float | None = None,
        n_neighbors: int | None = None,
    ) -> "KrigedSurface":
        """Build a `KrigedSurface` from estimate + variance arrays and a geotransform.

        `epsg` takes a code, a CRS specification string (WKT — what `crs_spec` returns for a
        projection the EPSG register does not name), or `None` to leave the surface without a
        CRS, matching what pyramids reports for an ungeoreferenced template rather than
        defaulting to WGS 84.
        """
        stacked = np.stack([np.asarray(estimate, dtype=float), np.asarray(variance, dtype=float)])
        dataset = Dataset.from_array(
            stacked, geo_ref=GeoReference(geo=geo, epsg=epsg), no_data_value=nodata
        )
        surface = cls(
            dataset.raster,
            model=model,
            nugget=nugget,
            sill=sill,
            range_=range_,
            n_neighbors=n_neighbors,
        )
        surface.persist_metadata()  # tag the raster so a written GeoTIFF is self-describing
        return surface

    def _band(self, index: int) -> Dataset:
        """Return band `index` as a standalone single-band `Dataset`, preserving the surface's own nodata."""
        array = np.asarray(self.read_array())
        band = array[index] if array.ndim == 3 else array
        nodata = self.no_data_value[index] if self.no_data_value else -9999.0
        # `epsg` alone is `None` for two different rasters: one with no CRS, and one whose CRS
        # simply carries no EPSG authority (geostationary, rotated pole, spherical-earth GRIB).
        # `crs_spec` tells them apart, handing back the WKT for the second so the band keeps the
        # projection instead of being written out unlocatable.
        dataset = Dataset.from_array(
            band,
            geo_ref=GeoReference(
                geo=self.geotransform, epsg=crs_spec(self.epsg, self.crs)
            ),
            no_data_value=nodata,
        )
        return dataset

    @property
    def estimate(self) -> Dataset:
        """The kriged estimate (band 0) as a single-band `Dataset`, carrying the surface's georeference and nodata."""
        return self._band(0)

    @property
    def variance(self) -> Dataset:
        """The kriging variance (band 1) as a single-band `Dataset` on the same georeference — the reason to prefer kriging over IDW."""
        return self._band(1)

    def persist_metadata(self) -> None:
        """Write the variogram provenance to the raster's metadata tags (`GS_*`)."""
        self.meta_data = {
            "GS_CLASS": type(self).__name__,
            "GS_MODEL": str(self.model),
            "GS_NUGGET": str(self.nugget),
            "GS_SILL": str(self.sill),
            "GS_RANGE": str(self.range_),
            "GS_NEIGHBORS": str(self.n_neighbors),
        }
