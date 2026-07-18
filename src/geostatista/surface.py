"""`KrigedSurface` — a 2-band `Dataset` carrying the kriged estimate + variance and its provenance.

Band 0 is the kriging estimate, band 1 the kriging variance. `Dataset` is a thin `gdal.Dataset` wrapper, so
subclassing it is safe (no pandas `_constructor` machinery). Provenance (the variogram that produced the surface)
is stored as typed attributes and, on `persist_metadata`, as raster tags so a written GeoTIFF is self-describing.
"""

import numpy as np
from pyramids.dataset import Dataset


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
        epsg: int,
        nodata: float = -9999.0,
        model: str | None = None,
        nugget: float | None = None,
        sill: float | None = None,
        range_: float | None = None,
        n_neighbors: int | None = None,
    ) -> "KrigedSurface":
        """Build a `KrigedSurface` from estimate + variance arrays and a geotransform."""
        stacked = np.stack([np.asarray(estimate, dtype=float), np.asarray(variance, dtype=float)])
        dataset = Dataset.create_from_array(stacked, geo=geo, epsg=int(epsg), no_data_value=nodata)
        surface = cls(
            dataset.raster,
            model=model,
            nugget=nugget,
            sill=sill,
            range_=range_,
            n_neighbors=n_neighbors,
        )
        return surface

    def _band(self, index: int) -> Dataset:
        """Return band `index` as a standalone single-band `Dataset`."""
        array = np.asarray(self.read_array())
        band = array[index] if array.ndim == 3 else array
        dataset = Dataset.create_from_array(band, geo=self.geotransform, epsg=self.epsg, no_data_value=-9999.0)
        return dataset

    @property
    def estimate(self) -> Dataset:
        """The kriged estimate (band 0) as a single-band `Dataset`."""
        return self._band(0)

    @property
    def variance(self) -> Dataset:
        """The kriging variance (band 1) as a single-band `Dataset` — the reason to prefer kriging over IDW."""
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
