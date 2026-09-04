"""`Samples` — a `FeatureCollection` of point observations with geostatistics methods.

Subclasses pyramids' `FeatureCollection` exactly as `FeatureCollection` subclasses geopandas' `GeoDataFrame`, and
adds `variogram()`, `interpolate_to_raster(method="kriging")` / `krige()`, and `cross_validate()`. `column` is a
method argument (never constructor state), so the subclass needs only the one-line `_constructor` override below.
"""

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from loguru import logger
from pyramids.base.crs import crs_equal, crs_spec
from pyramids.feature import FeatureCollection

from .kriging import OrdinaryKriging
from .variogram import Variogram, empirical_variogram

if TYPE_CHECKING:
    from pyramids.dataset import Dataset

#: Distinguishes "the caller passed `n_neighbors`" from "the caller said nothing". The two branches
#: of `interpolate_to_raster` have different natural defaults — 32 for kriging's moving
#: neighbourhood, all-points for pyramids' IDW (`invdist`) — so an unspoken 32 must not be forwarded
#: to IDW and silently turn it into `invdistnn`.
_UNSET: object = object()


class Samples(FeatureCollection):
    """Point observations with geostatistics methods (a `FeatureCollection` subclass)."""

    @property
    def _constructor(self):
        """Keep slices/copies typed as `Samples` (mirrors `FeatureCollection._constructor`)."""
        return Samples

    def _clean(self, op: str, column: str) -> tuple[np.ndarray, np.ndarray]:
        """Validate and extract `(coords, values)`, dropping NaN rows (validation lives in the method, as in pyramids)."""
        self._require_point_geometry(op)
        self._require_column(op, column)
        values = self[column].to_numpy(dtype=float)
        coords = np.column_stack([self.geometry.x.to_numpy(), self.geometry.y.to_numpy()])
        finite = np.isfinite(values)
        dropped = int((~finite).sum())
        if dropped:
            logger.info(f"{op}: dropped {dropped} row(s) with NaN in column {column!r}")
        coords, values = coords[finite], values[finite]
        if len(values) < 3:
            raise ValueError(f"{op}: need at least 3 valid points, got {len(values)}")
        return coords, values

    def _epsg(self) -> int | str | None:
        """The layer's CRS specification — its EPSG code, else its WKT, else `None` when it has no CRS.

        Deliberately does not fall back to 4326. A layer with no CRS has no CRS, and stamping
        WGS 84 on the kriged surface would assert a georeference the observations never carried;
        `crs_spec` also keeps a projection the EPSG register does not name (which `to_epsg()`
        alone reports as `None`) rather than discarding it.
        """
        if self.crs is None:
            return None
        return crs_spec(self.crs.to_epsg(), self.crs.to_wkt())

    def values(self, column: str) -> np.ndarray:
        """The numeric observations of `column` (NaN dropped), shape `(n,)`."""
        _, values = self._clean("values", column)
        return values

    def coords(self, column: str) -> np.ndarray:
        """The point coordinates for the valid observations of `column`, shape `(n, 2)`."""
        coords, _ = self._clean("coords", column)
        return coords

    def variogram(
        self,
        column: str,
        *,
        n_lags: int = 15,
        max_dist: float | None = None,
        estimator: str = "matheron",
    ) -> Variogram:
        """Empirical variogram of `column` — the spatial-structure cloud to fit a model to."""
        coords, values = self._clean("variogram", column)
        lags, semivariance, counts = empirical_variogram(coords, values, n_lags, max_dist, estimator)
        return Variogram(lags, semivariance, counts, estimator)

    def _resolve_variogram(self, column: str, variogram: "Variogram | str") -> Variogram:
        """Accept a fitted `Variogram` as-is, or auto-fit the named model to `column`."""
        if isinstance(variogram, Variogram):
            resolved = variogram
        else:
            resolved = self.variogram(column).fit(model=variogram)
        return resolved

    def _check_template_crs(self, template) -> None:
        """Refuse a `template` whose CRS disagrees with this layer's.

        `predict_grid` kriges the template's cell-centre coordinates against these samples'
        coordinates, which is only meaningful within one CRS. When the two disagree the estimates
        are computed in one system and the raster is labelled with another — an error of
        continental scale, produced silently. A template with no CRS is fine: it contributes only
        geometry, and the layer's own CRS still reaches the surface.
        """
        if template is None:
            return
        layer = self._epsg()
        other = crs_spec(template.epsg, template.crs)
        if layer is not None and other is not None and not crs_equal(layer, other):
            raise ValueError(
                f"interpolate_to_raster: the template's CRS ({other}) differs from the layer's "
                f"({layer}); reproject one to match the other before kriging"
            )

    def interpolate_to_raster(
        self,
        column: str,
        *,
        method: str = "idw",
        variogram: "Variogram | str" = "spherical",
        cell_size: float | None = None,
        bounds: tuple[float, float, float, float] | None = None,
        n_neighbors: int | None | object = _UNSET,
        nodata: float = -9999.0,
        template: "Dataset | None" = None,
        **idw_kwargs,
    ):
        """Interpolate `column` onto a raster. `method="kriging"` is ordinary kriging; other methods delegate to pyramids' IDW.

        The kriged surface takes its CRS from these samples, so a layer with no CRS yields an unreferenced
        surface rather than one stamped WGS 84, and a projection the EPSG register does not name is carried
        through as WKT. `template` is **kriging-only**: it supplies the grid geometry and, when it names
        a CRS of its own, the CRS too, while a template without one contributes only its geometry. A
        template whose CRS disagrees with this layer's is refused rather than kriged, because the two
        coordinate sets would not be comparable. The IDW branch has no template concept, so passing one
        there is an error rather than a silently different grid.

        `n_neighbors` reaches whichever branch runs: kriging's moving neighbourhood (default 32), or
        pyramids' `invdistnn` for IDW. It is forwarded to IDW only when you actually pass it, so
        leaving it alone keeps pyramids' own default of weighting every point (`invdist`). `None`
        means "use all points" to both.

        Raises:
            ValueError: If `template` is given for a non-kriging `method`, or if its CRS differs
                from this layer's.
        """
        if method == "kriging":
            coords, values = self._clean("interpolate_to_raster", column)
            self._check_template_crs(template)
            fitted = self._resolve_variogram(column, variogram)
            neighbors = 32 if n_neighbors is _UNSET else n_neighbors
            engine = OrdinaryKriging(coords, values, fitted, n_neighbors=neighbors)
            result = engine.predict_grid(
                cell_size=cell_size, bounds=bounds, template=template, epsg=self._epsg(), nodata=nodata
            )
        else:
            if template is not None:
                raise ValueError(
                    f"interpolate_to_raster: template is only supported for method='kriging', not {method!r}"
                )
            neighbors = {} if n_neighbors is _UNSET else {"n_neighbors": n_neighbors}
            result = super().interpolate_to_raster(
                column,
                method=method,
                cell_size=cell_size,
                bounds=bounds,
                nodata=nodata,
                **neighbors,
                **idw_kwargs,
            )
        return result

    def krige(
        self,
        column: str,
        variogram: "Variogram | str",
        *,
        cell_size: float | None = None,
        bounds: tuple[float, float, float, float] | None = None,
        n_neighbors: int = 32,
        nodata: float = -9999.0,
        template=None,
    ):
        """Convenience alias for `interpolate_to_raster(column, method="kriging", ...)`.

        The surface inherits this layer's CRS unless `template` names one of its own; samples with no CRS
        give an unreferenced surface.
        """
        return self.interpolate_to_raster(
            column,
            method="kriging",
            variogram=variogram,
            cell_size=cell_size,
            bounds=bounds,
            n_neighbors=n_neighbors,
            nodata=nodata,
            template=template,
        )

    def cross_validate(
        self,
        column: str,
        variogram: "Variogram | str",
        *,
        kind: str = "ordinary",
        n_neighbors: int = 32,
    ) -> pd.DataFrame:
        """Leave-one-out cross-validation. Returns a per-point `DataFrame`; summary stats in `frame.attrs["summary"]`."""
        coords, values = self._clean("cross_validate", column)
        fitted = self._resolve_variogram(column, variogram)
        n = len(values)
        predicted = np.empty(n)
        kriging_var = np.empty(n)
        for i in range(n):
            keep = np.arange(n) != i
            engine = OrdinaryKriging(coords[keep], values[keep], fitted, n_neighbors=n_neighbors)
            predicted[i], kriging_var[i] = engine.predict_point(coords[i])
        error = predicted - values
        std_dev = np.sqrt(np.where(kriging_var > 0, kriging_var, np.nan))
        standardized = error / std_dev
        frame = pd.DataFrame(
            {
                "observed": values,
                "predicted": predicted,
                "error": error,
                "kriging_variance": kriging_var,
                "standardized_error": standardized,
            }
        )
        frame.attrs["summary"] = {
            "ME": float(np.mean(error)),
            "RMSE": float(np.sqrt(np.mean(error**2))),
            "mean_standardized_error": float(np.nanmean(standardized)),
            "RMSE_standardized": float(np.sqrt(np.nanmean(standardized**2))),
            "correlation": float(np.corrcoef(values, predicted)[0, 1]),
        }
        return frame
