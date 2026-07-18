# Tools

Legacy spatial-interpolation helpers (inverse-distance weighting). These are slated
for removal — use pyramids' `FeatureCollection.interpolate_to_raster(method="idw")`
for IDW, and [`Samples.krige`](samples.md) / the [kriging engine](kriging.md) for the
kriging surface with its variance band. See
[ADR 0001](../adr/0001-pyramids-geostatista-boundary.md) for the boundary rationale.

::: geostatista.tools.Tools
