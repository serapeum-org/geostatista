# ADR 0001 — The pyramids ↔ geostatista boundary

- **Status:** accepted
- **Date:** 2026-07-18

## Context

Both pyramids and geostatista can turn scattered point observations into a continuous surface. It must be
unambiguous which package owns which operation, so capabilities are not duplicated or split across repos.

## Decision

**The test:** an operation belongs in **pyramids** when its output stays a geospatial object and spatial structure
is the mechanism; it belongs in **geostatista** when the output is a fitted model, a coefficient, or a p-value.

- **pyramids keeps** IDW and the free `gdal.Grid` algorithms via
  `FeatureCollection.interpolate_to_raster(method="idw", power=...)`.
- **geostatista owns** every variogram and every kriging variant, and the spatial-`Weights` subsystem with Moran's I
  / Geary's C / Local Moran / Getis-Ord Gi*.

Two facts force geostatistics out of pyramids and into its own package:

1. **GDAL cannot krige.** Verified against GDAL 3.13.1: `gdal.Grid` accepts `invdist`, `invdistnn`, `average`,
   `nearest`, `linear`, `minimum` — and **rejects** `kriging`. Kriging is numerical code someone must own, not a
   delegation.
2. **The `Weights` subsystem is a second domain.** Moran's I / Getis-Ord Gi* need a spatial-weights matrix
   (contiguity / knn / distance-band, row-standardization). That does not belong inside a GIS I/O package.

## Consequences

- pyramids' `interpolate_to_raster(method="kriging")` raises and names geostatista (it does not depend on
  geostatista — the dependency arrow points one way, geostatista → pyramids). The stale `pykrige` claim is removed
  (pyramids #770).
- geostatista adds kriging through `Samples.interpolate_to_raster(column, method="kriging", variogram=...)`, mirroring
  the pyramids signature; IDW on a `Samples` still delegates to pyramids.
- Kriging is implemented in exactly one place. A convenience `method="ordinary_kriging"` in pyramids would be
  superseded immediately and is not added.
