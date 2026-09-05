# Change log

## 0.3.0 (2026-09-05)

### BREAKING CHANGE

- a surface built from input with no CRS is now unreferenced instead of stamped
EPSG:4326, so code that writes or reprojects it will fail where it used to succeed on a false
georeference — pass `epsg=`, or set a CRS on the layer. A template whose CRS disagrees with the
layer, and a template passed to `method="idw"`, now raise `ValueError` instead of silently
producing a wrong or mis-shaped raster. Minimum `pyramids-gis` 0.59.0, minimum `cleopatra` 0.31.0.

### Fix

- **crs**: require pyramids-gis >=0.59.0 and stop inventing EPSG:4326 (#58)

## 0.2.0 (2026-07-18)

### Feat

- add geostatistics core (variograms, kriging, autocorrelation) (#50)

### Fix

- **ci**: remove template distributions test stub that fails the pure-wheel job (#51)

## 0.1.6 (2023-01-31)

- bump up versions

## 0.1.5 (2022-12-27)

- fix bug in pypi package names in the requirements.txt file

## 0.1.4 (2022-12-26)

- use environment.yml and conda instead of pyproject.toml and poetry

## 0.1.0 (2022-05-24)

- First release on PyPI.
