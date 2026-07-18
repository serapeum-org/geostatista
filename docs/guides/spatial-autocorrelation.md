# Spatial autocorrelation

The flow is **build a weights matrix → measure global autocorrelation → map local clusters / hotspots.** All the
statistics are free functions that take a `FeatureCollection`, a column, and a `Weights`.

## 1. Build a spatial-weights matrix

```python
from geostatista import Weights

w = Weights.queen(tracts)                        # or .rook(tracts)
w = Weights.knn(tracts, k=8)                      # k-nearest by centroid
w = Weights.distance_band(tracts, 5000)           # neighbors within 5 km
```

`Weights` exposes `.cardinalities`, `.neighbors`, `.islands`, and `.transform("r")` for row-standardization.
Islands (zero-neighbor features) are detected and keep a zero row under row-standardization.

## 2. Global autocorrelation — is the pattern clustered?

```python
from geostatista import morans_i, gearys_c

mi = morans_i(tracts, "income", w)                # -> MoranResult (I, EI, z, p_norm, p_sim)
mi.summary()
gc = gearys_c(tracts, "income", w)                # -> GearyResult (C, EC=1, z, p_sim)
```

Moran's I above its expectation `EI = -1/(n-1)` with a small `p_sim` indicates positive spatial autocorrelation.

## 3. Local clusters — LISA

Local Moran returns the input features annotated with per-feature cluster membership.

```python
from geostatista import local_morans

lisa = local_morans(tracts, "income", w)          # + local_i / z_sim / p_sim / quadrant / cluster
lisa[["cluster"]].value_counts()                  # HH / LL / HL / LH / ns
```

## 4. Hotspots — Getis-Ord Gi*

```python
from geostatista import getis_ord_gi

hot = getis_ord_gi(tracts, "income", w, star=True)  # + gi / z / p / hotspot (hot/cold/ns)
```

## One-call facade

The exact surface from the roadmap (pyramids #576 Item 5):

```python
from geostatista import spatial_autocorrelation, hotspots

spatial_autocorrelation(tracts, "income", weights="queen")   # {"I", "EI", "z", "p", "n", "weights"}
hotspots(tracts, "income", weights="queen")                   # annotated FeatureCollection (Gi*)
```
