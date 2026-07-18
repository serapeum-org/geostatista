"""Private numerics for kriging — system assembly, moving neighborhood, variogram fitting.

Kept separate from the public engines so the linear algebra is unit-testable in isolation. This subpackage must
never depend on `weights.py` (the Phase-3 spatial-weights matrix is a different abstraction — see the design docs).
"""
