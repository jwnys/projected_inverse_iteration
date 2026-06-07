r"""PII linear solvers — the PII analogue of :mod:`netket.optimizer.solver`.

- :func:`pii_default_solver` — direct general-LU (the non-symmetric-``Q`` analogue of NetKet's
  ``cholesky_with_fallback`` default);
- :func:`penrose_symmetrized_solver`, :func:`naive_symmetrized_solver` — symmetrized variants;
- :func:`gmres`, :func:`bicgstab` — matrix-free iterative solvers (``cg`` is intentionally absent:
  ``Q`` is not Hermitian PSD).
"""

from nkpii.optimizer.solver.solvers import (
    pii_default_solver as pii_default_solver,
    penrose_symmetrized_solver as penrose_symmetrized_solver,
    naive_symmetrized_solver as naive_symmetrized_solver,
    gmres as gmres,
    bicgstab as bicgstab,
)
