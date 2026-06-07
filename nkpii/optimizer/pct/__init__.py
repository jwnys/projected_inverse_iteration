r"""Projected Characteristic Tensor (PCT) operators — the PII analogue of ``netket.optimizer.qgt``.

These are **PCT operators** (not QGTs): semi-lazy :class:`netket.optimizer.LinearOperator`
objects representing the PII matrix ``Q = OᴴA − τ OᴴO (+ diag_shift·I)``.  Like NetKet's
``QGTJacobian{Dense,PyTree}`` they store only the Jacobians (here **two**: ``O`` of ``logψ``
and ``A`` of ``f_A``) and materialize ``Q`` only via ``.to_dense()``; ``Q @ v`` / ``Q.H @ v``
are lazy matrix-vector products that enable iterative solvers.

Usage::

    from nkpii.optimizer.pct import PCTJacobianDense
    Q = PCTJacobianDense(vstate, hamiltonian, tau=1.2 * E0, diag_shift=1e-4)
    Q @ v            # lazy mat-vec
    Q.H @ v          # lazy conjugate-transpose mat-vec (for CGNE/LSQR)
    Q.to_dense()     # assemble the dense Q
    dp = Q.solve(solver, rhs)   # rhs = ½∇E ; solver is operator-aware (NetKet contract)
"""

from .pct_jacobian import (
    PCTJacobianDense,
    PCTJacobianPyTree,
    PCTJacobian_DefaultConstructor,
)
from .pct_jacobian_dense import PCTJacobianDenseT
from .pct_jacobian_pytree import PCTJacobianPyTreeT

__all__ = [
    "PCTJacobianDense",
    "PCTJacobianPyTree",
    "PCTJacobianDenseT",
    "PCTJacobianPyTreeT",
    "PCTJacobian_DefaultConstructor",
]
