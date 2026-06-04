r"""PII gradient preconditioner — the PII analogue of :class:`netket.optimizer.SR`.

``PII`` **inherits from** :class:`netket.optimizer.SR` to reuse as much as possible: SR's
``__init__`` (constructor defaulting + ``qgt_constructor``/``qgt_kwargs`` plumbing + the
diagonal-shift fields) and the whole ``AbstractLinearPreconditioner`` solve machinery.

Only the two genuinely PII-specific bits are overridden:

- :meth:`lhs_constructor` builds a ``Q``-matrix (:mod:`pii.optimizer.q`) instead of a QGT,
  which additionally needs the **Hamiltonian** (for the ``A`` Jacobian) and the
  inverse-iteration shift **τ**;
- :meth:`__call__` halves the gradient, because PII solves ``Q ξ = ½∇E`` while SR (and the
  inherited machinery) solve ``lhs ξ = ∇E``.

So ``PII`` *is* "SR with explicit ``hamiltonian``/``tau`` and the ½".  Use it as the
``preconditioner`` of the standard :class:`pii.VMC` driver::

    pii.VMC(H, optax.sgd(1.0), variational_state=vs,
            preconditioner=pii.optimizer.PII(H, tau=1.2*E0, diag_shift=1e-4))
"""

from collections.abc import Callable

import jax
import jax.scipy as jsp
from jax.flatten_util import ravel_pytree

from netket.utils.types import ScalarOrSchedule
from netket.utils import struct
from netket.operator import AbstractOperator
from netket.optimizer import SR

from pii.optimizer.q import QJacobianDense


def q_default_solver(A, b, **kwargs):
    """Operator-aware general (non-symmetric) solve for a PII ``Q``-matrix operator.

    ``A`` is a ``QJacobian*`` operator and ``b`` the rhs (a flat array for the dense
    operator, a PyTree for the PyTree operator).  Solves the dense ``Q`` with a general
    LU (``Q`` is non-Hermitian) — the operator analogue of
    :func:`pii.driver._pii_default_solver`.
    """
    Qd = A.to_dense()
    if hasattr(b, "ndim"):
        return jsp.linalg.solve(Qd, b, assume_a="gen"), None
    b_flat, unravel = ravel_pytree(b)
    return unravel(jsp.linalg.solve(Qd, b_flat, assume_a="gen")), None


class PII(SR, mutable=True):
    r"""Projected Inverse Iteration preconditioner (PII analogue of :class:`netket.optimizer.SR`).

    Preconditions the energy gradient ``∇E`` so that the preconditioned gradient ``ξ`` solves
    the PII system ``(OᴴA − τ OᴴO + diag_shift·I) ξ = ½∇E``.  Inherits from ``SR``; reuses its
    ``__init__``, ``qgt_constructor``/``qgt_kwargs`` and diagonal-shift handling.
    """

    hamiltonian: AbstractOperator = struct.field(
        pytree_node=False, serialize=False, default=None
    )
    """The Hamiltonian (defines the ``A`` Jacobian of the PII matrix)."""

    tau: ScalarOrSchedule = struct.field(serialize=False, default=None)
    """The inverse-iteration shift ``τ`` (scalar or schedule)."""

    def __init__(
        self,
        hamiltonian: AbstractOperator,
        q: Callable | None = None,
        solver: Callable = q_default_solver,
        *,
        tau: ScalarOrSchedule,
        diag_shift: ScalarOrSchedule = 0.0,
        solver_restart: bool = False,
        **kwargs,
    ):
        """Construct the PII preconditioner.

        Args:
            hamiltonian: the Hamiltonian (needed to build the ``A`` Jacobian).
            q: the ``Q``-matrix constructor — :class:`pii.optimizer.q.QJacobianDense`
                (default) or :class:`~pii.optimizer.q.QJacobianPyTree`.  Plays the role of
                SR's ``qgt`` argument.
            solver: operator-aware ``(A, b) -> (x, info)`` solver.  Defaults to a general
                (non-symmetric) dense solve (``Q`` is non-Hermitian; SR's ``cg`` default
                would be wrong here).
            tau: the inverse-iteration shift ``τ`` (≈ ground-state energy).
            diag_shift: Tikhonov shift on the diagonal of ``Q``.
            solver_restart: warm-start the solver from the previous solution.
            **kwargs: forwarded to the ``Q`` constructor (e.g. ``mode='complex'``).
        """
        if q is None:
            q = QJacobianDense
        # reuse SR.__init__ for qgt_constructor=q, qgt_kwargs=kwargs, diag_shift, solver, ...
        super().__init__(
            q, solver, diag_shift=diag_shift, solver_restart=solver_restart, **kwargs
        )
        self.hamiltonian = hamiltonian
        self.tau = tau

    def lhs_constructor(self, vstate, step=None):
        """Build the PII ``Q``-matrix operator (the lhs of ``Q ξ = ½∇E``).

        Overrides ``SR.lhs_constructor``: passes the Hamiltonian and ``τ`` to the
        ``Q`` constructor (and omits SR's ``diag_scale``, which ``QJacobian`` doesn't use).
        """
        tau = self.tau(step) if callable(self.tau) else self.tau
        diag_shift = self.diag_shift(step) if callable(self.diag_shift) else self.diag_shift
        if tau is None:
            raise TypeError("The PII preconditioner requires `tau`.")
        return self.qgt_constructor(
            vstate, self.hamiltonian, tau=tau, diag_shift=diag_shift, **self.qgt_kwargs
        )

    def __call__(self, vstate, gradient, step=None, *args, **kwargs):
        # PII solves Q ξ = ½∇E (vs SR's S ξ = ∇E): halve the gradient, then inherit the solve.
        half_gradient = jax.tree_util.tree_map(lambda g: 0.5 * g, gradient)
        return super().__call__(vstate, half_gradient, step, *args, **kwargs)

    def __repr__(self):
        return (
            f"PII(tau={self.tau}, diag_shift={self.diag_shift}, "
            f"q_constructor={self.qgt_constructor}, solver={self.solver}, "
            f"solver_restart={self.solver_restart})"
        )
