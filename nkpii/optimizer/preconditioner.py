r"""PII gradient preconditioner — the PII analogue of :class:`netket.optimizer.SR`.

``PII`` is a **dedicated** preconditioner that subclasses the same shared base as ``SR``,
:class:`netket.optimizer.preconditioner.AbstractLinearPreconditioner` (it is *not* a subclass of
``SR``).  It reuses the base ``__init__``/``__call__``/solve machinery and only specializes the two
genuinely PII-specific pieces:

- :meth:`lhs_constructor` builds a ``Q``-matrix (:mod:`nkpii.optimizer.pct`) instead of a QGT — which
  additionally needs the **Hamiltonian** (for the ``A`` Jacobian) and the inverse-iteration shift
  **τ**;
- :meth:`__call__` halves the gradient, because PII solves ``Q ξ = ½∇E`` while ``SR`` (and the
  inherited machinery) solve ``lhs ξ = ∇E``.

It deliberately omits ``SR``'s ``diag_scale`` (PII has no diagonal-scale concept — this is the one
reason it is a sibling of ``SR`` rather than a subclass), and its default solver is a general
(non-symmetric) LU rather than ``cholesky``, since ``Q = OᴴA − τOᴴO + λI`` is not Hermitian PSD.

Use it as the ``preconditioner`` of the standard :class:`nkpii.driver.VMC` driver::

    gs = nkpii.driver.VMC(
        H, optax.sgd(1.0), variational_state=vs,
        preconditioner=nkpii.optimizer.PII(H, tau=1.2 * E0, diag_shift=1e-4),
    )
"""

from collections.abc import Callable

import jax

from netket.utils import struct
from netket.utils.types import ScalarOrSchedule
from netket.operator import AbstractOperator
from netket.optimizer.preconditioner import AbstractLinearPreconditioner

from nkpii.optimizer.pct import PCTJacobianDense
from nkpii.optimizer.solver import pii_default_solver


class PII(AbstractLinearPreconditioner, mutable=True):
    r"""Projected Inverse Iteration preconditioner (PII analogue of :class:`netket.optimizer.SR`).

    Preconditions the energy gradient ``∇E`` so the preconditioned gradient ``ξ`` solves the PII
    system ``(OᴴA − τ OᴴO + diag_shift·I) ξ = ½∇E``.  Pass it as the ``preconditioner`` of
    :class:`nkpii.driver.VMC`.
    """

    hamiltonian: AbstractOperator = struct.field(
        pytree_node=False, serialize=False, default=None
    )
    """The Hamiltonian (defines the ``A`` Jacobian of the PII ``Q``-matrix)."""

    tau: ScalarOrSchedule = struct.field(serialize=False, default=None)
    """The inverse-iteration shift ``τ`` (scalar or schedule); ``≈`` the ground-state energy."""

    diag_shift: ScalarOrSchedule = struct.field(serialize=False, default=0.0)
    """The Tikhonov shift added to the diagonal of ``Q`` (scalar or schedule)."""

    q_constructor: Callable = struct.static_field(default=None)
    """The ``Q``-matrix constructor (``PCTJacobianDense`` / ``PCTJacobianPyTree``) — the PII analogue of
    ``SR``'s ``qgt`` argument."""

    q_kwargs: dict = struct.field(serialize=False, default=None)
    """Extra keyword arguments forwarded to the ``Q``-matrix constructor."""

    def __init__(
        self,
        hamiltonian: AbstractOperator,
        q: Callable | None = None,
        solver: Callable = pii_default_solver,
        *,
        tau: ScalarOrSchedule,
        diag_shift: ScalarOrSchedule = 0.0,
        solver_restart: bool = False,
        **kwargs,
    ):
        r"""Construct the PII preconditioner.

        Args:
            hamiltonian: the Hamiltonian (needed to build the ``A`` Jacobian).
            q: the ``Q``-matrix constructor — :class:`nkpii.optimizer.pct.PCTJacobianDense` (default) or
                :class:`~nkpii.optimizer.pct.PCTJacobianPyTree`.  Plays the role of ``SR``'s ``qgt`` arg.
            solver: an operator-aware ``(A, b) -> (x, info)`` solver.  Defaults to
                :func:`nkpii.optimizer.solver.pii_default_solver` (direct general LU, since ``Q`` is
                non-symmetric).  Use :func:`~nkpii.optimizer.solver.gmres` /
                :func:`~nkpii.optimizer.solver.bicgstab` for a matrix-free solve.
            tau: the inverse-iteration shift ``τ`` (``≈ E0``).
            diag_shift: Tikhonov shift on the diagonal of ``Q``.
            solver_restart: warm-start the solver from the previous solution.
            **kwargs: forwarded to the ``Q`` constructor (e.g. ``mode='complex'``,
                ``chunk_size=...``).
        """
        if q is None:
            q = PCTJacobianDense
        self.hamiltonian = hamiltonian
        self.tau = tau
        self.diag_shift = diag_shift
        self.q_constructor = q
        self.q_kwargs = kwargs
        super().__init__(solver, solver_restart=solver_restart)

    def lhs_constructor(self, vstate, step=None):
        """Build the PII ``Q``-matrix operator (the lhs of ``Q ξ = ½∇E``).

        Unlike ``SR.lhs_constructor`` (which builds a QGT from ``vstate`` alone), this passes the
        Hamiltonian and ``τ`` to the ``Q`` constructor; it has no ``diag_scale``.
        """
        tau = self.tau
        if callable(tau):
            if step is None:
                raise TypeError(
                    "If you use a scheduled `tau`, you must call the preconditioner with a `step`."
                )
            tau = tau(step)
        diag_shift = self.diag_shift
        if callable(diag_shift):
            if step is None:
                raise TypeError(
                    "If you use a scheduled `diag_shift`, you must call the preconditioner "
                    "with a `step`."
                )
            diag_shift = diag_shift(step)
        if tau is None:
            raise TypeError("The PII preconditioner requires `tau` (the inverse-iteration shift).")
        return self.q_constructor(
            vstate, self.hamiltonian, tau=tau, diag_shift=diag_shift, **self.q_kwargs
        )

    def __call__(self, vstate, gradient, step=None):
        # PII solves Q ξ = ½∇E (vs SR's S ξ = ∇E): halve the gradient, then inherit the solve.
        half_gradient = jax.tree_util.tree_map(lambda g: 0.5 * g, gradient)
        return super().__call__(vstate, half_gradient, step)

    def __repr__(self):
        return (
            f"PII("
            f"\n  tau             = {self.tau},"
            f"\n  diag_shift      = {self.diag_shift},"
            f"\n  q_constructor   = {self.q_constructor},"
            f"\n  solver          = {self.solver},"
            f"\n  solver_restart  = {self.solver_restart},"
            f"\n)"
        )
