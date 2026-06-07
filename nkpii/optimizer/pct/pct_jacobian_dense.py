r"""Dense Projected Characteristic Tensor (PCT) operator for PII (analogue of ``QGTJacobianDense``).

Where NetKet's QGT stores the single Jacobian ``O`` and represents the metric
``S = OᴴO``, the PCT operator stores **two** Jacobians — ``O`` (of ``logψ``) and
``A`` (of ``f_A``) — and represents the (generally non-Hermitian) PII matrix

.. math::
    Q = Oᴴ A - τ\, Oᴴ O \;(+\; \mathrm{diag\_shift}\, I).

As in NetKet, the operator is **semi-lazy**: only ``O`` and ``A`` are stored; the dense
``Q`` is materialized **only** by :meth:`~PCTJacobianDenseT.to_dense`.  Matrix–vector
products ``Q @ v`` and ``Qᴴ @ v`` are computed directly from ``O`` and ``A`` without ever
forming ``Q`` (enabling iterative solvers — GMRES on ``Q``, or CGNE/LSQR via ``Q`` and
``Qᴴ``).  These are **PCT operators**, not QGTs.

Built on NetKet: subclasses :class:`netket.optimizer.LinearOperator` and reuses its
``solve``/``__add__``/``__call__``, plus ``convert_tree_to_dense_format`` and
``check_valid_vector_type``.  Supports ``'real'``/``'complex'``/``'holomorphic'`` modes; ``Q`` is
real in the realified ``'real'``/``'complex'`` modes and complex in ``'holomorphic'``.
"""

import jax
from jax import numpy as jnp
from flax import struct

from netket.utils.types import Scalar, PyTree
from netket.optimizer.linear_operator import LinearOperator, SolverT, Uninitialized
from netket.optimizer.qgt.qgt_jacobian_dense import convert_tree_to_dense_format
from netket.optimizer.qgt.common import check_valid_vector_type


@struct.dataclass
class PCTJacobianDenseT(LinearOperator):
    """Semi-lazy dense PII ``Q``-matrix as a :class:`LinearOperator`.

    Stores the two centered/√-rescaled Jacobians ``O`` and ``A``; ``Q`` itself is only
    assembled by :meth:`to_dense`.
    """

    O: jnp.ndarray = Uninitialized  # type: ignore
    """Centered, √-rescaled Jacobian ``O_ij = ∂logψ(σ_i)/∂p_j`` (same layout as NetKet's
    QGTJacobianDense: ``[n_samples, P]`` real, ``[n_samples, 2, P]`` complex)."""

    A: jnp.ndarray = Uninitialized  # type: ignore
    """Centered, √-rescaled Jacobian of ``f_A = E_L + sg(E_L) logψ`` — same shape as ``O``."""

    tau: Scalar = Uninitialized  # type: ignore
    """The inverse-iteration shift ``τ`` (≈ ground-state energy)."""

    scale: jnp.ndarray | None = None
    """Optional per-column rescaling (unused by default; kept for API parity)."""

    mode: str = struct.field(pytree_node=False, default=Uninitialized)
    """Differentiation mode: ``'real'``, ``'complex'`` or ``'holomorphic'``."""

    _in_solve: bool = struct.field(pytree_node=False, default=False)
    """Internal: inside ``_solve`` the input vector is already in dense format."""

    _conj_transpose: bool = struct.field(pytree_node=False, default=False)
    """Internal: if ``True`` this operator applies ``Qᴴ`` instead of ``Q`` (see ``.H``)."""

    _params_structure: PyTree = struct.field(pytree_node=False, default=Uninitialized)

    @property
    def H(self) -> "PCTJacobianDenseT":
        """The conjugate-transpose operator ``Qᴴ`` (for CGNE/LSQR-type iterative solvers)."""
        return self.replace(_conj_transpose=not self._conj_transpose)

    @jax.jit
    def __matmul__(self, vec: PyTree | jnp.ndarray) -> PyTree | jnp.ndarray:
        if not hasattr(vec, "ndim") and not self._in_solve:
            check_valid_vector_type(self._params_structure, vec)

        vec, reassemble = convert_tree_to_dense_format(
            vec, self.mode, disable=self._in_solve
        )
        if self.scale is not None:
            vec = vec * self.scale

        result = _q_mat_vec(
            vec, self.O, self.A, self.tau, self.diag_shift, self._conj_transpose
        )

        if self.scale is not None:
            result = result * self.scale
        return reassemble(result)

    @jax.jit
    def _solve(
        self, solve_fun: SolverT, y: PyTree, *, x0: PyTree | None = None
    ) -> PyTree:
        if not hasattr(y, "ndim"):
            check_valid_vector_type(self._params_structure, y)

        y, reassemble = convert_tree_to_dense_format(y, self.mode)
        if x0 is not None:
            x0, _ = convert_tree_to_dense_format(x0, self.mode)
            if self.scale is not None:
                x0 = x0 * self.scale
        if self.scale is not None:
            y = y / self.scale

        # pass the operator down (matvec/to_dense) but with scale removed.
        unscaled_self = self.replace(scale=None, _in_solve=True)
        out, info = solve_fun(unscaled_self, y, x0=x0)

        if self.scale is not None:
            out = out / self.scale
        return reassemble(out), info

    @jax.jit
    def to_dense(self) -> jnp.ndarray:
        """Materialize the dense ``Q = OᴴA − τ OᴴO + diag_shift·I`` (the only place ``Q`` is formed)."""
        if self.scale is None:
            O, A = self.O, self.A
            diag = jnp.eye(self.O.shape[-1], dtype=self.O.dtype)
        else:
            O = self.O * self.scale[jnp.newaxis, :]
            A = self.A * self.scale[jnp.newaxis, :]
            diag = jnp.diag(self.scale**2)

        O = O.reshape(-1, O.shape[-1])  # [n_samples(·2), P]
        A = A.reshape(-1, A.shape[-1])
        Q = O.conj().T @ A - self.tau * (O.conj().T @ O) + self.diag_shift * diag
        if self._conj_transpose:
            Q = Q.conj().T
        return Q

    def __repr__(self):
        return (
            f"PCTJacobianDense(tau={self.tau}, diag_shift={self.diag_shift}, "
            f"mode={self.mode}{', Hᵀ' if self._conj_transpose else ''})"
        )


def _q_mat_vec(v, O, A, tau, diag_shift, conj_transpose):
    """``Q @ v`` (or ``Qᴴ @ v``) from the stored Jacobians, never forming ``Q``.

    ``Q v  = Oᴴ((A − τO) v) + diag_shift·v`` ;
    ``Qᴴ v = (A − τO)ᴴ(O v) + diag_shift·v``.
    Mirrors NetKet's ``mat_vec`` (``w = O @ v`` then ``Oᴴ @ w`` via ``tensordot``).
    """
    if not conj_transpose:
        Bv = (A @ v) - tau * (O @ v)  # (A − τO) v   -> [n_samples(·2)]
        res = jnp.tensordot(Bv.conj(), O, axes=Bv.ndim).conj()  # Oᴴ (Bv)
    else:
        w = (O @ v).conj()  # (O v)*
        res = jnp.tensordot(w, A, axes=w.ndim).conj() - tau * jnp.tensordot(
            w, O, axes=w.ndim
        ).conj()  # (A − τO)ᴴ (O v)
    return res + diag_shift * v
