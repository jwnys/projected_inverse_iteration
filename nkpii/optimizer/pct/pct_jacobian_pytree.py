r"""PyTree Projected Characteristic Tensor (PCT) operator for PII (analogue of ``QGTJacobianPyTree``).

Like :class:`nkpii.optimizer.pct.PCTJacobianDenseT` but the two Jacobians ``O`` and ``A`` are
kept as **PyTrees** (one leaf per parameter group, shape ``[n_samples, ...]`` /
``[n_samples, 2, ...]``) instead of one flat dense array — cheaper when materializing the
dense Jacobian is expensive.  ``Q @ v`` / ``Qᴴ @ v`` use NetKet's tree ``_jvp``/``_vjp``;
``Q`` is assembled only by :meth:`~PCTJacobianPyTreeT.to_dense`.
"""

import jax
from jax import numpy as jnp
from jax.flatten_util import ravel_pytree
from flax import struct

from netket.utils.types import Array, PyTree, Scalar
from netket import jax as nkjax
from netket.optimizer.linear_operator import LinearOperator, SolverT, Uninitialized
from netket.optimizer.qgt.common import check_valid_vector_type
from netket.optimizer.qgt.qgt_jacobian_pytree import _jvp, _vjp


@struct.dataclass
class PCTJacobianPyTreeT(LinearOperator):
    """Semi-lazy PyTree PII ``Q``-matrix as a :class:`LinearOperator`."""

    O: PyTree = Uninitialized
    """Centered, √-rescaled Jacobian of ``logψ`` as a PyTree."""

    A: PyTree = Uninitialized
    """Centered, √-rescaled Jacobian of ``f_A`` as a PyTree (same structure as ``O``)."""

    tau: Scalar = Uninitialized  # type: ignore
    """The inverse-iteration shift ``τ``."""

    scale: PyTree | None = None
    """Optional per-parameter rescaling (unused by default; kept for API parity)."""

    mode: str = struct.field(pytree_node=False, default=Uninitialized)
    """Differentiation mode: ``'real'``, ``'complex'`` or ``'holomorphic'``."""

    _params_structure: PyTree = struct.field(pytree_node=False, default=Uninitialized)
    _in_solve: bool = struct.field(pytree_node=False, default=False)
    _conj_transpose: bool = struct.field(pytree_node=False, default=False)

    @property
    def H(self) -> "PCTJacobianPyTreeT":
        """The conjugate-transpose operator ``Qᴴ``."""
        return self.replace(_conj_transpose=not self._conj_transpose)

    @jax.jit
    def __matmul__(self, vec: PyTree | Array) -> PyTree | Array:
        if hasattr(vec, "ndim"):
            _, unravel = ravel_pytree(self._params_structure)
            vec = unravel(vec)
            ravel = True
        else:
            ravel = False

        check_valid_vector_type(self._params_structure, vec)

        reassemble = None
        if self.mode != "holomorphic" and not self._in_solve:
            vec, reassemble = nkjax.tree_to_real(vec)

        if self.scale is not None:
            vec = jax.tree_util.tree_map(jnp.multiply, vec, self.scale)

        result = _q_mat_vec_tree(
            vec, self.O, self.A, self.tau, self.diag_shift, self._conj_transpose
        )

        if self.scale is not None:
            result = jax.tree_util.tree_map(jnp.multiply, result, self.scale)
        if reassemble is not None:
            result = reassemble(result)
        if ravel:
            result, _ = ravel_pytree(result)
        return result

    @jax.jit
    def _solve(
        self, solve_fun: SolverT, y: PyTree, *, x0: PyTree | None = None
    ) -> PyTree:
        check_valid_vector_type(self._params_structure, y)

        if self.mode != "holomorphic":
            y, reassemble = nkjax.tree_to_real(y)
            if x0 is not None:
                x0, _ = nkjax.tree_to_real(x0)

        if self.scale is not None:
            y = jax.tree_util.tree_map(jnp.divide, y, self.scale)
            if x0 is not None:
                x0 = jax.tree_util.tree_map(jnp.multiply, x0, self.scale)

        unscaled_self = self.replace(scale=None, _in_solve=True)
        out, info = solve_fun(unscaled_self, y, x0=x0)

        if self.scale is not None:
            out = jax.tree_util.tree_map(jnp.divide, out, self.scale)
        if self.mode != "holomorphic":
            out = reassemble(out)
        return out, info

    @jax.jit
    def to_dense(self) -> jnp.ndarray:
        """Materialize the dense ``Q = OᴴA − τ OᴴO + diag_shift·I``."""

        def _flatten(tree):
            t = tree
            if self.mode == "complex":
                t = jax.tree_util.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), t)
            return jax.vmap(lambda leaf: ravel_pytree(leaf)[0])(t)

        O = _flatten(self.O)  # [n_samples(·2), P]
        A = _flatten(self.A)
        if self.scale is None:
            diag = jnp.eye(O.shape[-1], dtype=O.dtype)
        else:
            scale, _ = ravel_pytree(self.scale)
            O = O * scale[jnp.newaxis, :]
            A = A * scale[jnp.newaxis, :]
            diag = jnp.diag(scale**2)
        Q = O.conj().T @ A - self.tau * (O.conj().T @ O) + self.diag_shift * diag
        if self._conj_transpose:
            Q = Q.conj().T
        return Q

    def __repr__(self):
        return (
            f"PCTJacobianPyTree(tau={self.tau}, diag_shift={self.diag_shift}, "
            f"mode={self.mode}{', Hᵀ' if self._conj_transpose else ''})"
        )


def _q_mat_vec_tree(v, O, A, tau, diag_shift, conj_transpose):
    """``Q @ v`` / ``Qᴴ @ v`` for PyTree Jacobians, via NetKet's ``_jvp``/``_vjp``.

    ``Q v  = Oᴴ((A − τO) v)`` ;  ``Qᴴ v = (A − τO)ᴴ(O v)`` ; ``+ diag_shift·v``.
    """
    if not conj_transpose:
        Bv = _jvp(A, v) - tau * _jvp(O, v)  # (A − τO) v  -> array
        res = nkjax.tree_conj(_vjp(O, Bv.conjugate()))  # Oᴴ (Bv)
    else:
        wc = _jvp(O, v).conjugate()  # (O v)*
        rA = _vjp(A, wc)
        rO = _vjp(O, wc)
        res = nkjax.tree_conj(
            jax.tree_util.tree_map(lambda a, o: a - tau * o, rA, rO)
        )  # (A − τO)ᴴ (O v)
    res = nkjax.tree_cast(res, v)
    return nkjax.tree_axpy(diag_shift, v, res)
