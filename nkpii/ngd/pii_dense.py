r"""Dense PII update (``P×P`` preconditioner ``Q = H - \tau S + \lambda I``).

Models :func:`netket._src.ngd.sr._compute_sr_update`.  Given the centered/scaled
log-:math:`\psi` Jacobian ``O_L`` and the ``A`` Jacobian ``A_L`` (both ``[2M, P]``
in complex mode, ``[M, P]`` in real mode), it forms

.. math::
    S = O_L^T O_L,\qquad H = O_L^T A_L,\qquad \nabla E = O_L^T\,dv,

and solves :math:`(H - \tau S + \lambda I)\,\xi = \tfrac{1}{2}\nabla E`.

The factor 1/2 on the right-hand side is essential: NetKet's convention gives
``O_Lᵀ dv = ∇E`` (the full gradient, the factor 2 in ``dv`` already turned the
covariance into the gradient), while PII solves ``Q ξ = ½∇E``. Note ``Q`` is only
asymptotically Hermitian, so a general (non-Cholesky) solver must be used.
"""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp

from netket import jax as nkjax
from netket.utils.types import Array


@partial(jax.jit, static_argnames=("solver_fn", "mode"))
def _compute_pii_update_dense(
    O_L,
    A_L,
    dv,
    *,
    tau: float | Array,
    diag_shift: float | Array,
    solver_fn: Callable[[Array, Array], Array],
    mode: str,
    proj_reg: float | Array | None = None,
    momentum: float | Array | None = None,
    old_updates: Array | None = None,
    params_structure,
):
    if proj_reg is not None:
        raise ValueError("proj_reg is not implemented for the dense PII update.")

    S = O_L.T @ O_L
    H = O_L.T @ A_L
    grad = O_L.T @ dv  # = ∇E (factor 2 already in dv)

    # rhs = ½∇E (+ SPRING anchor). The dense SPRING update simplifies to
    # ξ = (H - τS + λI)⁻¹ (½∇E + λμ ξ_{k-1}).
    rhs = 0.5 * grad
    if momentum is not None:
        rhs = rhs + diag_shift * momentum * old_updates

    side = S.shape[-1]
    Q = H - tau * S + diag_shift * jnp.eye(side, dtype=S.dtype)

    updates = solver_fn(Q, rhs)
    if isinstance(updates, tuple):
        updates, info = updates
        if info is None:
            info = {}
    else:
        info = {}

    if momentum is not None:
        old_updates = updates

    if mode == "complex" and nkjax.tree_leaf_iscomplex(params_structure):
        np_ = updates.shape[-1] // 2
        updates = updates[:np_] + 1j * updates[np_:]

    return updates, old_updates, info
