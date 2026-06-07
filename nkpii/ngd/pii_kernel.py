r"""Kernel-trick / minPII update (``2M×2M`` push-through form).

Models :func:`netket._src.ngd.srt._compute_srt_update`.  Uses the push-through
identity (valid for the non-symmetric PII matrices)

.. math::
    (O^T(A - \tau O) + \lambda I_P)^{-1} O^T e
        = O^T ((A - \tau O) O^T + \lambda I_{2M})^{-1} e,

so we form the ``2M×2M`` kernel :math:`K = A O^T - \tau O O^T + \lambda I`,
solve :math:`K y = \tfrac{1}{2} e`, and set :math:`\xi = O^T y`.  Cheaper than
the dense path when ``2M < P``.

SPRING:

.. math::
    \xi = \mu\,\xi_{k-1} + O^T K^{-1}\big[\tfrac{1}{2} e - \mu (A - \tau O)\,\xi_{k-1}\big].
"""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp

from netket import jax as nkjax
from netket.utils import timing
from netket.utils.types import Array


@timing.timed
@partial(jax.jit, static_argnames=("solver_fn", "mode"))
def _compute_minpii_update(
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
        raise ValueError("proj_reg is not implemented for the minPII update.")

    # rhs in 2M space: ½ e (+ SPRING residual anchored to the previous update).
    rhs = 0.5 * dv
    if momentum is not None:
        rhs = rhs - momentum * ((A_L - tau * O_L) @ old_updates)

    OOt = O_L @ O_L.T
    AOt = A_L @ O_L.T
    side = OOt.shape[-1]
    K = AOt - tau * OOt + diag_shift * jnp.eye(side, dtype=OOt.dtype)

    y = solver_fn(K, rhs)
    if isinstance(y, tuple):
        y, info = y
        if info is None:
            info = {}
    else:
        info = {}

    updates = O_L.T @ y
    if momentum is not None:
        updates = updates + momentum * old_updates
        old_updates = updates

    if mode == "complex" and nkjax.tree_leaf_iscomplex(params_structure):
        np_ = updates.shape[-1] // 2
        updates = updates[:np_] + 1j * updates[np_:]

    return updates, old_updates, info
