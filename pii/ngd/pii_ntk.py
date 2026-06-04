r"""Two-function empirical NTK contraction.

The on-the-fly PII needs the cross term :math:`O A^T = J_{\log\psi}(x_i)\,J_{f_A}(x_j)^T`,
which involves **two different** functions.  NetKet's
:func:`netket.jax._ntk.empirical_ntk_by_jacobian` uses a single function ``f``
for both Jacobians, so we adapt its ``ntk_fn`` / ``sum_and_contract`` to accept a
pair ``(f1, f2)``.  When ``f1 is f2`` this reduces exactly to the standard NTK.
"""

import operator

import jax
import jax.numpy as jnp

from netket.jax._ntk import utils
from netket.jax._ntk.logic import _dot_general, _get_args, _get_f_params


def empirical_cross_ntk_by_jacobian(f1, f2, trace_axes=(), vmap_axes=None):
    r"""Return a function computing the cross-NTK ``J_{f1}(x1) J_{f2}(x2)^T``.

    Mirrors :func:`netket.jax._ntk.logic.empirical_ntk_by_jacobian` but contracts
    the Jacobian of ``f1`` (at ``x1``) with the Jacobian of ``f2`` (at ``x2``).
    """

    def sum_and_contract(fx, j1, j2):
        ndim = fx.ndim
        size = utils.size_at(fx, trace_axes)

        _trace_axes = utils.canonicalize_axis(trace_axes, ndim)

        def contract(x, y):
            param_axes = list(range(x.ndim))[ndim:]
            contract_axes = _trace_axes + param_axes
            return _dot_general(x, y, contract_axes, ()) / size

        return jax.tree.reduce(operator.add, jax.tree.map(contract, j1, j2))

    def ntk_fn(x1, x2, params, **apply_fn_kwargs):
        # f1 governs the output structure / vmap axes used for both Jacobians.
        args1, args2, fx1, fx2, fx_axis, keys, kw_axes, x_axis = _get_args(
            f1, apply_fn_kwargs, params, vmap_axes, x1, x2
        )

        def j_fn(fn, x, *args):
            _kwargs = {k: v for k, v in zip(keys, args)}
            fx = _get_f_params(fn, x, x_axis, fx_axis, kw_axes, **_kwargs)
            return jax.jacobian(fx)(params)

        j1_fn = lambda x, *a: j_fn(f1, x, *a)
        j2_fn = lambda x, *a: j_fn(f2, x, *a)

        if not utils.all_none(x_axis) or not utils.all_none(kw_axes):
            in_axes = [x_axis] + [kw_axes[k] if k in kw_axes else None for k in keys]
            j1_fn = jax.vmap(j1_fn, in_axes=in_axes, out_axes=fx_axis)
            j2_fn = jax.vmap(j2_fn, in_axes=in_axes, out_axes=fx_axis)

        j1 = j1_fn(x1, *args1)
        j2 = j2_fn(x2, *args2)
        return jax.tree.map(sum_and_contract, fx1, j1, j2)

    return ntk_fn
