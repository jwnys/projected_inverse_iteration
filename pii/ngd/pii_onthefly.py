r"""On-the-fly (matrix-free) minPII update.

Adapted from :func:`netket._src.ngd.srt_onthefly.srt_onthefly`.  Builds the
``2M×2M`` kernel ``K = O Aᵀ − τ O Oᵀ + λ I`` lazily, never materializing the
``[M, P]`` Jacobian:

- ``O Oᵀ`` uses NetKet's :func:`empirical_ntk_by_jacobian` (single function = logψ);
- ``O Aᵀ`` uses :func:`pii.ngd.pii_ntk.empirical_cross_ntk_by_jacobian` with the
  pair ``(logψ, f_A)``, chunked over the ``f_A`` (column) axis by
  ``chunk_size_dEloc`` so peak memory is ``chunk · n_conn`` passes — the
  local-energy-derivative memory never becomes a ``[M, n_conn, P]`` tensor.

This implementation drops the MPI/sharding machinery of the NetKet original for
simplicity; the ``pii=False`` on-the-fly path is served directly by NetKet's
``srt_onthefly`` in the driver, so SR-on-the-fly is reproduced exactly.
"""

from collections.abc import Callable
from functools import partial

from einops import rearrange

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

from netket import jax as nkjax
from netket.jax._ntk import empirical_ntk_by_jacobian
from netket.jax._jacobian.default_mode import JacobianMode
from netket.utils import timing
from netket.utils.types import Array

from pii.ngd.pii_ntk import empirical_cross_ntk_by_jacobian


def _center_ntk(ntk, N_mc: int, mode: str):
    r"""Double-center a raw ``[.,.]`` (or ``[.,.,2,2]``) NTK: ``\delta K \delta / N``."""
    if mode == "complex":
        ntk = ntk.reshape(N_mc, 2, N_mc, 2)
        col_means = ntk.mean(axis=0, keepdims=True)
        row_means = ntk.mean(axis=2, keepdims=True)
        global_mean = col_means.mean(axis=2, keepdims=True)
        ntk = ntk - col_means - row_means + global_mean
        ntk = ntk.reshape(2 * N_mc, 2 * N_mc)
    else:
        row_means = ntk.mean(axis=1, keepdims=True)
        col_means = ntk.mean(axis=0, keepdims=True)
        global_mean = col_means.mean()
        ntk = ntk - col_means - row_means + global_mean
    return ntk / N_mc


@timing.timed
@partial(
    jax.jit,
    static_argnames=(
        "log_psi",
        "f_A",
        "solver_fn",
        "chunk_size",
        "chunk_size_dEloc",
        "mode",
    ),
)
def pii_onthefly(
    log_psi,
    f_A,
    local_energies,
    parameters,
    model_state,
    samples,
    *,
    tau: float | Array,
    diag_shift: float | Array,
    solver_fn: Callable[[Array, Array], Array],
    mode: JacobianMode,
    proj_reg: float | Array | None = None,
    momentum: float | Array | None = None,
    old_updates: Array | None = None,
    chunk_size: int | None = None,
    chunk_size_dEloc: int | None = None,
    weights: Array | None = None,
):
    if weights is not None:
        raise NotImplementedError(
            "Weighted samples / FullSumState are not supported in pii_onthefly. "
            "Use the dense PII path for FullSumState."
        )
    if proj_reg is not None:
        raise NotImplementedError("proj_reg is not implemented for pii_onthefly.")

    N_mc = local_energies.size
    parameters_real, rss = nkjax.tree_to_real(parameters)

    def _apply(params_real, x, model_state):
        variables = {"params": rss(params_real), **model_state}
        log_amp = log_psi(variables, x)
        if mode == "complex":
            return jnp.concatenate(
                (log_amp.real[:, None], log_amp.imag[:, None]), axis=-1
            )
        return log_amp.real

    def _apply_A(params_real, x, model_state):
        variables = {"params": rss(params_real), **model_state}
        a = f_A(variables, x)
        if mode == "complex":
            return jnp.concatenate((a.real[:, None], a.imag[:, None]), axis=-1)
        return a.real

    def jvp_apply(fn, vector):
        f = lambda p: fn(p, samples, model_state)
        _, acc = jax.jvp(f, (parameters_real,), (vector,))
        return acc  # [N,2] or [N]

    # ---- right-hand side  dv_n = (2/√N)(E_L − ⟨E_L⟩) (+ SPRING residual) ----
    local_energies = local_energies.flatten()
    de = local_energies - jnp.mean(local_energies)
    dv = 2.0 * de / jnp.sqrt(N_mc)
    if mode == "complex":
        dv = jnp.stack([jnp.real(dv), jnp.imag(dv)], axis=-1)  # [N,2]
    else:
        dv = jnp.real(dv)

    rhs = 0.5 * dv
    if momentum is not None:
        if old_updates is None:
            old_updates = tree_map(jnp.zeros_like, parameters_real)
        else:
            # still inefficient
            # residual μ (A − τ O) ξ_{k-1}, centered & scaled like dv
            acc_O = jvp_apply(_apply, old_updates)
            acc_A = jvp_apply(_apply_A, old_updates)
            res = acc_A - tau * acc_O
            res = (res - jnp.mean(res, axis=0)) / jnp.sqrt(N_mc)
            rhs = rhs - momentum * res

    if mode == "complex":
        rhs = jax.lax.collapse(rhs, 0, 2)  # [2N]

    # ---- kernels ----
    _oo = empirical_ntk_by_jacobian(f=_apply, trace_axes=(), vmap_axes=0)
    OO_raw = _oo(samples, None, parameters_real, model_state=model_state).real

    # Cross term A Oᵀ = J_{f_A}(x_i) · J_{logψ}(x_j)ᵀ  (push-through, Eq. 30).
    # The f_A (row / x_i) axis carries the local-energy derivative, so we chunk it
    # by chunk_size_dEloc to bound peak memory to chunk · n_conn passes.
    _ao = empirical_cross_ntk_by_jacobian(f1=_apply_A, f2=_apply, trace_axes=(), vmap_axes=0)
    if chunk_size_dEloc is None or N_mc % chunk_size_dEloc != 0:
        AO_raw = _ao(samples, samples, parameters_real, model_state=model_state).real
    else:
        rows = samples.reshape(N_mc // chunk_size_dEloc, chunk_size_dEloc, -1)
        AO_chunks = jax.lax.map(
            lambda b: _ao(b, samples, parameters_real, model_state=model_state).real,
            rows,
        )
        if mode == "complex":
            AO_raw = rearrange(AO_chunks, "nb i j z w -> (nb i) j z w")
        else:
            AO_raw = rearrange(AO_chunks, "nb i j -> (nb i) j")

    if mode == "complex":
        OO_raw = rearrange(OO_raw, "i j z w -> (i z) (j w)")
        AO_raw = rearrange(AO_raw, "i j z w -> (i z) (j w)")

    OO = _center_ntk(OO_raw, N_mc, mode)
    AO = _center_ntk(AO_raw, N_mc, mode)

    side = OO.shape[0]
    K = AO - tau * OO + diag_shift * jnp.eye(side, dtype=OO.dtype)

    y = solver_fn(K, rhs)
    if isinstance(y, tuple):
        y, info = y
        if info is None:
            info = {}
    else:
        info = {}

    y = jnp.squeeze(y)
    if mode == "complex":
        y = y.reshape((N_mc, 2))

    # ξ = Oᵀ y : center y (δ/√N) then vjp with logψ Jacobian.
    y = (y - jnp.mean(y, axis=0, keepdims=True)) / jnp.sqrt(N_mc)

    vjp_fun = nkjax.vjp_chunked(
        _apply,
        parameters_real,
        samples,
        model_state,
        chunk_size=chunk_size,
        chunk_argnums=1,
        nondiff_argnums=(1, 2),
    )
    (updates,) = vjp_fun(y)

    if momentum is not None:
        updates = tree_map(lambda x, z: x + momentum * z, updates, old_updates)
        old_updates = updates

    return rss(updates), old_updates, info
