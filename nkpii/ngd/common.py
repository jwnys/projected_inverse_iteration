r"""Common machinery for the dense and kernel (minPII) PII updates.

Mirrors :mod:`netket._src.ngd.sr_srt_common`: it computes the centered, scaled
log-:math:`\psi` Jacobian ``O`` exactly as SR does, additionally computes the
PII ``A`` Jacobian (via :func:`nkpii.ngd.local_energy.make_local_energy_funs`),
and dispatches to the dense (``Oᵀ A``, ``P×P``) or kernel/minPII
(``O Aᵀ``, ``2M×2M``) solver.
"""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

import netket.jax as nkjax
from netket.utils import timing
from netket.utils.types import Array, PyTree
from netket.vqs import FullSumState

from nkpii.ngd.local_energy import make_local_energy_funs
from nkpii.ngd.pii_dense import _compute_pii_update_dense
from nkpii.ngd.pii_kernel import _compute_minpii_update


def get_samples_and_pdf(vstate):
    """Return ``(samples, pdf)``; ``pdf`` is set only for :class:`FullSumState`."""
    if isinstance(vstate, FullSumState):
        samples = vstate.hilbert.all_states()
        pdf = vstate.probability_distribution()
    else:
        samples = vstate.samples
        samples = jax.lax.collapse(samples, 0, samples.ndim - 1)
        pdf = None
    return samples, pdf


def _prepare_weights(weights, n_samples):
    if weights is not None:
        weights = weights / jnp.mean(weights)
        pdf = weights / n_samples
        mass = pdf
    else:
        pdf = None
        mass = 1 / n_samples
    return pdf, mass


@partial(jax.jit, static_argnames=("mode",))
def _prepare_input(O_L, A_L, local_grad, *, mode: str, scaling_factor):
    r"""Center & scale the Jacobians ``O_L``, ``A_L`` and build the rhs ``dv``.

    Same conventions as :func:`netket._src.ngd.sr_srt_common._prepare_input`:
    ``dv_n = (2/\sqrt{M})(E_{L,n} - \langle E_L\rangle)`` so that ``Oᵀ dv = ∇E``,
    and ``O_L``, ``A_L`` carry the ``1/\sqrt{M}`` (or ``\sqrt{pdf}``) factor.
    """
    local_grad = local_grad.flatten()
    de = local_grad - jnp.sum(scaling_factor * local_grad)
    dv = 2.0 * de * jnp.sqrt(scaling_factor)

    sf = scaling_factor
    if jnp.ndim(scaling_factor) != 0:
        sf = jax.lax.broadcast_in_dim(scaling_factor, O_L.shape, (0,))
    O_L = O_L * jnp.sqrt(sf)
    A_L = A_L * jnp.sqrt(sf)

    if mode == "complex":
        O_L = jax.lax.collapse(O_L, 0, 2)
        A_L = jax.lax.collapse(A_L, 0, 2)
        dv2 = jnp.stack([jnp.real(dv), jnp.imag(dv)], axis=-1)
        dv = jax.lax.collapse(dv2, 0, 2)
    elif mode == "real":
        dv = dv.real
    else:
        raise NotImplementedError()
    return O_L, A_L, dv


@timing.timed
@partial(
    jax.jit,
    static_argnames=(
        "log_psi",
        "f_A",
        "solver_fn",
        "mode",
        "chunk_size",
        "chunk_size_dEloc",
        "use_ntk",
    ),
)
def _pii_common(
    log_psi,
    f_A,
    local_grad,
    parameters,
    model_state,
    samples,
    *,
    tau: float | Array,
    diag_shift: float | Array,
    solver_fn: Callable[[Array, Array], Array],
    mode: str,
    proj_reg: float | Array | None = None,
    momentum: float | Array | None = None,
    old_updates: PyTree | None = None,
    chunk_size: int | None = None,
    chunk_size_dEloc: int | None = None,
    use_ntk: bool = False,
    weights: Array | None = None,
):
    r"""Compute the PII parameter update for the dense or kernel (minPII) path.

    Args:
        log_psi: apply function for the log-wavefunction.
        f_A: the PII ``A``-function (``E_L + sg(E_L) log_psi``); its Jacobian is ``A``.
        local_grad: per-sample local energies (used to build the rhs ``dv``).
        tau: the inverse-iteration shift :math:`\tau` (≈ ground-state energy).
        diag_shift: Tikhonov regularization :math:`\lambda`.
        chunk_size: chunking of the ``O`` Jacobian (backward pass).
        chunk_size_dEloc: chunking of the ``A`` Jacobian (local-energy derivative).
        use_ntk: if ``True`` use the push-through/minPII (``2M×2M``) form.
    """
    _, unravel_params_fn = ravel_pytree(parameters)
    _params_structure = jax.tree_util.tree_map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), parameters
    )

    pdf, mass = _prepare_weights(weights, samples.shape[0])

    # O: centered, scaled log-psi Jacobian (exactly as in SR).
    O = nkjax.jacobian(
        log_psi,
        parameters,
        samples,
        model_state,
        mode=mode,
        dense=True,
        center=True,
        chunk_size=chunk_size,
        pdf=pdf,
    )
    # A: Jacobian of f_A = E_L + sg(E_L) log_psi. Same flattening/mode as O, so
    # H = Oᵀ A is well defined. Chunked over samples with chunk_size_dEloc.
    A = nkjax.jacobian(
        f_A,
        parameters,
        samples,
        model_state,
        mode=mode,
        dense=True,
        center=True,
        chunk_size=chunk_size_dEloc,
        pdf=pdf,
    )

    O_L, A_L, dv = _prepare_input(O, A, local_grad, mode=mode, scaling_factor=mass)

    if old_updates is None and momentum is not None:
        old_updates = jnp.zeros(O_L.shape[-1], dtype=O_L.dtype)

    compute_update = _compute_minpii_update if use_ntk else _compute_pii_update_dense

    updates, old_updates, info = compute_update(
        O_L,
        A_L,
        dv,
        tau=tau,
        diag_shift=diag_shift,
        solver_fn=solver_fn,
        mode=mode,
        proj_reg=proj_reg,
        momentum=momentum,
        old_updates=old_updates,
        params_structure=_params_structure,
    )

    return unravel_params_fn(updates), old_updates, info
