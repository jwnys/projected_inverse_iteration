"""Correctness of the PII paths: mutual agreement, convergence, SPRING, FullSum."""

import jax
import numpy as np
import optax
import pytest

import pii

from .common import tfim, make_mcstate, make_fullsum, rel_error


def _one_step_dp(H, hi, **kw):
    """Return the (flattened) parameter update of the first PII step."""
    vs = make_mcstate(hi, seed=0)
    d = pii.VMC(H, optax.sgd(1.0), variational_state=vs, **kw)
    d.run(n_iter=1, show_progress=False)
    return jax.flatten_util.ravel_pytree(d._dp)[0]


def test_dense_kernel_onthefly_agree():
    """Dense, minPII and on-the-fly solve the same system → same update."""
    _, hi, H, E0 = tfim()
    common = dict(diag_shift=0.1, pii=True, tau=1.2 * E0, mode="real")
    dp_dense = _one_step_dp(H, hi, use_ntk=False, **common)
    dp_kernel = _one_step_dp(H, hi, use_ntk=True, on_the_fly=False, **common)
    dp_otf = _one_step_dp(H, hi, use_ntk=True, on_the_fly=True, **common)
    np.testing.assert_allclose(dp_kernel, dp_dense, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(dp_otf, dp_dense, rtol=1e-5, atol=1e-6)


def test_onthefly_chunking_is_exact():
    """chunk_size_dEloc must not change the result (up to numerics)."""
    _, hi, H, E0 = tfim()
    common = dict(diag_shift=0.1, pii=True, tau=1.2 * E0, mode="real",
                  use_ntk=True, on_the_fly=True)
    dp_full = _one_step_dp(H, hi, **common)
    dp_chunked = _one_step_dp(H, hi, chunk_size_dEloc=256, **common)
    np.testing.assert_allclose(dp_chunked, dp_full, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "kw",
    [
        dict(use_ntk=False),
        dict(use_ntk=True, on_the_fly=False),
        dict(use_ntk=True, on_the_fly=True),
        dict(use_ntk=True, on_the_fly=True, momentum=0.8),  # PII-SPRING
    ],
    ids=["dense", "minpii", "onthefly", "pii_spring"],
)
def test_pii_converges(kw):
    _, hi, H, E0 = tfim()
    vs = make_mcstate(hi, seed=0)
    d = pii.VMC(
        H, optax.sgd(1.0), variational_state=vs, diag_shift=0.1, pii=True,
        tau=1.2 * E0, mode="real", **kw,
    )
    d.run(n_iter=50, show_progress=False)
    assert rel_error(vs, H, E0) < 5e-3


def test_pii_fullsum_dense_converges():
    _, hi, H, E0 = tfim()
    vs = make_fullsum(hi, seed=0)
    d = pii.VMC(
        H, optax.sgd(1.0), variational_state=vs, diag_shift=0.1, pii=True,
        tau=1.2 * E0, mode="real",
    )
    d.run(n_iter=50, show_progress=False)
    assert rel_error(vs, H, E0) < 1e-3


def test_tau_required_for_pii():
    _, hi, H, _ = tfim()
    vs = make_mcstate(hi, seed=0)
    with pytest.raises(ValueError):
        pii.VMC(H, optax.sgd(1.0), variational_state=vs, diag_shift=0.1, pii=True)


def test_mode_real_rejects_complex_params():
    """mode='real' with complex parameters must raise a clear error, not crash later."""
    import netket as nk

    _, hi, H, _ = tfim()
    vs = nk.vqs.MCState(
        nk.sampler.MetropolisLocal(hi, n_chains=16),
        nk.models.RBM(alpha=2, param_dtype=complex), n_samples=512, seed=0,
    )
    with pytest.raises(ValueError, match="mode='real'"):
        pii.VMC(H, optax.sgd(1.0), variational_state=vs, diag_shift=0.1, mode="real")
    # complex mode is fine with complex params
    pii.VMC(H, optax.sgd(1.0), variational_state=vs, diag_shift=0.1, pii=True,
            tau=1.2 * (-8.0), mode="complex")


def test_tau_schedule():
    """tau can be a schedule Callable[[int], float]."""
    _, hi, H, E0 = tfim()
    vs = make_mcstate(hi, seed=0)
    d = pii.VMC(
        H, optax.sgd(1.0), variational_state=vs, diag_shift=0.1, pii=True,
        tau=lambda step: 1.2 * E0, mode="real",
    )
    d.run(n_iter=30, show_progress=False)
    assert rel_error(vs, H, E0) < 5e-3
