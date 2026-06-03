"""The RBMRealParams / LogStateVectorRealParams models: complex output, real parameters."""

import jax.tree_util as jtu
import numpy as np
import optax
import pytest
import netket as nk

import pii

from .common import tfim, rel_error


def test_rbm_real_params_complex_output_real_params():
    _, hi, _, _ = tfim()
    vs = nk.vqs.MCState(
        nk.sampler.MetropolisLocal(hi, n_chains=16),
        pii.models.RBMRealParams(alpha=2), n_samples=256, seed=0,
    )
    # output is complex
    out = vs._apply_fun(vs.variables, hi.all_states()[:4])
    assert np.iscomplexobj(np.asarray(out))
    # every parameter is real
    assert all(not np.iscomplexobj(np.asarray(p)) for p in jtu.tree_leaves(vs.parameters))


def test_pii_converges_with_rbm_real_params():
    """PII (mode auto -> complex) converges with the real-param complex ansatz."""
    _, hi, H, E0 = tfim()
    vs = nk.vqs.MCState(
        nk.sampler.MetropolisLocal(hi, n_chains=16),
        pii.models.RBMRealParams(alpha=2), n_samples=2048, seed=0,
    )
    d = pii.VMC(H, optax.sgd(1.0), variational_state=vs, diag_shift=0.1, pii=True,
                tau=1.2 * E0)
    assert d.mode == "complex"  # auto-detected from the complex output
    d.run(n_iter=40, show_progress=False)
    assert rel_error(vs, H, E0) < 5e-3


def test_logstatevector_real_params_complex_output_real_params():
    _, hi, _, _ = tfim()
    vs = nk.vqs.FullSumState(hi, pii.models.LogStateVectorRealParams(hi), seed=0)
    # output is complex
    out = vs._apply_fun(vs.variables, hi.all_states()[:4])
    assert np.iscomplexobj(np.asarray(out))
    # every parameter is real, two arrays of length n_states
    leaves = jtu.tree_leaves(vs.parameters)
    assert all(not np.iscomplexobj(np.asarray(p)) for p in leaves)
    assert all(p.shape == (hi.n_states,) for p in leaves)
    # default init (ones on re & im) is a constant log-amplitude = the uniform state
    psi = np.asarray(vs.to_array())
    assert np.allclose(np.abs(psi), np.abs(psi[0]))


def test_pii_converges_with_logstatevector_real_params():
    """PII converges with the exact complex-output vector ansatz (FullSum)."""
    _, hi, H, E0 = tfim()
    vs = nk.vqs.FullSumState(hi, pii.models.LogStateVectorRealParams(hi), seed=0)
    d = pii.VMC(H, optax.sgd(1.0), variational_state=vs, diag_shift=1e-8, pii=True,
                tau=1.2 * E0, mode="complex")
    d.run(n_iter=50, show_progress=False)
    assert rel_error(vs, H, E0) < 1e-4


def test_logstatevector_real_params_rejects_complex_dtype():
    _, hi, _, _ = tfim()
    with pytest.raises(ValueError, match="real parameters"):
        nk.vqs.FullSumState(
            hi, pii.models.LogStateVectorRealParams(hi, param_dtype=complex), seed=0
        ).parameters
