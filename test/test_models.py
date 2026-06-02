"""The RBMRealParams model: complex output, real parameters."""

import jax.tree_util as jtu
import numpy as np
import optax
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
