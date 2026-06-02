"""Paper Fig. 1 toy benchmark: Ĥ = diag(1, 10, 0) on a 3-dim Hilbert space.

Exact linear ansatz (LogStateVector) + FullSumState. PII (τ → E0) is
gap-insensitive and converges far faster than SR (η = 0.1).
"""

import numpy as np
import optax
import netket as nk

import pii


def _build():
    hi = nk.hilbert.Spin(s=1, N=1)  # local dimension 3
    H = nk.operator.LocalOperator(hi, operators=[np.diag([1.0, 10.0, 0.0])], acting_on=[[0]])
    return hi, H


def _state(hi, seed=2):
    return nk.vqs.FullSumState(hi, nk.models.LogStateVector(hi, param_dtype=float), seed=seed)


def test_pii_beats_sr_on_diag_hamiltonian():
    hi, H = _build()
    assert hi.n_states == 3
    np.testing.assert_allclose(np.sort(np.linalg.eigvalsh(np.array(H.to_dense()))), [0, 1, 10])

    sr = pii.VMC(H, optax.sgd(0.1), variational_state=_state(hi), diag_shift=0.0,
                 pii=False, mode="real")
    pii_drv = pii.VMC(H, optax.sgd(1.0), variational_state=_state(hi), diag_shift=1e-8,
                      pii=True, tau=1e-8, mode="real")

    sr.run(n_iter=20, show_progress=False)
    pii_drv.run(n_iter=20, show_progress=False)

    e_sr = float(sr.state.expect(H).mean.real)
    e_pii = float(pii_drv.state.expect(H).mean.real)

    assert e_pii < 1e-6        # PII essentially at the ground state E0 = 0
    assert e_pii < e_sr        # and well ahead of SR at the same iteration count
