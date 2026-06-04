"""``pii.driver.VMC_PII(pii=False)`` must reproduce NetKet's ``VMC_SR`` bit-for-bit."""

import jax
import numpy as np
import optax
import pytest

import pii
from netket._src.driver.vmc_sr import VMC_SR

from .common import tfim, make_mcstate


def _params_after(Driver, H, hi, n_iter=5, **kw):
    vs = make_mcstate(hi, seed=0)
    d = Driver(H, optax.sgd(0.05), variational_state=vs, diag_shift=0.01, **kw)
    d.run(n_iter=n_iter, show_progress=False)
    return jax.flatten_util.ravel_pytree(vs.parameters)[0]


@pytest.mark.parametrize(
    "use_ntk,on_the_fly",
    [(False, False), (True, False), (True, True)],
    ids=["sr", "minsr", "srt_onthefly"],
)
def test_reproduces_vmc_sr(use_ntk, on_the_fly):
    _, hi, H, _ = tfim()
    ref = _params_after(VMC_SR, H, hi, use_ntk=use_ntk, on_the_fly=on_the_fly, mode="real")
    got = _params_after(
        pii.driver.VMC_PII, H, hi, pii=False, use_ntk=use_ntk, on_the_fly=on_the_fly, mode="real"
    )
    np.testing.assert_allclose(got, ref, atol=0, rtol=0)


def test_sr_converges_with_small_lr():
    _, hi, H, E0 = tfim()
    vs = make_mcstate(hi, seed=0)
    d = pii.driver.VMC_PII(H, optax.sgd(0.05), variational_state=vs, diag_shift=0.01, mode="real")
    d.run(n_iter=150, show_progress=False)
    assert abs((float(vs.expect(H).mean.real) - E0) / E0) < 1e-3
