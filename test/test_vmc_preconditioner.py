"""The preconditioner path (`pii.driver.VMC` + `pii.optimizer.PII`) must agree with the
integrated `pii.driver.VMC_PII`, across the Dense/PyTree Q-operators and the matrix-free
solvers — and the SR path must agree with `VMC_PII(pii=False)` (≡ NetKet `VMC_SR`)."""

import numpy as np
import jax
from jax.flatten_util import ravel_pytree
import optax
import pytest
import netket as nk

import pii
from pii.optimizer.q import QJacobianDense, QJacobianPyTree

from .common import tfim, make_fullsum, rel_error

TAU_FACTOR = 1.2
DIAG_SHIFT = 0.1


def _params_after(make_driver, n=1):
    """Run ``n`` steps from a fresh seeded FullSumState; return the flat final parameters."""
    _, hi, _, _ = tfim()
    vs = make_fullsum(hi, seed=0)
    make_driver(vs).run(n, show_progress=False)
    return ravel_pytree(vs.parameters)[0]


@pytest.mark.parametrize("q", [QJacobianDense, QJacobianPyTree], ids=["dense", "pytree"])
def test_pii_preconditioner_matches_vmc_pii(q):
    """`VMC + PII(q=...)` == integrated `VMC_PII(pii=True, use_ntk=False)`, step for step."""
    _, _, H, E0 = tfim()
    tau = TAU_FACTOR * E0

    p_pre = _params_after(
        lambda vs: pii.driver.VMC(
            H, optax.sgd(1.0), variational_state=vs,
            preconditioner=pii.optimizer.PII(
                H, q=q, tau=tau, diag_shift=DIAG_SHIFT, mode="real"
            ),
        ),
        n=3,
    )
    p_int = _params_after(
        lambda vs: pii.driver.VMC_PII(
            H, optax.sgd(1.0), variational_state=vs,
            diag_shift=DIAG_SHIFT, pii=True, tau=tau, mode="real", use_ntk=False,
        ),
        n=3,
    )
    np.testing.assert_allclose(p_pre, p_int, rtol=1e-5, atol=1e-7)


def test_pii_gmres_matrixfree_matches_dense():
    """The matrix-free `gmres` solver gives the same update as the direct dense default."""
    _, _, H, E0 = tfim()
    tau = TAU_FACTOR * E0

    p_direct = _params_after(
        lambda vs: pii.driver.VMC(
            H, optax.sgd(1.0), variational_state=vs,
            preconditioner=pii.optimizer.PII(H, tau=tau, diag_shift=DIAG_SHIFT, mode="real"),
        ),
    )
    p_gmres = _params_after(
        lambda vs: pii.driver.VMC(
            H, optax.sgd(1.0), variational_state=vs,
            preconditioner=pii.optimizer.PII(
                H, tau=tau, diag_shift=DIAG_SHIFT, mode="real",
                solver=pii.optimizer.solver.gmres(tol=1e-10, restart=200, maxiter=4),
            ),
        ),
    )
    np.testing.assert_allclose(p_gmres, p_direct, rtol=1e-4, atol=1e-6)


def test_pii_preconditioner_converges():
    """`VMC + PII` reaches the ground-state energy on a FullSumState."""
    _, hi, H, E0 = tfim()
    vs = make_fullsum(hi, seed=0)
    pii.driver.VMC(
        H, optax.sgd(1.0), variational_state=vs,
        preconditioner=pii.optimizer.PII(H, tau=TAU_FACTOR * E0, diag_shift=DIAG_SHIFT, mode="real"),
    ).run(n_iter=50, show_progress=False)
    assert rel_error(vs, H, E0) < 1e-3


def test_sr_preconditioner_matches_vmc_sr():
    """`VMC + nk.optimizer.SR` (the classic-VMC reuse) == `VMC_PII(pii=False)` (≡ NetKet VMC_SR)."""
    _, _, H, _ = tfim()
    solver = nk.optimizer.solver.cholesky_with_fallback

    p_pre = _params_after(
        lambda vs: pii.driver.VMC(
            H, optax.sgd(0.05), variational_state=vs,
            preconditioner=nk.optimizer.SR(
                qgt=nk.optimizer.qgt.QGTJacobianDense, solver=solver,
                diag_shift=0.01, mode="real",
            ),
        ),
        n=3,
    )
    p_sr = _params_after(
        lambda vs: pii.driver.VMC_PII(
            H, optax.sgd(0.05), variational_state=vs,
            diag_shift=0.01, pii=False, mode="real", use_ntk=False,
            linear_solver=solver,
        ),
        n=3,
    )
    np.testing.assert_allclose(p_pre, p_sr, rtol=1e-5, atol=1e-7)
