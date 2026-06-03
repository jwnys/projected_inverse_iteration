"""Correctness of the PII paths: mutual agreement, convergence, SPRING, FullSum."""

from functools import partial

import jax
import numpy as np
import optax
import pytest

import pii
from pii import symmetrized_solver

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


# --- Symmetrized PII (regularized pseudo-inverse solver) ---------------------


def test_symmetrized_solver_math():
    """symmetrized_solver returns (QᵀQ+reg·I)⁻¹Qᵀb, → Q⁻¹b as reg→0."""
    rng = np.random.default_rng(0)
    n = 20
    Q = np.eye(n) + 0.3 * rng.standard_normal((n, n))  # well-conditioned
    b = rng.standard_normal(n)

    reg = 1e-3
    x, info = symmetrized_solver(Q, b, diag_shift=reg)
    expected = np.linalg.solve(Q.T @ Q + reg * np.eye(n), Q.T @ b)
    assert info is None
    np.testing.assert_allclose(np.asarray(x), expected, rtol=1e-6, atol=1e-8)

    # reg → 0 reproduces the plain solve Q⁻¹b (Q well-conditioned).
    x0, _ = symmetrized_solver(Q, b, diag_shift=1e-10)
    np.testing.assert_allclose(np.asarray(x0), np.linalg.solve(Q, b), rtol=1e-5, atol=1e-6)

    # an inner `solver` (A, b) -> (x, info) is honored for the SPD normal system.
    seen = {}

    def my_solver(A, rhs):
        seen["called"] = True
        return jax.scipy.linalg.solve(A, rhs, assume_a="sym"), {"k": 1}

    xs, infos = symmetrized_solver(Q, b, diag_shift=reg, solver=my_solver)
    assert seen.get("called") and infos == {"k": 1}
    np.testing.assert_allclose(np.asarray(xs), expected, rtol=1e-6, atol=1e-8)


def test_pii_symmetrized_fullsum_converges():
    """Symmetrized PII (diag_shift=0, solver carries reg) converges in FullSum."""
    _, hi, H, E0 = tfim()
    vs = make_fullsum(hi, seed=0)
    d = pii.VMC(
        H, optax.sgd(1.0), variational_state=vs, diag_shift=0.0, pii=True,
        tau=1.2 * E0, mode="real",
        linear_solver=partial(symmetrized_solver, diag_shift=1e-3),
    )
    d.run(n_iter=50, show_progress=False)
    assert rel_error(vs, H, E0) < 1e-3


def test_pii_symmetrized_converges():
    """Symmetrized PII converges under Monte Carlo sampling."""
    _, hi, H, E0 = tfim()
    vs = make_mcstate(hi, seed=0)
    d = pii.VMC(
        H, optax.sgd(1.0), variational_state=vs, diag_shift=0.0, pii=True,
        tau=1.2 * E0, mode="real",
        linear_solver=partial(symmetrized_solver, diag_shift=1e-3),
    )
    d.run(n_iter=50, show_progress=False)
    assert rel_error(vs, H, E0) < 5e-3


def test_symmetrized_solver_pseudoinverse_rank_deficient():
    """On a *singular* Q the solver gives the min-norm least-squares (pinv) solution.

    For a rank-deficient Q (the common case with over-parametrized / gauge-redundant
    ansätze, where Q = H − τS is singular and Q⁻¹ does not exist), the Tikhonov-
    regularized Euclidean least-squares problem converges to the Moore-Penrose
    pseudo-inverse solution `pinv(Q) @ b` as reg → 0. (Standard PII regularizes the
    same step differently — in the b-inner-product geometry, paper Eq. 32-33 — so
    this is a different update on the null space, not a strictly "more correct" one.)
    """
    rng = np.random.default_rng(1)
    n, rank = 20, 14
    U, _ = np.linalg.qr(rng.standard_normal((n, n)))
    V, _ = np.linalg.qr(rng.standard_normal((n, n)))
    s = np.concatenate([rng.uniform(0.5, 2.0, rank), np.zeros(n - rank)])  # 6 zeros
    Q = (U * s) @ V.T
    b = rng.standard_normal(n)

    x, _ = symmetrized_solver(Q, b, diag_shift=1e-10)
    np.testing.assert_allclose(np.asarray(x), np.linalg.pinv(Q) @ b, rtol=1e-4, atol=1e-6)


def test_symmetrized_solver_complex_hermitian():
    """For a genuinely complex Q the solver must use the Hermitian transpose Qᴴ.

    The conjugation only matters for a *singular* complex Q (for invertible Q both
    Qᴴ and Qᵀ collapse to Q⁻¹). On a rank-deficient complex Q, ξ = (QᴴQ+reg·I)⁻¹Qᴴb
    → pinv(Q)·b (complex Moore-Penrose, which uses Qᴴ); a plain Qᵀ would give a
    materially different, wrong vector. The in-code Q is always real, but `Q.conj().T`
    keeps this correct in either representation.
    """
    rng = np.random.default_rng(2)
    n, rank = 16, 10
    U, _ = np.linalg.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
    V, _ = np.linalg.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
    s = np.concatenate([rng.uniform(0.5, 2.0, rank), np.zeros(n - rank)])
    Q = U @ np.diag(s).astype(complex) @ V.conj().T  # rank-deficient complex
    b = rng.standard_normal(n) + 1j * rng.standard_normal(n)

    x, _ = symmetrized_solver(Q, b, diag_shift=1e-10)
    np.testing.assert_allclose(np.asarray(x), np.linalg.pinv(Q) @ b, rtol=1e-4, atol=1e-6)
    # the conjugate transpose matters: plain Qᵀ gives a materially different vector.
    x_wrong = np.linalg.solve(Q.T @ Q + 1e-10 * np.eye(n), Q.T @ b)
    assert not np.allclose(np.asarray(x), x_wrong, rtol=1e-2)
