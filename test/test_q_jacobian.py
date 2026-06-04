"""Tests for the PII Q-matrix operators (:mod:`pii.optimizer.q`).

Covers, across common NetKet models and our two models (incl. holomorphic
complex-parameter networks):

- no-crash construction + ``Q@v`` / ``Qᴴ@v`` matvecs + ``to_dense`` + ``solve``;
- Dense and PyTree operators give the same dense ``Q``;
- the QJacobian update equals the **existing dense-PII** implementation;
- holomorphic and realified modes give the same **physical** update (δψ), and a
  holomorphic complex-param exact ansatz matches our real-param one (both reach E0).
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import pytest
import netket as nk

import pii
from pii.optimizer.q import QJacobianDense, QJacobianPyTree

from .common import tfim


def _ravel(x):
    return np.asarray(x if hasattr(x, "ndim") else ravel_pytree(x)[0])


def _finite(x):
    return bool(np.all(np.isfinite(_ravel(x))))


# operator-aware dense solver (NetKet's solver contract: receives the operator and a rhs
# that is a flat array for the Dense operator but a PyTree for the PyTree operator).
def _op_solver(op, y, **kwargs):
    if hasattr(y, "ndim"):
        return jnp.linalg.solve(op.to_dense(), y), None
    y_flat, unravel = ravel_pytree(y)
    return unravel(jnp.linalg.solve(op.to_dense(), y_flat)), None


def _model(name, hi):
    return {
        "nk_RBM_real": nk.models.RBM(alpha=1, param_dtype=float),
        "nk_RBM_complex": nk.models.RBM(alpha=1, param_dtype=complex),
        "nk_LogStateVector": nk.models.LogStateVector(hi, param_dtype=complex),
        "pii_RBMRealParams": pii.models.RBMRealParams(alpha=1),
        "pii_LogStateVector": pii.models.LogStateVectorRealParams(hi),
    }[name]


# (model name, jacobian mode) — covers real, realified-complex, and holomorphic.
CASES = [
    ("nk_RBM_real", "real"),
    ("nk_RBM_complex", "complex"),
    ("nk_RBM_complex", "holomorphic"),
    ("nk_LogStateVector", "holomorphic"),
    ("pii_RBMRealParams", "complex"),
    ("pii_LogStateVector", "complex"),
]
_IDS = [f"{n}-{m}" for n, m in CASES]


@pytest.mark.parametrize("Qctor", [QJacobianDense, QJacobianPyTree], ids=["dense", "pytree"])
@pytest.mark.parametrize("name,mode", CASES, ids=_IDS)
def test_q_jacobian_runs(Qctor, name, mode):
    """Construct + ``Q@v`` + ``Qᴴ@v`` + ``to_dense`` + ``solve``, all finite (no crash)."""
    _, hi, H, E0 = tfim()
    vs = nk.vqs.FullSumState(hi, _model(name, hi), seed=0)
    Q = Qctor(vs, H, tau=1.2 * E0, diag_shift=1e-3, mode=mode)

    v = jax.tree_util.tree_map(jnp.ones_like, vs.parameters)
    assert _finite(Q @ v)
    assert _finite(Q.H @ v)

    Qd = np.asarray(Q.to_dense())
    assert Qd.ndim == 2 and Qd.shape[0] == Qd.shape[1] and np.all(np.isfinite(Qd))

    x, _info = Q.solve(_op_solver, v)
    assert _finite(x)


@pytest.mark.parametrize("name,mode", CASES, ids=_IDS)
def test_q_jacobian_dense_pytree_agree(name, mode):
    """The dense and PyTree Q-operators produce the same dense ``Q``."""
    _, hi, H, E0 = tfim()
    vs = nk.vqs.FullSumState(hi, _model(name, hi), seed=0)
    Qd = np.asarray(QJacobianDense(vs, H, tau=1.2 * E0, diag_shift=1e-3, mode=mode).to_dense())
    Qp = np.asarray(QJacobianPyTree(vs, H, tau=1.2 * E0, diag_shift=1e-3, mode=mode).to_dense())
    np.testing.assert_allclose(Qp, Qd, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize(
    "name,mode", [("nk_RBM_real", "real"), ("pii_RBMRealParams", "complex")]
)
def test_q_jacobian_matches_existing_dense_pii(name, mode):
    """``QJacobian.solve`` gives the same update as the existing dense-PII path."""
    from pii.ngd.common import get_samples_and_pdf, _prepare_weights, _prepare_input
    from pii.ngd.local_energy import make_local_energy_funs
    from pii.ngd.pii_dense import _compute_pii_update_dense
    from pii.optimizer.solver import pii_default_solver
    import netket.jax as nkjax

    _, hi, H, E0 = tfim()
    tau, lam = 1.2 * E0, 1e-3
    vs = nk.vqs.FullSumState(hi, _model(name, hi), seed=0)

    # existing dense-PII update, built exactly as the driver does
    samples, weights = get_samples_and_pdf(vs)
    pdf, mass = _prepare_weights(weights, samples.shape[0])
    f_EL, f_A = make_local_energy_funs(vs, H)
    eloc = f_EL(vs.variables, samples)
    O = nkjax.jacobian(vs._apply_fun, vs.parameters, samples, vs.model_state, mode=mode, dense=True, center=True, pdf=pdf)
    A = nkjax.jacobian(f_A, vs.parameters, samples, vs.model_state, mode=mode, dense=True, center=True, pdf=pdf)
    O_L, A_L, dv = _prepare_input(O, A, eloc, mode=mode, scaling_factor=mass)
    ps = jax.tree_util.tree_map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), vs.parameters)
    dp_existing, _, _ = _compute_pii_update_dense(
        O_L, A_L, dv, tau=tau, diag_shift=lam, solver_fn=pii_default_solver, mode=mode, params_structure=ps
    )

    # QJacobian update with rhs = ½∇E (= ½ · the energy gradient)
    Q = QJacobianDense(vs, H, tau=tau, diag_shift=lam, mode=mode)
    _, grad = vs.expect_and_grad(H)
    rhs = jax.tree_util.tree_map(lambda g: 0.5 * g, grad)
    dp_q, _ = Q.solve(_op_solver, rhs)

    np.testing.assert_allclose(_ravel(dp_q), _ravel(dp_existing), rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("name", ["nk_RBM_complex", "nk_LogStateVector"])
def test_holomorphic_matches_realified(name):
    """Holomorphic and realified (complex) modes give the same physical update δψ = Oξ."""
    _, hi, H, E0 = tfim()
    tau, lam = 1.2 * E0, 1e-3
    vs = nk.vqs.FullSumState(hi, _model(name, hi), seed=0)
    st = hi.all_states()

    def dpsi(mode):
        Q = QJacobianDense(vs, H, tau=tau, diag_shift=lam, mode=mode)
        _, grad = vs.expect_and_grad(H)
        dp, _ = Q.solve(_op_solver, jax.tree_util.tree_map(lambda g: 0.5 * g, grad))
        _, d = jax.jvp(
            lambda p: vs._apply_fun({"params": p, **vs.model_state}, st),
            (vs.parameters,), (dp,),
        )
        return np.asarray(d)

    np.testing.assert_allclose(dpsi("holomorphic"), dpsi("complex"), rtol=1e-5, atol=1e-7)


def test_logstatevector_holo_matches_realparams():
    """holomorphic nk.LogStateVector and our LogStateVectorRealParams both reach E0 via PII."""
    _, hi, H, E0 = tfim()
    tau, lam = 1.2 * E0, 1e-3

    def converge(model, mode, n=40):
        vs = nk.vqs.FullSumState(hi, model, seed=0)
        for _ in range(n):
            Q = QJacobianDense(vs, H, tau=tau, diag_shift=lam, mode=mode)
            _, grad = vs.expect_and_grad(H)
            dp, _ = Q.solve(_op_solver, jax.tree_util.tree_map(lambda g: 0.5 * g, grad))
            vs.parameters = jax.tree_util.tree_map(lambda p, d: p - d, vs.parameters, dp)
        return abs((float(vs.expect(H).mean.real) - E0) / E0)

    assert converge(nk.models.LogStateVector(hi, param_dtype=complex), "holomorphic") < 1e-6
    assert converge(pii.models.LogStateVectorRealParams(hi), "complex") < 1e-6


@pytest.mark.parametrize("kind", ["rbm", "logstatevector"])
def test_complex_params_vs_realparams_same_update(kind):
    """The PII update is *independent of the parametrization*: a complex-parameter ansatz
    (differentiated holomorphically) and its real-parameter twin (the RealParams trick, complex
    mode), set to the **same** wavefunction ψ, produce the **same** physical update δψ = O ξ.

    This is the strongest implementation-independence check: same ψ, same physics, two entirely
    different parameter encodings (complex weights vs. stacked real/imag weights) → identical δψ.
    """
    _, hi, H, E0 = tfim()
    tau, lam = 1.2 * E0, 1e-3
    st = hi.all_states()
    rng = np.random.default_rng(1)

    def dpsi(model, params, mode):
        vs = nk.vqs.FullSumState(hi, model, seed=0)
        vs.parameters = params  # set both encodings to the same ψ
        Q = QJacobianDense(vs, H, tau=tau, diag_shift=lam, mode=mode)
        _, grad = vs.expect_and_grad(H)
        dp, _ = Q.solve(_op_solver, jax.tree_util.tree_map(lambda g: 0.5 * g, grad))
        _, d = jax.jvp(
            lambda p: vs._apply_fun({"params": p, **vs.model_state}, st),
            (vs.parameters,), (dp,),
        )
        return np.asarray(d)

    def re_im(shape):
        return jnp.asarray(rng.standard_normal(shape)), jnp.asarray(rng.standard_normal(shape))

    if kind == "logstatevector":
        a, b = re_im((hi.n_states,))
        holo = dpsi(nk.models.LogStateVector(hi, param_dtype=complex),
                    {"logstate": a + 1j * b}, "holomorphic")
        real = dpsi(pii.models.LogStateVectorRealParams(hi),
                    {"logstate_re": a, "logstate_im": b}, "complex")
    else:  # rbm (alpha=1): map complex weights <-> stacked real/imag weights
        N = hi.size
        kre, kim = re_im((N, N)); hre, him = re_im((N,)); vre, vim = re_im((N,))
        holo = dpsi(nk.models.RBM(alpha=1, param_dtype=complex),
                    {"Dense": {"kernel": kre + 1j * kim, "bias": hre + 1j * him},
                     "visible_bias": vre + 1j * vim}, "holomorphic")
        real = dpsi(pii.models.RBMRealParams(alpha=1),
                    {"kernel_re": kre, "kernel_im": kim, "hidden_bias_re": hre,
                     "hidden_bias_im": him, "visible_bias_re": vre, "visible_bias_im": vim}, "complex")

    np.testing.assert_allclose(holo, real, rtol=1e-5, atol=1e-7)
