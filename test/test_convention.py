r"""Lock the SR↔PII factor-of-2, η=1, and SPRING conventions.

These tests pin, against independent references (NetKet's own energy gradient and
hand-assembled linear systems), that:

- SR solves ``S ξ = ∇E``           (no ½),
- PII solves ``Q ξ = ½∇E`` with ``Q = H − τS + λI``, and the driver with ``η=1``
  applies exactly ``θ ← θ − Q⁻¹(½∇E)`` (paper Eq. 5),
- the ½ is the *only* difference (adversarial "missing-½ would 2×" control, and the
  τ→−∞ asymptotic ``‖ξ_PII‖/‖ξ_SR‖ → 1/(2|τ|)``),
- SPRING carries the ½ and the ``λμ`` anchor correctly (Eq. 36), with all paths
  consistent and ``momentum=0`` ≡ plain PII.

All deterministic: exact ``FullSumState`` (no MC noise) for the gradient/step checks,
and fixed sampler-free inputs for the SPRING unit checks.
"""

import jax
import jax.numpy as jnp
import jax.flatten_util as fu
import numpy as np
import optax
import netket as nk

import pii
from pii.ngd.common import _prepare_input, _prepare_weights, get_samples_and_pdf
from pii.ngd.local_energy import make_local_energy_funs
from pii.ngd.pii_dense import _compute_pii_update_dense
from pii.ngd.pii_kernel import _compute_minpii_update

from .common import tfim, make_fullsum, fixed_inputs, gen_solver

LAM = 1e-3


def _ravel(tree):
    return np.asarray(fu.ravel_pytree(tree)[0])


def _assemble_fullsum(vs, H):
    """Reconstruct the centered/scaled (O_L, A_L, dv) exactly as ``_pii_common`` does."""
    samples, pdf_raw = get_samples_and_pdf(vs)
    pdf, mass = _prepare_weights(pdf_raw, samples.shape[0])
    f_EL, f_A = make_local_energy_funs(vs, H)
    eloc = f_EL(vs.variables, samples)
    jac_kw = dict(mode="real", dense=True, center=True, pdf=pdf)
    O = nk.jax.jacobian(vs._apply_fun, vs.parameters, samples, vs.model_state, **jac_kw)
    A = nk.jax.jacobian(f_A, vs.parameters, samples, vs.model_state, **jac_kw)
    return _prepare_input(O, A, eloc, mode="real", scaling_factor=mass)


def _one_step_param_delta(vs, H, **vmc_kw):
    """Run one driver step with η=1 and return ξ = θ_before − θ_after."""
    th0 = _ravel(vs.parameters)
    d = pii.driver.VMC_PII(H, optax.sgd(1.0), variational_state=vs, mode="real", **vmc_kw)
    d.run(n_iter=1, show_progress=False)
    return th0 - _ravel(vs.parameters)


# --------------------------------------------------------------------------- #
# factor of 2 in the gradient (independent: NetKet's expect_and_grad)
# --------------------------------------------------------------------------- #
def test_gradient_assembly_matches_netket():
    _, hi, H, _ = tfim()
    vs = make_fullsum(hi, seed=0)
    O_L, A_L, dv = _assemble_fullsum(vs, H)
    grad_assembled = np.asarray(O_L.T @ dv)
    _, grad_tree = vs.expect_and_grad(H)
    grad_netket = _ravel(grad_tree)
    # Oᵀdv must equal NetKet's own energy gradient (the factor 2 in dv is what makes
    # this hold). This is the netket-independent anchor for the factor-of-2.
    np.testing.assert_allclose(grad_assembled, grad_netket, rtol=1e-7, atol=1e-9)


# --------------------------------------------------------------------------- #
# PII step = Q⁻¹(½∇E) with η=1  (+ adversarial missing-½ control)
# --------------------------------------------------------------------------- #
def test_pii_step_is_Qinv_half_grad():
    _, hi, H, _ = tfim()
    vs = make_fullsum(hi, seed=0)
    tau = -5.0

    O_L, A_L, dv = _assemble_fullsum(vs, H)  # assemble BEFORE the step mutates vs
    grad = O_L.T @ dv
    S = O_L.T @ O_L
    Hm = O_L.T @ A_L
    Q = Hm - tau * S + LAM * jnp.eye(S.shape[0], dtype=S.dtype)
    xi_ref = jnp.linalg.solve(Q, 0.5 * grad)

    dp = _one_step_param_delta(vs, H, diag_shift=LAM, pii=True, tau=tau)

    # η=1 applies exactly Q⁻¹(½∇E)
    np.testing.assert_allclose(dp, np.asarray(xi_ref), rtol=1e-6, atol=1e-8)
    # adversarial: dropping the ½ would have produced 2× this step
    xi_no_half = np.asarray(jnp.linalg.solve(Q, grad))
    np.testing.assert_allclose(xi_no_half, 2.0 * dp, rtol=1e-6, atol=1e-8)


# --------------------------------------------------------------------------- #
# SR step = S⁻¹ ∇E  (no ½) — the contrast that defines the factor-of-2 asymmetry
# --------------------------------------------------------------------------- #
def test_sr_step_is_Sinv_grad_no_half():
    _, hi, H, _ = tfim()
    vs = make_fullsum(hi, seed=0)

    O_L, A_L, dv = _assemble_fullsum(vs, H)
    grad = O_L.T @ dv
    S = O_L.T @ O_L
    xi_ref = jnp.linalg.solve(S + LAM * jnp.eye(S.shape[0], dtype=S.dtype), grad)

    dp = _one_step_param_delta(vs, H, diag_shift=LAM, pii=False)
    np.testing.assert_allclose(dp, np.asarray(xi_ref), rtol=1e-6, atol=1e-8)


# --------------------------------------------------------------------------- #
# SPRING: dense update matches the Eq. 36 closed form  ξ = Q⁻¹(½∇E + λμ ξ_{k-1})
# --------------------------------------------------------------------------- #
def _fixed_OAdv(n_samples=64):
    log_psi, f_A, eloc, params, samples = fixed_inputs(n_samples=n_samples)
    jac_kw = dict(mode="real", dense=True, center=True)
    O = nk.jax.jacobian(log_psi, params, samples, {}, **jac_kw)
    A = nk.jax.jacobian(f_A, params, samples, {}, **jac_kw)
    O_L, A_L, dv = _prepare_input(O, A, eloc, mode="real", scaling_factor=1.0 / n_samples)
    ps = jax.tree_util.tree_map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), params)
    return O_L, A_L, dv, ps


def test_spring_dense_matches_eq36():
    O_L, A_L, dv, ps = _fixed_OAdv()
    grad = O_L.T @ dv
    S = O_L.T @ O_L
    Hm = O_L.T @ A_L
    tau, mu, lam = -5.0, 0.8, 1e-2
    P = O_L.shape[-1]
    xi_prev = jnp.asarray(np.random.default_rng(1).standard_normal(P))

    upd, _, _ = _compute_pii_update_dense(
        O_L, A_L, dv, tau=tau, diag_shift=lam, solver_fn=gen_solver, mode="real",
        momentum=mu, old_updates=xi_prev, params_structure=ps,
    )
    Q = Hm - tau * S + lam * jnp.eye(P, dtype=S.dtype)
    ref = jnp.linalg.solve(Q, 0.5 * grad + lam * mu * xi_prev)
    np.testing.assert_allclose(np.asarray(upd), np.asarray(ref), rtol=1e-6, atol=1e-8)


def test_spring_dense_equals_kernel_over_iterations():
    O_L, A_L, dv, ps = _fixed_OAdv()
    tau, mu, lam = -5.0, 0.8, 1e-2
    P = O_L.shape[-1]
    kw = dict(tau=tau, diag_shift=lam, solver_fn=gen_solver, mode="real",
              params_structure=ps)
    od = jnp.zeros(P)
    ok = jnp.zeros(P)
    for _ in range(3):
        d, od, _ = _compute_pii_update_dense(O_L, A_L, dv, momentum=mu, old_updates=od, **kw)
        k, ok, _ = _compute_minpii_update(O_L, A_L, dv, momentum=mu, old_updates=ok, **kw)
    np.testing.assert_allclose(np.asarray(d), np.asarray(k), rtol=1e-6, atol=1e-8)


def test_momentum_zero_equals_none():
    O_L, A_L, dv, ps = _fixed_OAdv()
    tau, lam = -5.0, 1e-2
    P = O_L.shape[-1]
    kw = dict(tau=tau, diag_shift=lam, solver_fn=gen_solver, mode="real",
              params_structure=ps)
    upd_none, _, _ = _compute_pii_update_dense(O_L, A_L, dv, momentum=None, old_updates=None, **kw)
    upd_zero, _, _ = _compute_pii_update_dense(O_L, A_L, dv, momentum=0.0, old_updates=jnp.zeros(P), **kw)
    np.testing.assert_allclose(np.asarray(upd_zero), np.asarray(upd_none), rtol=1e-7, atol=1e-9)
