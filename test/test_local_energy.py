"""The differentiable local energy and the ``A``-function (Eq. 28)."""

import jax
import jax.numpy as jnp
import numpy as np

from pii._ngd.local_energy import make_local_energy_funs

from .common import tfim, make_fullsum


def test_local_energy_matches_expectation():
    """E_L averaged over the Born distribution equals ⟨H⟩ (FullSum, exact)."""
    _, hi, H, _ = tfim()
    vs = make_fullsum(hi, seed=0)
    f_EL, _ = make_local_energy_funs(vs, H)
    samples = hi.all_states()
    eloc = f_EL({"params": vs.parameters}, samples)
    pdf = vs.probability_distribution()
    e_mc = jnp.sum(pdf * eloc)
    np.testing.assert_allclose(e_mc.real, vs.expect(H).mean.real, rtol=1e-10)


def test_local_energy_derivative_finite_difference():
    """J_{E_L} from autodiff agrees with central finite differences."""
    _, hi, H, _ = tfim()
    vs = make_fullsum(hi, seed=0)
    f_EL, _ = make_local_energy_funs(vs, H)
    flat, unravel = jax.flatten_util.ravel_pytree(vs.parameters)
    x = hi.all_states()[3]

    def EL(flat_p):
        return f_EL({"params": unravel(flat_p)}, x[None])[0].real

    jac_ad = np.array(jax.grad(EL)(flat))
    eps = 1e-5
    jac_fd = np.zeros_like(np.array(flat))
    for i in range(len(flat)):
        jac_fd[i] = (EL(flat.at[i].add(eps)) - EL(flat.at[i].add(-eps))) / (2 * eps)
    np.testing.assert_allclose(jac_ad, jac_fd, atol=1e-6)


def test_fA_jacobian_equals_A_definition():
    """∂_θ f_A = ∂_θ E_L + E_L ∂_θ logψ  (the rows of the A matrix, Eq. 28)."""
    _, hi, H, _ = tfim()
    vs = make_fullsum(hi, seed=0)
    f_EL, f_A = make_local_energy_funs(vs, H)
    flat, unravel = jax.flatten_util.ravel_pytree(vs.parameters)
    x = hi.all_states()[2][None]

    def fA(flat_p):
        return f_A({"params": unravel(flat_p)}, x)[0]

    def fEL(flat_p):
        return f_EL({"params": unravel(flat_p)}, x)[0]

    def flogpsi(flat_p):
        return vs._apply_fun({"params": unravel(flat_p)}, x)[0]

    # real-valued model here, so grads are real.
    g_fA = np.array(jax.grad(lambda p: fA(p).real)(flat))
    g_EL = np.array(jax.grad(lambda p: fEL(p).real)(flat))
    eloc = float(fEL(flat).real)
    g_logpsi = np.array(jax.grad(lambda p: flogpsi(p).real)(flat))
    np.testing.assert_allclose(g_fA, g_EL + eloc * g_logpsi, atol=1e-8)
