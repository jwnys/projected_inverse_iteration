"""Shared fixtures/helpers for the PII test suite."""

import numpy as np
import jax
import jax.numpy as jnp
import netket as nk


def tfim(L1: int = 2, L2: int = 3, h: float = 2.0):
    """Small 2D transverse-field Ising model and its exact ground-state energy."""
    g = nk.graph.Grid([L1, L2], pbc=True)
    hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes)
    H = nk.operator.Ising(hilbert=hi, graph=g, h=h)
    E0 = float(np.linalg.eigvalsh(H.to_dense())[0])
    return g, hi, H, E0


def make_mcstate(hi, *, alpha: int = 2, n_samples: int = 512, seed: int = 0):
    ma = nk.models.RBM(alpha=alpha, param_dtype=float)
    sa = nk.sampler.MetropolisLocal(hi, n_chains=16)
    return nk.vqs.MCState(sa, ma, n_samples=n_samples, seed=seed)


def make_fullsum(hi, *, alpha: int = 2, seed: int = 0):
    ma = nk.models.RBM(alpha=alpha, param_dtype=float)
    return nk.vqs.FullSumState(hi, ma, seed=seed)


def rel_error(vstate, H, E0) -> float:
    return abs((float(vstate.expect(H).mean.real) - E0) / E0)


def gen_solver(A, b):
    """General (non-PSD) solver, same as the PII default."""
    return jax.scipy.linalg.solve(A, b, assume_a="gen"), None


def fixed_inputs(n_samples: int = 128, seed: int = 0):
    """Deterministic (sampler-free) inputs for unit-level update comparisons.

    Returns ``(log_psi, f_A, local_energies, params, samples)`` so that a single
    PII update can be computed directly via the ``_ngd`` kernels, independent of
    sampling RNG, device count or chunking.
    """
    from pii.ngd.local_energy import make_local_energy_funs

    _, hi, H, _ = tfim()
    ma = nk.models.RBM(alpha=2, param_dtype=float)
    vs = nk.vqs.MCState(
        nk.sampler.MetropolisLocal(hi, n_chains=16), ma, n_samples=n_samples, seed=seed
    )
    rng = np.random.default_rng(seed)
    samples = jnp.asarray(rng.choice([-1.0, 1.0], size=(n_samples, hi.size)))
    f_EL, f_A = make_local_energy_funs(vs, H)
    eloc = f_EL(vs.variables, samples)
    return vs._apply_fun, f_A, eloc, vs.parameters, samples
