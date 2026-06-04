r"""Compare **unsymmetrized vs symmetrized PII** on a small TFIM (Monte Carlo + FullSum).

Same exactly-solvable periodic TFIM chain, model, starting parameters and PII
hyperparameters as :mod:`pii.examples.compare_sr_pii_tfim`, so the only thing that
changes between the two PII variants is the *linear solver*:

- **PII** (unsymmetrized) solves the (generally non-symmetric) dense system
  ``Q ξ = ½∇E`` with ``Q = H − τS + λ I`` (``λ = diag_shift_pii``), via a general
  LU solve.
- **PII symmetrized** solves the Tikhonov-regularized normal equations
  ``ξ = (QᴴQ + λ·I)⁻¹ Qᴴ (½∇E)`` with the *unregularized* ``Q = H − τS`` — i.e.
  the regularized pseudo-inverse of ``Q``. This requires the driver ``diag_shift=0``
  (the regularization is carried by the solver's ``diag_shift``). See
  :func:`pii.optimizer.solver.penrose_symmetrized_solver`. Note that ``diag_shift_pii_symm`` has **energy²
  units** and — because forming ``QᴴQ`` squares the (large) condition number of the
  rank-deficient ``Q`` — empirically needs to be ``O(0.1–1)`` here, *far* larger than
  ``diag_shift_pii`` (the energy-unit heuristic gives only the dimension, not the
  magnitude).

FullSum (noise-free) references for both variants separate "does symmetrization help
under Monte Carlo noise" from the deterministic limit (where both share the same
fixed point as ``reg → 0``).

Run with::

    python examples/compare_pii_symmetrized_tfim.py
"""

import time
from functools import partial
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from jax.nn.initializers import normal
import optax
import netket as nk
import matplotlib.pyplot as plt

import pii

HERE = Path(__file__).parent

L = 4
h = 2.0
N_ITER = 100

g = nk.graph.Chain(L, pbc=True)
hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes)
H = nk.operator.Ising(hilbert=hi, graph=g, h=h)

# Exact spectrum endpoints -> ground-state energy, gap, and spectral spread Γ.
import scipy.sparse.linalg as sla

E0, E1 = nk.exact.lanczos_ed(H, k=2, compute_eigenvectors=False)
Emax = float(sla.eigsh(H.to_sparse(), k=1, which="LA", return_eigenvectors=False)[0])
gap = E1 - E0
Gamma = Emax - E0  # spectral spread
tau = E0 - 0.1 * gap  # undershoot by fraction of the gap

# PII's natural learning rate is η = 1; halved for extra Monte Carlo stability
# (identical to compare_sr_pii_tfim.py).
lr_pii = 1.0 / 2.0
diag_shift_pii = 1e-4  # energy units; regularizes Q in the unsymmetrized variant.

# Symmetrized regularization for (QᵀQ + λ·I): ENERGY² units (Q has energy units),
# so this is NOT comparable to diag_shift_pii and is tuned on its own scale. Forming
# QᵀQ squares the condition number of Q, so this typically needs to be much larger
# than diag_shift_pii.
diag_shift_pii_symm = diag_shift_pii**2
# diag_shift_pii_symm = 1e-2 #3e-1

print(f"TFIM {g.extent}, h/J={h};  E0 = {E0:.6f}, gap Δ = {gap:.4f}, "
      f"spread Γ = {Gamma:.4f},  τ = {tau:.6f}")
print(f"PII η = {lr_pii};  diag_shift (unsym) = {diag_shift_pii};  "
      f"diag_shift_symm = {diag_shift_pii_symm}")

# The symmetrized solver is bound once (partial on the shift) and shared by the MC
# and FullSum runs. NOTE: the driver diag_shift must be 0 for these runs (the
# regularization is carried by the solver's diag_shift here).
# note: the default solver is now Cholesky!
sym_solver = partial(pii.optimizer.solver.penrose_symmetrized_solver, diag_shift=diag_shift_pii_symm)
# sym_solver = partial(pii.optimizer.solver.naive_symmetrized_solver, diag_shift=diag_shift_pii)

runs = {
    "PII symmetrized FullSum": (
        lr_pii,
        dict(diag_shift=0.0, pii=True, tau=tau, use_ntk=False, linear_solver=sym_solver, fullsum=True),
    ),
    "PII symmetrized": (
        lr_pii,
        dict(diag_shift=0.0, pii=True, tau=tau, use_ntk=False, linear_solver=sym_solver),
    ),
    "minPII symmetrized": (
        lr_pii,
        dict(diag_shift=0.0, pii=True, tau=tau, use_ntk=True, linear_solver=sym_solver),
    ),
    "PII FullSum": (
        lr_pii,
        dict(diag_shift=diag_shift_pii, pii=True, tau=tau, use_ntk=False, fullsum=True),
    ),
    "PII": (lr_pii, dict(diag_shift=diag_shift_pii, pii=True, tau=tau, use_ntk=False)),
}

# One shared model + identical starting parameters used by *every* method (same init
# as compare_sr_pii_tfim.py: stddev 0.3 to avoid the special |+x⟩^N low-energy start).
_init = normal(stddev=0.3)
model = pii.models.RBMRealParams(
    alpha=1, param_dtype=jnp.float64,
    kernel_init=_init, # hidden_bias_init=_init, visible_bias_init=_init,
)
# model = pii.models.LogStateVectorRealParams(
#     hi, param_dtype=jnp.float64,
# )
print("Model = ", model)
init_params = nk.vqs.FullSumState(hi, model, seed=0).parameters

results = {}
for label, (lr, kw) in runs.items():
    kw = dict(kw)
    if kw.pop("fullsum", False):
        vstate = nk.vqs.FullSumState(hi, model, seed=0)
    else:
        sampler = nk.sampler.MetropolisLocal(hi, n_chains=512, sweep_size=hi.size * 3)
        vstate = nk.vqs.MCState(
            sampler, model, n_samples=1024, seed=0, n_discard_per_chain=4
        )
    vstate.parameters = init_params  # identical starting parameters for every method
    opt = optax.sgd(lr)
    # RBMRealParams has a complex log-amplitude (real params), so use mode="complex".
    driver = pii.driver.VMC_PII(H, opt, variational_state=vstate, **kw)
    log = nk.logging.RuntimeLog()
    # warm up to trigger JIT compilation (not timed, not logged), then reset to the
    # shared start so the comparison still begins from identical parameters.
    driver.run(n_iter=1, show_progress=False)
    driver.state.parameters = init_params
    jax.block_until_ready(driver.state.parameters)

    t0 = time.perf_counter()
    driver.run(n_iter=N_ITER, out=log, show_progress=True)
    jax.block_until_ready(driver.state.parameters)
    total = time.perf_counter() - t0

    energies = np.asarray(log.data["Energy"].Mean).real  # N_ITER points
    # per-iteration wall time from the single accurate total (steady state ⇒ ~uniform
    # per step); avoids the per-step block_until_ready sync overhead.
    walltime = np.linspace(total / N_ITER, total, N_ITER)
    results[label] = (energies, walltime)
    print(f"  {label:26s} final E = {energies[-1]:.5f}  rel.err = "
          f"{abs((energies[-1] - E0) / E0):.2e}  ({total:.2f}s)")

# Plot the relative energy error for every entry in `runs`. Convention: unsymmetrized
# dashed, symmetrized solid; exact (FullSum) runs get markers.
_palette = plt.cm.tab10.colors
colors = {label: _palette[i % len(_palette)] for i, label in enumerate(results)}


def plot_convergence(x_of, xlabel, fname):
    plt.figure(figsize=(8, 5))
    for label, (energies, walltime) in results.items():
        kw = runs[label][1]
        is_sym = "linear_solver" in kw
        is_fullsum = kw.get("fullsum", False)
        rel = np.abs((energies - E0) / E0)
        rel = np.maximum(rel, 1e-16)  # floor: rel==0 is −∞ on a log axis (would vanish)
        plt.semilogy(
            x_of(energies, walltime), rel,
            label=label, color=colors[label],
            ls="-" if is_sym else "--",
            lw=2.2 if is_fullsum else 1.6,
            marker="o" if is_fullsum else None,
            markevery=max(len(rel) // 12, 1), ms=4, alpha=0.9,
        )
    plt.xlabel(xlabel)
    plt.ylabel(r"relative error $|(E - E_0)/E_0|$")
    plt.title(f"PII: symmetrized vs unsymmetrized — TFIM N={hi.size} spins, "
              f"h/J={h}  (Δ={gap:.3f})")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(HERE / fname, dpi=150)
    print(f"saved {HERE / fname}")


plot_convergence(lambda e, t: np.arange(len(e)), "iteration",
                 "compare_pii_symmetrized_tfim.png")
plot_convergence(lambda e, t: t, "wall time (s)",
                 "compare_pii_symmetrized_tfim_walltime.png")
