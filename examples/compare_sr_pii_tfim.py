r"""Compare SR and PII on a small, exactly-solvable transverse-field Ising model.

A periodic TFIM chain of ``L`` spins (``Ĥ = −J Σ σᶻσᶻ − h Σ σˣ``), small enough to
diagonalize exactly, so the relative energy error ``|(E − E0)/E0|`` against the
true ground state can be tracked every iteration.

All methods start from *identical* parameters (see ``init_params``) for a fair
comparison, and the script exercises every variant of :class:`pii.driver.VMC_PII`:

- **SR / minSR**            (``pii=False``) -- Stochastic Reconfiguration; its step
  size is capped by the critical value ``η < 1/Γ`` (paper Theorem 2).
- **PII dense / minPII / minPII on-the-fly** (``pii=True``) -- Projected Inverse
  Iteration at its natural ``η = 1``. The three are mathematically identical and
  overlap.
- **minPII-SPRING**         (``pii=True, momentum``).
- **SR FullSum / PII FullSum** -- exact (no Monte Carlo noise) references.

With Monte Carlo, PII drives the energy to ~machine precision while SR is limited
to a higher (sampling-noise) floor; the FullSum runs confirm both are exact in the
noise-free limit. This system is tiny, so it demonstrates correctness and the
SR-vs-PII contrast rather than the large-system, gap-closing regime of the paper.

Run with::

    python examples/compare_sr_pii_tfim.py
"""

import time
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

# g = nk.graph.Grid([L, L], pbc=True)
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

# SR has a hard step-size threshold η < 1/Γ = 1/(Emax−E0): for η ≥ 1/Γ it diverges
# (paper Theorem 2). PII has no such limit and uses η = 1.
eta_sr_max = 1.0 / Gamma
print(f"TFIM {g.extent}, h/J={h};  E0 = {E0:.6f}, gap Δ = {gap:.4f}, "
      f"spread Γ = {Gamma:.4f},  τ = {tau:.6f}")
print(f"SR critical learning rate  η_max = 1/Γ = {eta_sr_max:.4g}  "
      f"(SR diverges for η ≥ this; PII's natural rate is η = 1).")

# Learning rates: SR's stable range is η < 1/Γ (= eta_sr_max); PII's natural rate
# is η = 1. Both are halved below for extra Monte Carlo stability.
lr_sr = eta_sr_max
diag_shift_sr = 1e-4
# PII's natural learning rate is η = 1: the update ξ = Q⁻¹(½∇E) already *is* the
# (Galerkin-projected) inverse-iteration step (paper Eq. 5), so η=1 applies it
# exactly.
lr_pii = 1.0
diag_shift_pii = 1e-4

# For stability, we will halve both optimal learning rates.
lr_sr /= 2.0
lr_pii /= 2.0
# with SPRING, the learning rate must typically be reduced (also the case for SR-SPRING)
lr_spring_pii = lr_pii / 4.0 

runs = {
    # FullSumState (exact, no Monte Carlo noise) -- dense PII only.
    "SR FullSum": (
        lr_sr,
        dict(diag_shift=diag_shift_sr, pii=False, use_ntk=False, fullsum=True),
    ),
    "SR": (lr_sr, dict(diag_shift=diag_shift_sr, pii=False, use_ntk=False)),
    "minSR": (lr_sr, dict(diag_shift=diag_shift_sr, pii=False, use_ntk=True)),
    # FullSumState (exact, no Monte Carlo noise) -- dense PII only.
    "PII FullSum": (
        lr_pii,
        dict(diag_shift=diag_shift_pii, pii=True, tau=tau, use_ntk=False, fullsum=True),
    ),
    "PII": (lr_pii, dict(diag_shift=diag_shift_pii, pii=True, tau=tau, use_ntk=False)),
    "minPII": (
        lr_pii,
        dict(diag_shift=diag_shift_pii, pii=True, tau=tau, use_ntk=True, on_the_fly=False),
    ),
    "minPII on-the-fly": (
        lr_pii,
        dict(diag_shift=diag_shift_pii, pii=True, tau=tau, use_ntk=True, on_the_fly=True),
    ),
    # Combinations with SPRING
    "PII-SPRING": (
        lr_spring_pii,
        dict(
            diag_shift=diag_shift_pii,
            pii=True,
            tau=tau,
            use_ntk=False,
            momentum=0.8,
        ),
    ),
    "minPII-SPRING": (
        lr_spring_pii,
        dict(
            diag_shift=diag_shift_pii,
            pii=True,
            tau=tau,
            use_ntk=True,
            on_the_fly=True,
            momentum=0.8,
        ),
    ),
}

# One shared model + one set of starting parameters used by *every* method, so the
# comparison starts from an identical state. We use a larger init stddev (0.3) than
# the default (0.01): the tiny default puts logψ≈0, i.e. ψ ≈ the uniform |+x⟩^N
# state, which for this small TFIM is a special low-energy point the optimizer must
# climb out of (the transient energy "bump"). A generic start removes that.
# model = nk.models.RBM(alpha=1, param_dtype=complex)
_init = normal(stddev=0.3)
model = pii.models.RBMRealParams(
    alpha=1, param_dtype=jnp.float64,
    kernel_init=_init, #hidden_bias_init=_init, visible_bias_init=_init,
)
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
    # (mode="real" would truncate the phase and is only for real-output ansätze.)
    driver = pii.driver.VMC_PII(H, opt, variational_state=vstate, mode="complex", **kw)
    log = nk.logging.RuntimeLog()
    # warm up to trigger JIT compilation (not timed, not logged), then reset to the
    # shared start so the comparison still begins from identical parameters.
    driver.run(n_iter=1, show_progress=False)
    driver.state.parameters = init_params
    jax.block_until_ready(driver.state.parameters)

    t0 = time.perf_counter()
    driver.run(n_iter=N_ITER, out=log, show_progress=False)
    jax.block_until_ready(driver.state.parameters)
    total = time.perf_counter() - t0

    energies = np.asarray(log.data["Energy"].Mean).real  # N_ITER points
    # per-iteration wall time from the single accurate total (steady state ⇒ ~uniform
    # per step); avoids the per-step block_until_ready sync overhead.
    walltime = np.linspace(total / N_ITER, total, N_ITER)
    results[label] = (energies, walltime)
    print(f"  {label:18s} final E = {energies[-1]:.5f}  rel.err = "
          f"{abs((energies[-1] - E0) / E0):.2e}  ({total:.2f}s)")

# Plot the relative energy error for every entry in `runs` (so any experiment added
# above is plotted automatically), against two x-axes: iteration count and cumulative
# wall time. Convention: SR family dashed, PII family solid; exact (FullSum) runs get
# markers; the coinciding PII MC variants overlap.
_palette = plt.cm.tab10.colors
colors = {label: _palette[i % len(_palette)] for i, label in enumerate(results)}


def plot_convergence(x_of, xlabel, fname):
    plt.figure(figsize=(8, 5))
    for label, (energies, walltime) in results.items():
        kw = runs[label][1]
        is_pii = kw.get("pii", False)
        is_fullsum = kw.get("fullsum", False)
        rel = np.abs((energies - E0) / E0)
        # rel = np.minimum.accumulate(rel)  # uncomment for best-so-far error
        rel = np.maximum(rel, 1e-16)  # floor: rel==0 is −∞ on a log axis (would vanish)
        plt.semilogy(
            x_of(energies, walltime), rel,
            label=label, color=colors[label],
            ls="-" if is_pii else "--",
            lw=2.2 if is_fullsum else 1.6,
            marker="o" if is_fullsum else None,
            markevery=max(len(rel) // 12, 1), ms=4, alpha=0.9,
        )
    plt.xlabel(xlabel)
    plt.ylabel(r"relative error $|(E - E_0)/E_0|$")
    plt.title(f"SR vs PII — TFIM N={hi.size} spins, h/J={h}  (Δ={gap:.3f})")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(HERE / fname, dpi=150)
    print(f"saved {HERE / fname}")


plot_convergence(lambda e, t: np.arange(len(e)), "iteration", "compare_sr_pii_tfim.png")
plot_convergence(lambda e, t: t, "wall time (s)", "compare_sr_pii_tfim_walltime.png")
