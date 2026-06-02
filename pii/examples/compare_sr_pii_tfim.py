r"""Compare SR and PII on a small 2D transverse-field Ising model (TFIM).

At small ``h/J`` the TFIM is in the ordered phase where the finite-size spectral
gap ``Δ = E1 - E0`` is small. SR is then "gap-limited" and converges slowly
(rate ``ρ_SR ≈ 1 - ½ Δ/Γ``), whereas PII is gap-insensitive: with the shift
undershooting the ground state by half the gap, ``τ = E0 - ½Δ``, the rate is
``ρ_PII = |E0-τ|/|E1-τ| = 1/3`` independent of the gap, so it converges in a few
iterations with learning rate ``η = 1`` (paper Fig. 2).

This script exercises every variant of :class:`pii.VMC`:

- SR / minSR                        (``pii=False``)
- PII dense / minPII / on-the-fly   (``pii=True``) -- these are mathematically
  identical and produce the same trajectory (so they overlap on the plot)
- PII-SPRING                        (``pii=True, momentum``)
- PII dense on a FullSumState       (exact, no Monte Carlo noise)

Run with::

    conda activate pii && python pii/examples/compare_sr_pii_tfim.py
"""

from pathlib import Path

import numpy as np
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
      f"(SR diverges for η ≥ this; we use lr_sr below it). PII uses η = 1.")

# Each entry: a learning rate and the pii.VMC keyword arguments. SR needs a small
# learning rate; PII uses η = 1.

lr_sr = eta_sr_max
diag_shift_sr = 1e-4
# PII's natural learning rate is η = 1: the update ξ = Q⁻¹(½∇E) already *is* the
# (Galerkin-projected) inverse-iteration step (paper Eq. 5), so η=1 applies it
# exactly.
lr_pii = 1.0
diag_shift_pii = 1e-3

# For stability, we will halve both optimal learning rates.
lr_pii /= 2.0
lr_sr /= 2.0

runs = {
    "SR": (lr_sr, dict(diag_shift=diag_shift_sr, pii=False, use_ntk=False)),
    "minSR": (lr_sr, dict(diag_shift=diag_shift_sr, pii=False, use_ntk=True)),
    "PII dense": (lr_pii, dict(diag_shift=diag_shift_sr, pii=True, tau=tau, use_ntk=False)),
    "minPII": (
        lr_pii,
        dict(diag_shift=diag_shift_sr, pii=True, tau=tau, use_ntk=True, on_the_fly=False),
    ),
    "PII on-the-fly": (
        lr_pii,
        dict(diag_shift=diag_shift_sr, pii=True, tau=tau, use_ntk=True, on_the_fly=True),
    ),
    "PII-SPRING": (
        lr_pii,
        dict(
            diag_shift=diag_shift_sr,
            pii=True,
            tau=tau,
            use_ntk=True,
            on_the_fly=True,
            momentum=0.0,
        ),
    ),
    # FullSumState (exact, no Monte Carlo noise) -- dense PII only.
    "SR FullSum": (
        lr_sr,
        dict(diag_shift=diag_shift_sr, pii=False, use_ntk=False, fullsum=True),
    ),
    "PII FullSum": (
        lr_pii,
        dict(diag_shift=diag_shift_sr, pii=True, tau=tau, use_ntk=False, fullsum=True),
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
    kernel_init=_init, hidden_bias_init=_init, visible_bias_init=_init,
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
    driver = pii.VMC(H, opt, variational_state=vstate, mode="complex", **kw)
    log = nk.logging.RuntimeLog()
    driver.run(n_iter=N_ITER, out=log, show_progress=False)
    energies = np.asarray(log.data["Energy"].Mean).real
    results[label] = energies
    print(f"  {label:16s} final E = {energies[-1]:.5f}  rel.err = {abs((energies[-1] - E0) / E0):.2e}")

# Plot the best-so-far (running-minimum) relative error: this removes Monte Carlo
# jitter so the convergence *speed* of each method is legible. Every entry in
# `runs` is plotted automatically, so any extra experiment added above shows up
# here too. Convention: SR family dashed, PII family solid; exact (FullSum) runs
# get markers; the coinciding PII MC variants (dense/minPII/on-the-fly) overlap.
_palette = plt.cm.tab10.colors
colors = {label: _palette[i % len(_palette)] for i, label in enumerate(results)}

plt.figure(figsize=(8, 5))
for label, energies in results.items():
    kw = runs[label][1]
    is_pii = kw.get("pii", False)
    is_fullsum = kw.get("fullsum", False)
    rel = np.abs((energies - E0) / E0)
    # rel = np.minimum.accumulate(np.abs((energies - E0) / E0))  # best so far
    plt.semilogy(
        rel,
        label=label,
        color=colors[label],
        ls="-" if is_pii else "--",
        lw=2.2 if is_fullsum else 1.6,
        marker="o" if is_fullsum else None,
        markevery=max(N_ITER // 12, 1),
        ms=4,
        alpha=0.9,
    )
plt.xlabel("iteration")
plt.ylabel(r"best relative error $|(E - E_0)/E_0|$")
plt.title(f"SR vs PII — TFIM N={hi.size} spins, h/J={h}  (Δ={gap:.3f})")
plt.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig(HERE / "compare_sr_pii_tfim.png", dpi=150)
print(f"saved {HERE / 'compare_sr_pii_tfim.png'}")
