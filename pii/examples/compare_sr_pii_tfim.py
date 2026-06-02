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

# Exact two lowest eigenvalues -> ground-state energy and spectral gap.
E0, E1 = nk.exact.lanczos_ed(H, k=2, compute_eigenvectors=False)
gap = E1 - E0
tau = E0 - 0.1 * gap  # undershoot by fraction of the gap
print(f"TFIM {g.extent}, h/J={h};  E0 = {E0:.6f}, gap Δ = {gap:.4f},  τ = {tau:.6f}")

# Each entry: a learning rate and the pii.VMC keyword arguments. SR needs a small
# learning rate; PII uses η = 1.

lr_sr = 2e-2
diag_shift_sr = 1e-4
lr_pii = 0.5
diag_shift_pii = 1e-2

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

results = {}
for label, (lr, kw) in runs.items():
    kw = dict(kw)
    # model = nk.models.RBM(alpha=1, param_dtype=complex)
    model = pii.models.RBMRealParams(alpha=1, param_dtype=jnp.float64)
    if kw.pop("fullsum", False):
        vstate = nk.vqs.FullSumState(hi, model, seed=0)
    else:
        sampler = nk.sampler.MetropolisLocal(hi, n_chains=512, sweep_size=hi.size*3)
        vstate = nk.vqs.MCState(
            sampler,
            model, 
            n_samples=1024, 
            seed=0,
            n_discard_per_chain=4
        )
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
    rel = np.minimum.accumulate(np.abs((energies - E0) / E0))  # best so far
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
