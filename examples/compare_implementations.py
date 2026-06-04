r"""Consistency of PII's two entry points (and PII vs SR) — the NetKet-parallel API.

PII mirrors NetKet's architecture, so there are two equivalent ways to run it, exactly as
NetKet offers ``VMC`` + ``SR`` and the integrated ``VMC_SR``:

1. **Preconditioner + standard driver** — ``pii.driver.VMC`` (≡ ``netket.driver.VMC``) with
   ``pii.optimizer.PII`` as the ``preconditioner`` (the analogue of ``netket.optimizer.SR``).
   The ``Q``-matrix can be ``QJacobianDense`` or ``QJacobianPyTree``, and the linear solve can be
   the direct default or a **matrix-free** ``gmres`` (the analogue of ``SR(solver=cg)`` — there is
   no ``cg`` for PII because ``Q`` is non-symmetric/indefinite).
2. **Integrated driver** — ``pii.driver.VMC_PII`` (≡ ``netket.driver.VMC_SR``) with ``pii=True``.

This script runs all of them on the same small, exactly-solvable TFIM as ``compare_sr_pii_tfim.py``
(``FullSumState``, so every run is deterministic), from *identical* initial parameters, and
overlays the energy-error curves against both iteration count and wall time:

- the four **PII** curves (Dense / PyTree / matrix-free / integrated) lie on top of each other —
  same physical update, three storage/solve strategies plus the integrated kernel;
- the two **SR** curves (``VMC`` + ``netket.optimizer.SR`` and NetKet's integrated ``VMC_SR``)
  coincide;
- and PII (``η = 1``) converges where SR (``η < 1/Γ``) is slower — the point of the method.

Run with::

    python examples/compare_implementations.py
"""

import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from jax.nn.initializers import normal
import optax
import netket as nk
import scipy.sparse.linalg as sla
import matplotlib.pyplot as plt

import pii

HERE = Path(__file__).parent

L = 4
h = 2.0
N_ITER = 100

g = nk.graph.Chain(L, pbc=True)
hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes)
H = nk.operator.Ising(hilbert=hi, graph=g, h=h)

# Exact spectrum endpoints -> ground-state energy, gap, and spectral spread Γ (as in
# compare_sr_pii_tfim.py).
E0, E1 = nk.exact.lanczos_ed(H, k=2, compute_eigenvectors=False)
Emax = float(sla.eigsh(H.to_sparse(), k=1, which="LA", return_eigenvectors=False)[0])
gap = E1 - E0
Gamma = Emax - E0  # spectral spread
tau = E0 - 0.1 * gap  # undershoot by a fraction of the gap

# SR has a hard step-size threshold η < 1/Γ (paper Theorem 2); PII's natural rate is η = 1.
eta_sr_max = 1.0 / Gamma
lr_sr = eta_sr_max
lr_pii = 1.0
# For stability, halve both optimal learning rates (as in compare_sr_pii_tfim.py).
lr_sr /= 2.0
lr_pii /= 2.0
diag_shift_sr = 1e-4
diag_shift_pii = 1e-4

# Use the same solver for both SR paths: NetKet's SR defaults to `cg`, but VMC_SR to
# `cholesky_with_fallback` — match them so the two SR curves coincide.
sr_solver = nk.optimizer.solver.cholesky_with_fallback

print(f"TFIM chain L={L}, h/J={h};  E0 = {E0:.6f}, gap Δ = {gap:.4f}, "
      f"spread Γ = {Gamma:.4f},  τ = {tau:.6f}")

# One shared model + one set of starting parameters used by *every* method, so the comparison
# starts from an identical state (a larger init stddev than the default avoids the uniform-state
# transient; see compare_sr_pii_tfim.py). mode is auto-detected ('complex' for RBMRealParams).
model = pii.models.RBMRealParams(alpha=1, param_dtype=jnp.float64, kernel_init=normal(stddev=0.3))
init_params = nk.vqs.FullSumState(hi, model, seed=0).parameters


# --- builders: each returns a configured driver for a given variational state ---------------
def vmc_pii(vstate, q=pii.optimizer.q.QJacobianDense, solver=None):
    """`pii.driver.VMC` + `pii.optimizer.PII` (the preconditioner route)."""
    kw = {} if solver is None else {"solver": solver}
    return pii.driver.VMC(
        H, optax.sgd(lr_pii), variational_state=vstate,
        preconditioner=pii.optimizer.PII(H, q=q, tau=tau, diag_shift=diag_shift_pii, **kw),
    )


def vmc_pii_integrated(vstate):
    """`pii.driver.VMC_PII` (the integrated driver)."""
    return pii.driver.VMC_PII(
        H, optax.sgd(lr_pii), variational_state=vstate,
        diag_shift=diag_shift_pii, pii=True, tau=tau, use_ntk=False,
    )


def vmc_sr(vstate):
    """`pii.driver.VMC` + `netket.optimizer.SR` (the preconditioner route, SR)."""
    return pii.driver.VMC(
        H, optax.sgd(lr_sr), variational_state=vstate,
        preconditioner=nk.optimizer.SR(
            qgt=nk.optimizer.qgt.QGTJacobianDense, solver=sr_solver, diag_shift=diag_shift_sr
        ),
    )


def vmc_sr_integrated(vstate):
    """NetKet's integrated `VMC_SR`."""
    return nk.driver.VMC_SR(
        H, optax.sgd(lr_sr), variational_state=vstate,
        diag_shift=diag_shift_sr, use_ntk=False, linear_solver=sr_solver,
    )


# label -> builder(vstate) -> driver.  PII family first (solid), SR family second (dashed).
runs = {
    "PII  VMC + PII(QJacobianDense)":     vmc_pii,
    "PII  VMC + PII(QJacobianPyTree)":    lambda vs: vmc_pii(vs, q=pii.optimizer.q.QJacobianPyTree),
    "PII  VMC + PII(gmres, matrix-free)": lambda vs: vmc_pii(
        vs, solver=pii.optimizer.solver.gmres(tol=1e-10, restart=200, maxiter=4)),
    "PII  VMC_PII (integrated)":          vmc_pii_integrated,
    "SR   VMC + nk.optimizer.SR":         vmc_sr,
    "SR   VMC_SR (NetKet, integrated)":   vmc_sr_integrated,
}

results = {}
for label, build in runs.items():
    vstate = nk.vqs.FullSumState(hi, model, seed=0)
    vstate.parameters = init_params  # identical starting parameters for every method
    driver = build(vstate)
    log = nk.logging.RuntimeLog()
    # warm up to trigger JIT compilation (not timed, not logged), then reset to the shared
    # start so the comparison still begins from identical parameters.
    driver.run(n_iter=1, show_progress=False)
    driver.state.parameters = init_params
    jax.block_until_ready(driver.state.parameters)

    t0 = time.perf_counter()
    driver.run(n_iter=N_ITER, out=log, show_progress=False)
    jax.block_until_ready(driver.state.parameters)
    total = time.perf_counter() - t0

    energies = np.asarray(log.data["Energy"].Mean).real  # N_ITER points
    # per-iteration wall time from the single accurate total (steady state ⇒ ~uniform per step).
    walltime = np.linspace(total / N_ITER, total, N_ITER)
    results[label] = (energies, walltime)
    print(f"  {label:34s} final rel.err = {abs((energies[-1] - E0) / E0):.2e}  ({total:.2f}s)")

# Consistency: the four PII curves coincide (same physical update); the two SR curves coincide.
pii_curves = [e for k, (e, _) in results.items() if k.startswith("PII")]
sr_curves = [e for k, (e, _) in results.items() if k.startswith("SR")]
for e in pii_curves[1:]:
    np.testing.assert_allclose(e, pii_curves[0], rtol=1e-4, atol=1e-7)
for e in sr_curves[1:]:
    np.testing.assert_allclose(e, sr_curves[0], rtol=1e-4, atol=1e-7)
print("OK: the 4 PII paths coincide, and the 2 SR paths coincide.")

# Plot the relative energy error against two x-axes (iteration and wall time). Convention:
# PII family solid, SR family dashed; the coinciding curves overlap, so the *integrated*
# reference drivers (pii.driver.VMC_PII and NetKet's nk.driver.VMC_SR) get markers — you can
# see them sit exactly on top of the preconditioner-route curves.
_palette = plt.cm.tab10.colors
colors = {label: _palette[i % len(_palette)] for i, label in enumerate(results)}


def plot_convergence(x_of, xlabel, fname):
    plt.figure(figsize=(8, 5))
    for label, (energies, walltime) in results.items():
        is_integrated = "integrated" in label  # the all-in-one reference drivers (incl. NetKet's)
        rel = np.maximum(np.abs((energies - E0) / E0), 1e-16)  # floor: rel==0 is −∞ on a log axis
        plt.semilogy(x_of(energies, walltime), rel, label=label, color=colors[label],
                     ls="-" if label.startswith("PII") else "--", lw=2,
                     marker="o" if is_integrated else None,
                     markevery=max(len(rel) // 10, 1), ms=8, markeredgecolor="black",
                     markeredgewidth=1.4)
    plt.xlabel(xlabel)
    plt.ylabel(r"relative error $|(E - E_0)/E_0|$")
    plt.title(f"PII vs SR — two entry points, TFIM N={hi.size} spins, h/J={h}  (Δ={gap:.3f})")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(HERE / fname, dpi=150)
    print(f"saved {HERE / fname}")


plot_convergence(lambda e, t: np.arange(len(e)), "iteration", "compare_implementations.png")
plot_convergence(lambda e, t: t, "wall time (s)", "compare_implementations_walltime.png")
