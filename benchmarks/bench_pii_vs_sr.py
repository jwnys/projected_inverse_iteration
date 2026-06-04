r"""Benchmark: how much slower is one PII step than one SR step, and why?

Expectation (user): PII should be only a small constant (~3-5×) slower than SR per step. SR needs
local-energy **values** ``E_L`` (forward passes over the ``n_conn`` connected configs); PII
additionally needs local-energy **gradients** (the ``A``-Jacobian of ``f_A = E_L + sg(E_L)·logψ``).
A backward pass is ~3-4× a forward pass, and **both** methods already pay the ``n_conn`` factor, so
PII's *inherent* cost should be ~3-5× SR — not the ~400× seen in the wall-time example.

This script (STEP 1 — measurement only, it changes nothing in ``pii/``) separates the **inherent
compute** from a **per-step recompilation artifact**, by timing the jitted kernels directly:

- ``t_fwd``  : a forward pass ``apply(params, samples)``;
- ``t_bwd``  : one O-Jacobian ``nkjax.jacobian(apply, ...)``  → report ``t_bwd/t_fwd`` (model reference);
- ``t_EL``   : local energies ``f_EL(...)``;
- ``t_A``    : the A-Jacobian ``nkjax.jacobian(f_A, ...)``     → report ``t_A/t_EL``;
- ``t_SR``       : one SR kernel step ``netket ...sr(apply, ...)``  (``apply`` is a *stable* object);
- ``t_PII_steady``: one PII kernel step ``_pii_common(apply, f_A, ...)`` reusing the **same** ``f_A``
  object (jit cache hit) — the **inherent** PII step cost;
- ``t_PII_fresh`` : one PII kernel step building a **fresh** ``f_A`` each call (what the driver does
  today at ``vmc_pii.py:283``) — exposes the recompilation.

Key ratios printed: ``t_PII_steady/t_SR`` (should be ~3-5×, the real answer) vs ``t_PII_fresh/t_SR``
(inflated) and ``t_PII_fresh/t_PII_steady`` (the recompile waste). Run::

    python benchmarks/bench_pii_vs_sr.py
"""

import time
from pathlib import Path

import numpy as np
import jax
import netket as nk
import netket.jax as nkjax
import matplotlib.pyplot as plt

from netket._src.ngd.sr_srt_common import sr
from netket.optimizer.solver import cholesky_with_fallback

from pii.ngd.common import _pii_common, get_samples_and_pdf
from pii.ngd.local_energy import make_local_energy_funs
from pii.optimizer.solver import pii_default_solver

HERE = Path(__file__).parent

N_SAMPLES = 1024
ALPHA = 1
MODE = "real"          # real RBM → real log-amplitude
DIAG_SHIFT = 0.01
TAU = -1.0             # timing only; value irrelevant
SIZES = [6, 10, 14]    # chain length L


def median_ms(thunk, reps, warmup):
    """Median wall time (ms) of `thunk()` over `reps`, after `warmup` calls. `thunk` may build a
    fresh object each call (to measure recompilation) or reuse a stable one (steady state)."""
    for _ in range(warmup):
        jax.block_until_ready(thunk())
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(thunk())
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1e3


def bench_size(L):
    g = nk.graph.Chain(L, pbc=True)
    hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes)
    H = nk.operator.Ising(hilbert=hi, graph=g, h=2.0)
    vs = nk.vqs.MCState(
        nk.sampler.MetropolisLocal(hi, n_chains=16), nk.models.RBM(alpha=ALPHA, param_dtype=float),
        n_samples=N_SAMPLES, seed=0,
    )
    vs.sample()
    apply_fun, params, model_state = vs._apply_fun, vs.parameters, vs.model_state
    samples, pdf = get_samples_and_pdf(vs)
    f_EL, f_A = make_local_energy_funs(vs, H)
    eloc = f_EL(vs.variables, samples)

    jac = lambda f: nkjax.jacobian(  # noqa: E731
        f, params, samples, model_state, mode=MODE, dense=True, center=True, pdf=pdf
    )
    sr_step = lambda: sr(  # noqa: E731
        apply_fun, eloc, params, model_state, samples,
        diag_shift=DIAG_SHIFT, solver_fn=cholesky_with_fallback, mode=MODE, weights=pdf,
    )
    pii_step = lambda fA: _pii_common(  # noqa: E731
        apply_fun, fA, eloc, params, model_state, samples,
        tau=TAU, diag_shift=DIAG_SHIFT, solver_fn=pii_default_solver, mode=MODE,
        use_ntk=False, weights=pdf,
    )

    # jit the forward / E_L so they are apples-to-apples with the (internally jitted) jacobians
    # — otherwise un-jitted Python dispatch dominates these cheap ops and the ratio is meaningless.
    jit_apply, jit_fEL = jax.jit(apply_fun), jax.jit(f_EL)
    t_fwd = median_ms(lambda: jit_apply(vs.variables, samples), reps=30, warmup=3)
    t_bwd = median_ms(lambda: jac(apply_fun), reps=20, warmup=3)
    t_EL = median_ms(lambda: jit_fEL(vs.variables, samples), reps=30, warmup=3)
    t_A = median_ms(lambda: jac(f_A), reps=20, warmup=3)               # stable f_A → steady
    t_SR = median_ms(sr_step, reps=20, warmup=3)
    t_PII_steady = median_ms(lambda: pii_step(f_A), reps=20, warmup=3)  # same f_A → cache hit
    t_PII_fresh = median_ms(                                            # fresh f_A → recompiles
        lambda: pii_step(make_local_energy_funs(vs, H)[1]), reps=8, warmup=1
    )
    return dict(
        L=L, P=int(vs.n_parameters),
        t_fwd=t_fwd, bwd_fwd=t_bwd / t_fwd, t_EL=t_EL, A_EL=t_A / t_EL,
        t_SR=t_SR, t_PII_steady=t_PII_steady, steady_SR=t_PII_steady / t_SR,
        t_PII_fresh=t_PII_fresh, fresh_SR=t_PII_fresh / t_SR,
        fresh_steady=t_PII_fresh / t_PII_steady,
    )


rows = []
for L in SIZES:
    r = bench_size(L)
    rows.append(r)
    print(
        f"L={r['L']:2d} P={r['P']:4d} | "
        f"bwd/fwd={r['bwd_fwd']:4.1f}  A/EL={r['A_EL']:4.1f} | "
        f"SR={r['t_SR']:6.1f}ms  PII_steady={r['t_PII_steady']:7.1f}ms "
        f"(x{r['steady_SR']:4.1f}) | PII_fresh={r['t_PII_fresh']:8.1f}ms "
        f"(x{r['fresh_SR']:6.1f})  recompile_waste=x{r['fresh_steady']:6.1f}"
    )

print(
    "\nReading: PII_steady/SR is the *inherent* PII cost (reuse f_A); PII_fresh/SR is what the driver "
    "shows today (fresh f_A each step). A large fresh/steady ⇒ per-step recompilation."
)

# Plot the ratios vs n_params.
P = [r["P"] for r in rows]
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(P, [r["bwd_fwd"] for r in rows], "o--", color="gray", label="model: backward/forward (ref)")
ax.plot(P, [r["steady_SR"] for r in rows], "o-", color="C0", label="PII_steady / SR  (inherent)")
ax.plot(P, [r["fresh_SR"] for r in rows], "s-", color="C3", label="PII_fresh / SR  (driver today)")
ax.set_yscale("log")
ax.set_xlabel("n_parameters")
ax.set_ylabel("per-step time ratio vs SR")
ax.set_title(f"PII vs SR per-step cost (TFIM chain, {N_SAMPLES} samples, RBM α={ALPHA})")
ax.legend(fontsize=9)
fig.tight_layout()
out = HERE / "bench_pii_vs_sr.png"
fig.savefig(out, dpi=150)
print(f"saved {out}")
