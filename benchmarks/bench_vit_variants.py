r"""Per-VMC-step cost of SR vs PII across *all* variants, on a ViT ansatz of varying depth.

The question: how much more expensive is one PII update step than one SR step, for a realistic deep
ansatz, across every implementation form (dense, kernel/NTK = minSR/minPII, and matrix-free
on-the-fly)? PII's only extra ingredient is the local-energy-gradient ``A``-Jacobian, so the cost
should be a small constant (around 4× or less) once the per-step recompilation is avoided (``f_A`` is
a ``HashablePartial`` shared by all PII paths).

Setup: a small 2D square-lattice Heisenberg model (the physics is incidental here, only the ansatz
matters) with a Vision-Transformer wavefunction (:class:`nkpii.models.ViT`, real params / complex
output, architecture after Viteritti-Rende-Becca, PRL 130, 236401). The number of transformer layers
is swept, and one update step is timed per variant: ``driver.compute_loss_and_update`` on a fixed set
of samples (sampling is common to SR and PII, so excluding it isolates the SR-vs-PII difference).

Run::

    python benchmarks/bench_vit_variants.py
"""

import time
from pathlib import Path

import numpy as np
import jax
import optax
import netket as nk
import matplotlib.pyplot as plt

import nkpii

HERE = Path(__file__).parent

# --- system (2D square-lattice Heisenberg; physics incidental, only the ansatz matters) ---
L = 4
N_SAMPLES = 512
NUM_LAYERS = [1, 2, 3]
# ViT hyperparameters (small — "a few layers", not huge)
D_MODEL, N_HEADS, PATCH = 16, 2, 2
TAU, DIAG_SHIFT = -8.0, 1e-3   # timing only; values irrelevant

g = nk.graph.Square(L, pbc=True)
hi = nk.hilbert.Spin(0.5, N=g.n_nodes, total_sz=0)
H = nk.operator.Heisenberg(hi, g, J=1.0)
sampler = nk.sampler.MetropolisExchange(hi, graph=g, n_chains=16)

# (label, family, kwargs). dense is only run at the smallest depth (it builds a P×P matrix).
VARIANTS = [
    ("SR dense",        "SR",  dict(pii=False, use_ntk=False), True),
    ("PII dense",       "PII", dict(pii=True,  use_ntk=False, tau=TAU), True),
    ("minSR ntk",       "SR",  dict(pii=False, use_ntk=True,  on_the_fly=False), False),
    ("minPII ntk",      "PII", dict(pii=True,  use_ntk=True,  on_the_fly=False, tau=TAU), False),
    ("minSR onthefly",  "SR",  dict(pii=False, use_ntk=True,  on_the_fly=True), False),
    ("minPII onthefly", "PII", dict(pii=True,  use_ntk=True,  on_the_fly=True,  tau=TAU), False),
]


def median_ms(thunk, reps, warmup):
    """Median wall time (ms) of `thunk()` over `reps`, after `warmup` (compile) calls."""
    for _ in range(warmup):
        jax.block_until_ready(thunk())
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(thunk())
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1e3


def bench_layers(num_layers, include_dense):
    model = nkpii.models.ViT(
        num_layers=num_layers, d_model=D_MODEL, n_heads=N_HEADS, patch_size=PATCH,
        transl_invariant=True,
    )
    vs = nk.vqs.MCState(sampler, model, n_samples=N_SAMPLES, seed=0)
    vs.sample()  # fix one set of samples; we time the *update*, not the sampling
    times = {}
    for label, _family, kw, is_dense in VARIANTS:
        if is_dense and not include_dense:
            continue
        driver = nkpii.driver.VMC_PII(
            H, optax.sgd(0.01), variational_state=vs, diag_shift=DIAG_SHIFT, mode="complex", **kw
        )
        # dense is much slower → fewer reps
        reps = 5 if is_dense else 15
        times[label] = median_ms(driver.compute_loss_and_update, reps=reps, warmup=2)
    return int(vs.n_parameters), times


print(f"ViT on {L}x{L} Heisenberg, {N_SAMPLES} samples; d_model={D_MODEL}, n_heads={N_HEADS}, "
      f"patch={PATCH}.  Timing one update step (compute_loss_and_update, fixed samples).\n")

results = {}
for i, nl in enumerate(NUM_LAYERS):
    P, times = bench_layers(nl, include_dense=(i == 0))  # dense only at the smallest depth
    results[nl] = (P, times)
    print(f"--- num_layers={nl}  (n_params={P}) ---")
    for label, _family, _kw, _is_dense in VARIANTS:
        if label in times:
            print(f"    {label:16s} {times[label]:8.2f} ms")
    # PII/SR ratios per implementation form
    for form, sr_l, pii_l in [("dense", "SR dense", "PII dense"),
                              ("ntk", "minSR ntk", "minPII ntk"),
                              ("on-the-fly", "minSR onthefly", "minPII onthefly")]:
        if sr_l in times and pii_l in times:
            print(f"      → PII/SR ({form}) = {times[pii_l] / times[sr_l]:.2f}x")
    print()

# --- plot per-step time vs num_layers, SR family solid, PII family dashed ---
plt.figure(figsize=(7.5, 5))
colors = {"minSR ntk": "C0", "minPII ntk": "C0", "minSR onthefly": "C1", "minPII onthefly": "C1",
          "SR dense": "C2", "PII dense": "C2"}
for label, family, _kw, _is_dense in VARIANTS:
    xs = [nl for nl in NUM_LAYERS if label in results[nl][1]]
    ys = [results[nl][1][label] for nl in xs]
    if not xs:
        continue
    plt.plot(xs, ys, ("--o" if family == "PII" else "-o"), color=colors[label],
             label=label, lw=2, markerfacecolor=("none" if family == "PII" else colors[label]))
plt.xlabel("number of ViT layers")
plt.ylabel("per-step update time (ms)")
plt.title(f"SR vs PII per-step cost — ViT, {L}x{L} Heisenberg, {N_SAMPLES} samples")
plt.xticks(NUM_LAYERS)
plt.ylim(0, None)
plt.legend(fontsize=8, ncol=2)
plt.tight_layout()
out = HERE / "bench_vit_variants.png"
plt.savefig(out, dpi=150)
print(f"saved {out}")
