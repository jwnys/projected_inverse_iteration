# PII — Projected Inverse Iteration for Neural Quantum States

The `nkpii` package implements **Projected Inverse Iteration (PII)**, an
eigenvalue approach to ground-state computation with neural quantum states,
built on top of [NetKet](https://www.netket.org/).

PII reframes the ground-state search as an eigenvalue problem and replaces
Stochastic Reconfiguration's (SR) overlap-matrix preconditioner `S` with

```
Q = H − τ S + λ I
```

where `H` is the Hamiltonian projected onto the variational tangent space and
`τ` is the inverse-iteration shift (an estimate of, or mild undershoot of, the
ground-state energy `E0`). Unlike SR, PII is **robust to small spectral gaps**
and typically converges in far fewer iterations with learning rate `η = 1`.

The public API follows NetKet's, so PII drops into an existing NetKet workflow with little
change. The correspondence is:

| PII | NetKet |
|---|---|
| `nkpii.driver.VMC` (= `nkpii.VMC`) | `netket.driver.VMC` (standard, preconditioner-based) |
| `nkpii.driver.VMC_PII` | `netket.driver.VMC_SR` (integrated driver) |
| `nkpii.optimizer.PII` | `netket.optimizer.SR` (gradient preconditioner) |
| `nkpii.optimizer.pct.PCTJacobian{Dense,PyTree}` | `netket.optimizer.qgt.QGTJacobian{Dense,PyTree}` |
| `nkpii.optimizer.solver.*` | `netket.optimizer.solver.*` |

For example, use `nkpii.optimizer.PII` as the `preconditioner` of a standard `VMC` driver the same
way you would `netket.optimizer.SR`.

## Install

PII tracks the latest NetKet (installed from GitHub, per
[NetKet's docs](https://github.com/netket/netket)):

```bash
pip install -e .          # pulls NetKet (latest, from GitHub) + jax + einops
# or, with uv:
uv pip install -e .
```

## Usage

There are two equivalent entry points, mirroring NetKet's two:

### 1. Preconditioner + standard driver (`nkpii.optimizer.PII` + `nkpii.driver.VMC`)

The NetKet-idiomatic route — `nkpii.optimizer.PII` is a drop-in replacement for
`netket.optimizer.SR`:

```python
import optax, netket as nk
import nkpii

# ... build hamiltonian H, exact/estimated E0, and a variational state vstate ...

gs = nkpii.driver.VMC(                 # ≡ nkpii.VMC ; same as netket.driver.VMC
    H, optax.sgd(1.0),               # PII uses learning rate η = 1
    variational_state=vstate,
    preconditioner=nkpii.optimizer.PII(H, tau=1.1 * E0, diag_shift=1e-2),
)
gs.run(n_iter=100)
```

`nkpii.optimizer.PII` accepts `q=` (the PCT operator type — `PCTJacobianDense` (default) or
`PCTJacobianPyTree`) and `solver=` (a linear solver from `nkpii.optimizer.solver`). For a
**matrix-free** solve (no dense `Q`), use the iterative solvers — the PII analogue of
`SR(solver=cg)` (note: `cg` is unavailable here because `Q` is non-symmetric/indefinite,
so PII provides `gmres`/`bicgstab` instead):

```python
nkpii.optimizer.PII(H, tau=1.1 * E0, diag_shift=1e-2,
                  q=nkpii.optimizer.pct.PCTJacobianPyTree,
                  solver=nkpii.optimizer.solver.gmres)   # matrix-free
```

### 2. Integrated driver (`nkpii.driver.VMC_PII`)

`nkpii.driver.VMC_PII` is a **drop-in superset** of NetKet's `VMC_SR` (it adds the NTK /
on-the-fly / SPRING machinery). With `pii=False` it reproduces SR / minSR / on-the-fly SR
*exactly*; with `pii=True` it runs Projected Inverse Iteration:

```python
gs = nkpii.driver.VMC_PII(
    H, optax.sgd(1.0), variational_state=vstate,
    diag_shift=1e-2, pii=True, tau=1.1 * E0,
)
gs.run(n_iter=100)
```

#### Key options (`VMC_PII`)

| option | meaning |
|---|---|
| `pii` | `False` → SR/minSR (NetKet); `True` → Projected Inverse Iteration |
| `tau` | inverse-iteration shift `τ` (float or schedule `Callable[[int], float]`); **required** when `pii=True` |
| `diag_shift` | Tikhonov regularization `λ` (float or schedule) |
| `use_ntk` | `True` → kernel trick (minSR / **minPII**, `2M×2M`); `False` → dense (`P×P`) |
| `on_the_fly` | `True` → matrix-free / lazy NTK (lowest memory) |
| `momentum` | SPRING / PII-SPRING damping (≈ 0.8) — see the performance note below |
| `linear_solver` | `(Q, b) -> (x, info)` solver (e.g. `nkpii.optimizer.solver.penrose_symmetrized_solver`) |
| `chunk_size_bwd` | chunking of the `O` Jacobian / NTK (backward pass) |
| `chunk_size_dEloc` | chunking of the **local-energy-derivative** (`A`) computation; defaults to `chunk_size_bwd` |

The `chunk_size_dEloc` knob keeps the new ingredient of PII — the derivative of
the local energy `E_{L,θ}` — memory-bounded: the scalar `E_L` is differentiated
*after* summing over connected configurations, and chunking over samples bounds
peak memory to `chunk · n_conn` passes (never a `[M, n_conn, P]` tensor).

`FullSumState` (exact enumeration, no Monte Carlo noise) is supported on the
dense PII path.

> **Note:** **PII-SPRING** (`momentum` with `pii=True`) is **not yet optimized** and is
> currently **slow** — it works correctly, but its per-step cost is higher than it should be.
> Plain PII (no `momentum`) is the optimized path.

### Solvers

`Q = OᴴA − τOᴴO + λI` is in general **non-symmetric and indefinite**, so
`nkpii.optimizer.solver` provides:

- `pii_default_solver` — direct general-LU (the default; analogue of NetKet's
  `cholesky_with_fallback`, but LU since `Q` is not Hermitian PSD);
- `penrose_symmetrized_solver` — regularized least-squares `(QᴴQ + λI)⁻¹Qᴴb`
  (the inner SPD system uses `cholesky_with_fallback`);
- `naive_symmetrized_solver` — solves the (indefinite) Hermitian part `½(Q+Qᴴ)`;
- `gmres`, `bicgstab` — matrix-free iterative solvers (no `cg`: `Q` is not PSD).

## Examples

```bash
python examples/diag_hamiltonian_fig1.py        # diagonal toy Hamiltonian
python examples/compare_sr_pii_tfim.py          # SR vs PII on a small TFIM chain
python examples/compare_implementations.py      # VMC+PII vs VMC_PII vs netket VMC_SR (consistency)
python examples/compare_pii_symmetrized_tfim.py # unsymmetrized vs symmetrized PII solvers
```

- `diag_hamiltonian_fig1.py` uses the toy Hamiltonian
  `Ĥ = diag(1, 10, 0)` (dim-3 Hilbert space, exact `LogStateVector` ansatz),
  showing PII converging almost immediately while SR oscillates slowly.
- `compare_sr_pii_tfim.py` compares SR and every PII variant on a small,
  exactly-solvable 1D transverse-field Ising chain, all starting from identical
  parameters.
- `compare_implementations.py` shows the two entry points agree: `nkpii.driver.VMC` +
  `nkpii.optimizer.PII` (with `PCTJacobianDense`, `PCTJacobianPyTree`, and the matrix-free
  `gmres`) gives the same trajectory as the integrated `nkpii.driver.VMC_PII`, and the
  SR path matches NetKet's `VMC_SR`.
- `compare_pii_symmetrized_tfim.py` compares standard PII (general LU on `Q ξ = ½∇E`) with the
  symmetrized solvers (the regularized pseudo-inverse `(QᴴQ + λI)⁻¹Qᴴ`), with Monte Carlo and
  FullSum references.

## Benchmarks

```bash
python benchmarks/bench_pii_vs_sr.py       # per-step PII vs SR cost (RBM, system-size sweep)
python benchmarks/bench_vit_variants.py    # per-step cost across all SR/PII variants (ViT, depth sweep)
```

PII's only per-step overhead vs SR is the local-energy-gradient (`A`-Jacobian), so one PII step
should cost only a small constant (~3-4×) more than one SR step. `bench_pii_vs_sr.py` confirms this
for the dense path and isolates the (now-fixed) per-step recompilation that previously inflated it;
`bench_vit_variants.py` times every variant (dense, NTK/minSR-minPII, on-the-fly) on a Vision
Transformer of varying depth.

## Tests

```bash
python -m pytest test/                          # single device
PII_TEST_DEVICES=2 python -m pytest test/       # multi-device / sharding
```

The suite checks: exact reproduction of NetKet `VMC_SR` (`pii=False`), mutual
agreement of the dense / minPII / on-the-fly PII paths, the preconditioner path
(`VMC` + `PII`) against the integrated `VMC_PII`, convergence of every variant
(incl. PII-SPRING) and `FullSumState`, the local-energy derivative against finite
differences, chunk-size invariance, device-count-invariant results under sharding,
and the SR/PII factor-of-2, `η=1`, and SPRING conventions.

## Layout

```
nkpii/
  driver/                  # like netket.driver
    vmc.py                 # nkpii.driver.VMC  — standard preconditioner-based driver
    vmc_pii.py             # nkpii.driver.VMC_PII — integrated driver (superset of VMC_SR)
  optimizer/               # like netket.optimizer
    preconditioner.py      # nkpii.optimizer.PII  — preconditioner (like SR)
    pct/                   # Projected Characteristic Tensor operators (like netket.optimizer.qgt)
    solver/                # linear solvers (like netket.optimizer.solver)
  ngd/                     # the dense / minPII / on-the-fly kernels
    local_energy.py        # differentiable E_L and the A-function (E_L + sg(E_L)·logψ)
    common.py              # O & A Jacobians, dispatch to dense / minPII
    pii_dense.py           # dense:  Q = H − τS + λI            (P×P)
    pii_kernel.py          # minPII: K = A Oᵀ − τ O Oᵀ + λI     (2M×2M, push-through)
    pii_ntk.py             # two-function cross-NTK for the on-the-fly A Oᵀ term
    pii_onthefly.py        # matrix-free minPII
  models/                  # like netket.models — one ansatz per file
    rbm.py                 # RBMRealParams
    log_state_vector.py    # LogStateVectorRealParams
    vit.py                 # ViT (Vision Transformer, PRL 130, 236401)
examples/
benchmarks/                # per-VMC-step cost: SR vs PII (RBM size sweep; ViT depth sweep)
test/
```

## Citing

If you use this package, please cite:

> H. Zhang, V. Armegioiu, J. Carrasquilla, S. Mishra, J. Müller, J. Nys,
> M. Zeinhofer. *Projected Inverse Iteration: An Eigenvalue Approach to
> Ground-State Computation with Neural Quantum States.*
