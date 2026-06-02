# PII — Projected Inverse Iteration for Neural Quantum States

A standalone package implementing **Projected Inverse Iteration (PII)**, an
eigenvalue approach to ground-state computation with neural quantum states,
built on top of [NetKet](https://netket.org).

PII reframes the ground-state search as an eigenvalue problem and replaces
Stochastic Reconfiguration's (SR) overlap-matrix preconditioner `S` with

```
Q = H − τ S + λ I        (paper Eq. 17/29)
```

where `H` is the Hamiltonian projected onto the variational tangent space and
`τ` is the inverse-iteration shift (an estimate of, or mild undershoot of, the
ground-state energy `E0`). Unlike SR, PII is **robust to small spectral gaps**
and typically converges in far fewer iterations with learning rate `η = 1`.

See the paper: *"Projected Inverse Iteration: An Eigenvalue Approach to
Ground-State Computation with Neural Quantum States"*.

## Install

```bash
conda activate pii          # an env with netket already installed
pip install -e . --no-deps
```

## Usage

`pii.VMC` is a **drop-in superset** of NetKet's `VMC_SR` driver. With
`pii=False` (default) it reproduces SR / minSR / on-the-fly SR *exactly*; with
`pii=True` it runs Projected Inverse Iteration.

```python
import optax, netket as nk
import pii

# ... build hamiltonian H, exact/estimated E0, and a variational state vstate ...

driver = pii.VMC(
    H,
    optax.sgd(1.0),                 # PII uses learning rate η = 1
    variational_state=vstate,
    diag_shift=1e-2,                # Tikhonov regularization λ
    pii=True,
    tau=1.1 * E0,                   # shift τ (mild "undershoot": α ≥ 1)
)
driver.run(n_iter=100)
```

### Key options

| option | meaning |
|---|---|
| `pii` | `False` → SR/minSR (NetKet); `True` → Projected Inverse Iteration |
| `tau` | inverse-iteration shift `τ` (float or schedule `Callable[[int], float]`); **required** when `pii=True` |
| `diag_shift` | Tikhonov regularization `λ` (float or schedule) |
| `use_ntk` | `True` → kernel trick (minSR / **minPII**, `2M×2M`); `False` → dense (`P×P`) |
| `on_the_fly` | `True` → matrix-free / lazy NTK (lowest memory) |
| `momentum` | SPRING / PII-SPRING damping (≈ 0.8) |
| `chunk_size_bwd` | chunking of the `O` Jacobian / NTK (backward pass) |
| `chunk_size_dEloc` | chunking of the **local-energy-derivative** (`A`) computation; defaults to `chunk_size_bwd` |

The `chunk_size_dEloc` knob keeps the new ingredient of PII — the derivative of
the local energy `E_{L,θ}` — memory-bounded: the scalar `E_L` is differentiated
*after* summing over connected configurations, and chunking over samples bounds
peak memory to `chunk · n_conn` passes (never a `[M, n_conn, P]` tensor).

`FullSumState` (exact enumeration, no Monte Carlo noise) is supported on the
dense PII path.

### Bundled model

`pii.models.RBMRealParams` is an RBM with a **complex log-amplitude but real
parameters** — each complex weight is stored as a real/imaginary pair and
combined internally — matching the paper's "real parameters, complex output"
convention (`log Ψ = f + i g`). Use it with `mode="complex"` (or `mode=None`,
which auto-detects).

## Examples

```bash
python pii/examples/diag_hamiltonian_fig1.py   # paper Fig. 1 toy benchmark
python pii/examples/compare_sr_pii_tfim.py     # SR vs PII on a small TFIM chain
```

- `diag_hamiltonian_fig1.py` reproduces Figure 1: the toy Hamiltonian
  `Ĥ = diag(1, 10, 0)` (dim-3 Hilbert space, exact `LogStateVector` ansatz),
  showing PII converging almost immediately while SR oscillates slowly.
- `compare_sr_pii_tfim.py` compares SR and every PII variant on a small,
  exactly-solvable 1D transverse-field Ising chain, all starting from identical
  parameters. PII reaches ~machine precision while SR is limited by the
  Monte Carlo sampling-noise floor; the `FullSumState` runs confirm both are
  exact in the noise-free limit.

## Tests

```bash
python -m pytest test/                          # single device
PII_TEST_DEVICES=2 python -m pytest test/       # multi-device / sharding
```

The suite checks: exact reproduction of NetKet `VMC_SR` (`pii=False`), mutual
agreement of the dense / minPII / on-the-fly PII paths, convergence of every
variant (incl. PII-SPRING) and `FullSumState`, the local-energy derivative
against finite differences, chunk-size invariance, device-count-invariant
results under sharding, and the SR↔PII factor-of-2 / `η=1` / SPRING conventions.

## Layout

```
pii/
  driver.py              # pii.VMC — superset of VMC_SR (+ pii / tau / chunk_size_dEloc)
  _ngd/
    local_energy.py      # differentiable E_L and the A-function (E_L + sg(E_L)·logψ)
    common.py            # O & A Jacobians, dispatch to dense / minPII
    pii_dense.py         # dense:  Q = H − τS + λI            (P×P)
    pii_kernel.py        # minPII: K = A Oᵀ − τ O Oᵀ + λI     (2M×2M, push-through)
    pii_ntk.py           # two-function cross-NTK for the on-the-fly A Oᵀ term
    pii_onthefly.py      # matrix-free minPII
  examples/
    diag_hamiltonian_fig1.py   # paper Fig. 1 (Ĥ = diag(1, 10, 0))
    compare_sr_pii_tfim.py
test/
```

## Citing

If you use this package, please cite:

> H. Zhang, V. Armegioiu, J. Carrasquilla, S. Mishra, J. Müller, J. Nys,
> M. Zeinhofer. *Projected Inverse Iteration: An Eigenvalue Approach to
> Ground-State Computation with Neural Quantum States.*
