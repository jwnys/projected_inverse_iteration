r"""Diagonal toy benchmark ``Ĥ = diag(1, 10, 0)``.

A minimal illustrative example: a 3-dimensional Hilbert space with an
**exact, full** state-vector ansatz on the sphere ``S``. We use
:class:`netket.models.LogStateVector` (one free parameter per basis amplitude)
and a :class:`netket.vqs.FullSumState`, so the tangent space spans the whole
Hilbert space and the Galerkin projection is exact.

The Hamiltonian has eigenvalues ``{E0, E1, Emax} = {0, 1, 10}`` so the spectral
gap is ``Δ = 1`` and the spread is ``Γ = 10``.

- **PII** (``τ → E0 = 0``) converges almost immediately and is gap-insensitive.
- **SR** (``η = 0.1``) converges slowly with oscillating iterates because of the
  large eigenvalue spread and small gap.

Note: NetKet ansätze parametrize the **log**-amplitude, whereas a linear state
vector would be the direct choice here. The PII update is therefore the
inverse-iteration step pulled back through the (nonlinear) ``ψ = exp(logψ)`` map,
so convergence is extremely fast and gap-insensitive but not literally a single
step. The qualitative SR-vs-PII contrast is the same either way.

Run with::

    python examples/diag_hamiltonian_fig1.py
"""

from pathlib import Path

import numpy as np
import optax
import netket as nk
import matplotlib.pyplot as plt

import nkpii

HERE = Path(__file__).parent

E0 = 0.0  # Ĥ = diag(1, 10, 0) -> eigenvalues {0, 1, 10}
N_ITER = 20
SEED = 2

# 3-dimensional Hilbert space: a single site with local dimension 3.
hi = nk.hilbert.Spin(s=1, N=1)
H = nk.operator.LocalOperator(hi, operators=[np.diag([1.0, 10.0, 0.0])], acting_on=[[0]])

# LogStateVector: every basis amplitude is a free (real) parameter -> exact ansatz.
# Both drivers start from the same state (same seed).
sr = nkpii.driver.VMC_PII(
    H, optax.sgd(0.1),
    variational_state=nk.vqs.FullSumState(hi, nk.models.LogStateVector(hi, param_dtype=float), seed=SEED),
    diag_shift=0.0, pii=False,
)
pii_drv = nkpii.driver.VMC_PII(
    H, optax.sgd(1.0),
    variational_state=nk.vqs.FullSumState(hi, nk.models.LogStateVector(hi, param_dtype=float), seed=SEED),
    diag_shift=1e-8, pii=True, tau=1e-8,
)

# Run one iteration at a time, recording the energy error and the state vector.
e_sr, s_sr = [], []
e_pii, s_pii = [], []
for driver, energies, states in ((sr, e_sr, s_sr), (pii_drv, e_pii, s_pii)):
    for step in range(N_ITER + 1):
        if step > 0:
            driver.run(n_iter=1, show_progress=False)
        psi = np.asarray(driver.state.to_array()).real
        psi = psi / np.linalg.norm(psi)
        states.append(psi * np.sign(psi[np.argmax(np.abs(psi))]))  # fix global sign
        energies.append(float(driver.state.expect(H).mean.real) - E0)

e_sr, s_sr = np.array(e_sr), np.array(s_sr)
e_pii, s_pii = np.array(e_pii), np.array(s_pii)

print("Ĥ = diag(1, 10, 0);  E0=0, E1=1, Emax=10  (Δ=1, Γ=10)")
print(f"  SR  (η=0.1):  E-E0 after {N_ITER} steps = {e_sr[-1]:.3e}")
print(f"  PII (τ=1e-8): E-E0 after 1 step = {e_pii[1]:.3e}, "
      f"after {N_ITER} steps = {e_pii[-1]:.3e}")

fig = plt.figure(figsize=(10, 4))

# (A) trajectory on the sphere
axA = fig.add_subplot(1, 2, 1, projection="3d")
u, v = np.mgrid[0 : 2 * np.pi : 40j, 0 : np.pi : 20j]
axA.plot_surface(
    np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v),
    color="lightgray", alpha=0.15, linewidth=0,
)
axA.plot(*s_sr.T, "-o", ms=3, color="tab:red", label="SR")
axA.plot(*s_pii.T, "-o", ms=3, color="tab:cyan", label="PII")
axA.set_title("A  optimization on the sphere $\\mathbb{S}$")
axA.legend()

# (B) energy error vs iteration
axB = fig.add_subplot(1, 2, 2)
axB.axhline(1.0, ls="--", color="gray", lw=1, label="$E_1$ energy")
axB.semilogy(np.maximum(e_sr, 1e-16), "-s", color="tab:red", label="SR")
axB.semilogy(np.maximum(e_pii, 1e-16), "-o", color="tab:cyan", label="PII")
axB.set_xlabel("iteration")
axB.set_ylabel("$E(\\psi_k) - E_0$")
axB.set_title("B  energy error")
axB.legend()

fig.tight_layout()
fig.savefig(HERE / "diag_hamiltonian_fig1.png", dpi=150)
print(f"saved {HERE / 'diag_hamiltonian_fig1.png'}")
