"""Projected Inverse Iteration (PII) for neural quantum states.

This package provides :class:`pii.VMC`, a drop-in superset of NetKet's
``VMC_SR`` driver. With ``pii=False`` it reproduces Stochastic Reconfiguration
(SR), minSR and their on-the-fly variants exactly. With ``pii=True`` it runs
**Projected Inverse Iteration**, which reframes the ground-state search as an
eigenvalue problem and is robust to small spectral gaps.

See the paper "Projected Inverse Iteration: An Eigenvalue Approach to
Ground-State Computation with Neural Quantum States".
"""

from pii.driver import VMC
from pii import models  # exposes pii.models.RBMRealParams

__all__ = ["VMC", "models"]
