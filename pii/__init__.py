"""Projected Inverse Iteration (PII) for neural quantum states.

This package mirrors NetKet's architecture as closely as possible, for seamless integration:

- :class:`pii.driver.VMC` ↔ ``netket.driver.VMC`` (standard, preconditioner-based);
- :class:`pii.driver.VMC_PII` ↔ ``netket.driver.VMC_SR`` (integrated PII/SR driver);
- :class:`pii.optimizer.PII` ↔ ``netket.optimizer.SR`` (gradient preconditioner);
- :mod:`pii.optimizer.q` ↔ ``netket.optimizer.qgt`` (the ``Q``-matrix linear operators);
- :mod:`pii.optimizer.solver` ↔ ``netket.optimizer.solver`` (linear solvers).

As in NetKet, only the standard driver is aliased at the top level (``pii.VMC``); everything else
is reached through its subpackage (``pii.driver.VMC_PII``, ``pii.optimizer.PII``,
``pii.optimizer.solver.gmres``, ...).

See the paper "Projected Inverse Iteration: An Eigenvalue Approach to Ground-State Computation
with Neural Quantum States".
"""

from pii import driver, models, optimizer
from pii.driver import VMC

__all__ = ["VMC", "driver", "models", "optimizer"]
