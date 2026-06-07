"""Projected Inverse Iteration (PII) for neural quantum states.

The package layout follows NetKet's:

- ``nkpii.driver.VMC`` is the standard preconditioner-based driver, like ``netket.driver.VMC``;
- ``nkpii.driver.VMC_PII`` is the integrated PII/SR driver, like ``netket.driver.VMC_SR``;
- ``nkpii.optimizer.PII`` is the gradient preconditioner, like ``netket.optimizer.SR``;
- ``nkpii.optimizer.pct`` holds the Projected Characteristic Tensor (PCT) operators, like ``netket.optimizer.qgt``;
- ``nkpii.optimizer.solver`` holds the linear solvers, like ``netket.optimizer.solver``.

Only the standard driver is aliased at the top level (``nkpii.VMC``); everything else is reached
through its subpackage (``nkpii.driver.VMC_PII``, ``nkpii.optimizer.PII``, ``nkpii.optimizer.solver.gmres``).
"""

from nkpii import driver, models, optimizer
from nkpii.driver import VMC

__all__ = ["VMC", "driver", "models", "optimizer"]
