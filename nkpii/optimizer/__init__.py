r"""PII optimizer subpackage (mirrors ``netket.optimizer``).

- :mod:`nkpii.optimizer.pct` — the Projected Characteristic Tensor (PCT) linear operators
  (analogue of ``netket.optimizer.qgt``).
- :mod:`nkpii.optimizer.solver` — PII linear solvers (analogue of ``netket.optimizer.solver``).
- :class:`nkpii.optimizer.PII` — the PII gradient preconditioner (analogue of
  ``netket.optimizer.SR``), usable as the ``preconditioner`` of :class:`nkpii.driver.VMC`.
"""

from nkpii.optimizer import pct
from nkpii.optimizer import solver
from nkpii.optimizer.preconditioner import PII

__all__ = ["pct", "solver", "PII"]
