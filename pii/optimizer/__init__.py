r"""PII optimizer subpackage (mirrors ``netket.optimizer``).

- :mod:`pii.optimizer.q` — the PII ``Q``-matrix linear operators (analogue of
  ``netket.optimizer.qgt``).
- :mod:`pii.optimizer.solver` — PII linear solvers (analogue of ``netket.optimizer.solver``).
- :class:`pii.optimizer.PII` — the PII gradient preconditioner (analogue of
  ``netket.optimizer.SR``), usable as the ``preconditioner`` of :class:`pii.driver.VMC`.
"""

from pii.optimizer import q
from pii.optimizer import solver
from pii.optimizer.preconditioner import PII

__all__ = ["q", "solver", "PII"]
