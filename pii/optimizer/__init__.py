r"""PII optimizer subpackage (mirrors ``netket.optimizer``).

- :mod:`pii.optimizer.q` — the PII ``Q``-matrix linear operators (analogue of
  ``netket.optimizer.qgt``).
- :class:`pii.optimizer.PII` — the PII gradient preconditioner (analogue of
  ``netket.optimizer.SR``), usable as the ``preconditioner`` of :class:`pii.VMC`.
"""

from pii.optimizer import q
from pii.optimizer.preconditioner import PII, q_default_solver

__all__ = ["q", "PII", "q_default_solver"]
