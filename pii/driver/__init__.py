r"""PII drivers — the PII analogue of :mod:`netket.driver`.

- :class:`pii.driver.VMC` — the standard preconditioner-based driver (↔ ``netket.driver.VMC``),
  used with :class:`pii.optimizer.PII` or :class:`netket.optimizer.SR`.
- :class:`pii.driver.VMC_PII` — the integrated PII driver (↔ ``netket.driver.VMC_SR``).
"""

from pii.driver.vmc import VMC
from pii.driver.vmc_pii import VMC_PII

__all__ = ["VMC", "VMC_PII"]
