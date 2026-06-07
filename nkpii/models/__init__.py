r"""Variational models for PII (real parameters, complex log-amplitude).

- :class:`RBMRealParams` — a complex-output RBM with real parameters (``nkpii.models.rbm``).
- :class:`LogStateVectorRealParams` — the exact log-state-vector ansatz (``nkpii.models.log_state_vector``).
- :class:`ViT` — a Vision-Transformer ansatz for 2D spin systems (``nkpii.models.vit``).
"""

from nkpii.models.rbm import RBMRealParams
from nkpii.models.log_state_vector import LogStateVectorRealParams
from nkpii.models.vit import ViT

__all__ = ["RBMRealParams", "LogStateVectorRealParams", "ViT"]
