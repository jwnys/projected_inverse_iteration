r"""Exact log-state-vector ansatz with real parameters and a complex log-amplitude.

:class:`LogStateVectorRealParams` applies the real-parameter trick of :class:`nkpii.models.RBMRealParams`
to NetKet's exact :class:`netket.models.LogStateVector` (one log-coefficient per basis state): a
**linear/exact** ansatz with a **complex log-amplitude** and **real parameters**, useful for testing
the PII machinery on a full state vector with a sign/phase structure.

Use it with ``mode='complex'`` (the output log-amplitude is complex; the parameters are not split
further, so the update is a single real vector of length ``P``).
"""

from typing import Any

import jax.numpy as jnp
import flax.linen as nn

from netket.hilbert import DiscreteHilbert
from netket.utils.types import NNInitFunc, Array


class LogStateVectorRealParams(nn.Module):
    r"""Complex-output exact state vector parametrized by real parameters.

    The real-parameter analogue of :class:`netket.models.LogStateVector`: it stores
    one complex log-coefficient per basis state as two real arrays
    ``logstate = logstate_re + 1j logstate_im``, so the wavefunction is complex while
    all variational parameters are real. Like ``LogStateVector`` it only works with
    indexable Hilbert spaces and initialises to a (uniform) constant state by default.

    Use with ``mode='complex'`` (the output log-amplitude is complex; the parameters
    are not split further, so the update is a single real vector of length ``P``).
    """

    hilbert: DiscreteHilbert
    """The Hilbert space (must be indexable)."""
    param_dtype: Any = jnp.float64
    """Dtype of the (real) parameters. Must be a real dtype; the log-amplitude is
    complex regardless. Provided for parity with ``netket.models.LogStateVector``."""
    logstate_init: NNInitFunc = nn.initializers.ones
    """Initializer for the real and imaginary log-coefficient arrays. The default
    (``ones`` on both) is a constant log-amplitude, i.e. the uniform state."""

    @nn.compact
    def __call__(self, x_in: Array):
        if jnp.issubdtype(jnp.dtype(self.param_dtype), jnp.complexfloating):
            raise ValueError(
                "LogStateVectorRealParams has real parameters by construction; "
                "`param_dtype` must be a real dtype. For a complex-parameter exact "
                "state use `netket.models.LogStateVector(param_dtype=complex)`."
            )
        if not self.hilbert.is_indexable:
            raise ValueError(
                "LogStateVectorRealParams can only be used with indexable Hilbert spaces."
            )

        shape = (self.hilbert.n_states,)
        re = self.param("logstate_re", self.logstate_init, shape, self.param_dtype)
        im = self.param("logstate_im", self.logstate_init, shape, self.param_dtype)
        logstate = re + 1j * im
        return logstate[self.hilbert.states_to_numbers(x_in)]
