r"""Variational models for PII.

:class:`RBMRealParams` is a Restricted Boltzmann Machine with a **complex
log-amplitude** but **purely real parameters**: every complex weight/bias is
stored as two real arrays (real and imaginary part) and combined internally as
``w = w_re + 1j w_im``.  The wavefunction is therefore complex while all
variational parameters are real — the "real parameters, complex output"
convention used e.g. by the paper's ViT ansatz (``log Ψ = f_θ + i g_θ``, App. A).

:class:`LogStateVectorRealParams` applies the same real-parameter trick to
NetKet's exact :class:`netket.models.LogStateVector` (one log-coefficient per
basis state): a **linear/exact** ansatz with a **complex log-amplitude** and
**real parameters**, useful for testing the PII machinery on a full state vector
(as in the paper's Fig. 1 toy) with a sign/phase structure.

Use either with ``mode='complex'``: the Jacobian splits the (complex) *output*
into real/imaginary parts, but the parameters are not further split, so the
update is a single real vector of length ``P`` (no parameter doubling).
"""

from typing import Any

import jax.numpy as jnp
import flax.linen as nn

from jax.nn.initializers import normal

from netket import nn as nknn
from netket.hilbert import DiscreteHilbert
from netket.utils.types import NNInitFunc, Array

default_kernel_init = normal(stddev=0.01)


class RBMRealParams(nn.Module):
    r"""Complex-output RBM parametrized by real parameters.

    Functionally identical to :class:`netket.models.RBM` with complex weights,
    but each complex parameter ``w`` is represented by two real parameters
    ``w_re`` and ``w_im`` combined as ``w = w_re + 1j w_im``.
    """

    alpha: int | float = 1
    """Hidden-unit density: number of hidden units = ``alpha * n_visible``."""
    param_dtype: Any = jnp.float64
    """Dtype of the (real) parameters. Must be a real dtype; the log-amplitude is
    complex regardless. Provided for drop-in compatibility with ``netket.models.RBM``."""
    use_hidden_bias: bool = True
    use_visible_bias: bool = True
    activation: Any = nknn.activation.log_cosh
    """Nonlinearity applied to the hidden pre-activations (default ``log cosh``)."""
    kernel_init: NNInitFunc = default_kernel_init
    hidden_bias_init: NNInitFunc = default_kernel_init
    visible_bias_init: NNInitFunc = default_kernel_init

    @nn.compact
    def __call__(self, input):
        if jnp.issubdtype(jnp.dtype(self.param_dtype), jnp.complexfloating):
            raise ValueError(
                "RBMRealParams has real parameters by construction; `param_dtype` "
                "must be a real dtype. For a complex-parameter RBM use "
                "`netket.models.RBM(param_dtype=complex)`."
            )

        n_visible = input.shape[-1]
        n_features = int(self.alpha * n_visible)

        def complex_param(name, init, shape):
            re = self.param(f"{name}_re", init, shape, self.param_dtype)
            im = self.param(f"{name}_im", init, shape, self.param_dtype)
            return re + 1j * im

        kernel = complex_param("kernel", self.kernel_init, (n_visible, n_features))
        y = jnp.dot(input, kernel)  # real input @ complex kernel -> complex
        if self.use_hidden_bias:
            y = y + complex_param("hidden_bias", self.hidden_bias_init, (n_features,))

        y = self.activation(y)
        y = jnp.sum(y, axis=-1)

        if self.use_visible_bias:
            v_bias = complex_param("visible_bias", self.visible_bias_init, (n_visible,))
            y = y + jnp.dot(input, v_bias)

        return y


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
