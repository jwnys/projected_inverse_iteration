"""Multi-device / sharding correctness.

These tests run under whatever JAX mesh is active (single device by default,
``N`` simulated devices when launched with ``PII_TEST_DEVICES=N``). On fixed,
deterministic inputs the PII updates must be identical across the dense, minPII
and on-the-fly paths — and therefore identical regardless of the device count,
which is exactly the property NetKet's sharding tests assert.

To exercise multiple devices::

    PII_TEST_DEVICES=2 python -m pytest test/test_sharding.py
"""

import jax.flatten_util as fu
import numpy as np

from nkpii.ngd.common import _pii_common
from nkpii.ngd.pii_onthefly import pii_onthefly

from .common import fixed_inputs, gen_solver


def test_pii_paths_agree_under_active_mesh():
    """Dense, minPII and on-the-fly give the same update under the active mesh."""
    log_psi, f_A, eloc, params, samples = fixed_inputs()
    common = dict(tau=-20.0, diag_shift=0.1, solver_fn=gen_solver, mode="real", weights=None)

    dp_dense, _, _ = _pii_common(log_psi, f_A, eloc, params, {}, samples, use_ntk=False, **common)
    dp_kernel, _, _ = _pii_common(log_psi, f_A, eloc, params, {}, samples, use_ntk=True, **common)
    dp_otf, _, _ = pii_onthefly(log_psi, f_A, eloc, params, {}, samples, **common)

    a = fu.ravel_pytree(dp_dense)[0]
    b = fu.ravel_pytree(dp_kernel)[0]
    c = fu.ravel_pytree(dp_otf)[0]
    np.testing.assert_allclose(b, a, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(c, a, rtol=1e-6, atol=1e-7)
