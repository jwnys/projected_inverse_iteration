"""Chunk sizes must never change the PII update (up to floating-point noise).

Both chunking knobs are covered:

- ``chunk_size``        -> the ``O`` Jacobian / NTK (and the final vjp on-the-fly);
- ``chunk_size_dEloc``  -> the local-energy-derivative ``A`` computation.

Tests use deterministic, sampler-free inputs (``fixed_inputs``) and compare the
one-step update against the unchunked reference.
"""

import jax.flatten_util as fu
import numpy as np
import pytest

from nkpii.ngd.common import _pii_common
from nkpii.ngd.pii_onthefly import pii_onthefly

from .common import fixed_inputs, gen_solver

# 128 fixed samples; chunk sizes below all divide 128.
_BASE = dict(tau=-20.0, diag_shift=0.1, solver_fn=gen_solver, mode="real", weights=None)


def _ravel(dp):
    return fu.ravel_pytree(dp)[0]


def _assert_close(dp, ref):
    np.testing.assert_allclose(_ravel(dp), _ravel(ref), rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("use_ntk", [False, True], ids=["dense", "minpii"])
@pytest.mark.parametrize("chunk_size", [None, 16, 32, 128])
@pytest.mark.parametrize("chunk_size_dEloc", [None, 16, 64])
def test_dense_kernel_chunking_invariant(use_ntk, chunk_size, chunk_size_dEloc):
    log_psi, f_A, eloc, params, samples = fixed_inputs()
    ref, _, _ = _pii_common(log_psi, f_A, eloc, params, {}, samples, use_ntk=use_ntk, **_BASE)
    dp, _, _ = _pii_common(
        log_psi, f_A, eloc, params, {}, samples,
        use_ntk=use_ntk, chunk_size=chunk_size, chunk_size_dEloc=chunk_size_dEloc, **_BASE,
    )
    _assert_close(dp, ref)


@pytest.mark.parametrize("chunk_size", [None, 16, 64])
@pytest.mark.parametrize("chunk_size_dEloc", [None, 16, 32, 128])
def test_onthefly_chunking_invariant(chunk_size, chunk_size_dEloc):
    log_psi, f_A, eloc, params, samples = fixed_inputs()
    ref, _, _ = pii_onthefly(log_psi, f_A, eloc, params, {}, samples, **_BASE)
    dp, _, _ = pii_onthefly(
        log_psi, f_A, eloc, params, {}, samples,
        chunk_size=chunk_size, chunk_size_dEloc=chunk_size_dEloc, **_BASE,
    )
    _assert_close(dp, ref)


def test_onthefly_nondividing_chunk_falls_back_cleanly():
    """A chunk size that does not divide N_mc still gives the exact result."""
    log_psi, f_A, eloc, params, samples = fixed_inputs(n_samples=128)
    ref, _, _ = pii_onthefly(log_psi, f_A, eloc, params, {}, samples, **_BASE)
    dp, _, _ = pii_onthefly(
        log_psi, f_A, eloc, params, {}, samples, chunk_size_dEloc=30, **_BASE
    )  # 30 does not divide 128 -> full computation
    _assert_close(dp, ref)
