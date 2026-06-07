r"""Linear solvers for PII — the PII analogue of :mod:`netket.optimizer.solver.solvers`.

These follow NetKet's solver contract exactly: each is a callable ``(A, b) -> (x, info)``
where ``A`` is **either** a dense matrix **or** a :class:`~netket.optimizer.LinearOperator`
(here a :class:`nkpii.optimizer.pct.PCTJacobianDenseT` / ``PCTJacobianPyTreeT``), and ``b`` is a vector
or PyTree.  As in NetKet, the **direct** solvers materialize the operator with
``A.to_dense()`` and flatten the rhs with :func:`jax.flatten_util.ravel_pytree`; the
**iterative** solvers stay matrix-free, using only the lazy ``A @ v``.

They plug into both the PII preconditioner — ``nkpii.optimizer.PII(..., solver=...)`` — and the
integrated driver — ``nkpii.driver.VMC_PII(..., linear_solver=...)``.  Decorated with NetKet's
:func:`~netket.utils.api_utils.partial_from_kwargs`, so a keyword-only call returns a partial::

    nkpii.optimizer.solver.penrose_symmetrized_solver(diag_shift=1e-2)   # (Q, b) -> (x, info)

**Why not Cholesky/CG by default?**  PII's ``Q = OᴴA − τOᴴO + λI`` is in general **non-symmetric
and indefinite** (only asymptotically Hermitian), so the default direct solver is a general LU
(``pii_default_solver``) — the PII analogue of NetKet's ``cholesky_with_fallback`` default for the
(symmetric PSD) QGT — and the matrix-free iterative options are ``gmres``/``bicgstab`` (NOT ``cg``,
which requires Hermitian PSD).
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax.flatten_util import ravel_pytree

from netket.utils.api_utils import partial_from_kwargs
from netket.optimizer.solver import cholesky_with_fallback


@partial_from_kwargs
def pii_default_solver(A, b, *, x0=None):
    r"""Default PII solve: a **direct general-LU** solve of the (non-symmetric) ``Q``.

    The PII analogue of NetKet's ``SR`` default ``cholesky_with_fallback`` — direct, but LU
    rather than Cholesky because ``Q = OᴴA − τOᴴO + λI`` is not Hermitian PSD.  Accepts a dense
    ``Q`` (from the ngd kernels) or a ``PCTJacobian`` operator (from the preconditioner), and a
    flat or PyTree rhs.

    Args:
        A: the dense PII matrix ``Q`` or a ``PCTJacobian`` operator.
        b: the right-hand side ``½∇E`` (vector or PyTree).
        x0: unused (kept for the NetKet solver signature).

    Returns:
        ``(x, None)`` with ``x = Q⁻¹ b`` in the structure of ``b``.
    """
    del x0
    if not isinstance(A, jax.Array):
        A = A.to_dense()
    b, unravel = ravel_pytree(b)
    x = jsp.linalg.solve(A, b, assume_a="gen")
    return unravel(x), None


@partial_from_kwargs
def penrose_symmetrized_solver(A, b, *, diag_shift, solver=cholesky_with_fallback, x0=None):
    r"""Regularized-least-squares ("Penrose"/symmetrized) solver for the PII matrix ``Q``.

    Returns the Tikhonov-regularized **Euclidean** least-squares solution

    .. math::
        \xi = (Q^H Q + \mathrm{diag\_shift}\,I)^{-1} Q^H\, b
            = \arg\min_\xi \; \|Q\xi - b\|_2^2 + \mathrm{diag\_shift}\,\|\xi\|_2^2 ,

    i.e. the Tikhonov pseudo-inverse of ``Q``.  As ``diag_shift → 0`` (and when ``Q`` is
    invertible) this equals ``Q⁻¹ b``; when ``Q`` is singular it stays well-defined and returns
    the minimum-norm least-squares solution (→ ``pinv(Q) b``).

    The normal-equations matrix ``M = QᴴQ + diag_shift·I`` is **Hermitian positive-definite**, so
    the default inner ``solver`` is NetKet's :func:`~netket.optimizer.solver.cholesky_with_fallback`
    (Cholesky, automatically falling back to ``pinv_smooth`` when ``QᴴQ`` is rank-deficient / the
    shift is tiny).  Pass any ``(A, b) -> (x, info)`` solver to override it.

    **Hermitian vs. plain transpose.**  The normal equations use the *conjugate* (Hermitian)
    transpose ``Qᴴ``; for a genuinely complex ``Q`` a plain ``Qᵀ`` would be wrong.  In this code
    ``Q`` is **always real** even for complex ψ/parameters (NetKet realifies ``mode='complex'`` by
    stacking Re/Im into extra real rows/cols, so ``S = Re(S_ℂ)`` and ``Q`` is real), so
    ``Q.conj().T`` is effectively ``Qᵀ`` here; we keep it explicit so the math stays the correct
    ``Qᴴ`` if a genuinely complex matrix were ever passed.

    .. important::
        Use with ``diag_shift=0`` on the driver/preconditioner — the regularization is carried
        entirely by this solver's ``diag_shift``, which has **energy² units** (forming ``QᴴQ``
        squares the condition number of ``Q``), so it typically needs to be substantially larger
        than a comparable driver ``diag_shift``.

    .. note::
        Dense (parameter-space) path: the solver expects the ``P×P`` ``Q``, not the ``2M×2M``
        minPII kernel ``K`` — keep ``use_ntk=False``/``on_the_fly=False`` when used via the driver.

    Args:
        A: the dense PII matrix ``Q`` (or a ``PCTJacobian`` operator).
        b: the right-hand side ``½∇E``.
        diag_shift: Tikhonov regularization on the normal equations (keyword-only, energy² units).
        solver: inner ``(A, b) -> (x, info)`` solver for the SPD system
            ``(QᴴQ + diag_shift·I) ξ = Qᴴb``.  Defaults to ``cholesky_with_fallback``.
        x0: unused (NetKet solver signature).

    Returns:
        ``(x, info)`` with ``x = (QᴴQ + diag_shift·I)⁻¹ Qᴴ b`` and ``info`` from the inner solver.
    """
    del x0
    if not isinstance(A, jax.Array):
        A = A.to_dense()
    b, unravel = ravel_pytree(b)
    Qh = A.conj().T
    QhQ = Qh @ A
    M = QhQ + diag_shift * jnp.eye(QhQ.shape[-1], dtype=QhQ.dtype)
    rhs = Qh @ b
    x, info = solver(M, rhs)
    return unravel(x), info


@partial_from_kwargs
def naive_symmetrized_solver(A, b, *, diag_shift, solver=None, x0=None):
    r"""'Naive' symmetrization: solve the Hermitian part ``½(Q+Qᴴ)`` of the PII matrix.

    Replaces ``Q`` with its Hermitian part and solves

    .. math::
        \big(\tfrac12 (Q + Q^H) + \mathrm{diag\_shift}\,I\big)\,\xi = b .

    Contrast with :func:`penrose_symmetrized_solver`, which forms the normal equations ``QᴴQ``.
    This route is **additive** (the Hermitian part) rather than **multiplicative** (``QᴴQ``), so it
    does *not* square ``Q``'s condition number — ``diag_shift`` therefore has **energy units** (like
    an ordinary PII ``diag_shift``).

    .. warning::
        ``½(Q+Qᴴ)`` is symmetric but **indefinite** in general (a field-of-values effect: the
        non-normal ``Q`` is accretive — eigenvalues in the right half-plane — yet the smallest
        eigenvalue of its Hermitian part dips negative).  The default inner solver is therefore a
        **general LU** solve (``assume_a='sym'`` — which JAX maps to a plain LU, not a Cholesky),
        which never NaNs on an indefinite ``M`` (unlike Cholesky / ``cholesky_with_fallback``, whose
        PSD-oriented ``pinv_smooth`` fallback is the wrong tool here).  A *too-small* ``diag_shift``
        leaves ``M`` near-singular and the
        update overshoots; empirically ``diag_shift`` must be large enough to lift ``M`` comfortably
        positive-definite (``O(0.1–1)`` for small spin systems, despite the energy units).

    .. important::
        Use with ``diag_shift=0`` on the driver/preconditioner — regularization is carried entirely
        by this solver's ``diag_shift``.

    Args:
        A: the dense PII matrix ``Q`` (or NTK kernel ``K``, or a ``PCTJacobian`` operator).
        b: the right-hand side ``½∇E``.
        diag_shift: Tikhonov regularization added to ``½(Q+Qᴴ)`` (keyword-only, energy units).
        solver: optional inner ``(A, b) -> (x, info)`` solver for ``(½(Q+Qᴴ) + diag_shift·I) ξ = b``.
            Defaults (``None``) to a general LU solve (``jax.scipy.linalg.solve`` with
            ``assume_a='sym'``, which JAX maps to LU — indefinite-safe, unlike Cholesky).
        x0: unused (NetKet solver signature).

    Returns:
        ``(x, info)`` with ``x = (½(Q+Qᴴ) + diag_shift·I)⁻¹ b``.
    """
    del x0
    if not isinstance(A, jax.Array):
        A = A.to_dense()
    b, unravel = ravel_pytree(b)
    Qh = A.conj().T
    M = 0.5 * (Qh + A) + diag_shift * jnp.eye(A.shape[-1], dtype=A.dtype)
    if solver is None:
        return unravel(jsp.linalg.solve(M, b, assume_a="sym")), None
    x, info = solver(M, b)
    return unravel(x), info


@partial_from_kwargs
def gmres(A, b, *, x0=None, **kwargs):
    r"""Matrix-free **GMRES** for the (non-symmetric) PII ``Q`` — the analogue of ``SR(solver=cg)``.

    Uses only the lazy ``A @ v`` (never materializes ``Q``), so it scales to large parameter
    counts.  GMRES handles general nonsingular non-symmetric systems (unlike ``cg``, which needs
    Hermitian PSD — wrong for ``Q``).  ``kwargs`` are forwarded to
    :func:`jax.scipy.sparse.linalg.gmres` (e.g. ``tol``, ``atol``, ``restart``, ``maxiter``, ``M``).

    Args:
        A: a ``PCTJacobian`` operator (or a dense matrix); only ``A @ v`` is used.
        b: the right-hand side ``½∇E``.
        x0: optional initial guess.

    Returns:
        ``(x, info)`` from :func:`jax.scipy.sparse.linalg.gmres`.
    """
    return jax.scipy.sparse.linalg.gmres(lambda v: A @ v, b, x0=x0, **kwargs)


@partial_from_kwargs
def bicgstab(A, b, *, x0=None, **kwargs):
    r"""Matrix-free **BiCGSTAB** for the (non-symmetric) PII ``Q``.

    Like :func:`gmres` but with the BiCGSTAB iteration (cheaper per step, no growing Krylov basis;
    can break down on hard/non-normal spectra).  ``kwargs`` are forwarded to
    :func:`jax.scipy.sparse.linalg.bicgstab`.

    Args:
        A: a ``PCTJacobian`` operator (or a dense matrix); only ``A @ v`` is used.
        b: the right-hand side ``½∇E``.
        x0: optional initial guess.

    Returns:
        ``(x, info)`` from :func:`jax.scipy.sparse.linalg.bicgstab`.
    """
    return jax.scipy.sparse.linalg.bicgstab(lambda v: A @ v, b, x0=x0, **kwargs)
