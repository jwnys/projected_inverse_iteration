r"""Custom linear solvers for the PII dense path.

These plug into ``pii.VMC(..., linear_solver=...)``.  A solver follows the same
contract as :func:`pii.driver._pii_default_solver`: it is a callable
``(A, b) -> (x, info)`` that solves ``A x = b`` (or, here, an approximation
thereof).  The dense PII kernel ends with ``updates = solver_fn(Q, ½∇E)``
(:func:`pii._ngd.pii_dense._compute_pii_update_dense`), so anything we do to the
pair ``(Q, b)`` here defines a new optimization variant without touching the
kernels or the driver dispatch.

Bind the keyword-only ``diag_shift`` with :func:`functools.partial`, e.g.::

    from functools import partial
    import pii
    solver = partial(pii.symmetrized_solver, diag_shift=1e-2)   # (Q, b) -> (x, info)
"""

import jax.numpy as jnp
import jax.scipy as jsp


def symmetrized_solver(Q, b, *, diag_shift, solver=None):
    r"""Regularized-least-squares ("symmetrized") solver for the dense PII matrix ``Q``.

    Bind ``diag_shift`` (and optionally ``solver``) with :func:`functools.partial` to
    obtain a ``(Q, b) -> (x, info)`` solver for ``pii.VMC(linear_solver=...)``.

    The standard dense PII update solves the (generally non-symmetric) system
    ``Q ξ = ½∇E`` with ``Q = H - τS``.  This solver instead returns the
    Tikhonov-regularized **Euclidean** least-squares solution

    .. math::
        \xi = (Q^H Q + \mathrm{diag\_shift}\,I)^{-1} Q^H\, b
            = \arg\min_\xi \; \|Q\xi - b\|_2^2 + \mathrm{diag\_shift}\,\|\xi\|_2^2 ,

    i.e. the Tikhonov pseudo-inverse of ``Q``.  As ``diag_shift → 0`` (and when
    ``Q`` is invertible) this equals ``Q⁻¹ b``; when ``Q`` is singular it stays
    well-defined and returns the minimum-norm least-squares solution
    (→ ``pinv(Q) b``).

    **Hermitian vs. plain transpose.**  The least-squares normal equations use the
    *conjugate* (Hermitian) transpose ``Qᴴ`` — for a genuinely complex ``Q`` a plain
    ``Qᵀ`` would be wrong (``QᴴQ`` is the Hermitian PSD object whose inverse defines
    the pseudo-inverse).  In this code ``Q`` is **always real**, even for complex
    wavefunctions or complex parameters, because of how NetKet handles ``mode='complex'``:
    it realifies by splitting Re/Im into extra *real* rows of the Jacobians ``O_L``,
    ``A_L`` (and, for complex parameters, extra real columns), so that
    ``S = OᵀO = Re⟨∂ᵢψ|∂ⱼψ⟩ = Re(S_ℂ)`` (paper Eq. 22) and likewise ``H = Re(H_ℂ)``,
    and hence ``Q = H - τS`` is real.  The solver therefore receives the **P×P** (or
    2P×2P for complex parameters) *parameter-space* ``Q`` — never the 2M×2M kernel
    ``K`` of the minPII path — with no complex entries left: every imaginary part now
    lives in a doubled real dimension.  Under that (isometric) realification the
    complex ``Qᴴ`` corresponds exactly to the real ``Qᵀ``, so ``Q.conj().T`` is a
    *no-op* here (``Q.conj() is Q`` for real ``Q``); we keep it only so the line is
    literally the Hermitian normal equations and stays correct if a genuinely complex
    matrix were ever passed.

    .. important::
        Use this with ``diag_shift=0`` **on the driver** (a separate knob).  The dense
        kernel forms ``Q = H - τS + driver_diag_shift·I``, so a nonzero driver
        ``diag_shift`` would square an *already regularized* ``Q``.  The
        regularization is carried entirely by this solver's ``diag_shift``, which has
        **energy² units** (``Q`` has energy units) — unlike the driver's energy-unit
        ``diag_shift`` — so it is tuned independently.  Forming ``QᴴQ`` squares the
        condition number of ``Q``, so this ``diag_shift`` typically needs to be
        substantially larger than a comparable driver ``diag_shift``.

    .. note::
        Dense path only: keep ``use_ntk=False``/``on_the_fly=False``.  On the
        kernel (minPII) or on-the-fly paths the solver would receive the ``2M×2M``
        kernel matrix instead of ``Q`` and compute something different.

    Args:
        Q: the dense PII matrix (passed by the kernel).
        b: the right-hand side ``½∇E`` (passed by the kernel).
        diag_shift: Tikhonov regularization on the normal equations (keyword-only;
            energy² units, ``> 0`` for an SPD system).  Bind via ``partial``.
        solver: optional inner solver ``(A, b) -> (x, info)`` for the SPD normal-
            equations system ``(QᴴQ + diag_shift·I) ξ = Qᴴb`` (e.g. a NetKet solver,
            a CG, or a custom least-squares routine).  Defaults (``None``) to a
            Cholesky solve :func:`jax.scipy.linalg.solve` with ``assume_a='pos'``.

    Returns:
        ``(x, info)`` where ``x = (QᴴQ + diag_shift·I)⁻¹ Qᴴ b`` (``info`` is ``None``
        for the default solver, else whatever ``solver`` returns).
    """
    # Hermitian transpose. NetKet realifies complex ψ/params into stacked real Re/Im
    # rows/cols, so the parameter-space Q here is real and this is just Qᵀ (.conj() is
    # a no-op); kept explicit so the math is the correct Qᴴ in general.
    Qh = Q.conj().T
    QhQ = Qh @ Q
    side = QhQ.shape[-1]
    M = QhQ + diag_shift * jnp.eye(side, dtype=QhQ.dtype)
    rhs = Qh @ b
    if solver is None:
        return jsp.linalg.solve(M, rhs, assume_a="pos"), None
    return solver(M, rhs)
