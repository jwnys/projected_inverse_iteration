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
    solver = partial(pii.penrose_symmetrized_solver, diag_shift=1e-2)   # (Q, b) -> (x, info)
"""

import netket as nk
import jax.numpy as jnp
import jax.scipy as jsp
from functools import partial


def penrose_symmetrized_solver(Q, b, *, diag_shift, solver=None):
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
        solver = nk.optimizer.solver.cholesky
        # return jsp.linalg.solve(M, rhs, assume_a="pos"), None
    return solver(M, rhs)


def naive_symmetrized_solver(Q, b, *, diag_shift, solver=None):
    r"""'Naive' symmetrization: solve the Hermitian part ``½(Q+Qᴴ)`` of the PII matrix.

    Bind ``diag_shift`` (and optionally ``solver``) with :func:`functools.partial` to
    obtain a ``(Q, b) -> (x, info)`` solver for ``pii.VMC(linear_solver=...)``.

    The standard dense PII update solves ``Q ξ = ½∇E`` with ``Q = H - τS``.  In the
    infinite-sample limit ``Q`` is Hermitian (the Gram matrix of the inner product
    ``⟨v|Ĥ-τ|w⟩``); finite Monte-Carlo sampling adds an ``O(1/√M)`` anti-Hermitian part.
    This solver simply *imposes* that asymptotic symmetry by replacing ``Q`` with its
    Hermitian part and solving

    .. math::
        \big(\tfrac12 (Q + Q^H) + \mathrm{diag\_shift}\,I\big)\,\xi = b .

    Contrast with :func:`penrose_symmetrized_solver`, which forms the normal equations
    ``QᴴQ``.  This route is **additive** (the Hermitian part) rather than **multiplicative**
    (``QᴴQ``), so it does *not* square the condition number of ``Q`` — it keeps ``Q``'s
    spectral scale, and ``diag_shift`` therefore has **energy units** (like an ordinary PII
    ``diag_shift``, not the energy² of the Penrose variant).

    .. warning::
        ``½(Q+Qᴴ)`` is symmetric but **not positive-definite** in general.  This is a
        field-of-values effect: the (non-normal) ``Q`` is *positive-real* — all its
        eigenvalues lie in the right half-plane, which is why standard PII's ``(Q+λI)⁻¹``
        is well-behaved at tiny ``λ`` — yet the smallest eigenvalue of its Hermitian part
        is the leftmost point of ``Q``'s numerical range and dips **negative** (≈ −7e-4 in
        the small spin systems here).  So symmetrizing *introduces* a negative eigenvalue
        that ``Q`` itself does not have — the anti-Hermitian part one discards is precisely
        what keeps ``Q`` accretive.  The default inner solver is therefore a symmetric
        **LDL** solve (``assume_a='sym'``), which never NaNs on an indefinite ``M`` (unlike
        a Cholesky ``assume_a='pos'``, which would).  But a *too-small* ``diag_shift`` leaves
        ``M = ½(Q+Qᴴ)+λI`` **near-singular** (its smallest eigenvalue barely lifted above
        that negative one); the right-hand side excites that near-null mode and ``‖ξ‖``
        blows up (≈ 15× the standard-PII update at ``λ=1e-3``).  With the natural step size
        ``η=1`` the update then **overshoots** and the optimization oscillates / stalls — it
        stays finite, and the first-order ``∇Eᵀξ`` is even a descent; it is the *nonlinear*
        step that is too large.  Empirically ``diag_shift`` must be large enough to lift
        ``M`` comfortably positive-definite — ``O(0.1–1)`` here, *despite* the energy units —
        which shrinks ``‖ξ‖`` back to normal at the cost of over-regularization (slower
        convergence and a higher floor than standard PII).

    .. important::
        Use with ``diag_shift=0`` **on the driver** (a separate knob); the regularization is
        carried entirely by this solver's ``diag_shift``.

    .. note::
        ``Q`` is **always real** here — NetKet realifies ``mode='complex'`` by stacking the
        Re/Im channels into extra real rows/columns — so ``Qᴴ = Qᵀ`` and ``½(Q+Qᵀ)`` is the
        realified Hermitian part (``= realify(½(C+Cᴴ))``); the complex case is handled
        correctly by construction.  On the ``use_ntk``/``on_the_fly`` paths the solver
        instead receives the ``2M×2M`` NTK kernel ``K`` and symmetrizes ``½(K+Kᴴ)`` — a valid
        sibling, with the same indefiniteness caveat.

    Args:
        Q: the dense PII matrix (or the NTK kernel ``K``) passed by the kernel.
        b: the right-hand side ``½∇E`` (passed by the kernel).
        diag_shift: Tikhonov regularization added to ``½(Q+Qᴴ)`` (keyword-only;
            **energy units**).  Bind via ``partial``; use with driver ``diag_shift=0``.
        solver: optional inner solver ``(A, b) -> (x, info)`` for the symmetric system
            ``(½(Q+Qᴴ) + diag_shift·I) ξ = b``.  Defaults (``None``) to a symmetric LDL
            solve :func:`jax.scipy.linalg.solve` with ``assume_a='sym'``.

    Returns:
        ``ξ = (½(Q+Qᴴ) + diag_shift·I)⁻¹ b`` (a bare array for the default solver, or
        whatever a custom ``solver`` returns).
    """
    
    Qh = Q.conj().T
    side = Q.shape[-1]
    M = 0.5 * (Qh + Q) + diag_shift * jnp.eye(side, dtype=Q.dtype)
    if solver is None:
        solver = partial(jsp.linalg.solve, assume_a="sym")
    return solver(M, b)
        