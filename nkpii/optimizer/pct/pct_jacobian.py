r"""Constructors for the PII PCT operators (analogue of NetKet's ``qgt_jacobian.py``).

``PCTJacobianDense(vstate, hamiltonian, *, tau, ...)`` and ``PCTJacobianPyTree(...)`` build the
two centered/√-rescaled Jacobians ``O`` (of ``logψ``) and ``A`` (of ``f_A``) — using the
*same* ``nkjax.jacobian`` machinery and conventions as NetKet's QGT and the existing dense
PII path — and return a :class:`PCTJacobianDenseT` / :class:`PCTJacobianPyTreeT` that lazily
represents ``Q = OᴴA − τ OᴴO (+ diag_shift·I)``.
"""

import jax

from netket import jax as nkjax
from netket.utils import timing

from nkpii.ngd.local_energy import make_local_energy_funs

from .pct_jacobian_dense import PCTJacobianDenseT
from .pct_jacobian_pytree import PCTJacobianPyTreeT


@timing.timed
def PCTJacobian_DefaultConstructor(
    apply_fun,
    f_A,
    parameters,
    model_state,
    samples,
    pdf=None,
    *,
    dense: bool,
    tau,
    mode: str | None = None,
    holomorphic: bool | None = None,
    diag_shift: float | None = 0.0,
    chunk_size: int | None = None,
    chunk_size_dEloc: int | None = None,
):
    """Build a :class:`PCTJacobianDenseT` / :class:`PCTJacobianPyTreeT` from raw pieces.

    Mirrors :func:`netket.optimizer.qgt.QGTJacobian_DefaultConstructor`, but builds **two**
    Jacobians (``O`` from ``apply_fun``, ``A`` from ``f_A``) and stores the shift ``tau``.
    ``mode`` is ``'real'``, ``'complex'`` or ``'holomorphic'`` (auto-detected if ``None``).
    """
    if mode is not None and holomorphic is not None:
        raise ValueError("Cannot specify both `mode` and `holomorphic`.")
    if mode is None:
        mode = nkjax.jacobian_default_mode(
            apply_fun, parameters, model_state, samples, holomorphic=holomorphic
        )
    if mode not in ("real", "complex", "holomorphic"):
        raise ValueError(
            f"PCTJacobian supports mode='real'/'complex'/'holomorphic', got {mode!r}."
        )

    if pdf is not None:
        if pdf.shape != samples.shape[:-1]:
            raise ValueError(
                f"pdf.shape={pdf.shape} must match samples.shape[:-1]={samples.shape[:-1]}"
            )
        if pdf.ndim >= 2:
            pdf = jax.jit(jax.lax.collapse, static_argnums=(1, 2))(pdf, 0, 2)
    if samples.ndim >= 3:
        samples = jax.jit(jax.lax.collapse, static_argnums=(1, 2))(samples, 0, 2)

    if chunk_size_dEloc is None:
        chunk_size_dEloc = chunk_size

    jac_kwargs = dict(
        mode=mode, pdf=pdf, dense=dense, center=True, _sqrt_rescale=True
    )
    O = nkjax.jacobian(
        apply_fun, parameters, samples, model_state, chunk_size=chunk_size, **jac_kwargs
    )
    A = nkjax.jacobian(
        f_A, parameters, samples, model_state, chunk_size=chunk_size_dEloc, **jac_kwargs
    )

    pars_struct = jax.tree_util.tree_map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), parameters
    )

    QT = PCTJacobianDenseT if dense else PCTJacobianPyTreeT
    return QT(
        O=O,
        A=A,
        tau=tau,
        scale=None,
        mode=mode,
        _params_structure=pars_struct,
        diag_shift=diag_shift if diag_shift is not None else 0.0,
    )


def _vstate_samples_pdf(vstate, chunk_size):
    from netket.vqs import FullSumState

    if isinstance(vstate, FullSumState):
        samples = vstate._all_states
        pdf = vstate.probability_distribution()
    else:
        samples = vstate.samples
        pdf = None
    if chunk_size is None:
        chunk_size = getattr(vstate, "chunk_size", None)
    return samples, pdf, chunk_size


def PCTJacobianDense(
    vstate,
    hamiltonian,
    *,
    tau,
    mode: str | None = None,
    holomorphic: bool | None = None,
    diag_shift: float | None = 0.0,
    chunk_size: int | None = None,
    chunk_size_dEloc: int | None = None,
) -> PCTJacobianDenseT:
    r"""Semi-lazy **dense** PII PCT operator ``Q = OᴴA − τ OᴴO (+ diag_shift·I)``.

    The two Jacobians ``O`` (of ``logψ``) and ``A`` (of ``f_A = E_L + sg(E_L)logψ``) are
    computed and stored densely; ``Q`` is materialized only by ``.to_dense()``. ``Q @ v`` and
    ``Q.H @ v`` are lazy. Solve with ``Q.solve(solver, rhs)`` where ``rhs = ½∇E`` (e.g. a
    GMRES/dense solver, since ``Q`` is non-symmetric).

    Args:
        vstate: the variational state.
        hamiltonian: the Hamiltonian (defines ``f_A``).
        tau: the inverse-iteration shift ``τ`` (≈ ground-state energy).
        mode: ``'real'``, ``'complex'`` or ``'holomorphic'`` (auto-detected if ``None``).
        diag_shift: Tikhonov shift added to the diagonal of ``Q``.
        chunk_size / chunk_size_dEloc: chunking of the ``O`` / ``A`` Jacobians.
    """
    _, f_A = make_local_energy_funs(vstate, hamiltonian)
    samples, pdf, chunk_size = _vstate_samples_pdf(vstate, chunk_size)
    return PCTJacobian_DefaultConstructor(
        vstate._apply_fun,
        f_A,
        vstate.parameters,
        vstate.model_state,
        samples,
        pdf=pdf,
        dense=True,
        tau=tau,
        mode=mode,
        holomorphic=holomorphic,
        diag_shift=diag_shift,
        chunk_size=chunk_size,
        chunk_size_dEloc=chunk_size_dEloc,
    )


def PCTJacobianPyTree(
    vstate,
    hamiltonian,
    *,
    tau,
    mode: str | None = None,
    holomorphic: bool | None = None,
    diag_shift: float | None = 0.0,
    chunk_size: int | None = None,
    chunk_size_dEloc: int | None = None,
) -> PCTJacobianPyTreeT:
    r"""Semi-lazy **PyTree** PII PCT operator (Jacobians stored as PyTrees).

    Same as :func:`PCTJacobianDense` but ``O`` and ``A`` are kept as PyTrees; ``Q`` is
    assembled only by ``.to_dense()``.
    """
    _, f_A = make_local_energy_funs(vstate, hamiltonian)
    samples, pdf, chunk_size = _vstate_samples_pdf(vstate, chunk_size)
    return PCTJacobian_DefaultConstructor(
        vstate._apply_fun,
        f_A,
        vstate.parameters,
        vstate.model_state,
        samples,
        pdf=pdf,
        dense=False,
        tau=tau,
        mode=mode,
        holomorphic=holomorphic,
        diag_shift=diag_shift,
        chunk_size=chunk_size,
        chunk_size_dEloc=chunk_size_dEloc,
    )
