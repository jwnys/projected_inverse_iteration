r"""Differentiable local energy and the PII ``A``-function.

The new ingredient of PII compared to SR is the derivative of the local energy
:math:`E_{L,\theta} = \partial_\theta E_L`.  The :math:`A` matrix of the paper
(Eq. 28) has rows

.. math::
    A_n = E_{L,\theta}(x_n) + E_L(x_n)\,(J(x_n) - \langle J\rangle),

where :math:`J = \partial_\theta \log\psi`.  Rather than computing
:math:`E_{L,\theta}` and the :math:`E_L\,J` term separately (and worrying about
real/imaginary channel mixing and parameter flattening), we observe that

.. math::
    \partial_\theta\big[\,E_L + \overline{E_L}\,\log\psi\,\big] = E_{L,\theta} + E_L\,\partial_\theta\log\psi,

where :math:`\overline{E_L}` denotes ``stop_gradient(E_L)`` (a constant complex
coefficient).  Hence the Jacobian of the single scalar function

    ``f_A(variables, x) = E_L(x) + stop_gradient(E_L(x)) * log_psi(variables, x)``

is exactly :math:`A` (before centering/scaling).  Differentiating it through the
**same** :func:`netket.jax.jacobian` machinery used for ``O`` guarantees that the
two matrices share parameter flattening and real/imaginary channel layout, so
that ``H = Oᵀ A`` is well defined.

The local energy itself is obtained through NetKet's *dispatch* machinery
(:func:`~netket.vqs.mc.get_local_kernel` / :func:`~netket.vqs.mc.get_local_kernel_arguments`)
rather than a hand-rolled ``get_conn_padded`` loop, so PII works for any operator
that registers a local-estimator kernel (discrete, jax-discrete, continuous,
sums, ...).  For an :class:`~netket.vqs.MCState` the dispatch returns the kernel
arguments as the (jax) operator itself — sample-independent, hence safe to chunk
over samples.  Because :math:`E_L(x)` already sums over the connected
configurations internally, the per-sample Jacobian of ``f_A`` is a single ``[P]``
(or ``[2, P]``) vector with no ``n_conn`` axis: chunking over samples therefore
bounds peak memory to ``chunk_size · n_conn`` forward/backward passes, never a
``[M, n_conn, P]`` tensor.
"""

import jax

from netket.vqs import MCState
from netket.vqs.mc import get_local_kernel, get_local_kernel_arguments
from netket.vqs.mc.kernels import local_value_kernel_jax


def _kernel_and_args(vstate, operator):
    """Return ``(kernel, args)`` for the local energy of ``operator`` on ``vstate``.

    For an :class:`~netket.vqs.MCState` this uses NetKet's dispatch, which works
    for any registered operator type and (for discrete-jax/continuous operators)
    returns sample-independent ``args``. :class:`~netket.vqs.FullSumState` does not
    register kernels, so we fall back to the standard discrete-jax kernel with the
    operator's jax form as the (sample-independent) argument.
    """
    if isinstance(vstate, MCState):
        kernel = get_local_kernel(vstate, operator)
        _, args = get_local_kernel_arguments(vstate, operator)
        return kernel, args

    op = operator.to_jax_operator() if hasattr(operator, "to_jax_operator") else operator
    return local_value_kernel_jax, op


def make_local_energy_funs(vstate, operator):
    r"""Build the local-energy and PII ``A``-function for a fixed operator.

    Args:
        vstate: the variational state (provides the log-ψ apply function and,
            for an MCState, drives the kernel dispatch).
        operator: the Hamiltonian (any type with a registered local kernel).

    Returns a pair ``(f_EL, f_A)`` of apply-style functions
    ``f(variables, samples) -> (M,)`` suitable for :func:`netket.jax.jacobian`:

    - ``f_EL`` returns the local energies :math:`E_L`.
    - ``f_A`` returns :math:`E_L + \mathrm{sg}(E_L)\,\log\psi`, whose Jacobian
      w.r.t. the parameters is the PII matrix ``A`` (Eq. 28).
    """
    apply_fun = vstate._apply_fun
    kernel, args = _kernel_and_args(vstate, operator)

    def f_EL(variables, samples):
        return kernel(apply_fun, variables, samples, args)

    def f_A(variables, samples):
        eloc = f_EL(variables, samples)
        return eloc + jax.lax.stop_gradient(eloc) * apply_fun(variables, samples)

    return f_EL, f_A
