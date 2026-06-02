r"""``pii.VMC``: a drop-in superset of NetKet's ``VMC_SR`` driver.

With ``pii=False`` this reproduces SR / minSR / on-the-fly SR by delegating to
NetKet's own ``sr`` / ``srt`` / ``srt_onthefly`` kernels.  With ``pii=True`` it
runs **Projected Inverse Iteration**, solving ``(H - τS + λI) ξ = ½∇E`` instead
of SR's ``S ξ = ∇E``.
"""

from typing import Any
from collections.abc import Callable
from warnings import warn

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax.flatten_util import ravel_pytree

from netket import jax as nkjax
from netket.hilbert import SpinOrbitalFermions
from netket.optimizer.solver import cholesky_with_fallback
from netket.vqs import MCState, FullSumState
from netket.utils import timing, struct
from netket.utils.types import ScalarOrSchedule, Optimizer, Array, PyTree
from netket.jax._jacobian.default_mode import JacobianMode
from netket.operator import AbstractOperator, to_sparse_cached
from netket import stats as nkstats

from netket._src.driver.abstract_optimization_driver import AbstractOptimizationDriver
from netket._src.stats.online_stats import OnlineStats
from netket._src.callbacks.auto_chunk_size import get_forward_operator
from netket._src.ngd.sr_srt_common import sr, srt, get_samples_and_pdf
from netket._src.ngd.srt_onthefly import srt_onthefly

from pii._ngd.common import _pii_common
from pii._ngd.pii_onthefly import pii_onthefly
from pii._ngd.local_energy import make_local_energy_funs


def _pii_default_solver(A: Array, b: Array):
    """General (non-symmetric) linear solve for the PII matrices ``Q`` / ``K``.

    With Monte Carlo estimation, PII's ``Q = H - τS + λI`` is in general neither
    symmetric nor positive semi-definite, so NetKet's default Cholesky solver is
    inappropriate. We use the general LU-based :func:`jax.scipy.linalg.solve`
    (``assume_a='gen'``). NetKet's defaults are kept untouched for SR/minSR.
    """
    return jsp.linalg.solve(A, b, assume_a="gen"), None


class VMC(AbstractOptimizationDriver):
    r"""Energy minimization via SR/minSR (``pii=False``) or Projected Inverse Iteration (``pii=True``).

    When ``pii=False`` the driver is mathematically identical to
    :class:`netket.driver.VMC_SR`. When ``pii=True`` the SR preconditioner ``S``
    is replaced by ``Q = H - τ S + λ I`` (paper Eq. 17/29), where ``H`` is the
    Hamiltonian projected onto the variational tangent space and ``τ`` is the
    inverse-iteration shift (an estimate of, or undershoot of, the ground-state
    energy ``E0``). PII is robust to small spectral gaps.
    """

    diag_shift: ScalarOrSchedule = struct.field(serialize=False)
    proj_reg: ScalarOrSchedule = struct.field(serialize=False)
    tau: ScalarOrSchedule = struct.field(serialize=False)
    momentum: ScalarOrSchedule = struct.field(serialize=False, default=None)

    _ham: AbstractOperator = struct.field(pytree_node=False, serialize=False)

    _pii: bool = struct.field(serialize=False)
    _mode: str = struct.field(serialize=False)
    _chunk_size_bwd: int | None = struct.field(serialize=False)
    _chunk_size_dEloc: int | None = struct.field(serialize=False)
    _use_ntk: bool = struct.field(serialize=False)
    _on_the_fly: bool = struct.field(serialize=False)
    _linear_solver: Any = struct.field(serialize=False)

    _loss_stats_online: OnlineStats | None
    _mcmc_convergence_diagnostics_ema_window: int = struct.field(
        serialize=False, default=50
    )
    _unravel_params_fn: Any = struct.field(serialize=False)

    _old_updates: PyTree = None
    info: Any | None = None

    def __init__(
        self,
        hamiltonian: AbstractOperator,
        optimizer: Optimizer,
        *,
        diag_shift: ScalarOrSchedule,
        pii: bool = False,
        tau: ScalarOrSchedule | None = None,
        proj_reg: ScalarOrSchedule | None = None,
        momentum: ScalarOrSchedule | None = None,
        linear_solver: Callable[[Array, Array], Array] | None = None,
        variational_state: MCState,
        chunk_size_bwd: int | None = None,
        chunk_size_dEloc: int | None = None,
        mode: JacobianMode | None = None,
        use_ntk: bool | None = None,
        on_the_fly: bool | None = None,
    ):
        r"""Initialize the driver.

        Args:
            hamiltonian: The Hamiltonian whose ground state is sought.
            optimizer: An optax optimizer (use ``optax.sgd`` for proper NGD/PII).
            variational_state: The variational state to optimize.
            diag_shift: Tikhonov regularization :math:`\lambda`. Scalar or schedule.
            pii: If ``True`` run Projected Inverse Iteration; if ``False`` (default)
                reproduce SR/minSR exactly.
            tau: The inverse-iteration shift :math:`\tau` (≈ ``E0``, often
                ``α·E0`` with ``α ≥ 1`` "undershoot"). Required when ``pii=True``.
                Scalar or schedule ``Callable[[int], float]``.
            proj_reg: SPRING projection regularization (SR/minSR only).
            momentum: SPRING damping factor in ``[0, 1]`` (``~0.8`` works well).
                Supported for PII too (PII-SPRING, paper App. G).
            linear_solver: Linear solver ``(A, b) -> (x, info)``. Defaults to
                :func:`netket.optimizer.solver.cholesky_with_fallback` when
                ``pii=False`` and to a general LU solve when ``pii=True`` (since
                the PII matrix is non-symmetric).
            chunk_size_bwd: Chunk size for the ``O`` Jacobian / NTK (backward pass).
            chunk_size_dEloc: Chunk size for the local-energy-derivative (``A``)
                computation. Defaults to ``chunk_size_bwd``.
            mode: Jacobian mode, ``'real'`` or ``'complex'``.
            use_ntk: Use the kernel-trick / minSR(minPII) form. Auto if ``None``.
            on_the_fly: Use the matrix-free implementation. Auto if ``None``.
        """
        super().__init__(variational_state, optimizer, minimized_quantity_name="Energy")

        self._pii = bool(pii)
        if self._pii and tau is None:
            raise ValueError("`tau` (the inverse-iteration shift) is required when `pii=True`.")

        if isinstance(variational_state, FullSumState):
            if use_ntk:
                raise ValueError(
                    "NTK/minSR/minPII makes no sense for a FullSumState and is not supported."
                )
            use_ntk = False
        if use_ntk is None:
            use_ntk = variational_state.n_parameters > variational_state.n_samples
            if jax.process_index() == 0:
                print("Automatic implementation choice: ", "NTK" if use_ntk else "QGT")

        if on_the_fly is None:
            on_the_fly = bool(use_ntk)
        elif on_the_fly and not use_ntk:
            raise ValueError("`on_the_fly=True` is only supported when `use_ntk=True`.")

        if mode is None and isinstance(variational_state.hilbert, SpinOrbitalFermions):
            warn(
                "`mode` not selected for a Fermionic system; falling back to 'complex'. "
                "For real-valued wavefunctions with a sign, `mode='real'` is cheaper."
            )

        if chunk_size_bwd is None:
            chunk_size_bwd = variational_state.chunk_size
        if chunk_size_dEloc is None:
            chunk_size_dEloc = chunk_size_bwd

        if linear_solver is None:
            linear_solver = _pii_default_solver if self._pii else cholesky_with_fallback

        self._ham = hamiltonian

        self.diag_shift = diag_shift
        self.proj_reg = proj_reg
        self.tau = tau
        self.momentum = momentum

        self.chunk_size_bwd = chunk_size_bwd
        self.chunk_size_dEloc = chunk_size_dEloc
        self._use_ntk = use_ntk
        self.mode = mode
        self._on_the_fly = on_the_fly
        self._linear_solver = linear_solver

        _, unravel_params_fn = ravel_pytree(self.state.parameters)
        self._unravel_params_fn = jax.jit(unravel_params_fn)

        self._old_updates = None
        self.info = None

        params_structure = jax.tree_util.tree_map(
            lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), self.state.parameters
        )
        if not nkjax.tree_ishomogeneous(params_structure):
            raise ValueError(
                "This driver only supports all-real or all-complex parameters."
            )
        if self._mode == "real" and nkjax.tree_leaf_iscomplex(params_structure):
            raise ValueError(
                "`mode='real'` requires real-valued parameters: it is the "
                "truncated-phase mode for sign-structured *real* wavefunctions. "
                "Your variational state has complex parameters, so use "
                "`mode='complex'` (or `mode=None` to auto-detect)."
            )
        if self._mode == "real":
            # mode='real' keeps only Re(log ψ); if the ansatz has a complex
            # log-amplitude (e.g. RBMRealParams) this silently discards the phase
            # and gives wrong energies. Warn loudly (it is only correct for
            # real-output, sign-structured wavefunctions).
            test_out = self.state._apply_fun(
                self.state.variables,
                self.state.hilbert.random_state(jax.random.key(0), 2),
            )
            if jnp.iscomplexobj(test_out):
                warn(
                    "`mode='real'` is being used with an ansatz that has a "
                    "complex-valued log-amplitude: this truncates the phase and "
                    "is almost certainly wrong (it is only valid for real-output, "
                    "sign-structured wavefunctions). Use `mode='complex'` "
                    "(or `mode=None` to auto-detect).",
                    stacklevel=2,
                )

        self._loss_stats = None
        self._loss_stats_online = None

    @timing.timed
    def compute_loss_and_update(self):
        # --- local energies and loss statistics (identical to VMC_SR) ---
        if isinstance(self.state, FullSumState):
            Ô = to_sparse_cached(self._ham)
            Ψ = self.state.to_array()
            OΨ = Ô @ Ψ
            expval_O = (Ψ.conj() * OΨ).sum()
            local_energies = OΨ / Ψ
            variance = jnp.sum(jnp.abs(OΨ - expval_O * Ψ) ** 2)
            self._loss_stats = nkstats.Stats(
                mean=expval_O, error_of_mean=0.0, variance=variance
            )
        else:
            local_energies = self.state.local_estimators(self._ham)
            self._loss_stats = nkstats.statistics(local_energies)
            decay = self._mcmc_convergence_diagnostics_ema_decay
            if decay is not None:
                self._loss_stats_online = nkstats.online_statistics(
                    local_energies,
                    old_estimator=self._loss_stats_online,
                    decay=decay,
                    max_lag=32,
                )

        # --- resolve (possibly scheduled) hyperparameters ---
        diag_shift = self.diag_shift
        proj_reg = self.proj_reg
        momentum = self.momentum
        tau = self.tau
        if callable(diag_shift):
            diag_shift = diag_shift(self.step_count)
        if callable(proj_reg):
            proj_reg = proj_reg(self.step_count)
        if callable(momentum):
            momentum = momentum(self.step_count)
        if callable(tau):
            tau = tau(self.step_count)

        samples, pdf = get_samples_and_pdf(self.state)

        if not self._pii:
            # --- SR / minSR / on-the-fly SR: delegate to NetKet ---
            if self.use_ntk:
                compute_fun = srt_onthefly if self.on_the_fly else srt
            else:
                compute_fun = sr
            self._dp, self._old_updates, self.info = compute_fun(
                self.state._apply_fun,
                local_energies,
                self.state.parameters,
                self.state.model_state,
                samples,
                diag_shift=diag_shift,
                solver_fn=self._linear_solver,
                mode=self.mode,
                proj_reg=proj_reg,
                momentum=momentum,
                old_updates=self._old_updates,
                chunk_size=self.chunk_size_bwd,
                weights=pdf,
            )
            return self._loss_stats, self._dp

        # --- Projected Inverse Iteration ---
        _, f_A = make_local_energy_funs(self.state, self._ham)

        if self.on_the_fly:
            self._dp, self._old_updates, self.info = pii_onthefly(
                self.state._apply_fun,
                f_A,
                local_energies,
                self.state.parameters,
                self.state.model_state,
                samples,
                tau=tau,
                diag_shift=diag_shift,
                solver_fn=self._linear_solver,
                mode=self.mode,
                proj_reg=proj_reg,
                momentum=momentum,
                old_updates=self._old_updates,
                chunk_size=self.chunk_size_bwd,
                chunk_size_dEloc=self.chunk_size_dEloc,
                weights=pdf,
            )
        else:
            self._dp, self._old_updates, self.info = _pii_common(
                self.state._apply_fun,
                f_A,
                local_energies,
                self.state.parameters,
                self.state.model_state,
                samples,
                tau=tau,
                diag_shift=diag_shift,
                solver_fn=self._linear_solver,
                mode=self.mode,
                proj_reg=proj_reg,
                momentum=momentum,
                old_updates=self._old_updates,
                chunk_size=self.chunk_size_bwd,
                chunk_size_dEloc=self.chunk_size_dEloc,
                use_ntk=self.use_ntk,
                weights=pdf,
            )
        return self._loss_stats, self._dp

    @timing.timed
    def _log_additional_data(self, log_dict: dict):
        super()._log_additional_data(log_dict)
        if self.info is not None:
            log_dict["info"] = self.info
        if self._loss_stats_online is not None:
            log_dict[self._loss_name + "_ema"] = self._loss_stats_online

    @property
    def _mcmc_convergence_diagnostics_ema_decay(self) -> float | None:
        if not hasattr(self.state, "sampler"):
            return None
        if self.state.chain_length >= self._mcmc_convergence_diagnostics_ema_window:
            return None
        return max(
            0.5,
            min(
                0.99,
                1.0
                - self.state.chain_length / self._mcmc_convergence_diagnostics_ema_window,
            ),
        )

    @property
    def pii(self) -> bool:
        """Whether Projected Inverse Iteration (``True``) or SR (``False``) is used."""
        return self._pii

    @property
    def mode(self) -> JacobianMode:
        """Jacobian mode: ``'real'`` or ``'complex'``."""
        return self._mode

    @mode.setter
    def mode(self, mode: str | JacobianMode | None):
        if mode is None:
            mode = nkjax.jacobian_default_mode(
                self.state._apply_fun,
                self.state.parameters,
                self.state.model_state,
                self.state.hilbert.random_state(jax.random.key(1), 3),
                warn=False,
            )
        if mode not in ["complex", "real"]:
            raise ValueError(
                f"`mode` only supports 'real' and 'complex'. You gave {mode}"
            )
        self._mode = mode

    @property
    def on_the_fly(self) -> bool:
        """Whether the matrix-free (lazy NTK) implementation is used."""
        return self._on_the_fly

    @property
    def use_ntk(self) -> bool:
        """Whether the kernel-trick (minSR / minPII) form is used."""
        return self._use_ntk

    @property
    def chunk_size_bwd(self) -> int:
        """Chunk size for the ``O`` Jacobian / NTK (backward pass)."""
        return self._chunk_size_bwd

    @chunk_size_bwd.setter
    def chunk_size_bwd(self, value: int | None):
        if not isinstance(value, int | None):
            raise TypeError("chunk_size_bwd must be an integer or None.")
        elif isinstance(value, int) and value <= 0:
            raise ValueError("chunk_size_bwd must be a positive integer.")
        self._chunk_size_bwd = value

    @property
    def chunk_size_dEloc(self) -> int:
        """Chunk size for the local-energy-derivative (``A``) computation."""
        return self._chunk_size_dEloc

    @chunk_size_dEloc.setter
    def chunk_size_dEloc(self, value: int | None):
        if not isinstance(value, int | None):
            raise TypeError("chunk_size_dEloc must be an integer or None.")
        elif isinstance(value, int) and value <= 0:
            raise ValueError("chunk_size_dEloc must be a positive integer.")
        self._chunk_size_dEloc = value


@get_forward_operator.dispatch
def get_forward_operator_VMC_PII(driver: VMC):
    return driver._ham
