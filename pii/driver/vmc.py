r"""``pii.driver.VMC`` — the standard preconditioner-based VMC driver.

A thin subclass of NetKet's classic :class:`netket.driver.VMC` (the analogue of
``netket/driver/vmc.py``).  It carries no PII-specific logic itself: pass
:class:`pii.optimizer.PII` as the ``preconditioner`` to run Projected Inverse Iteration,
or :class:`netket.optimizer.SR` to run Stochastic Reconfiguration — exactly as you would with
NetKet's own ``VMC`` + ``SR``.  This keeps PII drop-in compatible with the NetKet driver/optimizer
architecture; for the *integrated* PII driver (analogue of ``netket.driver.VMC_SR``) see
:class:`pii.driver.VMC_PII`.
"""

from netket.driver import VMC as _NetKetVMC


class VMC(_NetKetVMC):
    r"""Standard VMC driver (NetKet's preconditioner-based ``VMC``).

    Use with a ``preconditioner``:

    - :class:`pii.optimizer.PII` → Projected Inverse Iteration,
    - :class:`netket.optimizer.SR` → Stochastic Reconfiguration.

    Example::

        vs = nk.vqs.MCState(sampler, model, n_samples=512)
        gs = pii.driver.VMC(
            H, optax.sgd(1.0), variational_state=vs,
            preconditioner=pii.optimizer.PII(H, tau=1.2 * E0, diag_shift=1e-4),
        )
        gs.run(n_iter=300)
    """
