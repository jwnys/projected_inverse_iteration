"""Pytest configuration.

By default the (small) test suite runs on a single CPU device. To exercise
multi-device sharding, set ``PII_TEST_DEVICES=N`` (this must happen before JAX is
imported, which is why it is done here in ``conftest.py``)::

    PII_TEST_DEVICES=2 python -m pytest test/

NetKet's sharding (``NETKET_SHARDING=1``, the default) then distributes samples
across the ``N`` simulated devices, exactly as in NetKet's own sharding CI.
"""

import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

_n_devices = int(os.environ.get("PII_TEST_DEVICES", "1"))
if _n_devices > 1:
    _flags = os.environ.get("XLA_FLAGS", "")
    if "xla_force_host_platform_device_count" not in _flags:
        os.environ["XLA_FLAGS"] = (
            _flags + f" --xla_force_host_platform_device_count={_n_devices}"
        ).strip()
    os.environ["NETKET_SHARDING"] = "1"
