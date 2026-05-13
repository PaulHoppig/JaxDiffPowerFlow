"""
Backward-compatibility shim.

All numerical functions have moved to ``diffpf.core.ybus`` and
``diffpf.core.residuals``. This module re-exports them so that
existing imports continue to work.
"""

from diffpf.core.residuals import (  # noqa: F401
    calc_power_injection,
    power_flow_residual,
    residual_loss,
    state_to_voltage,
)
from diffpf.core.ybus import build_ybus  # noqa: F401
