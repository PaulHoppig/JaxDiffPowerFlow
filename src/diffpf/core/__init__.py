"""
diffpf.core – pure numerical primitives for the AC power-flow kernel.

Public re-exports cover types, units, Y-bus assembly, and residual
functions. Nothing in this package imports from ``io/``, ``models/``,
or any dict/JSON logic.
"""

from .residuals import (
    calc_power_injection,
    power_flow_residual,
    residual_loss,
    state_to_voltage,
)
from .observables import PowerFlowObservables, power_flow_observables
from .types import (
    BusSpec,
    CompiledTopology,
    LineSpec,
    NetworkParams,
    NetworkSpec,
    PFState,
    ShuntSpec,
    TrafoSpec,
)
from .units import BaseValues
from .ybus import build_ybus

__all__ = [
    # types
    "BusSpec",
    "LineSpec",
    "TrafoSpec",
    "ShuntSpec",
    "NetworkSpec",
    "CompiledTopology",
    "NetworkParams",
    "PFState",
    # units
    "BaseValues",
    # numerics
    "build_ybus",
    "state_to_voltage",
    "calc_power_injection",
    "power_flow_residual",
    "residual_loss",
    "PowerFlowObservables",
    "power_flow_observables",
]
