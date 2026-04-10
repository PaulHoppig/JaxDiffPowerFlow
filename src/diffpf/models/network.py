"""
Backward-compatibility shim.

All types have moved to ``diffpf.core.types``.
This module re-exports them so that existing imports continue to work.
"""

from diffpf.core.types import (  # noqa: F401
    BusSpec,
    CompiledTopology,
    LineSpec,
    NetworkParams,
    NetworkSpec,
    PFState,
)
