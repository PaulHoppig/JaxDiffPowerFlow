"""
Parser: RawNetwork → (CompiledTopology, NetworkParams, PFState).

This is the only module that bridges the raw Python world (dicts,
JSON-sourced dataclasses) and the JAX array world.  ``core/`` never
imports from here.

Responsibilities
----------------
1. Convert physical units (MW, MVAR) to per-unit via ``BaseValues``.
2. Build a contiguous bus-index map from potentially non-contiguous IDs.
3. Decompose the slack voltage setpoint into rectangular Vr/Vi.
4. Delegate the actual JAX array construction to ``compile_network``.
5. Return a flat-start ``PFState`` as a convenient default initial guess.

Public surface
--------------
``parse(raw)``         – RawNetwork → (topology, params, state)
``load_network(path)`` – convenience one-liner: path → same triple
"""

from __future__ import annotations

import math
from pathlib import Path

import jax.numpy as jnp

from diffpf.compile.network import compile_network
from diffpf.core.types import BusSpec, LineSpec, NetworkSpec, PFState
from diffpf.core.units import BaseValues
from diffpf.io.reader import RawBus, RawNetwork, load_json

# ---------------------------------------------------------------------------
# Internal conversion helpers
# ---------------------------------------------------------------------------


def _slack_rectangular(bus: RawBus) -> tuple[float, float]:
    """Return (vr_pu, vi_pu) for the slack bus from polar setpoint."""
    ang_rad = math.radians(bus.v_ang_deg)
    return bus.v_mag_pu * math.cos(ang_rad), bus.v_mag_pu * math.sin(ang_rad)


def _build_spec(raw: RawNetwork, base: BaseValues) -> NetworkSpec:
    """
    Convert a validated ``RawNetwork`` into a ``NetworkSpec``.

    Bus ordering follows the sorted ``raw.buses`` list (ascending id).
    Line endpoints are remapped from external IDs to internal 0-based indices.
    """
    # External bus id → internal 0-based index
    id_to_idx: dict[int, int] = {bus.id: idx for idx, bus in enumerate(raw.buses)}

    buses = tuple(
        BusSpec(name=b.name, is_slack=(b.type == "slack")) for b in raw.buses
    )
    lines = tuple(
        LineSpec(
            from_bus=id_to_idx[ln.from_bus],
            to_bus=id_to_idx[ln.to_bus],
            r_pu=ln.r_pu,
            x_pu=ln.x_pu,
            b_shunt_pu=ln.b_shunt_pu,
        )
        for ln in raw.lines
    )

    # Sign convention: JSON uses generator sign (+ = inject into bus).
    # BaseValues converts MW/MVAR → p.u. with the same sign.
    p_spec = tuple(base.mw_to_pu(b.p_mw) for b in raw.buses)
    q_spec = tuple(base.mvar_to_pu(b.q_mvar) for b in raw.buses)

    slack_bus = next(b for b in raw.buses if b.type == "slack")
    slack_vr, slack_vi = _slack_rectangular(slack_bus)

    return NetworkSpec(
        buses=buses,
        lines=lines,
        p_spec_pu=p_spec,
        q_spec_pu=q_spec,
        slack_vr_pu=slack_vr,
        slack_vi_pu=slack_vi,
    )


def _flat_start(n_var: int) -> PFState:
    """Flat-start initial guess: Vr = 1, Vi = 0 for all non-slack buses."""
    return PFState(
        vr_pu=jnp.ones(n_var, dtype=jnp.float64),
        vi_pu=jnp.zeros(n_var, dtype=jnp.float64),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse(raw: RawNetwork):
    """
    Convert a validated ``RawNetwork`` into JAX-ready structures.

    Parameters
    ----------
    raw : RawNetwork
        Output of ``reader.load_json()``.

    Returns
    -------
    topology : CompiledTopology
        Static integer index arrays.
    params : NetworkParams
        Differentiable float64 parameter arrays.
    state : PFState
        Flat-start initial voltage state for Newton-Raphson.
    """
    base = BaseValues(s_mva=raw.base.s_mva, v_kv=raw.base.v_kv)
    spec = _build_spec(raw, base)
    topology, params = compile_network(spec)
    state = _flat_start(topology.variable_buses.shape[0])
    return topology, params, state


def load_network(path: str | Path):
    """
    One-shot loader: JSON path → (topology, params, state).

    This is the primary entry point for users loading a network from disk.

    Parameters
    ----------
    path : str | Path
        Path to the network JSON file.

    Returns
    -------
    topology : CompiledTopology
    params : NetworkParams
    state : PFState  (flat-start)

    Example
    -------
    >>> topology, params, state = load_network("cases/three_bus_poc.json")
    >>> solution, norm, loss = solve_power_flow(topology, params, state)
    """
    return parse(load_json(path))
