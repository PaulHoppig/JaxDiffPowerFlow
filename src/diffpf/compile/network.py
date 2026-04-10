"""
Compile human-friendly NetworkSpec into static topology + differentiable params.

This module is the internal compiler used by ``diffpf.io.parser``.
Direct use is also fine for programmatic network construction (no JSON needed).
"""

from __future__ import annotations

import jax.numpy as jnp

from diffpf.core.types import CompiledTopology, NetworkParams, NetworkSpec


def _line_series_admittance(r_pu: float, x_pu: float) -> complex:
    z = complex(r_pu, x_pu)
    if abs(z) < 1e-12:
        raise ValueError("Line impedance must be non-zero (|z| < 1e-12 p.u.).")
    return 1.0 / z


def compile_network(spec: NetworkSpec) -> tuple[CompiledTopology, NetworkParams]:
    """
    Compile a ``NetworkSpec`` into JAX array structures.

    Parameters
    ----------
    spec : NetworkSpec
        Human-readable network definition with topology and per-unit parameters.

    Returns
    -------
    topology : CompiledTopology
        Static index arrays (not differentiated).
    params : NetworkParams
        Differentiable float64 parameter arrays.
    """
    n_bus = len(spec.buses)
    if n_bus == 0:
        raise ValueError("NetworkSpec must contain at least one bus.")

    slack_indices = [i for i, bus in enumerate(spec.buses) if bus.is_slack]
    if len(slack_indices) != 1:
        raise ValueError(f"Exactly one slack bus required, found {len(slack_indices)}.")

    if len(spec.p_spec_pu) != n_bus or len(spec.q_spec_pu) != n_bus:
        raise ValueError("p_spec_pu and q_spec_pu length must equal number of buses.")

    slack_bus = slack_indices[0]
    variable_buses = jnp.asarray(
        [i for i in range(n_bus) if i != slack_bus], dtype=jnp.int32
    )

    from_bus, to_bus, g_series, b_series, b_shunt = [], [], [], [], []
    for line in spec.lines:
        if not (0 <= line.from_bus < n_bus and 0 <= line.to_bus < n_bus):
            raise ValueError(
                f"Line ({line.from_bus}→{line.to_bus}): endpoint out of range."
            )
        if line.from_bus == line.to_bus:
            raise ValueError("Self-loops are not supported.")

        y = _line_series_admittance(line.r_pu, line.x_pu)
        from_bus.append(line.from_bus)
        to_bus.append(line.to_bus)
        g_series.append(y.real)
        b_series.append(y.imag)
        b_shunt.append(line.b_shunt_pu)

    topology = CompiledTopology(
        n_bus=n_bus,
        slack_bus=slack_bus,
        from_bus=jnp.asarray(from_bus, dtype=jnp.int32),
        to_bus=jnp.asarray(to_bus, dtype=jnp.int32),
        variable_buses=variable_buses,
    )
    params = NetworkParams(
        p_spec_pu=jnp.asarray(spec.p_spec_pu, dtype=jnp.float64),
        q_spec_pu=jnp.asarray(spec.q_spec_pu, dtype=jnp.float64),
        g_series_pu=jnp.asarray(g_series, dtype=jnp.float64),
        b_series_pu=jnp.asarray(b_series, dtype=jnp.float64),
        b_shunt_pu=jnp.asarray(b_shunt, dtype=jnp.float64),
        slack_vr_pu=jnp.asarray(spec.slack_vr_pu, dtype=jnp.float64),
        slack_vi_pu=jnp.asarray(spec.slack_vi_pu, dtype=jnp.float64),
    )
    return topology, params
