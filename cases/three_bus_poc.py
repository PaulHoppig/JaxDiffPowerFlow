"""Three-bus demonstrator for the differentiable AC power-flow core."""

from __future__ import annotations

import jax.numpy as jnp

from diffpf.compile import compile_network
from diffpf.models import BusSpec, LineSpec, NetworkSpec, PFState
from diffpf.numerics import build_ybus, calc_power_injection, residual_loss, state_to_voltage
from diffpf.solver import NewtonOptions, solve_power_flow


def build_three_bus_case():
    """Return a compiled 3-bus case with load and PV injection."""

    spec = NetworkSpec(
        buses=(
            BusSpec(name="grid", is_slack=True),
            BusSpec(name="load"),
            BusSpec(name="pv"),
        ),
        lines=(
            LineSpec(from_bus=0, to_bus=1, r_pu=0.02, x_pu=0.04, b_shunt_pu=0.02),
            LineSpec(from_bus=1, to_bus=2, r_pu=0.015, x_pu=0.03, b_shunt_pu=0.015),
            LineSpec(from_bus=0, to_bus=2, r_pu=0.03, x_pu=0.06, b_shunt_pu=0.01),
        ),
        p_spec_pu=(0.0, -0.9, 0.7),
        q_spec_pu=(0.0, -0.3, -0.05),
    )
    topology, params = compile_network(spec)
    initial_state = PFState(
        vr_pu=jnp.ones((topology.variable_buses.shape[0],), dtype=jnp.float64),
        vi_pu=jnp.zeros((topology.variable_buses.shape[0],), dtype=jnp.float64),
    )
    return topology, params, initial_state


def solve_three_bus_case():
    """Solve the 3-bus demonstrator and derive typical output quantities."""

    topology, params, initial_state = build_three_bus_case()
    solution, residual_norm, loss = solve_power_flow(
        topology,
        params,
        initial_state,
        NewtonOptions(max_iters=30, tolerance=1e-10, damping=1.0),
    )
    voltage = state_to_voltage(topology, params, solution)
    y_bus = build_ybus(topology, params)
    s_injection = calc_power_injection(y_bus, voltage)
    p_grid = jnp.real(s_injection[topology.slack_bus])
    voltage_mag = jnp.abs(voltage)
    p_loss = jnp.sum(jnp.real(s_injection))
    return {
        "solution": solution,
        "residual_norm": residual_norm,
        "residual_loss": loss,
        "p_grid_pu": p_grid,
        "voltage_mag_pu": voltage_mag,
        "p_loss_pu": p_loss,
    }

