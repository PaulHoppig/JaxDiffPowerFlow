"""Tests for solver-independent power-flow observables."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from diffpf.core import build_ybus, calc_power_injection, power_flow_observables, state_to_voltage


def test_observables_match_bus_power_balance(solved_three_bus):
    topology, params, solution, _, _ = solved_three_bus
    observables = power_flow_observables(topology, params, solution)
    voltage = state_to_voltage(topology, params, solution)
    s_bus = calc_power_injection(build_ybus(topology, params), voltage)

    np.testing.assert_allclose(
        float(observables.slack_p_pu),
        float(jnp.real(s_bus[topology.slack_bus])),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        float(observables.total_p_loss_pu),
        float(jnp.sum(jnp.real(s_bus))),
        rtol=1e-10,
        atol=1e-10,
    )


def test_observables_have_expected_shapes(solved_three_bus):
    topology, params, solution, _, _ = solved_three_bus
    observables = power_flow_observables(topology, params, solution)

    assert observables.voltage_mag_pu.shape == topology.variable_buses.shape
    assert observables.voltage_angle_rad.shape == topology.variable_buses.shape
    assert observables.line_p_from_pu.shape == topology.from_bus.shape
    assert observables.line_q_from_pu.shape == topology.from_bus.shape
