"""
Tests für diffpf.core.residuals  (Leistungsfluss-Residuen).

Abgedeckte Invarianten
----------------------
- state_to_voltage: Slack-Bus korrekt rekonstruiert, Dimension stimmt
- calc_power_injection: Leistungsbilanz (Summe S ≈ Verluste, nicht null)
- power_flow_residual: Form (2 * n_var,), Vorzeichen, Flat-Start endlich
- residual_loss: nicht-negativ, null genau beim konvergierten Zustand
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from diffpf.core import (
    build_ybus,
    calc_power_injection,
    power_flow_residual,
    residual_loss,
    state_to_voltage,
)


# ---------------------------------------------------------------------------
# state_to_voltage
# ---------------------------------------------------------------------------


def test_state_to_voltage_slack_correct(three_bus_case):
    topology, params, state = three_bus_case
    voltage = state_to_voltage(topology, params, state)
    slack = topology.slack_bus
    assert jnp.isclose(jnp.real(voltage[slack]), params.slack_vr_pu)
    assert jnp.isclose(jnp.imag(voltage[slack]), params.slack_vi_pu)


def test_state_to_voltage_full_dimension(three_bus_case):
    topology, params, state = three_bus_case
    voltage = state_to_voltage(topology, params, state)
    assert voltage.shape == (topology.n_bus,)
    assert voltage.dtype == jnp.complex128


def test_state_to_voltage_variable_buses_match_state(three_bus_case):
    """Nicht-Slack-Busse müssen exakt mit dem PFState übereinstimmen."""
    topology, params, state = three_bus_case
    voltage = state_to_voltage(topology, params, state)
    var = topology.variable_buses
    np.testing.assert_allclose(
        np.real(np.asarray(voltage[var])), np.asarray(state.vr_pu), atol=1e-15
    )
    np.testing.assert_allclose(
        np.imag(np.asarray(voltage[var])), np.asarray(state.vi_pu), atol=1e-15
    )


# ---------------------------------------------------------------------------
# calc_power_injection
# ---------------------------------------------------------------------------


def test_calc_power_injection_shape(three_bus_case):
    topology, params, state = three_bus_case
    y_bus = build_ybus(topology, params)
    voltage = state_to_voltage(topology, params, state)
    s = calc_power_injection(y_bus, voltage)
    assert s.shape == (topology.n_bus,)
    assert s.dtype == jnp.complex128


def test_calc_power_injection_at_solution_matches_spec(solved_three_bus):
    """Im konvergierten Zustand stimmt S_calc[non-slack] ≈ S_spec."""
    topology, params, solution, _, _ = solved_three_bus
    y_bus = build_ybus(topology, params)
    voltage = state_to_voltage(topology, params, solution)
    s_calc = calc_power_injection(y_bus, voltage)
    var = topology.variable_buses
    np.testing.assert_allclose(
        np.real(np.asarray(s_calc[var])),
        np.asarray(params.p_spec_pu[var]),
        atol=1e-7,
    )
    np.testing.assert_allclose(
        np.imag(np.asarray(s_calc[var])),
        np.asarray(params.q_spec_pu[var]),
        atol=1e-7,
    )


# ---------------------------------------------------------------------------
# power_flow_residual
# ---------------------------------------------------------------------------


def test_power_flow_residual_shape(three_bus_case):
    topology, params, state = three_bus_case
    r = power_flow_residual(topology, params, state)
    n_var = topology.variable_buses.shape[0]
    assert r.shape == (2 * n_var,)


def test_power_flow_residual_flat_start_finite(three_bus_case):
    """Flat-Start muss ein endliches (nicht-nan) Residuum liefern."""
    topology, params, state = three_bus_case
    r = power_flow_residual(topology, params, state)
    assert jnp.all(jnp.isfinite(r))


def test_power_flow_residual_near_zero_at_solution(solved_three_bus):
    topology, params, solution, _, _ = solved_three_bus
    r = power_flow_residual(topology, params, solution)
    np.testing.assert_allclose(np.asarray(r), 0.0, atol=1e-8)


# ---------------------------------------------------------------------------
# residual_loss
# ---------------------------------------------------------------------------


def test_residual_loss_non_negative(three_bus_case):
    topology, params, state = three_bus_case
    loss = residual_loss(topology, params, state)
    assert float(loss) >= 0.0


def test_residual_loss_is_scalar(three_bus_case):
    topology, params, state = three_bus_case
    loss = residual_loss(topology, params, state)
    assert loss.shape == ()


def test_residual_loss_near_zero_at_solution(solved_three_bus):
    topology, params, solution, _, _ = solved_three_bus
    loss = residual_loss(topology, params, solution)
    assert float(loss) < 1e-15


def test_residual_loss_equals_half_squared_norm(three_bus_case):
    """L = 0.5 * ||r||² muss mit direkter Berechnung übereinstimmen."""
    topology, params, state = three_bus_case
    r = power_flow_residual(topology, params, state)
    expected = 0.5 * float(jnp.vdot(r, r).real)
    np.testing.assert_allclose(float(residual_loss(topology, params, state)), expected, rtol=1e-12)
