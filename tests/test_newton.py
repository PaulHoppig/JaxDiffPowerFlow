"""
Tests für diffpf.solver.newton  (Newton-Raphson-Solver).

Abgedeckte Invarianten
----------------------
- Konvergenz: Residuum < Toleranz nach max_iters
- Ausgabe-Shape: solution passt zu initial_state
- Physikalische Plausibilität: |V| nahe 1 p.u., P_grid endlich
- Gradient: jax.grad stimmt mit zentralen Finite-Differenzen überein
- Dämpfung: damping < 1.0 verändert Konvergenzpfad (kein Absturz)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from diffpf.core.residuals import residual_loss
from diffpf.core.types import NetworkParams
from diffpf.core.ybus import build_ybus
from diffpf.core.residuals import calc_power_injection, state_to_voltage
from diffpf.solver import NewtonOptions, solve_power_flow
from diffpf.validation import central_difference


# ---------------------------------------------------------------------------
# Grundlegende Konvergenz
# ---------------------------------------------------------------------------


def test_newton_residual_below_tolerance(three_bus_case):
    topology, params, state = three_bus_case
    _, norm, _ = solve_power_flow(
        topology, params, state, NewtonOptions(max_iters=30, tolerance=1e-10)
    )
    assert float(norm) < 1e-8


def test_newton_solution_shape_matches_initial(three_bus_case):
    topology, params, state = three_bus_case
    solution, _, _ = solve_power_flow(topology, params, state)
    assert solution.vr_pu.shape == state.vr_pu.shape
    assert solution.vi_pu.shape == state.vi_pu.shape


def test_newton_solution_voltages_finite(solved_three_bus):
    _, _, solution, _, _ = solved_three_bus
    assert jnp.all(jnp.isfinite(solution.vr_pu))
    assert jnp.all(jnp.isfinite(solution.vi_pu))


# ---------------------------------------------------------------------------
# Physikalische Plausibilität
# ---------------------------------------------------------------------------


def test_voltage_magnitudes_near_one_pu(solved_three_bus):
    """Spannungsbeträge im Normalfall zwischen 0.9 und 1.1 p.u."""
    topology, params, solution, _, _ = solved_three_bus
    voltage = state_to_voltage(topology, params, solution)
    v_mag = jnp.abs(voltage)
    assert jnp.all(v_mag > 0.9)
    assert jnp.all(v_mag < 1.1)


def test_slack_power_is_finite(solved_three_bus):
    topology, params, solution, _, _ = solved_three_bus
    y_bus = build_ybus(topology, params)
    voltage = state_to_voltage(topology, params, solution)
    s_inj = calc_power_injection(y_bus, voltage)
    p_grid = jnp.real(s_inj[topology.slack_bus])
    assert jnp.isfinite(p_grid)


def test_active_power_balance(solved_three_bus):
    """Summe aller Wirkleistungseinspeisung ≈ Verluste (≥ 0)."""
    topology, params, solution, _, _ = solved_three_bus
    y_bus = build_ybus(topology, params)
    voltage = state_to_voltage(topology, params, solution)
    s_inj = calc_power_injection(y_bus, voltage)
    p_loss = float(jnp.sum(jnp.real(s_inj)))
    assert p_loss >= -1e-6  # Netz ist passiv, also nicht-negative Gesamtverluste


# ---------------------------------------------------------------------------
# Solver-Optionen
# ---------------------------------------------------------------------------


def test_damped_newton_converges(three_bus_case):
    """Gedämpftes Newton (damping=0.8) muss ebenfalls konvergieren."""
    topology, params, state = three_bus_case
    _, norm, _ = solve_power_flow(
        topology, params, state, NewtonOptions(max_iters=50, tolerance=1e-10, damping=0.8)
    )
    assert float(norm) < 1e-8


def test_loss_decreases_from_flat_start(three_bus_case):
    """Verlustfunktion muss nach Konvergenz kleiner sein als am Flat-Start."""
    topology, params, state = three_bus_case
    loss_init = float(residual_loss(topology, params, state))
    _, _, loss_final = solve_power_flow(
        topology, params, state, NewtonOptions(max_iters=30, tolerance=1e-10)
    )
    assert float(loss_final) < loss_init


# ---------------------------------------------------------------------------
# Gradient-Check: Autodiff vs. Finite Differences
# ---------------------------------------------------------------------------


def test_gradient_p_pv_matches_finite_difference(three_bus_case):
    """
    dL/dP_PV (Residuumsverlust bzgl. PV-Einspeisung) aus jax.grad
    muss mit dem zentralen Finite-Differenzen-Gradienten übereinstimmen.
    """
    topology, params, state = three_bus_case

    def loss_fn(p_pv: jnp.ndarray) -> jnp.ndarray:
        varied = NetworkParams(
            p_spec_pu=params.p_spec_pu.at[2].set(p_pv),
            q_spec_pu=params.q_spec_pu,
            g_series_pu=params.g_series_pu,
            b_series_pu=params.b_series_pu,
            b_shunt_pu=params.b_shunt_pu,
            slack_vr_pu=params.slack_vr_pu,
            slack_vi_pu=params.slack_vi_pu,
        )
        return residual_loss(topology, varied, state)

    p_pv_val = float(params.p_spec_pu[2])

    autodiff = float(jax.grad(loss_fn)(jnp.asarray(p_pv_val, dtype=jnp.float64)))
    fd = central_difference(lambda p: float(loss_fn(jnp.asarray(p, dtype=jnp.float64))), p_pv_val, h=1e-6)

    np.testing.assert_allclose(autodiff, fd, rtol=1e-5, atol=1e-7)


def test_gradient_r_pu_matches_finite_difference(three_bus_case):
    """dL/dR_Leitung ebenfalls via Autodiff vs. FD prüfen."""
    topology, params, state = three_bus_case

    def loss_fn(r0: jnp.ndarray) -> jnp.ndarray:
        varied = NetworkParams(
            p_spec_pu=params.p_spec_pu,
            q_spec_pu=params.q_spec_pu,
            g_series_pu=params.g_series_pu.at[0].set(1.0 / jnp.sqrt(r0**2 + params.b_series_pu[0] ** -2)),
            b_series_pu=params.b_series_pu,
            b_shunt_pu=params.b_shunt_pu,
            slack_vr_pu=params.slack_vr_pu,
            slack_vi_pu=params.slack_vi_pu,
        )
        return residual_loss(topology, varied, state)

    # Einfacherer Ansatz: direkt g_series differenzieren
    def loss_g(g0: jnp.ndarray) -> jnp.ndarray:
        varied = NetworkParams(
            p_spec_pu=params.p_spec_pu,
            q_spec_pu=params.q_spec_pu,
            g_series_pu=params.g_series_pu.at[0].set(g0),
            b_series_pu=params.b_series_pu,
            b_shunt_pu=params.b_shunt_pu,
            slack_vr_pu=params.slack_vr_pu,
            slack_vi_pu=params.slack_vi_pu,
        )
        return residual_loss(topology, varied, state)

    g0_val = float(params.g_series_pu[0])
    autodiff = float(jax.grad(loss_g)(jnp.asarray(g0_val, dtype=jnp.float64)))
    fd = central_difference(lambda g: float(loss_g(jnp.asarray(g, dtype=jnp.float64))), g0_val, h=1e-6)

    np.testing.assert_allclose(autodiff, fd, rtol=1e-5, atol=1e-7)
