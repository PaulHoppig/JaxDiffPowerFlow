"""
Tests für diffpf.core.ybus  (Y-Bus-Aufbau) und diffpf.compile (Netz-Compiler).

Abgedeckte Invarianten
----------------------
- Topologie-Korrektheit: Busse, Slack, variable Busse
- Symmetrie der Y-Bus-Matrix (Reziprozität)
- Vorzeichen der Diagonale (passiv-resistive Netze)
- Off-Diagonal-Elemente negativ (admittive Kopplung)
- Einheiten: g_series = Re(1/z) muss mit Handrechnung übereinstimmen
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from diffpf.compile import compile_network
from diffpf.core import build_ybus
from diffpf.core.types import BusSpec, LineSpec, NetworkSpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_two_bus() -> tuple:
    """Minimalnetz: Slack + PQ-Bus, eine Leitung."""
    spec = NetworkSpec(
        buses=(BusSpec(name="slack", is_slack=True), BusSpec(name="load")),
        lines=(LineSpec(from_bus=0, to_bus=1, r_pu=0.02, x_pu=0.04),),
        p_spec_pu=(0.0, -0.5),
        q_spec_pu=(0.0, -0.1),
    )
    return compile_network(spec)


def _three_bus() -> tuple:
    """3-Bus-Netz ohne Shunt-Kapazitäten (b_shunt=0)."""
    spec = NetworkSpec(
        buses=(
            BusSpec(name="grid", is_slack=True),
            BusSpec(name="load"),
            BusSpec(name="pv"),
        ),
        lines=(
            LineSpec(from_bus=0, to_bus=1, r_pu=0.02, x_pu=0.04),
            LineSpec(from_bus=1, to_bus=2, r_pu=0.015, x_pu=0.03),
            LineSpec(from_bus=0, to_bus=2, r_pu=0.03, x_pu=0.06),
        ),
        p_spec_pu=(0.0, -0.9, 0.7),
        q_spec_pu=(0.0, -0.3, -0.05),
    )
    return compile_network(spec)


# ---------------------------------------------------------------------------
# Compiler-Tests
# ---------------------------------------------------------------------------


def test_compile_two_bus_topology():
    topology, params = _simple_two_bus()
    assert topology.n_bus == 2
    assert topology.slack_bus == 0
    assert topology.variable_buses.tolist() == [1]
    assert topology.from_bus.shape == (1,)


def test_compile_three_bus_topology():
    topology, params = _three_bus()
    assert topology.n_bus == 3
    assert topology.slack_bus == 0
    assert topology.variable_buses.tolist() == [1, 2]
    assert topology.from_bus.shape == (3,)


def test_compile_param_shapes():
    topology, params = _three_bus()
    assert params.p_spec_pu.shape == (3,)
    assert params.q_spec_pu.shape == (3,)
    assert params.g_series_pu.shape == (3,)
    assert params.b_series_pu.shape == (3,)
    assert params.b_shunt_pu.shape == (3,)


def test_compile_series_admittance_value():
    """g = Re(1/z), b = Im(1/z) für z = r + jx."""
    topology, params = _simple_two_bus()
    r, x = 0.02, 0.04
    z = complex(r, x)
    y = 1.0 / z
    np.testing.assert_allclose(float(params.g_series_pu[0]), y.real, rtol=1e-10)
    np.testing.assert_allclose(float(params.b_series_pu[0]), y.imag, rtol=1e-10)


def test_compile_slack_not_in_variable_buses():
    topology, _ = _three_bus()
    assert topology.slack_bus not in topology.variable_buses.tolist()


def test_compile_requires_exactly_one_slack():
    spec_no_slack = NetworkSpec(
        buses=(BusSpec(name="a"), BusSpec(name="b")),
        lines=(LineSpec(from_bus=0, to_bus=1, r_pu=0.01, x_pu=0.02),),
        p_spec_pu=(0.0, -0.5),
        q_spec_pu=(0.0, -0.1),
    )
    with pytest.raises(ValueError, match="slack"):
        compile_network(spec_no_slack)


def test_compile_rejects_self_loop():
    spec = NetworkSpec(
        buses=(BusSpec(name="slack", is_slack=True), BusSpec(name="load")),
        lines=(LineSpec(from_bus=0, to_bus=0, r_pu=0.01, x_pu=0.02),),
        p_spec_pu=(0.0, -0.5),
        q_spec_pu=(0.0, -0.1),
    )
    with pytest.raises(ValueError, match="[Ss]elf"):
        compile_network(spec)


# ---------------------------------------------------------------------------
# Y-Bus-Tests (via fixture)
# ---------------------------------------------------------------------------


def test_ybus_symmetry(three_bus_case):
    """Y-Bus muss für reziproke Netze symmetrisch sein."""
    topology, params, _ = three_bus_case
    y_bus = build_ybus(topology, params)
    np.testing.assert_allclose(np.asarray(y_bus), np.asarray(y_bus.T), atol=1e-12)


def test_ybus_diagonal_positive_real(three_bus_case):
    """Diagonale muss positiven Realteil haben (Passivität)."""
    topology, params, _ = three_bus_case
    y_bus = build_ybus(topology, params)
    assert jnp.all(jnp.real(jnp.diag(y_bus)) > 0)


def test_ybus_off_diagonal_negative_real():
    """Außerdiagonale Elemente haben negativen Realteil bei reinen Serienelementen."""
    topology, params = _three_bus()
    y_bus = build_ybus(topology, params)
    # Leitungen ohne Shunt: Y[i,j] = -y_series, Re(y_series) > 0
    np.testing.assert_array_less(np.real(np.asarray(y_bus[0, 1])), 0)
    np.testing.assert_array_less(np.real(np.asarray(y_bus[1, 0])), 0)


def test_ybus_shape(three_bus_case):
    topology, params, _ = three_bus_case
    y_bus = build_ybus(topology, params)
    assert y_bus.shape == (topology.n_bus, topology.n_bus)


def test_ybus_two_bus_diagonal_value():
    """Handrechnung für 2-Bus: Y[0,0] = y_series + y_shunt/2."""
    r, x, b = 0.02, 0.04, 0.06
    spec = NetworkSpec(
        buses=(BusSpec(name="slack", is_slack=True), BusSpec(name="load")),
        lines=(LineSpec(from_bus=0, to_bus=1, r_pu=r, x_pu=x, b_shunt_pu=b),),
        p_spec_pu=(0.0, -0.5),
        q_spec_pu=(0.0, -0.1),
    )
    topology, params = compile_network(spec)
    y_bus = build_ybus(topology, params)
    y_series = 1.0 / complex(r, x)
    expected_diag = y_series + 0.5j * b
    np.testing.assert_allclose(
        complex(y_bus[0, 0]), expected_diag, rtol=1e-10
    )
