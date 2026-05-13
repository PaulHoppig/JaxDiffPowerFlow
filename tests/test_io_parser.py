"""Tests for the JSON reader and parser (diffpf.io)."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from diffpf.io import load_network
from diffpf.io.reader import load_json

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

THREE_BUS_JSON = Path(__file__).resolve().parents[1] / "cases" / "three_bus_poc.json"


# ---------------------------------------------------------------------------
# Reader tests
# ---------------------------------------------------------------------------


def test_load_json_returns_correct_bus_count():
    net = load_json(THREE_BUS_JSON)
    assert len(net.buses) == 3


def test_load_json_buses_sorted_by_id():
    net = load_json(THREE_BUS_JSON)
    ids = [b.id for b in net.buses]
    assert ids == sorted(ids)


def test_load_json_lines_sorted_by_id():
    net = load_json(THREE_BUS_JSON)
    ids = [ln.id for ln in net.lines]
    assert ids == sorted(ids)


def test_load_json_exactly_one_slack():
    net = load_json(THREE_BUS_JSON)
    slack_buses = [b for b in net.buses if b.type == "slack"]
    assert len(slack_buses) == 1


def test_load_json_base_values():
    net = load_json(THREE_BUS_JSON)
    assert net.base.s_mva == pytest.approx(1.0)
    assert net.base.v_kv == pytest.approx(0.4)


def test_load_json_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_json("/nonexistent/path/network.json")


def test_load_json_duplicate_bus_id_raises(tmp_path):
    bad = {
        "base": {"s_mva": 1.0, "v_kv": 0.4},
        "buses": [
            {"id": 0, "name": "slack", "type": "slack"},
            {"id": 0, "name": "dup", "type": "pq"},
        ],
        "lines": [],
    }
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="[Dd]uplikat|[Dd]uplicate|[Dd]oppelte"):
        load_json(p)


def test_load_json_no_slack_raises(tmp_path):
    bad = {
        "base": {"s_mva": 1.0, "v_kv": 0.4},
        "buses": [
            {"id": 0, "name": "a", "type": "pq"},
            {"id": 1, "name": "b", "type": "pq"},
        ],
        "lines": [],
    }
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="[Ss]lack"):
        load_json(p)


def test_load_json_zero_impedance_line_raises(tmp_path):
    bad = {
        "base": {"s_mva": 1.0, "v_kv": 0.4},
        "buses": [
            {"id": 0, "name": "slack", "type": "slack"},
            {"id": 1, "name": "load", "type": "pq"},
        ],
        "lines": [
            {"id": 0, "from_bus": 0, "to_bus": 1, "r_ohm": 0.0, "x_ohm": 0.0}
        ],
    }
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="[Ii]mpedanz|[Ii]mpedance"):
        load_json(p)


# ---------------------------------------------------------------------------
# Parser / load_network tests
# ---------------------------------------------------------------------------


def test_load_network_topology_shape():
    topology, params, state = load_network(THREE_BUS_JSON)
    assert topology.n_bus == 3
    assert topology.from_bus.shape == (3,)
    assert topology.variable_buses.shape == (2,)


def test_load_network_slack_bus_is_zero():
    topology, _, _ = load_network(THREE_BUS_JSON)
    assert topology.slack_bus == 0


def test_load_network_p_spec_matches_json():
    """p_mw values in JSON: [0.0, -0.9, 0.7] with s_mva=1 → same in p.u."""
    _, params, _ = load_network(THREE_BUS_JSON)
    np.testing.assert_allclose(
        np.asarray(params.p_spec_pu),
        [0.0, -0.9, 0.7],
        atol=1e-12,
    )


def test_load_network_slack_voltage_rectangular():
    """Slack: v_mag=1.0, ang=0° → vr=1.0, vi=0.0."""
    _, params, _ = load_network(THREE_BUS_JSON)
    assert float(params.slack_vr_pu) == pytest.approx(1.0)
    assert float(params.slack_vi_pu) == pytest.approx(0.0)


def test_load_network_flat_start_state():
    topology, _, state = load_network(THREE_BUS_JSON)
    n_var = topology.variable_buses.shape[0]
    np.testing.assert_array_equal(np.asarray(state.vr_pu), np.ones(n_var))
    np.testing.assert_array_equal(np.asarray(state.vi_pu), np.zeros(n_var))


def test_load_network_g_series_positive():
    """Series conductance must be positive for resistive lines."""
    _, params, _ = load_network(THREE_BUS_JSON)
    assert jnp.all(params.g_series_pu > 0)


def test_load_network_non_contiguous_ids_remap_correctly(tmp_path):
    """Bus IDs 10, 20, 30 must map to internal indices 0, 1, 2."""
    net_def = {
        "base": {"s_mva": 1.0, "v_kv": 0.4},
        "buses": [
            {"id": 10, "name": "slack", "type": "slack", "v_mag_pu": 1.0},
            {"id": 20, "name": "load",  "type": "pq", "p_mw": -0.5, "q_mvar": -0.1},
            {"id": 30, "name": "pv",    "type": "pq", "p_mw": 0.3,  "q_mvar": 0.0},
        ],
        "lines": [
            {"id": 0, "from_bus": 10, "to_bus": 20, "r_ohm": 0.02, "x_ohm": 0.04},
            {"id": 1, "from_bus": 10, "to_bus": 30, "r_ohm": 0.03, "x_ohm": 0.06},
        ],
    }
    p = tmp_path / "nc.json"
    p.write_text(json.dumps(net_def))
    topology, _, _ = load_network(p)
    assert topology.n_bus == 3
    # from_bus of line 0 should map to internal index 0 (id=10 → idx=0)
    assert int(topology.from_bus[0]) == 0
    assert int(topology.to_bus[0]) == 1


def test_load_network_roundtrip_with_solver():
    """Full integration: load from JSON, solve, check convergence."""
    from diffpf.solver import NewtonOptions, solve_power_flow

    topology, params, state = load_network(THREE_BUS_JSON)
    _, norm, _ = solve_power_flow(
        topology, params, state, NewtonOptions(max_iters=30, tolerance=1e-10)
    )
    assert float(norm) < 1e-8


# ---------------------------------------------------------------------------
# Neue Tests: physikalische Leitungsparameter (Form A + Form B)
# ---------------------------------------------------------------------------

# Hilfsfunktion: minimales 2-Bus-Netz als Dict konstruieren
def _two_bus_net(line_dict: dict, base_extra: dict | None = None) -> dict:
    base = {"s_mva": 1.0, "v_kv": 0.4}
    if base_extra:
        base.update(base_extra)
    return {
        "base": base,
        "buses": [
            {"id": 0, "name": "slack", "type": "slack"},
            {"id": 1, "name": "load",  "type": "pq", "p_mw": -0.5, "q_mvar": -0.1},
        ],
        "lines": [{"id": 0, "from_bus": 0, "to_bus": 1, **line_dict}],
    }


# ---- Form A ----------------------------------------------------------------


def test_form_a_parses_without_error(tmp_path):
    """Form A (r_ohm, x_ohm) muss ohne Fehler geladen werden."""
    p = tmp_path / "net.json"
    p.write_text(json.dumps(_two_bus_net({"r_ohm": 0.032, "x_ohm": 0.064})))
    raw = load_json(p)
    assert raw.lines[0].r_ohm == pytest.approx(0.032)
    assert raw.lines[0].x_ohm == pytest.approx(0.064)
    assert raw.lines[0].b_shunt_s is None  # optional, nicht angegeben


def test_form_a_pu_conversion_exact(tmp_path):
    """
    Form A: r_ohm=0.032 Ω, x_ohm=0.064 Ω, b_shunt_s=0.625 S
    mit S_base=1 MVA, V_base=0.4 kV  →  Z_base=0.16 Ω, Y_base=6.25 S
    →  r_pu=0.2, x_pu=0.4, b_shunt_pu=0.1
    """
    p = tmp_path / "net.json"
    p.write_text(json.dumps(_two_bus_net(
        {"r_ohm": 0.032, "x_ohm": 0.064, "b_shunt_s": 0.625}
    )))
    _, params, _ = load_network(p)
    np.testing.assert_allclose(float(params.g_series_pu[0]),
                               np.real(1.0 / complex(0.2, 0.4)), rtol=1e-10)
    np.testing.assert_allclose(float(params.b_shunt_pu[0]), 0.1, rtol=1e-10)


def test_form_a_b_shunt_defaults_to_zero(tmp_path):
    """Form A ohne b_shunt_s → b_shunt_pu == 0."""
    p = tmp_path / "net.json"
    p.write_text(json.dumps(_two_bus_net({"r_ohm": 0.01, "x_ohm": 0.02})))
    _, params, _ = load_network(p)
    assert float(params.b_shunt_pu[0]) == pytest.approx(0.0)


# ---- Form B mit b_shunt_s_per_km -------------------------------------------


def test_form_b_b_shunt_s_per_km_parses_and_converts(tmp_path):
    """
    Form B mit b_shunt_s_per_km:
      length=0.2 km, r=0.16 Ω/km, x=0.32 Ω/km, b_shunt=0.625 S/km
      → r_total=0.032 Ω, x_total=0.064 Ω, b_total=0.125 S
      → r_pu=0.2, x_pu=0.4, b_shunt_pu=0.02
    """
    p = tmp_path / "net.json"
    p.write_text(json.dumps(_two_bus_net({
        "length_km": 0.2,
        "r_ohm_per_km": 0.16,
        "x_ohm_per_km": 0.32,
        "b_shunt_s_per_km": 0.625,
    })))
    _, params, _ = load_network(p)
    # r_pu = 0.032 / 0.16 = 0.2
    np.testing.assert_allclose(
        float(params.g_series_pu[0]),
        np.real(1.0 / complex(0.2, 0.4)),
        rtol=1e-10,
    )
    # b_shunt_pu = 0.125 * 0.16 = 0.02
    np.testing.assert_allclose(float(params.b_shunt_pu[0]), 0.02, rtol=1e-10)


def test_form_b_no_shunt_defaults_to_zero(tmp_path):
    """Form B ohne Shunt-Angabe → b_shunt_pu == 0."""
    p = tmp_path / "net.json"
    p.write_text(json.dumps(_two_bus_net({
        "length_km": 0.1,
        "r_ohm_per_km": 0.2,
        "x_ohm_per_km": 0.1,
    })))
    _, params, _ = load_network(p)
    assert float(params.b_shunt_pu[0]) == pytest.approx(0.0)


# ---- Form B mit c_nf_per_km ------------------------------------------------


def test_form_b_c_nf_per_km_converts_correctly(tmp_path):
    """
    Form B mit c_nf_per_km:
      f=50 Hz, C=200 nF/km, length=1 km
      b_total = 2π·50·200e-9·1 = 6.2832e-5 S
      b_pu = 6.2832e-5 · 0.16 = 1.00531e-5
    """
    import math as _math
    f, c_nf, L = 50.0, 200.0, 1.0
    b_s_expected = 2.0 * _math.pi * f * c_nf * 1e-9 * L
    b_pu_expected = b_s_expected * 0.16   # × Z_base

    p = tmp_path / "net.json"
    p.write_text(json.dumps(_two_bus_net(
        {"length_km": L, "r_ohm_per_km": 0.1, "x_ohm_per_km": 0.05,
         "c_nf_per_km": c_nf},
        base_extra={"f_hz": f},
    )))
    _, params, _ = load_network(p)
    np.testing.assert_allclose(float(params.b_shunt_pu[0]), b_pu_expected, rtol=1e-10)


def test_form_b_c_nf_requires_f_hz(tmp_path):
    """c_nf_per_km ohne base.f_hz muss ValueError auslösen."""
    p = tmp_path / "net.json"
    p.write_text(json.dumps(_two_bus_net(
        {"length_km": 1.0, "r_ohm_per_km": 0.1, "x_ohm_per_km": 0.05,
         "c_nf_per_km": 200.0},
        # kein f_hz in base
    )))
    with pytest.raises(ValueError, match="f_hz"):
        load_json(p)


# ---- Fehlerhafte Eingaben --------------------------------------------------


def test_mixed_form_raises(tmp_path):
    """Form A und Form B gleichzeitig → ValueError."""
    p = tmp_path / "net.json"
    p.write_text(json.dumps(_two_bus_net({
        "r_ohm": 0.01,
        "x_ohm": 0.02,
        "length_km": 0.1,          # Form B
        "r_ohm_per_km": 0.1,
        "x_ohm_per_km": 0.05,
    })))
    with pytest.raises(ValueError, match="[Mm]isch"):
        load_json(p)


def test_no_form_raises(tmp_path):
    """Weder Form A noch Form B → ValueError."""
    p = tmp_path / "net.json"
    # Leitung hat nur Topologie, keine Parameter
    p.write_text(json.dumps(_two_bus_net({})))
    with pytest.raises(ValueError):
        load_json(p)


def test_form_b_both_shunt_specs_raises(tmp_path):
    """b_shunt_s_per_km und c_nf_per_km gleichzeitig → ValueError."""
    p = tmp_path / "net.json"
    p.write_text(json.dumps(_two_bus_net(
        {"length_km": 1.0, "r_ohm_per_km": 0.1, "x_ohm_per_km": 0.05,
         "b_shunt_s_per_km": 0.01, "c_nf_per_km": 200.0},
        base_extra={"f_hz": 50.0},
    )))
    with pytest.raises(ValueError, match="c_nf_per_km"):
        load_json(p)


def test_form_b_zero_impedance_after_expansion_raises(tmp_path):
    """Beläge > 0, aber length_km = 0 ist verboten (length_km > 0 Pflicht)."""
    p = tmp_path / "net.json"
    p.write_text(json.dumps(_two_bus_net({
        "length_km": 0.0,    # ungültig
        "r_ohm_per_km": 0.1,
        "x_ohm_per_km": 0.05,
    })))
    with pytest.raises(ValueError, match="length_km"):
        load_json(p)


def test_form_b_zero_per_km_values_raises(tmp_path):
    """r_ohm_per_km=0, x_ohm_per_km=0 → Nullimpedanz nach Expansion."""
    p = tmp_path / "net.json"
    p.write_text(json.dumps(_two_bus_net({
        "length_km": 1.0,
        "r_ohm_per_km": 0.0,
        "x_ohm_per_km": 0.0,
    })))
    with pytest.raises(ValueError, match="[Ii]mpedanz|[Ii]mpedance"):
        load_json(p)


def test_base_f_hz_validation(tmp_path):
    """base.f_hz <= 0 muss abgelehnt werden."""
    p = tmp_path / "net.json"
    p.write_text(json.dumps(_two_bus_net(
        {"r_ohm": 0.01, "x_ohm": 0.02},
        base_extra={"f_hz": -1.0},
    )))
    with pytest.raises(ValueError, match="f_hz"):
        load_json(p)
