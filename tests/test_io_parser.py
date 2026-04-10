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
    with pytest.raises(ValueError, match="Duplicate bus IDs"):
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
    with pytest.raises(ValueError, match="slack bus"):
        load_json(p)


def test_load_json_zero_impedance_line_raises(tmp_path):
    bad = {
        "base": {"s_mva": 1.0, "v_kv": 0.4},
        "buses": [
            {"id": 0, "name": "slack", "type": "slack"},
            {"id": 1, "name": "load", "type": "pq"},
        ],
        "lines": [
            {"id": 0, "from_bus": 0, "to_bus": 1, "r_pu": 0.0, "x_pu": 0.0}
        ],
    }
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="impedance"):
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
            {"id": 0, "from_bus": 10, "to_bus": 20, "r_pu": 0.02, "x_pu": 0.04},
            {"id": 1, "from_bus": 10, "to_bus": 30, "r_pu": 0.03, "x_pu": 0.06},
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
