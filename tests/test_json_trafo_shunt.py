"""
Tests für das erweiterte JSON-Format mit Transformatoren und Shunts.

Testet reader.py (RawTrafo/RawShunt-Parsing und Validierung) sowie
parser.py (_raw_trafo_to_spec, _raw_shunt_to_spec) und compile_network.
"""

from __future__ import annotations

import json
import math

import pytest

from diffpf.io import load_network
from diffpf.io.reader import load_json


# ---------------------------------------------------------------------------
# Hilfs-Bausteine
# ---------------------------------------------------------------------------


def _minimal_net(
    *,
    buses: list[dict] | None = None,
    lines: list[dict] | None = None,
    trafos: list[dict] | None = None,
    shunts: list[dict] | None = None,
    base: dict | None = None,
) -> dict:
    """Baut ein minimales Netz-Dict mit optionalen Erweiterungen."""
    if buses is None:
        buses = [
            {"id": 0, "name": "slack", "type": "slack"},
            {"id": 1, "name": "load", "type": "pq", "p_mw": -0.5, "q_mvar": -0.1},
        ]
    if lines is None:
        lines = [
            {"id": 0, "from_bus": 0, "to_bus": 1, "r_ohm": 0.01, "x_ohm": 0.02}
        ]
    if base is None:
        base = {"s_mva": 1.0, "v_kv": 0.4}
    net = {"base": base, "buses": buses, "lines": lines}
    if trafos is not None:
        net["trafos"] = trafos
    if shunts is not None:
        net["shunts"] = shunts
    return net


def _minimal_trafo(
    hv_bus: int = 0,
    lv_bus: int = 1,
    sn_mva: float = 0.4,
    vn_hv_kv: float = 0.4,
    vn_lv_kv: float = 0.2,
    vk_percent: float = 4.0,
    vkr_percent: float = 1.0,
    **kwargs,
) -> dict:
    return {
        "id": 0,
        "hv_bus": hv_bus,
        "lv_bus": lv_bus,
        "sn_mva": sn_mva,
        "vn_hv_kv": vn_hv_kv,
        "vn_lv_kv": vn_lv_kv,
        "vk_percent": vk_percent,
        "vkr_percent": vkr_percent,
        **kwargs,
    }


def _three_bus_net_with_trafo() -> dict:
    """3-Bus-Netz: slack - HV-Bus - Trafo - LV-Bus."""
    return {
        "base": {"s_mva": 1.0, "v_kv": 110.0},
        "buses": [
            {"id": 0, "name": "slack", "type": "slack"},
            {"id": 1, "name": "hv",    "type": "pq"},
            {"id": 2, "name": "lv",    "type": "pq", "p_mw": -0.5, "q_mvar": -0.1},
        ],
        "lines": [
            {
                "id": 0,
                "from_bus": 0,
                "to_bus": 1,
                "r_ohm_per_km": 0.06,
                "x_ohm_per_km": 0.144,
                "length_km": 10.0,
            }
        ],
        "trafos": [
            {
                "id": 0,
                "hv_bus": 1,
                "lv_bus": 2,
                "sn_mva": 0.4,
                "vn_hv_kv": 110.0,
                "vn_lv_kv": 20.0,
                "vk_percent": 4.0,
                "vkr_percent": 1.0,
                "tap_ratio": 1.0,
                "shift_rad": 0.0,
            }
        ],
    }


# ---------------------------------------------------------------------------
# Test 1: JSON mit Trafo wird gelesen
# ---------------------------------------------------------------------------


def test_json_with_trafo_parses(tmp_path):
    """JSON mit trafo-Eintrag wird korrekt geparst."""
    net = _minimal_net(trafos=[_minimal_trafo()])
    # Brauchen 3 Busse für HV/LV zu trennen
    net["buses"] = [
        {"id": 0, "name": "slack", "type": "slack"},
        {"id": 1, "name": "hv",    "type": "pq"},
        {"id": 2, "name": "lv",    "type": "pq", "p_mw": -0.5, "q_mvar": -0.1},
    ]
    net["lines"] = [
        {"id": 0, "from_bus": 0, "to_bus": 1, "r_ohm": 0.05, "x_ohm": 0.1}
    ]
    net["trafos"] = [_minimal_trafo(hv_bus=1, lv_bus=2)]
    p = tmp_path / "net.json"
    p.write_text(json.dumps(net))

    raw = load_json(p)
    assert len(raw.trafos) == 1
    assert raw.trafos[0].sn_mva == pytest.approx(0.4)
    assert raw.trafos[0].vk_percent == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Test 2: JSON mit Shunt wird gelesen
# ---------------------------------------------------------------------------


def test_json_with_shunt_parses(tmp_path):
    """JSON mit shunt-Eintrag wird korrekt geparst."""
    net = _minimal_net(shunts=[{"id": 0, "bus": 1, "p_mw": 0.0, "q_mvar": -0.1}])
    p = tmp_path / "net.json"
    p.write_text(json.dumps(net))

    raw = load_json(p)
    assert len(raw.shunts) == 1
    assert raw.shunts[0].q_mvar == pytest.approx(-0.1)


# ---------------------------------------------------------------------------
# Test 3: Trafo + Shunt + Leitung → compile_network läuft
# ---------------------------------------------------------------------------


def test_json_trafo_shunt_compilable(tmp_path):
    """Trafo + Shunt + Leitung → compile_network ohne Fehler."""
    net = _three_bus_net_with_trafo()
    net["shunts"] = [{"id": 0, "bus": 1, "p_mw": 0.0, "q_mvar": -0.05}]
    p = tmp_path / "net.json"
    p.write_text(json.dumps(net))

    topology, params, state = load_network(p)
    assert topology.n_bus == 3
    assert len(params.trafo_hv_bus) == 1
    assert len(params.shunt_bus) == 1


# ---------------------------------------------------------------------------
# Test 4: Ungültiges sn_mva
# ---------------------------------------------------------------------------


def test_json_trafo_invalid_sn_mva_raises(tmp_path):
    """sn_mva <= 0 → ValueError."""
    net = _three_bus_net_with_trafo()
    net["trafos"][0]["sn_mva"] = 0.0
    p = tmp_path / "net.json"
    p.write_text(json.dumps(net))
    with pytest.raises(ValueError, match="sn_mva"):
        load_json(p)


# ---------------------------------------------------------------------------
# Test 5: Ungültiges vk_percent
# ---------------------------------------------------------------------------


def test_json_trafo_invalid_vk_raises(tmp_path):
    """vk_percent <= 0 → ValueError."""
    net = _three_bus_net_with_trafo()
    net["trafos"][0]["vk_percent"] = 0.0
    p = tmp_path / "net.json"
    p.write_text(json.dumps(net))
    with pytest.raises(ValueError, match="vk_percent"):
        load_json(p)


# ---------------------------------------------------------------------------
# Test 6: vkr_percent > vk_percent
# ---------------------------------------------------------------------------


def test_json_trafo_vkr_exceeds_vk_raises(tmp_path):
    """vkr_percent > vk_percent → ValueError."""
    net = _three_bus_net_with_trafo()
    net["trafos"][0]["vkr_percent"] = 10.0   # > vk_percent=4
    p = tmp_path / "net.json"
    p.write_text(json.dumps(net))
    with pytest.raises(ValueError, match="vkr_percent"):
        load_json(p)


# ---------------------------------------------------------------------------
# Test 7: Self-Loop-Trafo
# ---------------------------------------------------------------------------


def test_json_trafo_self_loop_raises(tmp_path):
    """hv_bus == lv_bus → ValueError."""
    net = _three_bus_net_with_trafo()
    net["trafos"][0]["hv_bus"] = 1
    net["trafos"][0]["lv_bus"] = 1
    p = tmp_path / "net.json"
    p.write_text(json.dumps(net))
    with pytest.raises(ValueError, match="[Ss]elbst|[Ss]elf|[Ss]elf-[Ll]oop|hv_bus|lv_bus"):
        load_json(p)


# ---------------------------------------------------------------------------
# Test 8: Unbekannte Bus-ID im Trafo
# ---------------------------------------------------------------------------


def test_json_trafo_unknown_bus_raises(tmp_path):
    """Trafo referenziert unbekannte Bus-ID → ValueError."""
    net = _three_bus_net_with_trafo()
    net["trafos"][0]["lv_bus"] = 999   # nicht vorhanden
    p = tmp_path / "net.json"
    p.write_text(json.dumps(net))
    with pytest.raises(ValueError, match="lv_bus|Bus|bus"):
        load_json(p)


# ---------------------------------------------------------------------------
# Test 9: Unbekannte Bus-ID im Shunt
# ---------------------------------------------------------------------------


def test_json_shunt_unknown_bus_raises(tmp_path):
    """Shunt referenziert unbekannte Bus-ID → ValueError."""
    net = _minimal_net(shunts=[{"id": 0, "bus": 999, "p_mw": 0.0, "q_mvar": -0.1}])
    p = tmp_path / "net.json"
    p.write_text(json.dumps(net))
    with pytest.raises(ValueError, match="bus|Bus"):
        load_json(p)


# ---------------------------------------------------------------------------
# Bonus: p.u.-Umrechnung des Trafos prüfen
# ---------------------------------------------------------------------------


def test_json_trafo_pu_values_reasonable(tmp_path):
    """TrafoSpec-Werte liegen in plausiblem Bereich."""
    net = _three_bus_net_with_trafo()
    p = tmp_path / "net.json"
    p.write_text(json.dumps(net))

    topology, params, state = load_network(p)

    # r_pu und x_pu des Trafos sollten klein aber positiv sein
    r_pu = float(params.trafo_g_series_pu[0])   # this is g = 1/r actually
    b_pu = float(params.trafo_b_series_pu[0])

    # Check that the series impedance is non-zero
    y_trafo = complex(r_pu, b_pu)
    assert abs(y_trafo) > 0, "Trafo series admittance should be non-zero"


def test_json_shunt_b_sign(tmp_path):
    """
    Kapazitiver Shunt (negatives q_mvar = erzeugt Q) → positives b_pu.
    Induktiver Shunt (positives q_mvar = absorbiert Q) → negatives b_pu.
    """
    # Kapazitiv: q_mvar = -0.1 (erzeugt Q)
    net = _minimal_net(shunts=[{"id": 0, "bus": 1, "p_mw": 0.0, "q_mvar": -0.1}])
    p = tmp_path / "cap.json"
    p.write_text(json.dumps(net))
    _, params_cap, _ = load_network(p)
    b_cap = float(params_cap.shunt_b_pu[0])
    assert b_cap > 0, f"Capacitive shunt should have b_pu > 0, got {b_cap}"

    # Induktiv: q_mvar = +0.1 (absorbiert Q)
    net2 = _minimal_net(shunts=[{"id": 0, "bus": 1, "p_mw": 0.0, "q_mvar": 0.1}])
    p2 = tmp_path / "ind.json"
    p2.write_text(json.dumps(net2))
    _, params_ind, _ = load_network(p2)
    b_ind = float(params_ind.shunt_b_pu[0])
    assert b_ind < 0, f"Inductive shunt should have b_pu < 0, got {b_ind}"
