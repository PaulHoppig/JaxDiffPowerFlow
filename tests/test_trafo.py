"""
Tests for transformer Y-bus stamping.

Covered invariants
------------------
- Trafo with tap=1, shift=0, g_mag=0, b_mag=0 equals a series line stamp
- Trafo with empty list leaves line-only network unchanged
- Tap ratio correctly scales HV diagonal entry
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from diffpf.compile import compile_network
from diffpf.core.types import BusSpec, LineSpec, NetworkSpec, TrafoSpec
from diffpf.core.ybus import build_ybus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _two_bus_line_spec(r: float, x: float) -> NetworkSpec:
    """2-bus network with one line."""
    return NetworkSpec(
        buses=(BusSpec(name="slack", is_slack=True), BusSpec(name="load")),
        lines=(LineSpec(from_bus=0, to_bus=1, r_pu=r, x_pu=x),),
        p_spec_pu=(0.0, -0.5),
        q_spec_pu=(0.0, -0.1),
    )


def _two_bus_trafo_spec(r: float, x: float, tap: float = 1.0, shift: float = 0.0,
                         g_mag: float = 0.0, b_mag: float = 0.0) -> NetworkSpec:
    """2-bus network with one transformer (no lines)."""
    return NetworkSpec(
        buses=(BusSpec(name="slack", is_slack=True), BusSpec(name="load")),
        lines=(),
        trafos=(TrafoSpec(
            hv_bus=0, lv_bus=1,
            r_pu=r, x_pu=x,
            g_mag_pu=g_mag, b_mag_pu=b_mag,
            tap_ratio=tap, shift_rad=shift,
        ),),
        p_spec_pu=(0.0, -0.5),
        q_spec_pu=(0.0, -0.1),
    )


# ---------------------------------------------------------------------------
# Tests: trivial trafo = line
# ---------------------------------------------------------------------------


def test_trafo_tap1_shift0_equals_line():
    """
    Trafo with tap=1, shift=0, g_mag=0, b_mag=0 must produce the same
    Y-bus as a line with identical r_pu, x_pu (no shunt).
    """
    r, x = 0.02, 0.04

    spec_line = _two_bus_line_spec(r, x)
    topology_line, params_line = compile_network(spec_line)
    y_bus_line = build_ybus(topology_line, params_line)

    spec_trafo = _two_bus_trafo_spec(r, x, tap=1.0, shift=0.0)
    topology_trafo, params_trafo = compile_network(spec_trafo)
    y_bus_trafo = build_ybus(topology_trafo, params_trafo)

    np.testing.assert_allclose(
        np.asarray(y_bus_trafo),
        np.asarray(y_bus_line),
        rtol=1e-10,
        atol=1e-14,
    )


def test_trafo_tap_ratio_scales_hv_diagonal():
    """
    With tap_ratio=t, shift=0, g_mag=0, b_mag=0:
      Y[hv,hv] = y_series / t^2
    """
    r, x, t = 0.02, 0.04, 0.95
    spec = _two_bus_trafo_spec(r, x, tap=t)
    topology, params = compile_network(spec)
    y_bus = build_ybus(topology, params)

    y_series = 1.0 / complex(r, x)
    expected_hv_diag = y_series / (t ** 2)
    np.testing.assert_allclose(
        complex(y_bus[0, 0]),
        expected_hv_diag,
        rtol=1e-10,
    )


def test_trafo_tap_ratio_lv_diagonal_unchanged():
    """
    Y[lv,lv] = y_series  (independent of tap ratio when g_mag=0)
    """
    r, x, t = 0.02, 0.04, 0.95
    spec = _two_bus_trafo_spec(r, x, tap=t)
    topology, params = compile_network(spec)
    y_bus = build_ybus(topology, params)

    y_series = 1.0 / complex(r, x)
    np.testing.assert_allclose(
        complex(y_bus[1, 1]),
        y_series,
        rtol=1e-10,
    )


def test_empty_trafo_list_leaves_line_network_unchanged():
    """
    A network with no transformers must produce the same Y-bus
    whether trafos=() is explicit or left as the default.
    """
    r, x = 0.02, 0.04

    spec_default = NetworkSpec(
        buses=(BusSpec(name="slack", is_slack=True), BusSpec(name="load")),
        lines=(LineSpec(from_bus=0, to_bus=1, r_pu=r, x_pu=x),),
        p_spec_pu=(0.0, -0.5),
        q_spec_pu=(0.0, -0.1),
    )
    spec_explicit = NetworkSpec(
        buses=(BusSpec(name="slack", is_slack=True), BusSpec(name="load")),
        lines=(LineSpec(from_bus=0, to_bus=1, r_pu=r, x_pu=x),),
        trafos=(),
        p_spec_pu=(0.0, -0.5),
        q_spec_pu=(0.0, -0.1),
    )

    topo_d, params_d = compile_network(spec_default)
    topo_e, params_e = compile_network(spec_explicit)

    y_default = build_ybus(topo_d, params_d)
    y_explicit = build_ybus(topo_e, params_e)

    np.testing.assert_allclose(
        np.asarray(y_default), np.asarray(y_explicit), atol=1e-14
    )


def test_trafo_with_magnetising_admittance():
    """
    With g_mag and b_mag, the HV diagonal gets an extra y_mag / |a|^2 contribution.
    """
    r, x, t = 0.02, 0.04, 1.0
    g_mag, b_mag = 0.001, -0.005

    spec = _two_bus_trafo_spec(r, x, tap=t, g_mag=g_mag, b_mag=b_mag)
    topology, params = compile_network(spec)
    y_bus = build_ybus(topology, params)

    y_series = 1.0 / complex(r, x)
    y_mag_val = complex(g_mag, b_mag)
    expected_hv_diag = (y_series + y_mag_val) / (t ** 2)
    np.testing.assert_allclose(
        complex(y_bus[0, 0]),
        expected_hv_diag,
        rtol=1e-10,
    )


def test_trafo_off_diagonal_symmetry_no_shift():
    """
    With shift=0, the off-diagonal elements must be symmetric:
    Y[hv,lv] == Y[lv,hv] == -y_series / t.
    """
    r, x, t = 0.02, 0.04, 0.98
    spec = _two_bus_trafo_spec(r, x, tap=t, shift=0.0)
    topology, params = compile_network(spec)
    y_bus = build_ybus(topology, params)

    y_series = 1.0 / complex(r, x)
    expected_off = -y_series / t
    np.testing.assert_allclose(complex(y_bus[0, 1]), expected_off, rtol=1e-10)
    np.testing.assert_allclose(complex(y_bus[1, 0]), expected_off, rtol=1e-10)
