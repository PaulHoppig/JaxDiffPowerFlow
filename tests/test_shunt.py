"""
Tests for shunt Y-bus stamping.

Covered invariants
------------------
- Shunt g+jb adds to Y-bus diagonal of the correct bus
- Network without shunts is unchanged
- Multiple shunts on the same bus accumulate correctly
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from diffpf.compile import compile_network
from diffpf.core.types import BusSpec, LineSpec, NetworkSpec, ShuntSpec
from diffpf.core.ybus import build_ybus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _two_bus_no_shunt(r: float = 0.02, x: float = 0.04) -> tuple:
    spec = NetworkSpec(
        buses=(BusSpec(name="slack", is_slack=True), BusSpec(name="load")),
        lines=(LineSpec(from_bus=0, to_bus=1, r_pu=r, x_pu=x),),
        p_spec_pu=(0.0, -0.5),
        q_spec_pu=(0.0, -0.1),
    )
    return compile_network(spec)


def _two_bus_with_shunt(
    bus: int,
    g_pu: float,
    b_pu: float,
    r: float = 0.02,
    x: float = 0.04,
) -> tuple:
    spec = NetworkSpec(
        buses=(BusSpec(name="slack", is_slack=True), BusSpec(name="load")),
        lines=(LineSpec(from_bus=0, to_bus=1, r_pu=r, x_pu=x),),
        shunts=(ShuntSpec(bus=bus, g_pu=g_pu, b_pu=b_pu),),
        p_spec_pu=(0.0, -0.5),
        q_spec_pu=(0.0, -0.1),
    )
    return compile_network(spec)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_shunt_adds_to_diagonal_bus0():
    """Shunt at bus 0 adds g+jb to Y[0,0]."""
    g, b = 0.01, -0.05
    topo_base, params_base = _two_bus_no_shunt()
    topo_sh, params_sh = _two_bus_with_shunt(bus=0, g_pu=g, b_pu=b)

    y_base = build_ybus(topo_base, params_base)
    y_sh = build_ybus(topo_sh, params_sh)

    diff = np.asarray(y_sh - y_base)
    np.testing.assert_allclose(complex(diff[0, 0]), complex(g, b), rtol=1e-10)
    # All other entries must be unchanged
    diff_other = diff.copy()
    diff_other[0, 0] = 0.0
    np.testing.assert_allclose(np.abs(diff_other), 0.0, atol=1e-14)


def test_shunt_adds_to_diagonal_bus1():
    """Shunt at bus 1 adds g+jb to Y[1,1]."""
    g, b = 0.02, 0.03
    topo_base, params_base = _two_bus_no_shunt()
    topo_sh, params_sh = _two_bus_with_shunt(bus=1, g_pu=g, b_pu=b)

    y_base = build_ybus(topo_base, params_base)
    y_sh = build_ybus(topo_sh, params_sh)

    diff = np.asarray(y_sh - y_base)
    np.testing.assert_allclose(complex(diff[1, 1]), complex(g, b), rtol=1e-10)
    diff_other = diff.copy()
    diff_other[1, 1] = 0.0
    np.testing.assert_allclose(np.abs(diff_other), 0.0, atol=1e-14)


def test_network_without_shunts_unchanged():
    """Empty shunts tuple produces the same Y-bus as the default."""
    spec_default = NetworkSpec(
        buses=(BusSpec(name="slack", is_slack=True), BusSpec(name="load")),
        lines=(LineSpec(from_bus=0, to_bus=1, r_pu=0.02, x_pu=0.04),),
        p_spec_pu=(0.0, -0.5),
        q_spec_pu=(0.0, -0.1),
    )
    spec_explicit = NetworkSpec(
        buses=(BusSpec(name="slack", is_slack=True), BusSpec(name="load")),
        lines=(LineSpec(from_bus=0, to_bus=1, r_pu=0.02, x_pu=0.04),),
        shunts=(),
        p_spec_pu=(0.0, -0.5),
        q_spec_pu=(0.0, -0.1),
    )

    topo_d, params_d = compile_network(spec_default)
    topo_e, params_e = compile_network(spec_explicit)

    np.testing.assert_allclose(
        np.asarray(build_ybus(topo_d, params_d)),
        np.asarray(build_ybus(topo_e, params_e)),
        atol=1e-14,
    )


def test_multiple_shunts_same_bus_accumulate():
    """Two shunts on the same bus accumulate their admittances."""
    r, x = 0.02, 0.04
    g1, b1 = 0.01, -0.02
    g2, b2 = 0.005, 0.01
    spec = NetworkSpec(
        buses=(BusSpec(name="slack", is_slack=True), BusSpec(name="load")),
        lines=(LineSpec(from_bus=0, to_bus=1, r_pu=r, x_pu=x),),
        shunts=(
            ShuntSpec(bus=1, g_pu=g1, b_pu=b1),
            ShuntSpec(bus=1, g_pu=g2, b_pu=b2),
        ),
        p_spec_pu=(0.0, -0.5),
        q_spec_pu=(0.0, -0.1),
    )
    topo, params = compile_network(spec)
    y_bus = build_ybus(topo, params)

    topo_base, params_base = _two_bus_no_shunt(r, x)
    y_base = build_ybus(topo_base, params_base)

    diff = np.asarray(y_bus - y_base)
    expected = complex(g1 + g2, b1 + b2)
    np.testing.assert_allclose(complex(diff[1, 1]), expected, rtol=1e-10)
