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
from diffpf.core.types import (
    BusSpec,
    CompiledTopology,
    LineSpec,
    NetworkParams,
    NetworkSpec,
    TrafoSpec,
)
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


def _two_bus_direct_trafo_params(
    *,
    g_series: float = 0.0,
    b_series: float = 0.0,
    g_mag: float = 0.0,
    b_mag: float = 0.0,
    tap: float = 1.0,
    shift: float = 0.0,
) -> tuple[CompiledTopology, NetworkParams]:
    """Build a two-bus topology with raw trafo admittance arrays.

    This helper bypasses ``compile_network`` so tests can isolate pure
    magnetizing-admittance stamps without relaxing the compiler's non-zero
    series-impedance validation for real transformer specs.
    """

    topology = CompiledTopology(
        n_bus=2,
        slack_bus=0,
        from_bus=jnp.asarray([], dtype=jnp.int32),
        to_bus=jnp.asarray([], dtype=jnp.int32),
        variable_buses=jnp.asarray([1], dtype=jnp.int32),
        is_pq_mask=jnp.asarray([True]),
        is_pv_mask=jnp.asarray([False]),
    )
    params = NetworkParams(
        p_spec_pu=jnp.asarray([0.0, 0.0], dtype=jnp.float64),
        q_spec_pu=jnp.asarray([0.0, 0.0], dtype=jnp.float64),
        v_set_pu=jnp.asarray([1.0], dtype=jnp.float64),
        g_series_pu=jnp.asarray([], dtype=jnp.float64),
        b_series_pu=jnp.asarray([], dtype=jnp.float64),
        b_shunt_pu=jnp.asarray([], dtype=jnp.float64),
        slack_vr_pu=jnp.asarray(1.0, dtype=jnp.float64),
        slack_vi_pu=jnp.asarray(0.0, dtype=jnp.float64),
        trafo_g_series_pu=jnp.asarray([g_series], dtype=jnp.float64),
        trafo_b_series_pu=jnp.asarray([b_series], dtype=jnp.float64),
        trafo_g_mag_pu=jnp.asarray([g_mag], dtype=jnp.float64),
        trafo_b_mag_pu=jnp.asarray([b_mag], dtype=jnp.float64),
        trafo_tap_ratio=jnp.asarray([tap], dtype=jnp.float64),
        trafo_shift_rad=jnp.asarray([shift], dtype=jnp.float64),
        trafo_hv_bus=(0,),
        trafo_lv_bus=(1,),
        shunt_g_pu=jnp.asarray([], dtype=jnp.float64),
        shunt_b_pu=jnp.asarray([], dtype=jnp.float64),
        shunt_bus=(),
    )
    return topology, params


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
    With total y_mag, both transformer terminals get half of the magnetizing
    admittance. b_mag is stored as a positive inductive magnitude, so the
    complex admittance is g_mag - j*b_mag.
    """
    r, x, t = 0.02, 0.04, 1.0
    g_mag, b_mag = 0.001, 0.005

    spec = _two_bus_trafo_spec(r, x, tap=t, g_mag=g_mag, b_mag=b_mag)
    topology, params = compile_network(spec)
    y_bus = build_ybus(topology, params)

    y_series = 1.0 / complex(r, x)
    y_mag_half = 0.5 * complex(g_mag, -b_mag)
    expected_hv_diag = y_series + y_mag_half
    expected_lv_diag = y_series + y_mag_half
    np.testing.assert_allclose(
        complex(y_bus[0, 0]),
        expected_hv_diag,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        complex(y_bus[1, 1]),
        expected_lv_diag,
        rtol=1e-10,
    )


def test_trafo_magnetising_admittance_not_counted_twice():
    """The sum of both pi shunts equals one total y_mag, not two."""
    topology, params = _two_bus_direct_trafo_params(g_mag=0.014, b_mag=0.0105)
    y_bus = build_ybus(topology, params)

    expected_half = 0.5 * complex(0.014, -0.0105)
    np.testing.assert_allclose(complex(y_bus[0, 0]), expected_half, rtol=1e-12)
    np.testing.assert_allclose(complex(y_bus[1, 1]), expected_half, rtol=1e-12)
    np.testing.assert_allclose(
        complex(y_bus[0, 0] + y_bus[1, 1]),
        complex(0.014, -0.0105),
        rtol=1e-12,
    )


def test_pure_magnetising_admittance_consumes_real_power_once():
    """At V=1 on both terminals, P consumption is g_mag once, not twice."""
    topology, params = _two_bus_direct_trafo_params(g_mag=0.014, b_mag=0.0105)
    y_bus = build_ybus(topology, params)
    voltage = jnp.asarray([1.0 + 0.0j, 1.0 + 0.0j], dtype=jnp.complex128)
    current = y_bus @ voltage
    s_bus = voltage * jnp.conj(current)

    np.testing.assert_allclose(float(jnp.real(jnp.sum(s_bus))), 0.014, rtol=1e-12)
    np.testing.assert_allclose(float(jnp.imag(jnp.sum(s_bus))), 0.0105, rtol=1e-12)


def test_trafo_zero_magnetising_admittance_keeps_series_stamp():
    """With y_m=0, the corrected stamp is identical to the old series branch."""
    r, x, tap, shift = 0.02, 0.04, 0.93, 0.4
    spec = _two_bus_trafo_spec(r, x, tap=tap, shift=shift, g_mag=0.0, b_mag=0.0)
    topology, params = compile_network(spec)
    y_bus = build_ybus(topology, params)

    y_series = 1.0 / complex(r, x)
    t = tap * np.exp(1j * shift)
    np.testing.assert_allclose(complex(y_bus[0, 0]), y_series / (t * np.conj(t)))
    np.testing.assert_allclose(complex(y_bus[1, 1]), y_series)
    np.testing.assert_allclose(complex(y_bus[0, 1]), -y_series / np.conj(t))
    np.testing.assert_allclose(complex(y_bus[1, 0]), -y_series / t)


def test_complex_tap_uses_abs_square_not_complex_square_for_hv_diagonal():
    """For phase shift, HV self-admittance must use t*conj(t), not t**2."""
    r, x, tap, shift = 0.02, 0.04, 0.95, 0.7
    g_mag, b_mag = 0.003, 0.004
    spec = _two_bus_trafo_spec(
        r, x, tap=tap, shift=shift, g_mag=g_mag, b_mag=b_mag
    )
    topology, params = compile_network(spec)
    y_bus = build_ybus(topology, params)

    y_series = 1.0 / complex(r, x)
    y_mag_half = 0.5 * complex(g_mag, -b_mag)
    t = tap * np.exp(1j * shift)
    expected = (y_series + y_mag_half) / (t * np.conj(t))
    wrong_complex_square = (y_series + y_mag_half) / (t ** 2)

    np.testing.assert_allclose(complex(y_bus[0, 0]), expected, rtol=1e-10)
    assert abs(complex(y_bus[0, 0]) - wrong_complex_square) > 1e-3


def test_complex_tap_off_diagonals_use_conjugated_factors():
    """Off-diagonal stamps follow pandapower pi convention for complex taps."""
    r, x, tap, shift = 0.02, 0.04, 0.95, 0.7
    spec = _two_bus_trafo_spec(r, x, tap=tap, shift=shift)
    topology, params = compile_network(spec)
    y_bus = build_ybus(topology, params)

    y_series = 1.0 / complex(r, x)
    t = tap * np.exp(1j * shift)
    np.testing.assert_allclose(complex(y_bus[0, 1]), -y_series / np.conj(t))
    np.testing.assert_allclose(complex(y_bus[1, 0]), -y_series / t)
    assert abs(complex(y_bus[0, 1]) - complex(y_bus[1, 0])) > 1e-3


def test_positive_b_mag_is_inductive_pandapower_sign():
    """Positive stored b_mag is stamped as negative imaginary admittance."""
    topology, params = _two_bus_direct_trafo_params(g_mag=0.0, b_mag=0.0105)
    y_bus = build_ybus(topology, params)

    np.testing.assert_allclose(float(jnp.imag(y_bus[0, 0])), -0.00525, rtol=1e-12)
    np.testing.assert_allclose(float(jnp.imag(y_bus[1, 1])), -0.00525, rtol=1e-12)


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
