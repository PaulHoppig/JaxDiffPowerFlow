from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pandapower.networks as pn

from diffpf.compile.network import compile_network
from diffpf.core.types import NetworkParams
from diffpf.io.pandapower_adapter import from_pandapower
from diffpf.io.topology_utils import merge_buses
from diffpf.models.pv import (
    PV_BASE_P_MW,
    PV_BASE_Q_MVAR,
    PV_COUPLING_BUS_NAME,
    PV_COUPLING_SGEN_NAME,
    PV_Q_OVER_P,
    inject_pq_at_bus,
    inject_pv_at_bus,
    pv_power_mw,
    pv_pq_injection,
    pv_q_mvar_from_ratio,
)


def _params() -> NetworkParams:
    return NetworkParams(
        p_spec_pu=jnp.asarray([0.0, -0.1, 0.2], dtype=jnp.float64),
        q_spec_pu=jnp.asarray([0.0, -0.05, 0.01], dtype=jnp.float64),
        v_set_pu=jnp.asarray([1.0, 1.0], dtype=jnp.float64),
        g_series_pu=jnp.asarray([], dtype=jnp.float64),
        b_series_pu=jnp.asarray([], dtype=jnp.float64),
        b_shunt_pu=jnp.asarray([], dtype=jnp.float64),
        slack_vr_pu=jnp.asarray(1.0, dtype=jnp.float64),
        slack_vi_pu=jnp.asarray(0.0, dtype=jnp.float64),
        trafo_g_series_pu=jnp.asarray([], dtype=jnp.float64),
        trafo_b_series_pu=jnp.asarray([], dtype=jnp.float64),
        trafo_g_mag_pu=jnp.asarray([], dtype=jnp.float64),
        trafo_b_mag_pu=jnp.asarray([], dtype=jnp.float64),
        trafo_tap_ratio=jnp.asarray([], dtype=jnp.float64),
        trafo_shift_rad=jnp.asarray([], dtype=jnp.float64),
        shunt_g_pu=jnp.asarray([], dtype=jnp.float64),
        shunt_b_pu=jnp.asarray([], dtype=jnp.float64),
    )


def test_pv_power_reference_conditions_returns_base_power():
    injection = pv_pq_injection(1000.0, 25.0, alpha=1.0, kappa=PV_Q_OVER_P)

    np.testing.assert_allclose(
        float(injection.p_pv_mw),
        PV_BASE_P_MW,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        float(injection.q_pv_mvar),
        PV_BASE_Q_MVAR,
        rtol=0.0,
        atol=1e-12,
    )


def test_pv_q_from_default_ratio():
    q_mvar = pv_q_mvar_from_ratio(2.0)
    np.testing.assert_allclose(float(q_mvar), -0.5, rtol=0.0, atol=1e-12)


def test_pv_power_increases_with_irradiance():
    low = pv_power_mw(600.0, 25.0)
    high = pv_power_mw(900.0, 25.0)

    assert float(high) > float(low)


def test_pv_power_decreases_with_temperature_for_negative_gamma():
    cool = pv_power_mw(1000.0, 20.0)
    hot = pv_power_mw(1000.0, 45.0)

    assert float(hot) < float(cool)


def test_pv_q_over_p_ratio_matches_kappa():
    injection = pv_pq_injection(750.0, 30.0, alpha=0.8, kappa=-0.2)

    np.testing.assert_allclose(
        float(injection.q_pv_mvar / injection.p_pv_mw),
        -0.2,
        rtol=1e-12,
        atol=1e-12,
    )


def test_pv_power_gradients_have_plausible_signs():
    grad_g = jax.grad(lambda g: pv_power_mw(g, 25.0))(jnp.asarray(1000.0))
    grad_t = jax.grad(lambda t: pv_power_mw(1000.0, t))(jnp.asarray(25.0))

    assert jnp.isfinite(grad_g)
    assert jnp.isfinite(grad_t)
    assert float(grad_g) > 0.0
    assert float(grad_t) < 0.0


def test_pv_pq_injection_is_differentiable_in_alpha_and_kappa():
    grad_alpha = jax.grad(
        lambda alpha: pv_pq_injection(1000.0, 25.0, alpha=alpha).p_pv_mw
    )(jnp.asarray(1.0))
    grad_kappa = jax.grad(
        lambda kappa: pv_pq_injection(1000.0, 25.0, kappa=kappa).q_pv_mvar
    )(jnp.asarray(PV_Q_OVER_P))

    np.testing.assert_allclose(
        float(grad_alpha),
        PV_BASE_P_MW,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        float(grad_kappa),
        PV_BASE_P_MW,
        rtol=1e-12,
        atol=1e-12,
    )


def test_inject_pq_at_bus_changes_only_target_bus():
    params = _params()
    updated = inject_pq_at_bus(
        params,
        bus_idx=1,
        p_mw=2.0,
        q_mvar=-0.5,
        s_base_mva=10.0,
    )

    np.testing.assert_allclose(updated.p_spec_pu[0], params.p_spec_pu[0])
    np.testing.assert_allclose(updated.p_spec_pu[2], params.p_spec_pu[2])
    np.testing.assert_allclose(updated.q_spec_pu[0], params.q_spec_pu[0])
    np.testing.assert_allclose(updated.q_spec_pu[2], params.q_spec_pu[2])
    np.testing.assert_allclose(float(updated.p_spec_pu[1]), 0.1)
    np.testing.assert_allclose(float(updated.q_spec_pu[1]), -0.1)


def test_example_simple_target_sgen_and_bus_are_coupled_by_adapter():
    net_original = pn.example_simple()
    target_sgen = net_original.sgen[
        (net_original.sgen["name"] == PV_COUPLING_SGEN_NAME)
        & (net_original.sgen["in_service"] == True)  # noqa: E712
    ]
    assert len(target_sgen) == 1

    sgen_idx = int(target_sgen.index[0])
    assert float(net_original.sgen.at[sgen_idx, "p_mw"]) == PV_BASE_P_MW
    assert float(net_original.sgen.at[sgen_idx, "q_mvar"]) == PV_BASE_Q_MVAR

    bus_matches = net_original.bus[net_original.bus["name"] == PV_COUPLING_BUS_NAME]
    assert len(bus_matches) == 1
    original_bus = int(bus_matches.index[0])
    assert int(net_original.sgen.at[sgen_idx, "bus"]) == original_bus

    bb_pairs = [
        (int(row["bus"]), int(row["element"]))
        for _, row in net_original.switch.iterrows()
        if row["et"] == "b" and bool(row["closed"])
    ]
    bus_to_repr = merge_buses(list(net_original.bus.index), bb_pairs)

    spec_original = from_pandapower(net_original)
    _, params_original = compile_network(spec_original)
    target_bus_idx = [bus.name for bus in spec_original.buses].index(
        str(bus_to_repr[original_bus])
    )

    net_without_sgen = pn.example_simple()
    net_without_sgen.sgen.at[sgen_idx, "in_service"] = False
    spec_without_sgen = from_pandapower(net_without_sgen)
    _, params_without_sgen = compile_network(spec_without_sgen)

    injection = pv_pq_injection(1000.0, 25.0)
    coupled = inject_pv_at_bus(
        params_without_sgen,
        bus_idx=target_bus_idx,
        injection=injection,
        s_base_mva=float(net_original.sn_mva),
    )

    np.testing.assert_allclose(
        np.asarray(coupled.p_spec_pu),
        np.asarray(params_original.p_spec_pu),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(coupled.q_spec_pu),
        np.asarray(params_original.q_spec_pu),
        rtol=0.0,
        atol=1e-12,
    )
