from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from diffpf.core.types import NetworkParams
from diffpf.models.pv import inject_pq_at_bus, pv_power_mw, pv_q_mvar_from_ratio


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
    p_mw = pv_power_mw(1000.0, 25.0)
    np.testing.assert_allclose(float(p_mw), 2.0, rtol=0.0, atol=1e-12)


def test_pv_q_from_default_ratio():
    q_mvar = pv_q_mvar_from_ratio(2.0)
    np.testing.assert_allclose(float(q_mvar), -0.5, rtol=0.0, atol=1e-12)


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


def test_pv_power_grad_with_respect_to_irradiance():
    grad = jax.grad(lambda g: pv_power_mw(g, 25.0))(jnp.asarray(1000.0))
    np.testing.assert_allclose(float(grad), 0.002, rtol=1e-12, atol=1e-12)
