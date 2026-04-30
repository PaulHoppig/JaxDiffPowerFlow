"""JAX-compatible PV coupling helpers for fixed P/Q bus injections."""

from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp

from diffpf.core.types import NetworkParams

PV_COUPLING_BUS_NAME = "MV Bus 2"
PV_COUPLING_SGEN_NAME = "static generator"
PV_BASE_P_MW = 2.0
PV_BASE_Q_MVAR = -0.5
PV_Q_OVER_P = -0.25


def pv_power_mw(
    irradiance_w_m2: jnp.ndarray,
    cell_temp_c: jnp.ndarray,
    p_stc_mw: float = PV_BASE_P_MW,
    g_ref_w_m2: float = 1000.0,
    t_ref_c: float = 25.0,
    gamma_p_per_c: float = -0.004,
) -> jnp.ndarray:
    """
    Compute PV active power from irradiance and cell temperature.

    Parameters
    ----------
    irradiance_w_m2
        Plane-of-array irradiance in W/m^2.
    cell_temp_c
        PV cell temperature in degree Celsius.
    p_stc_mw
        Active power at reference irradiance and temperature in MW.
    g_ref_w_m2
        Reference irradiance in W/m^2.
    t_ref_c
        Reference cell temperature in degree Celsius.
    gamma_p_per_c
        Relative power temperature coefficient per degree Celsius.

    Returns
    -------
    jnp.ndarray
        Active power injection in MW.
    """
    irradiance = jnp.asarray(irradiance_w_m2, dtype=jnp.float64)
    cell_temp = jnp.asarray(cell_temp_c, dtype=jnp.float64)
    irradiance_factor = irradiance / jnp.asarray(g_ref_w_m2, dtype=jnp.float64)
    temp_factor = 1.0 + gamma_p_per_c * (cell_temp - t_ref_c)
    return p_stc_mw * irradiance_factor * temp_factor


def pv_q_mvar_from_ratio(
    p_mw: jnp.ndarray,
    q_over_p: float = PV_Q_OVER_P,
) -> jnp.ndarray:
    """
    Compute reactive PV injection from a fixed Q/P ratio.

    Parameters
    ----------
    p_mw
        Active power injection in MW.
    q_over_p
        Reactive-over-active power ratio.

    Returns
    -------
    jnp.ndarray
        Reactive power injection in MVAr.
    """
    return jnp.asarray(p_mw, dtype=jnp.float64) * q_over_p


def inject_pq_at_bus(
    params: NetworkParams,
    bus_idx: int,
    p_mw: jnp.ndarray,
    q_mvar: jnp.ndarray,
    s_base_mva: float,
) -> NetworkParams:
    """
    Add a fixed P/Q injection at one bus on the system MVA base.

    The sign convention follows ``NetworkParams``: positive P/Q values are
    generator-sign injections into the network.
    """
    base = jnp.asarray(s_base_mva, dtype=jnp.float64)
    p_pu = jnp.asarray(p_mw, dtype=jnp.float64) / base
    q_pu = jnp.asarray(q_mvar, dtype=jnp.float64) / base
    return replace(
        params,
        p_spec_pu=params.p_spec_pu.at[bus_idx].add(p_pu),
        q_spec_pu=params.q_spec_pu.at[bus_idx].add(q_pu),
    )
