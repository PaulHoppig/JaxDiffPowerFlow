"""JAX-compatible PV coupling helpers for fixed P/Q bus injections.

The model in this module represents the PV plant in pandapower
``example_simple()`` as a weather-dependent PQ injection. It is deliberately
not a voltage-regulating PV bus model: voltage control, Q limits, controller
logic, and PV-to-PQ switching are outside the current project scope.
"""

from __future__ import annotations

from dataclasses import replace
from typing import NamedTuple

import jax.numpy as jnp

from diffpf.core.types import NetworkParams

PV_COUPLING_BUS_NAME = "MV Bus 2"
PV_COUPLING_SGEN_NAME = "static generator"
PV_BASE_P_MW = 2.0
PV_BASE_Q_MVAR = -0.5
PV_Q_OVER_P = -0.25
PV_G_REF_W_M2 = 1000.0
PV_T_REF_C = 25.0
PV_GAMMA_P_PER_C = -0.004
PV_ALPHA = 1.0
PV_NOCT_ADJ_C = 45.0
PV_ETA_REF = 0.18
PV_TAU_ALPHA = 0.90


class PVInjection(NamedTuple):
    """PV active and reactive power injection in physical units.

    Attributes
    ----------
    p_pv_mw
        Active power injected into the network in MW.
    q_pv_mvar
        Reactive power injected into the network in MVAr.

    Notes
    -----
    Positive values follow generator sign convention. The baseline model uses
    the explicit ratio ``Q_pv = kappa * P_pv`` rather than a cos(phi)
    parameterization.
    """

    p_pv_mw: jnp.ndarray
    q_pv_mvar: jnp.ndarray


def pv_power_mw(
    irradiance_w_m2: jnp.ndarray,
    cell_temp_c: jnp.ndarray,
    alpha: jnp.ndarray = PV_ALPHA,
    p_ref_mw: float = PV_BASE_P_MW,
    g_ref_w_m2: float = PV_G_REF_W_M2,
    t_ref_c: float = PV_T_REF_C,
    gamma_p_per_c: float = PV_GAMMA_P_PER_C,
    p_stc_mw: float | None = None,
) -> jnp.ndarray:
    """Compute analytical PV active power from irradiance and cell temperature.

    Parameters
    ----------
    irradiance_w_m2
        Plane-of-array irradiance in W/m^2.
    cell_temp_c
        PV cell temperature in degree Celsius.
    alpha
        Dimensionless scaling or curtailment factor. The reference value is 1.
    p_ref_mw
        Active power at reference irradiance and temperature in MW.
    g_ref_w_m2
        Reference irradiance in W/m^2.
    t_ref_c
        Reference cell temperature in degree Celsius.
    gamma_p_per_c
        Relative power temperature coefficient per degree Celsius.
    p_stc_mw
        Deprecated alias for ``p_ref_mw`` kept for backwards compatibility.

    Returns
    -------
    jnp.ndarray
        Active power injection in MW.

    Notes
    -----
    The implemented baseline relation is

    ``P = alpha * P_ref * (G / G_ref) * (1 + gamma * (T_cell - T_ref))``.

    No clipping or saturation is applied. This keeps the reference point exact
    and leaves any operational limits to explicit outer-model logic.
    """
    irradiance = jnp.asarray(irradiance_w_m2, dtype=jnp.float64)
    cell_temp = jnp.asarray(cell_temp_c, dtype=jnp.float64)
    alpha_arr = jnp.asarray(alpha, dtype=jnp.float64)
    p_ref = jnp.asarray(p_ref_mw if p_stc_mw is None else p_stc_mw, dtype=jnp.float64)
    irradiance_factor = irradiance / jnp.asarray(g_ref_w_m2, dtype=jnp.float64)
    temp_factor = 1.0 + jnp.asarray(gamma_p_per_c, dtype=jnp.float64) * (
        cell_temp - jnp.asarray(t_ref_c, dtype=jnp.float64)
    )
    return alpha_arr * p_ref * irradiance_factor * temp_factor


def cell_temperature_noct_sam(
    g_poa_wm2: jnp.ndarray,
    t_amb_c: jnp.ndarray,
    wind_ms: jnp.ndarray,
    t_noct_adj_c: float = PV_NOCT_ADJ_C,
    eta_ref: float = PV_ETA_REF,
    tau_alpha: float = PV_TAU_ALPHA,
) -> jnp.ndarray:
    """Estimate PV cell temperature with a reduced NOCT-SAM relation.

    Parameters
    ----------
    g_poa_wm2
        Plane-of-array irradiance in W/m^2.
    t_amb_c
        Ambient air temperature in degree Celsius.
    wind_ms
        Wind speed in m/s. In this first model version, ``wind_adj = wind_ms``;
        no height or mounting correction is applied.
    t_noct_adj_c
        Adjusted nominal operating cell temperature in degree Celsius.
    eta_ref
        Reference module efficiency as a fraction.
    tau_alpha
        Effective transmittance-absorptance product as a fraction.

    Returns
    -------
    jnp.ndarray
        Estimated cell temperature in degree Celsius.

    Notes
    -----
    The implemented reduced NOCT-SAM form is

    ``T_cell = T_amb + (G_poa / 800) * (T_noct_adj - 20)
    * (1 - eta_ref / tau_alpha) * 9.5 / (5.7 + 3.8 * wind_adj)``.

    Inputs remain in meteorological units and are not converted to the
    electrical per-unit system. No clipping is applied.
    """

    irradiance = jnp.asarray(g_poa_wm2, dtype=jnp.float64)
    ambient = jnp.asarray(t_amb_c, dtype=jnp.float64)
    wind_adj = jnp.asarray(wind_ms, dtype=jnp.float64)
    noct_delta = jnp.asarray(t_noct_adj_c, dtype=jnp.float64) - 20.0
    heat_fraction = 1.0 - (
        jnp.asarray(eta_ref, dtype=jnp.float64)
        / jnp.asarray(tau_alpha, dtype=jnp.float64)
    )
    wind_factor = 9.5 / (5.7 + 3.8 * wind_adj)
    return ambient + (irradiance / 800.0) * noct_delta * heat_fraction * wind_factor


def pv_q_mvar_from_ratio(
    p_mw: jnp.ndarray,
    kappa: jnp.ndarray = PV_Q_OVER_P,
    q_over_p: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Compute reactive PV injection from a fixed Q/P ratio.

    Parameters
    ----------
    p_mw
        Active power injection in MW.
    kappa
        Reactive-over-active power ratio.
    q_over_p
        Deprecated alias for ``kappa`` kept for backwards compatibility.

    Returns
    -------
    jnp.ndarray
        Reactive power injection in MVAr.
    """
    ratio = kappa if q_over_p is None else q_over_p
    return jnp.asarray(p_mw, dtype=jnp.float64) * jnp.asarray(ratio, dtype=jnp.float64)


def pv_pq_injection(
    irradiance_w_m2: jnp.ndarray,
    cell_temp_c: jnp.ndarray,
    alpha: jnp.ndarray = PV_ALPHA,
    kappa: jnp.ndarray = PV_Q_OVER_P,
    p_ref_mw: float = PV_BASE_P_MW,
    g_ref_w_m2: float = PV_G_REF_W_M2,
    t_ref_c: float = PV_T_REF_C,
    gamma_p_per_c: float = PV_GAMMA_P_PER_C,
) -> PVInjection:
    """Return the weather-dependent PV PQ injection in MW/MVAr.

    This is the main upstream-model API for the current scope. It is
    differentiable with respect to irradiance, cell temperature, ``alpha``, and
    ``kappa``. The coupling bus remains a PQ bus; the returned values are meant
    to be written into ``NetworkParams.p_spec_pu`` and ``q_spec_pu`` through an
    adapter function.

    At ``G = 1000 W/m^2``, ``T_cell = 25 degC``, ``alpha = 1``, and
    ``kappa = -0.25`` the function reproduces the ``example_simple()`` static
    generator baseline: ``P = 2.0 MW`` and ``Q = -0.5 MVAr``.
    """

    p_pv_mw = pv_power_mw(
        irradiance_w_m2=irradiance_w_m2,
        cell_temp_c=cell_temp_c,
        alpha=alpha,
        p_ref_mw=p_ref_mw,
        g_ref_w_m2=g_ref_w_m2,
        t_ref_c=t_ref_c,
        gamma_p_per_c=gamma_p_per_c,
    )
    q_pv_mvar = pv_q_mvar_from_ratio(p_pv_mw, kappa=kappa)
    return PVInjection(p_pv_mw=p_pv_mw, q_pv_mvar=q_pv_mvar)


def pv_pq_injection_from_weather(
    g_poa_wm2: jnp.ndarray,
    t_amb_c: jnp.ndarray,
    wind_ms: jnp.ndarray,
    alpha: jnp.ndarray = PV_ALPHA,
    kappa: jnp.ndarray = PV_Q_OVER_P,
    p_ref_mw: float = PV_BASE_P_MW,
    g_ref_w_m2: float = PV_G_REF_W_M2,
    t_ref_c: float = PV_T_REF_C,
    gamma_p_per_c: float = PV_GAMMA_P_PER_C,
    t_noct_adj_c: float = PV_NOCT_ADJ_C,
    eta_ref: float = PV_ETA_REF,
    tau_alpha: float = PV_TAU_ALPHA,
) -> PVInjection:
    """Return PV PQ injection from irradiance, ambient temperature, and wind.

    This V2 helper keeps the V1 electrical coupling unchanged. It first
    computes cell temperature with :func:`cell_temperature_noct_sam`, then
    evaluates the same analytical active-power model used by
    :func:`pv_pq_injection` and finally applies ``Q_pv = kappa * P_pv``.
    """

    cell_temp_c = cell_temperature_noct_sam(
        g_poa_wm2=g_poa_wm2,
        t_amb_c=t_amb_c,
        wind_ms=wind_ms,
        t_noct_adj_c=t_noct_adj_c,
        eta_ref=eta_ref,
        tau_alpha=tau_alpha,
    )
    return pv_pq_injection(
        irradiance_w_m2=g_poa_wm2,
        cell_temp_c=cell_temp_c,
        alpha=alpha,
        kappa=kappa,
        p_ref_mw=p_ref_mw,
        g_ref_w_m2=g_ref_w_m2,
        t_ref_c=t_ref_c,
        gamma_p_per_c=gamma_p_per_c,
    )


def inject_pq_at_bus(
    params: NetworkParams,
    bus_idx: int,
    p_mw: jnp.ndarray,
    q_mvar: jnp.ndarray,
    s_base_mva: float,
) -> NetworkParams:
    """Add a fixed P/Q injection at one bus on the system MVA base.

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


def replace_pq_contribution_at_bus(
    params: NetworkParams,
    bus_idx: int,
    old_p_mw: jnp.ndarray,
    old_q_mvar: jnp.ndarray,
    new_p_mw: jnp.ndarray,
    new_q_mvar: jnp.ndarray,
    s_base_mva: float,
) -> NetworkParams:
    """Replace one known P/Q contribution at a bus by adding its delta.

    This helper is useful when a compiled ``NetworkParams`` still contains the
    static ``sgen`` baseline and a model output should replace only that
    element, while preserving other contributions at the same bus such as load.
    """

    delta_p_mw = jnp.asarray(new_p_mw, dtype=jnp.float64) - jnp.asarray(
        old_p_mw, dtype=jnp.float64
    )
    delta_q_mvar = jnp.asarray(new_q_mvar, dtype=jnp.float64) - jnp.asarray(
        old_q_mvar, dtype=jnp.float64
    )
    return inject_pq_at_bus(params, bus_idx, delta_p_mw, delta_q_mvar, s_base_mva)


def inject_pv_at_bus(
    params: NetworkParams,
    bus_idx: int,
    injection: PVInjection,
    s_base_mva: float,
) -> NetworkParams:
    """Add a PV model output to one bus in ``NetworkParams``."""

    return inject_pq_at_bus(
        params=params,
        bus_idx=bus_idx,
        p_mw=injection.p_pv_mw,
        q_mvar=injection.q_pv_mvar,
        s_base_mva=s_base_mva,
    )
