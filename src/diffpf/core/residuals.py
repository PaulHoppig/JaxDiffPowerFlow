"""
AC power-flow residual functions in rectangular voltage coordinates.

Mathematical core
-----------------
Full voltage vector (all buses, rectangular):
  V = Vr + j*Vi

Current injection:
  I = Y_bus @ V

Complex power injection:
  S_calc = V * conj(I)

Residual for non-slack bus i (index in variable_buses):
  PQ-bus:
    r[2i]   = p_spec[i] - P_calc[i]
    r[2i+1] = q_spec[i] - Q_calc[i]
  PV-bus:
    r[2i]   = p_spec[i] - P_calc[i]
    r[2i+1] = v_set_pu[i]**2 - (Vr[i]**2 + Vi[i]**2)

Scalar loss:
  L = 0.5 * ||r||²

All functions are pure and JIT/grad-able. ``CompiledTopology`` carries
the static structure; ``NetworkParams`` and ``PFState`` are the
differentiable leaves.
"""

from __future__ import annotations

import jax.numpy as jnp

from diffpf.core.types import CompiledTopology, NetworkParams, PFState
from diffpf.core.ybus import build_ybus


def state_to_voltage(
    topology: CompiledTopology,
    params: NetworkParams,
    state: PFState,
) -> jnp.ndarray:
    """
    Reconstruct the full complex voltage vector from the non-slack state.

    Returns
    -------
    voltage : jnp.ndarray
        Complex128 array of shape (n_bus,).
    """
    vr = jnp.zeros(topology.n_bus, dtype=jnp.float64)
    vi = jnp.zeros(topology.n_bus, dtype=jnp.float64)
    vr = vr.at[topology.slack_bus].set(params.slack_vr_pu)
    vi = vi.at[topology.slack_bus].set(params.slack_vi_pu)
    vr = vr.at[topology.variable_buses].set(state.vr_pu)
    vi = vi.at[topology.variable_buses].set(state.vi_pu)
    return vr + 1j * vi


def calc_power_injection(
    y_bus: jnp.ndarray,
    voltage: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute complex nodal power injection S = V * conj(Y_bus @ V).

    Returns
    -------
    s : jnp.ndarray
        Complex128 array of shape (n_bus,).
    """
    current = y_bus @ voltage
    return voltage * jnp.conjugate(current)


def power_flow_residual(
    topology: CompiledTopology,
    params: NetworkParams,
    state: PFState,
) -> jnp.ndarray:
    """
    Compute the concatenated mismatch vector for non-slack buses.

    For each non-slack bus i (index into variable_buses):
      PQ-bus  (is_pq_mask[i] == True):
        r[2i]   = p_spec[i] - P_calc[i]
        r[2i+1] = q_spec[i] - Q_calc[i]
      PV-bus  (is_pv_mask[i] == True):
        r[2i]   = p_spec[i] - P_calc[i]
        r[2i+1] = v_set_pu[i]**2 - (Vr[i]**2 + Vi[i]**2)

    Returns
    -------
    r : jnp.ndarray
        Float64 array of shape (2 * n_var,).
    """
    y_bus = build_ybus(topology, params)
    voltage = state_to_voltage(topology, params, state)
    s_calc = calc_power_injection(y_bus, voltage)

    var = topology.variable_buses
    r_p = params.p_spec_pu[var] - jnp.real(s_calc[var])

    # Q-mismatch for PQ buses; voltage magnitude equation for PV buses
    r_q_pq = params.q_spec_pu[var] - jnp.imag(s_calc[var])
    v_mag_sq_calc = state.vr_pu ** 2 + state.vi_pu ** 2
    r_q_pv = params.v_set_pu ** 2 - v_mag_sq_calc

    r_q = jnp.where(topology.is_pv_mask, r_q_pv, r_q_pq)

    return jnp.concatenate([r_p, r_q], axis=0)


def residual_loss(
    topology: CompiledTopology,
    params: NetworkParams,
    state: PFState,
) -> jnp.ndarray:
    """
    Scalar loss 0.5 * ||r||² for use in gradient-based analyses.

    Returns
    -------
    loss : jnp.ndarray
        Non-negative float64 scalar.
    """
    r = power_flow_residual(topology, params, state)
    return 0.5 * jnp.vdot(r, r).real
