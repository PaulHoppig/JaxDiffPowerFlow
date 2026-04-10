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

Power mismatch (non-slack buses only):
  r_P = P_spec - Re(S_calc)   [shape: (n_var,)]
  r_Q = Q_spec - Im(S_calc)   [shape: (n_var,)]
  r   = [r_P | r_Q]            [shape: (2*n_var,)]

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
    Compute the concatenated PQ mismatch vector for non-slack buses.

    Returns
    -------
    r : jnp.ndarray
        Float64 array of shape (2 * n_var,) = [r_P | r_Q].
    """
    y_bus = build_ybus(topology, params)
    voltage = state_to_voltage(topology, params, state)
    s_calc = calc_power_injection(y_bus, voltage)

    var = topology.variable_buses
    r_p = params.p_spec_pu[var] - jnp.real(s_calc[var])
    r_q = params.q_spec_pu[var] - jnp.imag(s_calc[var])
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
