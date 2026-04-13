"""Solver-independent observable quantities derived from a solved PF state."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax.numpy as jnp
from jax import tree_util

from diffpf.core.residuals import calc_power_injection, state_to_voltage
from diffpf.core.types import CompiledTopology, NetworkParams, PFState
from diffpf.core.ybus import build_ybus


@partial(
    tree_util.register_dataclass,
    data_fields=[
        "voltage_mag_pu",
        "voltage_angle_rad",
        "slack_p_pu",
        "slack_q_pu",
        "total_p_loss_pu",
        "total_q_loss_pu",
        "line_p_from_pu",
        "line_q_from_pu",
        "line_p_to_pu",
        "line_q_to_pu",
        "line_p_loss_pu",
        "line_q_loss_pu",
    ],
    meta_fields=[],
)
@dataclass(frozen=True)
class PowerFlowObservables:
    """Differentiable electrical quantities for reporting and gradients."""

    voltage_mag_pu: jnp.ndarray
    voltage_angle_rad: jnp.ndarray
    slack_p_pu: jnp.ndarray
    slack_q_pu: jnp.ndarray
    total_p_loss_pu: jnp.ndarray
    total_q_loss_pu: jnp.ndarray
    line_p_from_pu: jnp.ndarray
    line_q_from_pu: jnp.ndarray
    line_p_to_pu: jnp.ndarray
    line_q_to_pu: jnp.ndarray
    line_p_loss_pu: jnp.ndarray
    line_q_loss_pu: jnp.ndarray


def power_flow_observables(
    topology: CompiledTopology,
    params: NetworkParams,
    state: PFState,
) -> PowerFlowObservables:
    """Compute differentiable non-slack voltages, slack power, losses, and flows."""

    voltage = state_to_voltage(topology, params, state)
    y_bus = build_ybus(topology, params)
    s_bus = calc_power_injection(y_bus, voltage)

    v_from = voltage[topology.from_bus]
    v_to = voltage[topology.to_bus]
    y_series = params.g_series_pu + 1j * params.b_series_pu
    y_shunt_half = 0.5j * params.b_shunt_pu
    i_from = (v_from - v_to) * y_series + v_from * y_shunt_half
    i_to = (v_to - v_from) * y_series + v_to * y_shunt_half
    s_from = v_from * jnp.conjugate(i_from)
    s_to = v_to * jnp.conjugate(i_to)
    s_loss = s_from + s_to

    non_slack_voltage = voltage[topology.variable_buses]
    return PowerFlowObservables(
        voltage_mag_pu=jnp.abs(non_slack_voltage),
        voltage_angle_rad=jnp.angle(non_slack_voltage),
        slack_p_pu=jnp.real(s_bus[topology.slack_bus]),
        slack_q_pu=jnp.imag(s_bus[topology.slack_bus]),
        total_p_loss_pu=jnp.sum(jnp.real(s_loss)),
        total_q_loss_pu=jnp.sum(jnp.imag(s_loss)),
        line_p_from_pu=jnp.real(s_from),
        line_q_from_pu=jnp.imag(s_from),
        line_p_to_pu=jnp.real(s_to),
        line_q_to_pu=jnp.imag(s_to),
        line_p_loss_pu=jnp.real(s_loss),
        line_q_loss_pu=jnp.imag(s_loss),
    )
