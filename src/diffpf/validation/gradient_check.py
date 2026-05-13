"""Gradient validation helpers for Experiment 2."""

from __future__ import annotations

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp

from diffpf.core.observables import power_flow_observables
from diffpf.core.types import CompiledTopology, NetworkParams, PFState
from diffpf.io.parser import parse
from diffpf.io.reader import RawNetwork
from diffpf.solver import NewtonOptions, solve_power_flow_implicit
from diffpf.validation.finite_diff import central_difference
from diffpf.validation.pandapower_ref import (
    PowerFlowValidationCase,
    default_validation_cases,
    make_operating_point,
)


INPUT_SPECS: dict[str, tuple[str, int]] = {
    "P_load": ("p_spec_pu", 1),
    "Q_load": ("q_spec_pu", 1),
    "P_pv": ("p_spec_pu", 2),
    "Q_pv": ("q_spec_pu", 2),
}

OUTPUT_SPECS: tuple[str, ...] = (
    "V1_mag",
    "V2_mag",
    "theta1_rad",
    "theta2_rad",
    "P_loss_total",
    "P_slack",
    "line0_P_from",
)


@dataclass(frozen=True)
class GradientValidationRow:
    """One AD-vs-FD comparison row."""

    scenario: str
    output_name: str
    input_name: str
    ad_grad: float
    fd_grad: float
    abs_error: float
    rel_error: float
    fd_step: float


@dataclass(frozen=True)
class ErrorSummaryRow:
    """Aggregated gradient error statistics for one scenario."""

    scenario: str
    max_abs_error: float
    max_rel_error: float
    mean_abs_error: float
    mean_rel_error: float
    n_gradients: int


@dataclass(frozen=True)
class StepStudyRow:
    """One finite-difference step-size study row."""

    scenario: str
    output_name: str
    input_name: str
    ad_grad: float
    fd_grad: float
    abs_error: float
    rel_error: float
    fd_step: float


def robust_relative_error(ad_grad: float, fd_grad: float, floor: float = 1e-12) -> float:
    """Return a stable relative error for gradients that may be close to zero."""

    return abs(ad_grad - fd_grad) / max(abs(fd_grad), abs(ad_grad), floor)


def vary_input(params: NetworkParams, input_name: str, value: jnp.ndarray) -> NetworkParams:
    """Set one named p.u. input in the existing NetworkParams pytree."""

    field_name, index = INPUT_SPECS[input_name]
    values = getattr(params, field_name).at[index].set(value)
    return replace(params, **{field_name: values})


def output_value(
    output_name: str,
    topology: CompiledTopology,
    params: NetworkParams,
    state: PFState,
    options: NewtonOptions = NewtonOptions(max_iters=30, tolerance=1e-10, damping=1.0),
) -> jnp.ndarray:
    """Solve implicitly and return one scalar observable."""

    solution = solve_power_flow_implicit(topology, params, state, options)
    obs = power_flow_observables(topology, params, solution)
    if output_name == "V1_mag":
        return obs.voltage_mag_pu[0]
    if output_name == "V2_mag":
        return obs.voltage_mag_pu[1]
    if output_name == "theta1_rad":
        return obs.voltage_angle_rad[0]
    if output_name == "theta2_rad":
        return obs.voltage_angle_rad[1]
    if output_name == "P_loss_total":
        return obs.total_p_loss_pu
    if output_name == "P_slack":
        return obs.slack_p_pu
    if output_name == "line0_P_from":
        return obs.line_p_from_pu[0]
    raise ValueError(f"Unknown output_name: {output_name}")


def output_vector(
    output_names: tuple[str, ...],
    topology: CompiledTopology,
    params: NetworkParams,
    state: PFState,
    options: NewtonOptions = NewtonOptions(max_iters=30, tolerance=1e-10, damping=1.0),
) -> jnp.ndarray:
    """Solve once and return a compact vector of named scalar observables."""

    solution = solve_power_flow_implicit(topology, params, state, options)
    obs = power_flow_observables(topology, params, solution)
    values = {
        "V1_mag": obs.voltage_mag_pu[0],
        "V2_mag": obs.voltage_mag_pu[1],
        "theta1_rad": obs.voltage_angle_rad[0],
        "theta2_rad": obs.voltage_angle_rad[1],
        "P_loss_total": obs.total_p_loss_pu,
        "P_slack": obs.slack_p_pu,
        "line0_P_from": obs.line_p_from_pu[0],
    }
    return jnp.stack([values[name] for name in output_names])


def _input_start_value(params: NetworkParams, input_name: str) -> jnp.ndarray:
    field_name, index = INPUT_SPECS[input_name]
    return getattr(params, field_name)[index]


def gradient_row(
    scenario: str,
    output_name: str,
    input_name: str,
    topology: CompiledTopology,
    params: NetworkParams,
    state: PFState,
    fd_step: float = 1e-5,
    options: NewtonOptions = NewtonOptions(max_iters=30, tolerance=1e-10, damping=1.0),
) -> GradientValidationRow:
    """Compare one implicit AD gradient against a central finite difference."""

    def scalar_fn(value: jnp.ndarray) -> jnp.ndarray:
        varied = vary_input(params, input_name, value)
        return output_value(output_name, topology, varied, state, options)

    x0 = _input_start_value(params, input_name)
    ad_grad = float(jax.grad(scalar_fn)(x0))
    fd_grad = central_difference(lambda x: float(scalar_fn(jnp.asarray(x))), float(x0), h=fd_step)
    abs_error = abs(ad_grad - fd_grad)
    return GradientValidationRow(
        scenario=scenario,
        output_name=output_name,
        input_name=input_name,
        ad_grad=ad_grad,
        fd_grad=fd_grad,
        abs_error=abs_error,
        rel_error=robust_relative_error(ad_grad, fd_grad),
        fd_step=fd_step,
    )


def scenario_from_raw(
    raw: RawNetwork,
    scenario: PowerFlowValidationCase,
) -> tuple[CompiledTopology, NetworkParams, PFState]:
    """Build compiled JAX inputs for one Experiment 2 scenario."""

    return parse(make_operating_point(raw, scenario))


def validate_scenario_gradients(
    scenario_name: str,
    topology: CompiledTopology,
    params: NetworkParams,
    state: PFState,
    output_names: tuple[str, ...] = OUTPUT_SPECS,
    input_names: tuple[str, ...] = tuple(INPUT_SPECS.keys()),
    fd_step: float = 1e-5,
) -> tuple[GradientValidationRow, ...]:
    """Run the full local Jacobian comparison for one operating point."""

    rows: list[GradientValidationRow] = []
    for input_name in input_names:
        x0 = _input_start_value(params, input_name)

        def vector_fn(value: jnp.ndarray) -> jnp.ndarray:
            varied = vary_input(params, input_name, value)
            return output_vector(output_names, topology, varied, state)

        ad_grads = jax.jacfwd(vector_fn)(x0)
        fd_grads = (
            vector_fn(x0 + fd_step) - vector_fn(x0 - fd_step)
        ) / (2.0 * fd_step)
        for output_idx, output_name in enumerate(output_names):
            ad_grad = float(ad_grads[output_idx])
            fd_grad = float(fd_grads[output_idx])
            abs_error = abs(ad_grad - fd_grad)
            rows.append(
                GradientValidationRow(
                    scenario=scenario_name,
                    output_name=output_name,
                    input_name=input_name,
                    ad_grad=ad_grad,
                    fd_grad=fd_grad,
                    abs_error=abs_error,
                    rel_error=robust_relative_error(ad_grad, fd_grad),
                    fd_step=fd_step,
                )
            )
    return tuple(rows)


def summarize_errors(rows: tuple[GradientValidationRow, ...]) -> tuple[ErrorSummaryRow, ...]:
    """Aggregate gradient errors by scenario."""

    scenario_names = tuple(dict.fromkeys(row.scenario for row in rows))
    summaries: list[ErrorSummaryRow] = []
    for scenario in scenario_names:
        selected = [row for row in rows if row.scenario == scenario]
        abs_errors = [row.abs_error for row in selected]
        rel_errors = [row.rel_error for row in selected]
        summaries.append(
            ErrorSummaryRow(
                scenario=scenario,
                max_abs_error=max(abs_errors),
                max_rel_error=max(rel_errors),
                mean_abs_error=sum(abs_errors) / len(abs_errors),
                mean_rel_error=sum(rel_errors) / len(rel_errors),
                n_gradients=len(selected),
            )
        )
    return tuple(summaries)


def finite_difference_step_study(
    scenario_name: str,
    topology: CompiledTopology,
    params: NetworkParams,
    state: PFState,
    samples: tuple[tuple[str, str], ...] = (
        ("V2_mag", "P_pv"),
        ("P_loss_total", "P_load"),
        ("P_slack", "P_pv"),
    ),
    steps: tuple[float, ...] = (1e-2, 1e-4, 1e-6),
) -> tuple[StepStudyRow, ...]:
    """Run a small finite-difference step-size study."""

    rows: list[StepStudyRow] = []
    for output_name, input_name in samples:
        def scalar_fn(value: jnp.ndarray) -> jnp.ndarray:
            varied = vary_input(params, input_name, value)
            return output_value(output_name, topology, varied, state)

        x0 = _input_start_value(params, input_name)
        ad_grad = float(jax.grad(scalar_fn)(x0))
        for step in steps:
            fd_grad = central_difference(
                lambda x: float(scalar_fn(jnp.asarray(x))),
                float(x0),
                h=step,
            )
            abs_error = abs(ad_grad - fd_grad)
            rows.append(
                StepStudyRow(
                    scenario=scenario_name,
                    output_name=output_name,
                    input_name=input_name,
                    ad_grad=ad_grad,
                    fd_grad=fd_grad,
                    abs_error=abs_error,
                    rel_error=robust_relative_error(ad_grad, fd_grad),
                    fd_step=step,
                )
            )
    return tuple(rows)


def experiment2_scenarios() -> tuple[PowerFlowValidationCase, ...]:
    """Reuse the established Experiment 1 operating points."""

    return default_validation_cases()
