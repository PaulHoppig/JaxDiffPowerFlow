"""Experiment 2b: gradient validation for pandapower ``example_simple()``.

This experiment validates a compact set of local implicit gradients against
central finite differences on the scope-matched ``example_simple()`` network.
The network topology is fixed; all varied inputs are continuous scale factors.

Run:
    python experiments/exp02_validate_gradients_example_simple.py
"""

from __future__ import annotations

import copy
import csv
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import NamedTuple

from jax import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp

from diffpf.compile.network import compile_network
from diffpf.core.residuals import calc_power_injection, state_to_voltage
from diffpf.core.types import CompiledTopology, NetworkParams, PFState
from diffpf.core.ybus import build_ybus
from diffpf.io.pandapower_adapter import from_pandapower
from diffpf.io.topology_utils import merge_buses
from diffpf.solver.implicit import solve_power_flow_implicit
from diffpf.solver.newton import NewtonOptions, solve_power_flow_result


RESULTS_DIR = (
    Path(__file__).resolve().parent / "results" / "exp02_example_simple_gradients"
)

NEWTON_OPTIONS = NewtonOptions(max_iters=50, tolerance=1e-10, damping=0.7)
FD_STEP_DEFAULT = 1e-4
FD_STEPS_STUDY = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)


@dataclass(frozen=True)
class ScenarioSpec:
    """One fixed operating point for the gradient validation."""

    name: str
    load_factor: float
    sgen_factor: float


@dataclass(frozen=True)
class InputParameterSpec:
    """Continuous scalar parameter used as AD/FD input."""

    name: str
    units: str
    description: str


@dataclass(frozen=True)
class OutputObservableSpec:
    """Scalar differentiable observable used as AD/FD output."""

    name: str
    units: str
    description: str


SCENARIOS: tuple[ScenarioSpec, ...] = (
    ScenarioSpec("base", 1.00, 1.00),
    ScenarioSpec("load_high", 1.25, 1.00),
    ScenarioSpec("sgen_high", 1.00, 1.50),
)

INPUT_PARAMETERS: tuple[InputParameterSpec, ...] = (
    InputParameterSpec(
        "load_scale_mv_bus_2",
        "dimensionless",
        "Scales P and Q of the existing load at MV Bus 2.",
    ),
    InputParameterSpec(
        "sgen_scale_static_generator",
        "dimensionless",
        "Scales P and Q of the existing static generator.",
    ),
    InputParameterSpec(
        "shunt_q_scale",
        "dimensionless",
        "Scales the susceptance contribution of the existing shunt.",
    ),
    InputParameterSpec(
        "trafo_x_scale",
        "dimensionless",
        "Scales the series reactance of the existing two-winding transformer.",
    ),
)

OUTPUT_OBSERVABLES: tuple[OutputObservableSpec, ...] = (
    OutputObservableSpec(
        "vm_mv_bus_2_pu",
        "p.u.",
        "Voltage magnitude at MV Bus 2 after bus-fusion mapping.",
    ),
    OutputObservableSpec(
        "p_slack_mw",
        "MW",
        "Active power injection at the slack bus.",
    ),
    OutputObservableSpec(
        "total_p_loss_mw",
        "MW",
        "Total active network losses from the solved bus power balance.",
    ),
    OutputObservableSpec(
        "p_trafo_hv_mw",
        "MW",
        "Active transformer power consumed from the HV bus side.",
    ),
)

STEP_STUDY_SELECTIONS: tuple[tuple[str, str, str], ...] = (
    ("base", "load_scale_mv_bus_2", "vm_mv_bus_2_pu"),
    ("base", "sgen_scale_static_generator", "p_slack_mw"),
    ("base", "shunt_q_scale", "total_p_loss_mw"),
)


@dataclass(frozen=True)
class ExampleSimpleMetadata:
    """Static mapping and base contributions for one compiled scenario."""

    s_base_mva: float
    target_bus_internal_idx: int
    load_bus_internal_idx: int
    sgen_bus_internal_idx: int
    shunt_idx: int
    trafo_idx: int
    load_p_injection_pu: float
    load_q_injection_pu: float
    sgen_p_injection_pu: float
    sgen_q_injection_pu: float
    bus_to_repr_json: str
    notes: str


@dataclass(frozen=True)
class GradientTableRow:
    """One AD-vs-FD comparison row."""

    scenario: str
    input_parameter: str
    output_observable: str
    theta0: float
    fd_step: float
    ad_grad: float
    fd_grad: float
    abs_error: float
    rel_error: float
    ad_converged: bool
    fd_plus_converged: bool
    fd_minus_converged: bool
    base_residual_norm: float
    plus_residual_norm: float
    minus_residual_norm: float
    base_iterations: int
    plus_iterations: int
    minus_iterations: int
    units_input: str
    units_output: str
    notes: str


@dataclass(frozen=True)
class ErrorSummaryRow:
    """Aggregated error statistics by scenario and input parameter."""

    scenario: str
    input_parameter: str
    n_gradients: int
    n_valid: int
    max_abs_error: float
    median_abs_error: float
    max_rel_error: float
    median_rel_error: float
    n_failed_ad: int
    n_failed_fd: int
    worst_output_observable: str


@dataclass(frozen=True)
class StepStudyRow:
    """One finite-difference step-size study row."""

    selected_gradient_id: str
    scenario: str
    input_parameter: str
    output_observable: str
    fd_step: float
    ad_grad: float
    fd_grad: float
    abs_error: float
    rel_error: float
    fd_plus_converged: bool
    fd_minus_converged: bool


class ScenarioInputs(NamedTuple):
    """Compiled inputs for one scenario."""

    topology: CompiledTopology
    params: NetworkParams
    state0: PFState
    metadata: ExampleSimpleMetadata


class SolveDiagnostics(NamedTuple):
    """Forward solve convergence metadata."""

    converged: bool
    residual_norm: float
    iterations: int


GRADIENT_TABLE_COLUMNS = tuple(field.name for field in fields(GradientTableRow))
ERROR_SUMMARY_COLUMNS = tuple(field.name for field in fields(ErrorSummaryRow))
FD_STEP_STUDY_COLUMNS = tuple(field.name for field in fields(StepStudyRow))


def robust_relative_error(ad_grad: float, fd_grad: float, floor: float = 1e-12) -> float:
    """Return a stable relative error for gradients close to zero."""

    if not math.isfinite(ad_grad) or not math.isfinite(fd_grad):
        return float("nan")
    return abs(ad_grad - fd_grad) / max(abs(fd_grad), abs(ad_grad), floor)


def _select_row_by_name(table, expected_name: str):
    if "name" in table.columns:
        matched = table[table["name"].astype(str) == expected_name]
        if len(matched) > 0:
            return matched.iloc[0]
    active = table[table["in_service"] == True] if "in_service" in table.columns else table  # noqa: E712
    if len(active) == 0:
        raise ValueError(f"No active element found for {expected_name!r}.")
    return active.iloc[0]


def make_scope_matched_example_simple(load_factor: float, sgen_factor: float):
    """Build ``example_simple()`` with the Experiment 1 scope-matched gen model."""

    import pandapower as pp
    import pandapower.networks as pn

    net = pn.example_simple()

    load = _select_row_by_name(net.load, "load")
    net.load.at[load.name, "scaling"] = float(net.load.at[load.name, "scaling"]) * load_factor

    sgen = _select_row_by_name(net.sgen, "static generator")
    net.sgen.at[sgen.name, "scaling"] = float(net.sgen.at[sgen.name, "scaling"]) * sgen_factor

    net = copy.deepcopy(net)
    for idx, row in net.gen.iterrows():
        if not bool(row["in_service"]):
            continue
        pp.create_sgen(
            net,
            bus=int(row["bus"]),
            p_mw=float(row["p_mw"]),
            q_mvar=0.0,
            name=f"gen_as_sgen_{idx}",
            in_service=True,
        )
        net.gen.at[idx, "in_service"] = False

    return net


def _build_switch_info(net) -> tuple[dict[int, int], set[int], set[int]]:
    """Return the bus-fusion map and disabled line/trafo sets."""

    bb_pairs: list[tuple[int, int]] = []
    disabled_lines: set[int] = set()
    disabled_trafos: set[int] = set()

    for _, sw in net.switch.iterrows():
        et = sw["et"]
        closed = bool(sw["closed"])
        element = int(sw["element"])
        if et == "b" and closed:
            bb_pairs.append((int(sw["bus"]), element))
        elif et == "l" and not closed:
            disabled_lines.add(element)
        elif et == "t" and not closed:
            disabled_trafos.add(element)

    return merge_buses(list(net.bus.index), bb_pairs), disabled_lines, disabled_trafos


def make_smart_initial_state(net, spec) -> PFState:
    """Use the Experiment 1 trafo-shift-aware initialisation."""

    slack_ang_rad = float(net.ext_grid.iloc[0]["va_degree"]) * math.pi / 180.0
    trafo_shift_deg = float(net.trafo.iloc[0]["shift_degree"]) if len(net.trafo) else 0.0
    lv_ang_rad = (float(net.ext_grid.iloc[0]["va_degree"]) - trafo_shift_deg) * math.pi / 180.0

    vr_list: list[float] = []
    vi_list: list[float] = []
    for bus in spec.buses:
        if bus.is_slack:
            continue
        original_bus_id = int(bus.name)
        vn_kv = float(net.bus.loc[original_bus_id, "vn_kv"])
        angle = slack_ang_rad if vn_kv >= 100.0 else lv_ang_rad
        vr_list.append(math.cos(angle))
        vi_list.append(math.sin(angle))

    return PFState(
        vr_pu=jnp.asarray(vr_list, dtype=jnp.float64),
        vi_pu=jnp.asarray(vi_list, dtype=jnp.float64),
    )


def _internal_bus_index_for_original_bus(spec, bus_to_repr: dict[int, int], original_bus: int) -> int:
    repr_bus = bus_to_repr[int(original_bus)]
    for internal_idx, bus in enumerate(spec.buses):
        if int(bus.name) == repr_bus:
            return internal_idx
    raise ValueError(f"Original bus {original_bus} / repr {repr_bus} not found in spec.")


def build_scenario_inputs(scenario: ScenarioSpec) -> ScenarioInputs:
    """Compile the scope-matched example_simple scenario for JAX."""

    net = make_scope_matched_example_simple(scenario.load_factor, scenario.sgen_factor)
    spec = from_pandapower(net)
    topology, params = compile_network(spec)
    state0 = make_smart_initial_state(net, spec)

    bus_to_repr, _, _ = _build_switch_info(net)
    s_base_mva = float(net.sn_mva) if float(net.sn_mva) > 0 else 1.0

    load_row = _select_row_by_name(net.load, "load")
    sgen_row = _select_row_by_name(net.sgen, "static generator")
    load_bus = _internal_bus_index_for_original_bus(spec, bus_to_repr, int(load_row["bus"]))
    sgen_bus = _internal_bus_index_for_original_bus(spec, bus_to_repr, int(sgen_row["bus"]))

    load_scaling = float(load_row.get("scaling", 1.0))
    sgen_scaling = float(sgen_row.get("scaling", 1.0))

    load_p_injection = -float(load_row["p_mw"]) * load_scaling / s_base_mva
    load_q_injection = -float(load_row["q_mvar"]) * load_scaling / s_base_mva
    sgen_p_injection = float(sgen_row["p_mw"]) * sgen_scaling / s_base_mva
    sgen_q_injection = float(sgen_row["q_mvar"]) * sgen_scaling / s_base_mva

    metadata = ExampleSimpleMetadata(
        s_base_mva=s_base_mva,
        target_bus_internal_idx=load_bus,
        load_bus_internal_idx=load_bus,
        sgen_bus_internal_idx=sgen_bus,
        shunt_idx=0,
        trafo_idx=0,
        load_p_injection_pu=load_p_injection,
        load_q_injection_pu=load_q_injection,
        sgen_p_injection_pu=sgen_p_injection,
        sgen_q_injection_pu=sgen_q_injection,
        bus_to_repr_json=json.dumps({str(k): v for k, v in sorted(bus_to_repr.items())}),
        notes=(
            "scope_matched: active gen is converted to sgen(P, Q=0); "
            "no voltage regulation or Q limits are enforced."
        ),
    )
    return ScenarioInputs(topology=topology, params=params, state0=state0, metadata=metadata)


def apply_input_parameter(
    params: NetworkParams,
    parameter_name: str,
    value: jnp.ndarray,
    metadata: ExampleSimpleMetadata,
) -> NetworkParams:
    """Apply one continuous scale parameter to an existing NetworkParams pytree."""

    theta = jnp.asarray(value, dtype=jnp.float64)

    if parameter_name == "load_scale_mv_bus_2":
        p_spec = params.p_spec_pu.at[metadata.load_bus_internal_idx].add(
            (theta - 1.0) * metadata.load_p_injection_pu
        )
        q_spec = params.q_spec_pu.at[metadata.load_bus_internal_idx].add(
            (theta - 1.0) * metadata.load_q_injection_pu
        )
        return replace(params, p_spec_pu=p_spec, q_spec_pu=q_spec)

    if parameter_name == "sgen_scale_static_generator":
        p_spec = params.p_spec_pu.at[metadata.sgen_bus_internal_idx].add(
            (theta - 1.0) * metadata.sgen_p_injection_pu
        )
        q_spec = params.q_spec_pu.at[metadata.sgen_bus_internal_idx].add(
            (theta - 1.0) * metadata.sgen_q_injection_pu
        )
        return replace(params, p_spec_pu=p_spec, q_spec_pu=q_spec)

    if parameter_name == "shunt_q_scale":
        shunt_b = params.shunt_b_pu.at[metadata.shunt_idx].set(
            params.shunt_b_pu[metadata.shunt_idx] * theta
        )
        return replace(params, shunt_b_pu=shunt_b)

    if parameter_name == "trafo_x_scale":
        idx = metadata.trafo_idx
        y_base = params.trafo_g_series_pu[idx] + 1j * params.trafo_b_series_pu[idx]
        z_base = 1.0 / y_base
        r_pu = jnp.real(z_base)
        x_pu = jnp.imag(z_base) * theta
        y_scaled = 1.0 / (r_pu + 1j * x_pu)
        trafo_g = params.trafo_g_series_pu.at[idx].set(jnp.real(y_scaled))
        trafo_b = params.trafo_b_series_pu.at[idx].set(jnp.imag(y_scaled))
        return replace(params, trafo_g_series_pu=trafo_g, trafo_b_series_pu=trafo_b)

    raise ValueError(f"Unknown input parameter: {parameter_name}")


def _trafo_hv_p_mw(
    topology: CompiledTopology,
    params: NetworkParams,
    state: PFState,
    metadata: ExampleSimpleMetadata,
) -> jnp.ndarray:
    """Return active power consumed from the transformer HV side in MW."""

    voltage = state_to_voltage(topology, params, state)
    idx = metadata.trafo_idx
    hv = params.trafo_hv_bus[idx]
    lv = params.trafo_lv_bus[idx]

    y_t = params.trafo_g_series_pu[idx] + 1j * params.trafo_b_series_pu[idx]
    y_m = params.trafo_g_mag_pu[idx] + 1j * params.trafo_b_mag_pu[idx]
    a = params.trafo_tap_ratio[idx]
    phi = params.trafo_shift_rad[idx]
    tap = a * jnp.exp(1j * phi)
    current_hv = ((y_t + y_m) / (a * a)) * voltage[hv] + (-y_t / jnp.conj(tap)) * voltage[lv]
    s_hv = voltage[hv] * jnp.conj(current_hv)
    return jnp.real(s_hv) * metadata.s_base_mva


def evaluate_solved_observable(
    topology: CompiledTopology,
    params: NetworkParams,
    solution: PFState,
    observable_name: str,
    metadata: ExampleSimpleMetadata,
) -> jnp.ndarray:
    """Evaluate one scalar observable from an already solved state."""

    voltage = state_to_voltage(topology, params, solution)
    y_bus = build_ybus(topology, params)
    s_bus = calc_power_injection(y_bus, voltage)

    if observable_name == "vm_mv_bus_2_pu":
        return jnp.abs(voltage[metadata.target_bus_internal_idx])
    if observable_name == "p_slack_mw":
        return jnp.real(s_bus[topology.slack_bus]) * metadata.s_base_mva
    if observable_name == "total_p_loss_mw":
        return jnp.sum(jnp.real(s_bus)) * metadata.s_base_mva
    if observable_name == "p_trafo_hv_mw":
        return _trafo_hv_p_mw(topology, params, solution, metadata)

    raise ValueError(f"Unknown output observable: {observable_name}")


def evaluate_observable(
    params: NetworkParams,
    observable_name: str,
    topology: CompiledTopology,
    state: PFState,
    metadata: ExampleSimpleMetadata,
    options: NewtonOptions = NEWTON_OPTIONS,
) -> jnp.ndarray:
    """Solve implicitly and return one scalar observable."""

    solution = solve_power_flow_implicit(topology, params, state, options)
    return evaluate_solved_observable(topology, params, solution, observable_name, metadata)


def _solve_diagnostics(
    topology: CompiledTopology,
    params: NetworkParams,
    state: PFState,
) -> SolveDiagnostics:
    """Run the forward Newton solver and return convergence metadata."""

    try:
        result = solve_power_flow_result(topology, params, state, NEWTON_OPTIONS)
        return SolveDiagnostics(
            converged=bool(result.converged),
            residual_norm=float(result.residual_norm),
            iterations=int(result.iterations),
        )
    except Exception:
        return SolveDiagnostics(
            converged=False,
            residual_norm=float("nan"),
            iterations=-1,
        )


def _scalar_function(
    scenario_inputs: ScenarioInputs,
    input_parameter: str,
    output_observable: str,
):
    def scalar_fn(theta: jnp.ndarray) -> jnp.ndarray:
        varied = apply_input_parameter(
            scenario_inputs.params,
            input_parameter,
            theta,
            scenario_inputs.metadata,
        )
        return evaluate_observable(
            varied,
            output_observable,
            scenario_inputs.topology,
            scenario_inputs.state0,
            scenario_inputs.metadata,
        )

    return scalar_fn


def _finite_difference(
    scalar_fn,
    theta0: float,
    step: float,
) -> float:
    plus = scalar_fn(jnp.asarray(theta0 + step, dtype=jnp.float64))
    minus = scalar_fn(jnp.asarray(theta0 - step, dtype=jnp.float64))
    return float((plus - minus) / (2.0 * step))


def gradient_row(
    scenario: ScenarioSpec,
    scenario_inputs: ScenarioInputs,
    input_spec: InputParameterSpec,
    output_spec: OutputObservableSpec,
    fd_step: float = FD_STEP_DEFAULT,
) -> GradientTableRow:
    """Compare one implicit AD gradient against a central finite difference."""

    theta0 = 1.0
    scalar_fn = _scalar_function(scenario_inputs, input_spec.name, output_spec.name)

    base_params = apply_input_parameter(
        scenario_inputs.params,
        input_spec.name,
        jnp.asarray(theta0, dtype=jnp.float64),
        scenario_inputs.metadata,
    )
    plus_params = apply_input_parameter(
        scenario_inputs.params,
        input_spec.name,
        jnp.asarray(theta0 + fd_step, dtype=jnp.float64),
        scenario_inputs.metadata,
    )
    minus_params = apply_input_parameter(
        scenario_inputs.params,
        input_spec.name,
        jnp.asarray(theta0 - fd_step, dtype=jnp.float64),
        scenario_inputs.metadata,
    )

    base_diag = _solve_diagnostics(scenario_inputs.topology, base_params, scenario_inputs.state0)
    plus_diag = _solve_diagnostics(scenario_inputs.topology, plus_params, scenario_inputs.state0)
    minus_diag = _solve_diagnostics(scenario_inputs.topology, minus_params, scenario_inputs.state0)

    notes = []
    ad_grad = float("nan")
    fd_grad = float("nan")
    abs_error = float("nan")
    rel_error = float("nan")

    try:
        ad_grad = float(jax.grad(scalar_fn)(jnp.asarray(theta0, dtype=jnp.float64)))
    except Exception as exc:
        notes.append(f"AD failed: {type(exc).__name__}")

    if plus_diag.converged and minus_diag.converged:
        try:
            fd_grad = _finite_difference(scalar_fn, theta0, fd_step)
        except Exception as exc:
            notes.append(f"FD failed: {type(exc).__name__}")
    else:
        notes.append("FD comparison unavailable because plus or minus point did not converge.")

    ad_converged = base_diag.converged and math.isfinite(ad_grad)
    if ad_converged and plus_diag.converged and minus_diag.converged and math.isfinite(fd_grad):
        abs_error = abs(ad_grad - fd_grad)
        rel_error = robust_relative_error(ad_grad, fd_grad)
    elif not base_diag.converged:
        notes.append("Base point did not converge.")

    return GradientTableRow(
        scenario=scenario.name,
        input_parameter=input_spec.name,
        output_observable=output_spec.name,
        theta0=theta0,
        fd_step=fd_step,
        ad_grad=ad_grad,
        fd_grad=fd_grad,
        abs_error=abs_error,
        rel_error=rel_error,
        ad_converged=ad_converged,
        fd_plus_converged=plus_diag.converged,
        fd_minus_converged=minus_diag.converged,
        base_residual_norm=base_diag.residual_norm,
        plus_residual_norm=plus_diag.residual_norm,
        minus_residual_norm=minus_diag.residual_norm,
        base_iterations=base_diag.iterations,
        plus_iterations=plus_diag.iterations,
        minus_iterations=minus_diag.iterations,
        units_input=input_spec.units,
        units_output=output_spec.units,
        notes="; ".join(notes),
    )


def build_gradient_table() -> list[GradientTableRow]:
    """Run the mandatory 3 x 4 x 4 gradient validation grid."""

    rows: list[GradientTableRow] = []
    for scenario in SCENARIOS:
        print(f"  Building scenario {scenario.name} ...", flush=True)
        scenario_inputs = build_scenario_inputs(scenario)
        for input_spec in INPUT_PARAMETERS:
            for output_spec in OUTPUT_OBSERVABLES:
                print(
                    f"    {input_spec.name} -> {output_spec.name}",
                    flush=True,
                )
                rows.append(
                    gradient_row(
                        scenario,
                        scenario_inputs,
                        input_spec,
                        output_spec,
                        fd_step=FD_STEP_DEFAULT,
                    )
                )
    return rows


def build_fd_step_study() -> list[StepStudyRow]:
    """Run the small finite-difference step study for three selected gradients."""

    scenario_inputs_by_name = {scenario.name: build_scenario_inputs(scenario) for scenario in SCENARIOS}
    input_specs = {spec.name: spec for spec in INPUT_PARAMETERS}
    output_specs = {spec.name: spec for spec in OUTPUT_OBSERVABLES}
    rows: list[StepStudyRow] = []

    for scenario_name, input_name, output_name in STEP_STUDY_SELECTIONS:
        scenario_inputs = scenario_inputs_by_name[scenario_name]
        scalar_fn = _scalar_function(scenario_inputs, input_name, output_name)
        selected_id = f"{scenario_name}:{input_name}->{output_name}"
        ad_grad = float(jax.grad(scalar_fn)(jnp.asarray(1.0, dtype=jnp.float64)))

        for step in FD_STEPS_STUDY:
            plus_params = apply_input_parameter(
                scenario_inputs.params,
                input_name,
                jnp.asarray(1.0 + step, dtype=jnp.float64),
                scenario_inputs.metadata,
            )
            minus_params = apply_input_parameter(
                scenario_inputs.params,
                input_name,
                jnp.asarray(1.0 - step, dtype=jnp.float64),
                scenario_inputs.metadata,
            )
            plus_diag = _solve_diagnostics(
                scenario_inputs.topology,
                plus_params,
                scenario_inputs.state0,
            )
            minus_diag = _solve_diagnostics(
                scenario_inputs.topology,
                minus_params,
                scenario_inputs.state0,
            )

            if plus_diag.converged and minus_diag.converged:
                fd_grad = _finite_difference(scalar_fn, 1.0, step)
                abs_error = abs(ad_grad - fd_grad)
                rel_error = robust_relative_error(ad_grad, fd_grad)
            else:
                fd_grad = abs_error = rel_error = float("nan")

            rows.append(
                StepStudyRow(
                    selected_gradient_id=selected_id,
                    scenario=scenario_name,
                    input_parameter=input_specs[input_name].name,
                    output_observable=output_specs[output_name].name,
                    fd_step=step,
                    ad_grad=ad_grad,
                    fd_grad=fd_grad,
                    abs_error=abs_error,
                    rel_error=rel_error,
                    fd_plus_converged=plus_diag.converged,
                    fd_minus_converged=minus_diag.converged,
                )
            )
    return rows


def summarize_errors(rows: list[GradientTableRow]) -> list[ErrorSummaryRow]:
    """Aggregate gradient errors by scenario and input parameter."""

    summaries: list[ErrorSummaryRow] = []
    groups = sorted({(row.scenario, row.input_parameter) for row in rows})

    for scenario, input_parameter in groups:
        selected = [
            row
            for row in rows
            if row.scenario == scenario and row.input_parameter == input_parameter
        ]
        valid = [
            row
            for row in selected
            if row.ad_converged
            and row.fd_plus_converged
            and row.fd_minus_converged
            and math.isfinite(row.abs_error)
            and math.isfinite(row.rel_error)
        ]
        if valid:
            abs_errors = [row.abs_error for row in valid]
            rel_errors = [row.rel_error for row in valid]
            worst = max(valid, key=lambda row: row.rel_error)
            max_abs = max(abs_errors)
            median_abs = median(abs_errors)
            max_rel = max(rel_errors)
            median_rel = median(rel_errors)
            worst_output = worst.output_observable
        else:
            max_abs = median_abs = max_rel = median_rel = float("nan")
            worst_output = ""

        summaries.append(
            ErrorSummaryRow(
                scenario=scenario,
                input_parameter=input_parameter,
                n_gradients=len(selected),
                n_valid=len(valid),
                max_abs_error=max_abs,
                median_abs_error=median_abs,
                max_rel_error=max_rel,
                median_rel_error=median_rel,
                n_failed_ad=sum(not row.ad_converged for row in selected),
                n_failed_fd=sum(
                    not (row.fd_plus_converged and row.fd_minus_converged)
                    for row in selected
                ),
                worst_output_observable=worst_output,
            )
        )
    return summaries


def _to_native(obj):
    """Recursively convert JAX/NumPy scalar-ish values for JSON/CSV export."""

    if isinstance(obj, dict):
        return {key: _to_native(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(value) for value in obj]
    try:
        return obj.item()
    except AttributeError:
        return obj


def _write_csv(path: Path, rows: list, columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow(_to_native(asdict(row)))


def _write_json(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump([_to_native(asdict(row)) for row in rows], handle, indent=2)


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def write_metadata(results_dir: Path) -> None:
    """Write reproducibility metadata for the experiment."""

    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_hash": _git_hash(),
        "python_version": sys.version,
        "network": "pandapower.networks.example_simple(), scope_matched",
        "scenarios": [asdict(scenario) for scenario in SCENARIOS],
        "input_parameters": [asdict(spec) for spec in INPUT_PARAMETERS],
        "output_observables": [asdict(spec) for spec in OUTPUT_OBSERVABLES],
        "solver_options": {
            "max_iters": NEWTON_OPTIONS.max_iters,
            "tolerance": NEWTON_OPTIONS.tolerance,
            "damping": NEWTON_OPTIONS.damping,
            "initialization": "trafo_shift_aware",
        },
        "fd_step_default": FD_STEP_DEFAULT,
        "fd_step_study_steps": list(FD_STEPS_STUDY),
        "fd_step_study_selection": [
            {
                "scenario": scenario,
                "input_parameter": input_parameter,
                "output_observable": output_observable,
            }
            for scenario, input_parameter, output_observable in STEP_STUDY_SELECTIONS
        ],
        "known_model_simplifications": [
            "The active gen is converted to sgen(P, Q=0) in the scope-matched model.",
            "gen is not treated as a voltage-regulating PV bus in diffpf.",
            "Generator Q limits are not considered.",
            "No PV-to-PQ switching or controller behaviour is modelled.",
            "The topology remains constant for all AD and finite-difference points.",
            "Only four continuous scale parameters are checked; no full Jacobian is exported.",
        ],
        "excluded_gradients": [
            "individual line parameters",
            "individual bus P/Q entries",
            "trafo phase shift and tap ratio",
            "generator Q limits and PV-bus voltage setpoints",
            "trafo_r_scale diagnostic parameter",
        ],
    }
    path = results_dir / "metadata.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def write_readme(results_dir: Path) -> None:
    """Write a compact human-readable description of the artifacts."""

    text = """# Experiment 2b - example_simple() Gradient Validation

This directory contains a compact AD-vs-central-finite-difference validation for
the scope-matched pandapower `example_simple()` network. The active `gen` is
converted to `sgen(P, Q=0)`, matching the current diffpf model scope.

`gradient_table.csv` is long-format: one row is one
scenario/input-parameter/output-observable gradient. `error_summary.csv`
aggregates those rows by scenario and input parameter. `fd_step_study.csv`
contains only three representative gradients over five finite-difference steps.

The run intentionally avoids a full Jacobian over all network parameters. It
does not validate Q limits, PV-to-PQ switching, controller behaviour, tap
control, or the original pandapower generator voltage-regulation semantics.
"""
    path = results_dir / "README.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def export_all(
    gradient_rows: list[GradientTableRow],
    summary_rows: list[ErrorSummaryRow],
    step_rows: list[StepStudyRow],
    results_dir: Path,
) -> None:
    """Export all mandatory CSV/JSON artifacts and metadata."""

    _write_csv(results_dir / "gradient_table.csv", gradient_rows, GRADIENT_TABLE_COLUMNS)
    _write_json(results_dir / "gradient_table.json", gradient_rows)
    _write_csv(results_dir / "error_summary.csv", summary_rows, ERROR_SUMMARY_COLUMNS)
    _write_json(results_dir / "error_summary.json", summary_rows)
    _write_csv(results_dir / "fd_step_study.csv", step_rows, FD_STEP_STUDY_COLUMNS)
    _write_json(results_dir / "fd_step_study.json", step_rows)
    write_metadata(results_dir)
    write_readme(results_dir)


def main() -> None:
    print("=" * 72)
    print("Experiment 2b: example_simple() implicit-gradient validation")
    print(f"Results directory: {RESULTS_DIR}")
    print("=" * 72)

    gradient_rows = build_gradient_table()
    summary_rows = summarize_errors(gradient_rows)
    step_rows = build_fd_step_study()
    export_all(gradient_rows, summary_rows, step_rows, RESULTS_DIR)

    print()
    print("Aggregated error summary:")
    for row in summary_rows:
        max_abs = f"{row.max_abs_error:.3e}" if math.isfinite(row.max_abs_error) else "nan"
        max_rel = f"{row.max_rel_error:.3e}" if math.isfinite(row.max_rel_error) else "nan"
        print(
            f"  {row.scenario:<10} {row.input_parameter:<30} "
            f"valid={row.n_valid}/{row.n_gradients} "
            f"max_abs={max_abs} max_rel={max_rel}"
        )

    print()
    print("Exported artifacts:")
    for path in sorted(RESULTS_DIR.iterdir()):
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
