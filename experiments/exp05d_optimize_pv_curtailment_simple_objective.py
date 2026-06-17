"""Experiment 5d - PV curtailment with a simple quadratic export objective.

This experiment solves the same selected one-dimensional curtailment problem as
Experiment 5b, using the analytical PV/weather block, but replaces the
target-tracking/softplus/regularized objective by

    J(theta) = ((p_export_proxy(theta) - 7.0) / p_scale_mw) ** 2

The Exp. 5b artifacts and objective are not modified.

Run:
    python experiments/exp05d_optimize_pv_curtailment_simple_objective.py
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path

from jax import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffpf.core.residuals import calc_power_injection, state_to_voltage
from diffpf.core.ybus import build_ybus
from diffpf.models.pv import PV_COUPLING_BUS_NAME, PV_COUPLING_SGEN_NAME
from diffpf.solver.implicit import solve_power_flow_implicit
from diffpf.solver.newton import solve_power_flow_result
from experiments import exp05a_network_screening as exp05a
from experiments import exp05b_optimize_pv_curtailment as exp05b


RESULTS_DIR = (
    Path(__file__).resolve().parent
    / "results"
    / "exp05d_optimize_pv_curtailment_simple_objective"
)
EXP05B_RESULTS_DIR = (
    Path(__file__).resolve().parent / "results" / "exp05b_optimize_pv_curtailment"
)

EXPERIMENT_NAME = "exp05d_optimize_pv_curtailment_simple_objective"
OBJECTIVE_VARIANT = "simple_target_7mw"
UPSTREAM_MODEL_NAME = "analytical_pv_weather"

CASE_ID = exp05b.CASE_ID
LOAD_MULTIPLIER_MV_BUS_2 = exp05b.LOAD_MULTIPLIER_MV_BUS_2
G_POA_WM2 = exp05b.G_POA_WM2
T_AMB_C = exp05b.T_AMB_C
WIND_MS = exp05b.WIND_MS
PV_SIZE_FACTOR = exp05b.PV_SIZE_FACTOR
KAPPA = exp05b.KAPPA
P_EXPORT_LIMIT_MW = 7.0
P_EXPORT_TARGET_MW = 7.0

C_MIN = exp05b.C_MIN
C_MAX = exp05b.C_MAX
C_INIT = 1.0
BOUNDARY_START_EPS = 1e-3
LEARNING_RATE = exp05b.LEARNING_RATE
MAX_ITER = exp05b.MAX_ITER
ADAM_BETA1 = exp05b.ADAM_BETA1
ADAM_BETA2 = exp05b.ADAM_BETA2
ADAM_EPS = exp05b.ADAM_EPS
P_SCALE_MW = 1.0
GRID_POINTS = exp05b.GRID_POINTS

REQUIRED_ARTIFACTS: tuple[str, ...] = exp05b.REQUIRED_ARTIFACTS


@dataclass(frozen=True)
class SelectedCaseBaselineRow:
    baseline_type: str
    experiment: str
    objective_variant: str
    case_id: str
    load_multiplier_mv_bus_2: float
    g_poa_wm2: float
    t_amb_c: float
    wind_ms: float
    curtailment_factor: float
    pv_size_factor: float
    kappa: float
    p_export_limit_mw: float
    p_export_target_mw: float
    p_scale_mw: float
    p_export_proxy_mw: float
    export_proxy_mw: float
    hard_export_violation_mw: float
    soft_export_violation_mw: float
    export_margin_mw: float
    objective: float
    p_export_mw: float
    p_slack_mw: float
    p_pv_mw: float
    q_pv_mvar: float
    vm_mv_bus_2_pu: float
    total_p_loss_mw: float
    s_trafo_hv_mva: float
    converged: bool
    iterations: int
    residual_norm: float


@dataclass(frozen=True)
class OptimizationTraceRow:
    iteration: int
    theta: float
    curtailment_factor: float
    objective: float
    grad_theta: float
    p_export_limit_mw: float
    p_export_target_mw: float
    p_scale_mw: float
    p_export_proxy_mw: float
    export_proxy_mw: float
    hard_export_violation_mw: float
    soft_export_violation_mw: float
    export_margin_mw: float
    p_export_mw: float
    export_violation_mw: float
    p_slack_mw: float
    p_pv_mw: float
    q_pv_mvar: float
    vm_mv_bus_2_pu: float
    total_p_loss_mw: float
    s_trafo_hv_mva: float
    converged: bool
    iterations: int
    residual_norm: float


@dataclass(frozen=True)
class GridReferenceRow:
    grid_index: int
    curtailment_factor: float
    p_export_limit_mw: float
    p_export_target_mw: float
    p_scale_mw: float
    objective: float
    p_export_mw: float
    p_export_proxy_mw: float
    export_proxy_mw: float
    hard_export_violation_mw: float
    soft_export_violation_mw: float
    export_margin_mw: float
    feasible: bool
    is_grid_best_objective: bool
    is_grid_best_feasible: bool
    is_grid_best: bool
    p_slack_mw: float
    p_pv_mw: float
    q_pv_mvar: float
    vm_mv_bus_2_pu: float
    total_p_loss_mw: float
    s_trafo_hv_mva: float
    converged: bool
    iterations: int
    residual_norm: float


@dataclass(frozen=True)
class ConstraintDiagnosticsRow:
    case_id: str
    experiment: str
    objective_variant: str
    p_export_limit_mw: float
    p_export_full_pv_mw: float
    p_export_zero_pv_mw: float
    full_pv_violates_limit: bool
    feasible_by_zero_pv: bool
    constraint_satisfied: bool
    notes: str


@dataclass(frozen=True)
class FinalSolutionRow:
    experiment: str
    objective_variant: str
    selected_case_id: str
    case_id: str
    final_iteration: int
    final_theta: float
    final_curtailment_factor: float
    final_pv_utilization_pct: float
    final_curtailment_pct: float
    final_objective: float
    objective_final: float
    final_grad_theta: float
    final_p_export_proxy_mw: float
    final_p_export_mw: float
    final_p_slack_mw: float
    final_export_margin_mw: float
    final_hard_export_violation_mw: float
    final_soft_export_violation_mw: float
    final_p_pv_mw: float
    final_q_pv_mvar: float
    final_vm_mv_bus_2_pu: float
    final_total_p_loss_mw: float
    final_s_trafo_hv_mva: float
    p_export_limit_mw: float
    p_export_target_mw: float
    p_scale_mw: float
    grid_best_objective_curtailment_factor: float
    grid_best_objective_p_export_mw: float
    grid_best_feasible_curtailment_factor: float
    grid_best_feasible_p_export_mw: float
    grid_best_curtailment_factor: float
    abs_c_difference_optimizer_vs_grid_objective: float
    abs_c_difference_optimizer_vs_grid_feasible: float
    abs_c_difference_optimizer_vs_grid: float
    constraint_satisfied: bool


@dataclass(frozen=True)
class RunSummaryRow:
    metric: str
    value: float
    unit: str
    notes: str


SELECTED_CASE_BASELINE_COLUMNS = tuple(field.name for field in fields(SelectedCaseBaselineRow))
OPTIMIZATION_TRACE_COLUMNS = tuple(field.name for field in fields(OptimizationTraceRow))
GRID_REFERENCE_COLUMNS = tuple(field.name for field in fields(GridReferenceRow))
CONSTRAINT_DIAGNOSTICS_COLUMNS = tuple(field.name for field in fields(ConstraintDiagnosticsRow))
FINAL_SOLUTION_COLUMNS = tuple(field.name for field in fields(FinalSolutionRow))
RUN_SUMMARY_COLUMNS = tuple(field.name for field in fields(RunSummaryRow))
CASE_METRIC_NAMES = exp05b.CASE_METRIC_NAMES


def sigmoid(theta: float | jnp.ndarray) -> jnp.ndarray:
    return exp05b.sigmoid(theta)


def logit(curtailment_factor: float, eps: float = 1e-12) -> float:
    return exp05b.logit(curtailment_factor, eps=eps)


def _optimizer_initial_c(c_init: float) -> float:
    if c_init >= C_MAX:
        return C_MAX - BOUNDARY_START_EPS
    if c_init <= C_MIN:
        return C_MIN + BOUNDARY_START_EPS
    return c_init


def curtailment_from_theta(theta: float | jnp.ndarray) -> jnp.ndarray:
    return exp05b.curtailment_from_theta(theta)


def selected_case_config() -> dict:
    """Return the fixed Exp. 5d operating point and objective settings."""

    return {
        "case_id": CASE_ID,
        "load_multiplier_mv_bus_2": LOAD_MULTIPLIER_MV_BUS_2,
        "g_poa_wm2": G_POA_WM2,
        "t_amb_c": T_AMB_C,
        "wind_ms": WIND_MS,
        "pv_size_factor": PV_SIZE_FACTOR,
        "kappa": KAPPA,
        "p_export_limit_mw": P_EXPORT_LIMIT_MW,
        "p_export_target_mw": P_EXPORT_TARGET_MW,
        "p_scale_mw": P_SCALE_MW,
        "objective_variant": OBJECTIVE_VARIANT,
        "upstream_model": UPSTREAM_MODEL_NAME,
    }


def build_selected_scenario() -> exp05a.ScenarioBase:
    return exp05b.build_selected_scenario()


def simple_objective_from_export_proxy(
    p_export_proxy_mw: float | jnp.ndarray,
    p_scale_mw: float = P_SCALE_MW,
) -> jnp.ndarray:
    """Simple Exp. 5d objective used in the differentiable path."""

    return ((p_export_proxy_mw - P_EXPORT_TARGET_MW) / p_scale_mw) ** 2


def objective_from_theta(theta: jnp.ndarray, scenario: exp05a.ScenarioBase) -> jnp.ndarray:
    """Simple quadratic 7-MW export target objective for implicit AD."""

    curtailment = curtailment_from_theta(theta)
    params, _ = exp05a._params_for_case(
        scenario,
        G_POA_WM2,
        T_AMB_C,
        WIND_MS,
        curtailment,
        PV_SIZE_FACTOR,
    )
    state = solve_power_flow_implicit(
        scenario.topology,
        params,
        scenario.state0,
        exp05a.NEWTON_OPTIONS,
    )
    voltage = state_to_voltage(scenario.topology, params, state)
    y_bus = build_ybus(scenario.topology, params)
    s_bus = calc_power_injection(y_bus, voltage)
    p_slack_mw = jnp.real(s_bus[scenario.topology.slack_bus]) * scenario.s_base_mva
    p_export_proxy_mw = -p_slack_mw
    return simple_objective_from_export_proxy(p_export_proxy_mw, P_SCALE_MW)


def hard_export_violation_mw(p_export_mw: float) -> float:
    """Reporting-only hard violation against the 7.0-MW export line."""

    return max(0.0, p_export_mw - P_EXPORT_LIMIT_MW)


def soft_export_violation_mw(_p_export_proxy_mw: float) -> float:
    """No soft penalty is part of Exp. 5d; keep the column as NaN."""

    return float("nan")


def _make_objective_grad(scenario: exp05a.ScenarioBase):
    return jax.jit(jax.value_and_grad(lambda theta: objective_from_theta(theta, scenario)))


def _make_case_evaluator(scenario: exp05a.ScenarioBase):
    @jax.jit
    def evaluate(curtailment_factor: jnp.ndarray) -> tuple:
        params, injection = exp05a._params_for_case(
            scenario,
            G_POA_WM2,
            T_AMB_C,
            WIND_MS,
            curtailment_factor,
            PV_SIZE_FACTOR,
        )
        result = solve_power_flow_result(
            scenario.topology,
            params,
            scenario.state0,
            exp05a.NEWTON_OPTIONS,
        )
        voltage = state_to_voltage(scenario.topology, params, result.solution)
        y_bus = build_ybus(scenario.topology, params)
        s_bus = calc_power_injection(y_bus, voltage)
        s_slack = s_bus[scenario.topology.slack_bus] * scenario.s_base_mva
        p_slack = jnp.real(s_slack)
        q_slack = jnp.imag(s_slack)
        vm = jnp.abs(voltage)
        s_trafo = exp05a._trafo_hv_complex_power_mva(scenario, params, result.solution)
        return (
            curtailment_factor,
            p_slack,
            q_slack,
            -p_slack,
            jnp.maximum(0.0, -p_slack),
            injection.p_pv_mw,
            injection.q_pv_mvar,
            vm[scenario.pv_bus_internal_idx],
            jnp.sum(jnp.real(s_bus)) * scenario.s_base_mva,
            jnp.sum(jnp.imag(s_bus)) * scenario.s_base_mva,
            jnp.abs(s_trafo),
            result.converged,
            result.iterations,
            result.residual_norm,
        )

    return evaluate


def _case_metrics_to_dict(values: tuple) -> dict:
    return exp05b._case_metrics_to_dict(values)


def _solve_at_curtailment(
    scenario: exp05a.ScenarioBase,
    curtailment_factor: float,
    case_type: str = "optimization_eval",
    evaluator=None,
) -> dict:
    evaluator = evaluator or _make_case_evaluator(scenario)
    metrics = _case_metrics_to_dict(
        evaluator(jnp.asarray(curtailment_factor, dtype=jnp.float64))
    )
    metrics.update(
        {
            "case_id": CASE_ID,
            "case_type": case_type,
            "load_multiplier_mv_bus_2": LOAD_MULTIPLIER_MV_BUS_2,
            "g_poa_wm2": G_POA_WM2,
            "t_amb_c": T_AMB_C,
            "wind_ms": WIND_MS,
            "pv_size_factor": PV_SIZE_FACTOR,
            "kappa": KAPPA,
            "upstream_model": UPSTREAM_MODEL_NAME,
        }
    )
    return metrics


def _objective_components(
    theta: float,
    scenario: exp05a.ScenarioBase,
    objective_grad=None,
    evaluator=None,
    display_c: float | None = None,
) -> dict:
    objective_grad = objective_grad or _make_objective_grad(scenario)
    evaluator = evaluator or _make_case_evaluator(scenario)
    theta_jnp = jnp.asarray(theta, dtype=jnp.float64)
    objective, grad = objective_grad(theta_jnp)
    curtailment = float(display_c) if display_c is not None else float(curtailment_from_theta(theta_jnp))
    row = _solve_at_curtailment(scenario, curtailment, evaluator=evaluator)
    p_export_proxy = -row["p_slack_mw"]
    hard_violation = hard_export_violation_mw(row["p_export_mw"])
    if display_c is not None:
        objective = simple_objective_from_export_proxy(p_export_proxy)
    return {
        "theta": theta,
        "curtailment_factor": curtailment,
        "objective": float(objective),
        "grad_theta": float(grad),
        "p_export_proxy_mw": float(p_export_proxy),
        "export_proxy_mw": float(p_export_proxy),
        "p_export_mw": row["p_export_mw"],
        "hard_export_violation_mw": hard_violation,
        "soft_export_violation_mw": soft_export_violation_mw(float(p_export_proxy)),
        "export_margin_mw": P_EXPORT_LIMIT_MW - row["p_export_mw"],
        "export_violation_mw": hard_violation,
        "p_slack_mw": row["p_slack_mw"],
        "p_pv_mw": row["p_pv_mw"],
        "q_pv_mvar": row["q_pv_mvar"],
        "vm_mv_bus_2_pu": row["vm_mv_bus_2_pu"],
        "total_p_loss_mw": row["total_p_loss_mw"],
        "s_trafo_hv_mva": row["s_trafo_hv_mva"],
        "converged": row["converged"],
        "iterations": row["iterations"],
        "residual_norm": row["residual_norm"],
    }


def build_baseline_rows(scenario: exp05a.ScenarioBase) -> list[SelectedCaseBaselineRow]:
    """Evaluate full-PV and zero-PV endpoints for the selected case."""

    rows: list[SelectedCaseBaselineRow] = []
    evaluator = _make_case_evaluator(scenario)
    for baseline_type, curtailment in (("full_pv", 1.0), ("zero_pv", 0.0)):
        row = _solve_at_curtailment(
            scenario,
            curtailment,
            case_type=baseline_type,
            evaluator=evaluator,
        )
        p_export_proxy = -row["p_slack_mw"]
        p_export = row["p_export_mw"]
        rows.append(
            SelectedCaseBaselineRow(
                baseline_type=baseline_type,
                experiment=EXPERIMENT_NAME,
                objective_variant=OBJECTIVE_VARIANT,
                case_id=CASE_ID,
                load_multiplier_mv_bus_2=LOAD_MULTIPLIER_MV_BUS_2,
                g_poa_wm2=G_POA_WM2,
                t_amb_c=T_AMB_C,
                wind_ms=WIND_MS,
                curtailment_factor=curtailment,
                pv_size_factor=PV_SIZE_FACTOR,
                kappa=KAPPA,
                p_export_limit_mw=P_EXPORT_LIMIT_MW,
                p_export_target_mw=P_EXPORT_TARGET_MW,
                p_scale_mw=P_SCALE_MW,
                p_export_proxy_mw=p_export_proxy,
                export_proxy_mw=p_export_proxy,
                hard_export_violation_mw=hard_export_violation_mw(p_export),
                soft_export_violation_mw=soft_export_violation_mw(p_export_proxy),
                export_margin_mw=P_EXPORT_LIMIT_MW - p_export,
                objective=float(simple_objective_from_export_proxy(p_export_proxy)),
                p_export_mw=p_export,
                p_slack_mw=row["p_slack_mw"],
                p_pv_mw=row["p_pv_mw"],
                q_pv_mvar=row["q_pv_mvar"],
                vm_mv_bus_2_pu=row["vm_mv_bus_2_pu"],
                total_p_loss_mw=row["total_p_loss_mw"],
                s_trafo_hv_mva=row["s_trafo_hv_mva"],
                converged=row["converged"],
                iterations=row["iterations"],
                residual_norm=row["residual_norm"],
            )
        )
    return rows


def build_constraint_diagnostics(
    baseline_rows: list[SelectedCaseBaselineRow],
) -> list[ConstraintDiagnosticsRow]:
    """Summarize endpoint feasibility for the selected 1D curtailment problem."""

    by_type = {row.baseline_type: row for row in baseline_rows}
    full = by_type["full_pv"]
    zero = by_type["zero_pv"]
    full_violates = full.p_export_mw > P_EXPORT_LIMIT_MW
    feasible_by_zero = zero.p_export_mw <= P_EXPORT_LIMIT_MW
    notes = (
        "Exp. 5d minimizes a target error at 7.0 MW and has no hard inequality penalty."
        if full_violates and feasible_by_zero
        else "Check selected case: the 7.0-MW line is not bracketed by c=1 and c=0."
    )
    return [
        ConstraintDiagnosticsRow(
            case_id=CASE_ID,
            experiment=EXPERIMENT_NAME,
            objective_variant=OBJECTIVE_VARIANT,
            p_export_limit_mw=P_EXPORT_LIMIT_MW,
            p_export_full_pv_mw=full.p_export_mw,
            p_export_zero_pv_mw=zero.p_export_mw,
            full_pv_violates_limit=full_violates,
            feasible_by_zero_pv=feasible_by_zero,
            constraint_satisfied=full_violates and feasible_by_zero,
            notes=notes,
        )
    ]


def run_optimizer(
    scenario: exp05a.ScenarioBase,
    max_iter: int = MAX_ITER,
    learning_rate: float = LEARNING_RATE,
    c_init: float = C_INIT,
) -> list[OptimizationTraceRow]:
    """Run the same local Adam loop as Exp. 5b over the simple objective."""

    theta = logit(_optimizer_initial_c(c_init))
    m = 0.0
    v = 0.0
    objective_grad = _make_objective_grad(scenario)
    evaluator = _make_case_evaluator(scenario)
    trace: list[OptimizationTraceRow] = []
    for iteration in range(max_iter + 1):
        display_c = c_init if iteration == 0 and c_init in (C_MIN, C_MAX) else None
        values = _objective_components(theta, scenario, objective_grad, evaluator, display_c)
        trace.append(
            OptimizationTraceRow(
                iteration=iteration,
                theta=values["theta"],
                curtailment_factor=values["curtailment_factor"],
                objective=values["objective"],
                grad_theta=values["grad_theta"],
                p_export_limit_mw=P_EXPORT_LIMIT_MW,
                p_export_target_mw=P_EXPORT_TARGET_MW,
                p_scale_mw=P_SCALE_MW,
                p_export_proxy_mw=values["p_export_proxy_mw"],
                export_proxy_mw=values["export_proxy_mw"],
                hard_export_violation_mw=values["hard_export_violation_mw"],
                soft_export_violation_mw=values["soft_export_violation_mw"],
                export_margin_mw=values["export_margin_mw"],
                p_export_mw=values["p_export_mw"],
                export_violation_mw=values["export_violation_mw"],
                p_slack_mw=values["p_slack_mw"],
                p_pv_mw=values["p_pv_mw"],
                q_pv_mvar=values["q_pv_mvar"],
                vm_mv_bus_2_pu=values["vm_mv_bus_2_pu"],
                total_p_loss_mw=values["total_p_loss_mw"],
                s_trafo_hv_mva=values["s_trafo_hv_mva"],
                converged=values["converged"],
                iterations=values["iterations"],
                residual_norm=values["residual_norm"],
            )
        )
        if iteration == max_iter:
            break
        grad = values["grad_theta"]
        m = ADAM_BETA1 * m + (1.0 - ADAM_BETA1) * grad
        v = ADAM_BETA2 * v + (1.0 - ADAM_BETA2) * grad * grad
        m_hat = m / (1.0 - ADAM_BETA1 ** (iteration + 1))
        v_hat = v / (1.0 - ADAM_BETA2 ** (iteration + 1))
        theta -= learning_rate * m_hat / (math.sqrt(v_hat) + ADAM_EPS)
    return trace


def build_grid_reference(
    scenario: exp05a.ScenarioBase,
    n_points: int = GRID_POINTS,
) -> tuple[list[GridReferenceRow], GridReferenceRow, GridReferenceRow | None]:
    """Evaluate a 1D grid and mark objective-best and feasible-best rows."""

    if n_points < 2:
        raise ValueError("n_points must be at least 2")
    raw_rows: list[dict] = []
    evaluator = _make_case_evaluator(scenario)
    for idx in range(n_points):
        curtailment = idx / (n_points - 1)
        row = _solve_at_curtailment(
            scenario,
            curtailment,
            case_type="grid_reference",
            evaluator=evaluator,
        )
        p_export = row["p_export_mw"]
        p_export_proxy = -row["p_slack_mw"]
        raw_rows.append(
            {
                "grid_index": idx,
                "curtailment_factor": curtailment,
                "p_export_limit_mw": P_EXPORT_LIMIT_MW,
                "p_export_target_mw": P_EXPORT_TARGET_MW,
                "p_scale_mw": P_SCALE_MW,
                "objective": float(simple_objective_from_export_proxy(p_export_proxy)),
                "p_export_mw": p_export,
                "p_export_proxy_mw": p_export_proxy,
                "export_proxy_mw": p_export_proxy,
                "hard_export_violation_mw": hard_export_violation_mw(p_export),
                "soft_export_violation_mw": soft_export_violation_mw(p_export_proxy),
                "export_margin_mw": P_EXPORT_LIMIT_MW - p_export,
                "feasible": p_export <= P_EXPORT_LIMIT_MW,
                "is_grid_best_objective": False,
                "is_grid_best_feasible": False,
                "is_grid_best": False,
                "p_slack_mw": row["p_slack_mw"],
                "p_pv_mw": row["p_pv_mw"],
                "q_pv_mvar": row["q_pv_mvar"],
                "vm_mv_bus_2_pu": row["vm_mv_bus_2_pu"],
                "total_p_loss_mw": row["total_p_loss_mw"],
                "s_trafo_hv_mva": row["s_trafo_hv_mva"],
                "converged": row["converged"],
                "iterations": row["iterations"],
                "residual_norm": row["residual_norm"],
            }
        )
    best_objective_raw = min(raw_rows, key=lambda row: row["objective"])
    feasible = [row for row in raw_rows if row["feasible"]]
    best_feasible_raw = (
        max(feasible, key=lambda row: row["curtailment_factor"]) if feasible else None
    )
    for row in raw_rows:
        row["is_grid_best_objective"] = row["grid_index"] == best_objective_raw["grid_index"]
        row["is_grid_best"] = row["is_grid_best_objective"]
        row["is_grid_best_feasible"] = (
            best_feasible_raw is not None
            and row["grid_index"] == best_feasible_raw["grid_index"]
        )
    rows = [GridReferenceRow(**row) for row in raw_rows]
    best_objective = rows[best_objective_raw["grid_index"]]
    best_feasible = rows[best_feasible_raw["grid_index"]] if best_feasible_raw else None
    return rows, best_objective, best_feasible


def build_final_solution(
    trace_rows: list[OptimizationTraceRow],
    grid_best_objective: GridReferenceRow,
    grid_best_feasible: GridReferenceRow | None,
) -> list[FinalSolutionRow]:
    """Create the compact final-solution artifact."""

    final = trace_rows[-1]
    feasible_c = (
        grid_best_feasible.curtailment_factor if grid_best_feasible is not None else float("nan")
    )
    feasible_export = (
        grid_best_feasible.p_export_mw if grid_best_feasible is not None else float("nan")
    )
    feasible_diff = (
        abs(final.curtailment_factor - feasible_c)
        if grid_best_feasible is not None
        else float("nan")
    )
    objective_diff = abs(
        final.curtailment_factor - grid_best_objective.curtailment_factor
    )
    return [
        FinalSolutionRow(
            experiment=EXPERIMENT_NAME,
            objective_variant=OBJECTIVE_VARIANT,
            selected_case_id=CASE_ID,
            case_id=CASE_ID,
            final_iteration=final.iteration,
            final_theta=final.theta,
            final_curtailment_factor=final.curtailment_factor,
            final_pv_utilization_pct=100.0 * final.curtailment_factor,
            final_curtailment_pct=100.0 * (1.0 - final.curtailment_factor),
            final_objective=final.objective,
            objective_final=final.objective,
            final_grad_theta=final.grad_theta,
            final_p_export_proxy_mw=final.p_export_proxy_mw,
            final_p_export_mw=final.p_export_mw,
            final_p_slack_mw=final.p_slack_mw,
            final_export_margin_mw=final.export_margin_mw,
            final_hard_export_violation_mw=final.hard_export_violation_mw,
            final_soft_export_violation_mw=final.soft_export_violation_mw,
            final_p_pv_mw=final.p_pv_mw,
            final_q_pv_mvar=final.q_pv_mvar,
            final_vm_mv_bus_2_pu=final.vm_mv_bus_2_pu,
            final_total_p_loss_mw=final.total_p_loss_mw,
            final_s_trafo_hv_mva=final.s_trafo_hv_mva,
            p_export_limit_mw=P_EXPORT_LIMIT_MW,
            p_export_target_mw=P_EXPORT_TARGET_MW,
            p_scale_mw=P_SCALE_MW,
            grid_best_objective_curtailment_factor=grid_best_objective.curtailment_factor,
            grid_best_objective_p_export_mw=grid_best_objective.p_export_mw,
            grid_best_feasible_curtailment_factor=feasible_c,
            grid_best_feasible_p_export_mw=feasible_export,
            grid_best_curtailment_factor=grid_best_objective.curtailment_factor,
            abs_c_difference_optimizer_vs_grid_objective=objective_diff,
            abs_c_difference_optimizer_vs_grid_feasible=feasible_diff,
            abs_c_difference_optimizer_vs_grid=objective_diff,
            constraint_satisfied=final.p_export_mw <= P_EXPORT_LIMIT_MW,
        )
    ]


def build_run_summary(
    baseline_rows: list[SelectedCaseBaselineRow],
    trace_rows: list[OptimizationTraceRow],
    final_rows: list[FinalSolutionRow],
    grid_best_objective: GridReferenceRow,
    grid_best_feasible: GridReferenceRow | None,
) -> list[RunSummaryRow]:
    by_type = {row.baseline_type: row for row in baseline_rows}
    final = final_rows[0]
    feasible_c = (
        grid_best_feasible.curtailment_factor if grid_best_feasible is not None else float("nan")
    )
    return [
        RunSummaryRow(
            "p_export_full_pv_mw",
            by_type["full_pv"].p_export_mw,
            "MW",
            "Selected case at c=1.",
        ),
        RunSummaryRow(
            "p_export_zero_pv_mw",
            by_type["zero_pv"].p_export_mw,
            "MW",
            "Selected case at c=0.",
        ),
        RunSummaryRow(
            "final_curtailment_factor",
            final.final_curtailment_factor,
            "dimensionless",
            "Optimizer result from sigmoid(theta).",
        ),
        RunSummaryRow(
            "final_p_export_mw",
            final.final_p_export_mw,
            "MW",
            "p_export_mw = max(0, -p_slack_mw), for reporting only.",
        ),
        RunSummaryRow(
            "final_p_export_proxy_mw",
            final.final_p_export_proxy_mw,
            "MW",
            "p_export_proxy_mw = -p_slack_mw, used in the objective.",
        ),
        RunSummaryRow(
            "final_objective",
            final.final_objective,
            "MW^2",
            "((p_export_proxy_mw - 7.0) / p_scale_mw)^2.",
        ),
        RunSummaryRow(
            "final_hard_export_violation_mw",
            final.final_hard_export_violation_mw,
            "MW",
            "Reporting-only max(0, p_export_mw - 7.0).",
        ),
        RunSummaryRow(
            "final_export_margin_mw",
            final.final_export_margin_mw,
            "MW",
            "Positive means p_export_mw is below 7.0 MW.",
        ),
        RunSummaryRow(
            "grid_best_objective_curtailment_factor",
            grid_best_objective.curtailment_factor,
            "dimensionless",
            "Grid row with the smallest simple quadratic objective.",
        ),
        RunSummaryRow(
            "grid_best_feasible_curtailment_factor",
            feasible_c,
            "dimensionless",
            "Largest grid c with p_export_mw <= 7.0 MW.",
        ),
        RunSummaryRow(
            "abs_c_difference_optimizer_vs_grid_objective",
            final.abs_c_difference_optimizer_vs_grid_objective,
            "dimensionless",
            "Distance between optimizer result and objective-best grid point.",
        ),
        RunSummaryRow(
            "abs_c_difference_optimizer_vs_grid_feasible",
            final.abs_c_difference_optimizer_vs_grid_feasible,
            "dimensionless",
            "Distance between optimizer result and largest feasible grid point.",
        ),
        RunSummaryRow(
            "n_optimization_trace_rows",
            float(len(trace_rows)),
            "count",
            "Includes iteration 0 and the final iteration.",
        ),
    ]


def validate_consistency(
    trace_rows: list[OptimizationTraceRow],
    final_rows: list[FinalSolutionRow],
    summary_rows: list[RunSummaryRow],
) -> None:
    if not trace_rows:
        raise ValueError("optimization_trace must contain at least one row")
    if len(final_rows) != 1:
        raise ValueError("final_solution must contain exactly one row")
    final = final_rows[0]
    last = trace_rows[-1]
    summary = {row.metric: row.value for row in summary_rows}
    checks = [
        (final.final_iteration, last.iteration, "final_iteration"),
        (final.final_curtailment_factor, last.curtailment_factor, "curtailment_factor"),
        (final.final_objective, last.objective, "objective"),
        (final.final_grad_theta, last.grad_theta, "grad_theta"),
        (final.final_p_export_mw, last.p_export_mw, "p_export_mw"),
        (final.final_p_export_proxy_mw, last.p_export_proxy_mw, "p_export_proxy_mw"),
        (summary["final_curtailment_factor"], final.final_curtailment_factor, "summary c"),
        (summary["final_p_export_mw"], final.final_p_export_mw, "summary export"),
    ]
    for left, right, name in checks:
        if not jnp.isclose(left, right, rtol=1e-10, atol=1e-10):
            raise ValueError(f"Inconsistent final run value for {name}: {left} != {right}")
    if not 0.0 <= final.final_curtailment_factor <= 1.0:
        raise ValueError("final_curtailment_factor must be in [0, 1]")


def run_experiment(
    max_iter: int = MAX_ITER,
    grid_points: int = GRID_POINTS,
) -> tuple[
    list[SelectedCaseBaselineRow],
    list[OptimizationTraceRow],
    list[FinalSolutionRow],
    list[GridReferenceRow],
    list[ConstraintDiagnosticsRow],
    list[RunSummaryRow],
]:
    """Run the full Exp. 5d workflow."""

    scenario = build_selected_scenario()
    baseline_rows = build_baseline_rows(scenario)
    diagnostics_rows = build_constraint_diagnostics(baseline_rows)
    trace_rows = run_optimizer(scenario, max_iter=max_iter)
    grid_rows, grid_best_objective, grid_best_feasible = build_grid_reference(
        scenario,
        n_points=grid_points,
    )
    final_rows = build_final_solution(trace_rows, grid_best_objective, grid_best_feasible)
    summary_rows = build_run_summary(
        baseline_rows,
        trace_rows,
        final_rows,
        grid_best_objective,
        grid_best_feasible,
    )
    validate_consistency(trace_rows, final_rows, summary_rows)
    return baseline_rows, trace_rows, final_rows, grid_rows, diagnostics_rows, summary_rows


def _to_native(obj):
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


def _read_exp05b_final_if_available() -> dict | None:
    path = EXP05B_RESULTS_DIR / "final_solution.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data[0] if data else None
    except Exception:
        return None


def write_metadata(
    results_dir: Path,
    final_rows: list[FinalSolutionRow],
    grid_rows: list[GridReferenceRow],
    diagnostics_rows: list[ConstraintDiagnosticsRow],
) -> None:
    final = final_rows[0]
    grid_best_objective = next(row for row in grid_rows if row.is_grid_best_objective)
    grid_best_feasible = next((row for row in grid_rows if row.is_grid_best_feasible), None)
    diagnostics = diagnostics_rows[0]
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_hash": _git_hash(),
        "python_version": sys.version,
        "experiment": EXPERIMENT_NAME,
        "objective_variant": OBJECTIVE_VARIANT,
        "purpose": (
            "Gradient-based optimization of one PV curtailment factor for the "
            "selected realistic high-PV case using only a simple quadratic "
            "7.0-MW export target objective."
        ),
        "network": "pandapower.networks.example_simple(), scope_matched",
        "coupling_bus": PV_COUPLING_BUS_NAME,
        "replaced_element": PV_COUPLING_SGEN_NAME,
        "selected_case": selected_case_config(),
        "upstream_model": {
            "model_type": UPSTREAM_MODEL_NAME,
            "p_curtailment": "P_pv_mw(c) = c * P_available_mw",
            "q_coupling": "Q_pv_mvar(c) = -0.25 * P_pv_mw(c)",
        },
        "optimization": {
            "variable": "theta",
            "bounded_parameter": "curtailment_factor = sigmoid(theta)",
            "c_min": C_MIN,
            "c_max": C_MAX,
            "c_init": C_INIT,
            "learning_rate": LEARNING_RATE,
            "max_iter": MAX_ITER,
            "optimizer": "Adam implemented locally, no Optax dependency",
            "p_export_proxy_mw": "-p_slack_mw",
            "p_export_mw": "max(0, -p_slack_mw), reporting only",
            "p_export_limit_mw": P_EXPORT_LIMIT_MW,
            "p_export_target_mw": P_EXPORT_TARGET_MW,
            "p_scale_mw": P_SCALE_MW,
            "objective": "((p_export_proxy_mw - 7.0) / p_scale_mw) ** 2",
            "contains_softplus_penalty": False,
            "contains_curtailment_regularization": False,
            "soft_export_violation_mw": (
                "Column retained as NaN for schema comparability; it is not "
                "used in the Exp. 5d objective."
            ),
        },
        "grid_reference": {
            "n_points": GRID_POINTS,
            "objective_selection": "minimum simple quadratic objective over c in [0, 1]",
            "feasible_selection": "largest curtailment_factor with p_export_mw <= 7.0 MW",
            "grid_best_objective_curtailment_factor": (
                grid_best_objective.curtailment_factor
            ),
            "grid_best_feasible_curtailment_factor": (
                grid_best_feasible.curtailment_factor if grid_best_feasible else None
            ),
        },
        "final_results": _to_native(asdict(final)),
        "constraint_diagnostics": _to_native(asdict(diagnostics)),
        "optional_exp05b_reference_final_solution": _read_exp05b_final_if_available(),
        "known_simplifications": [
            "PV plant is modelled as weather-dependent PQ injection, not as voltage-regulating PV bus.",
            "No Q limits, no PV-PQ switching, no controller logic.",
            "The 7.0 MW export value is a demonstrator-internal target, not a normative grid-code limit.",
            "No thermal equipment-limit assessment is claimed.",
            "Optimization covers only one selected operating point.",
            "The simple objective is a target search, not a hard inequality optimizer.",
        ],
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def write_readme(
    results_dir: Path,
    final_rows: list[FinalSolutionRow],
    diagnostics_rows: list[ConstraintDiagnosticsRow],
) -> None:
    final = final_rows[0]
    diagnostics = diagnostics_rows[0]
    soft_note = (
        "`soft_export_violation_mw` is retained as `NaN` for schema "
        "comparability only; it is not part of the Exp. 5d objective."
    )
    text = f"""# Experiment 5d - Simple Objective PV Curtailment

Experiment 5d is a deliberately simplified sibling of Experiment 5b. It uses
the same selected case, the same analytical PV weather model, the same
sigmoid-parameterized curtailment factor, and the same AC power-flow chain, but
it does not replace or modify Experiment 5b.

Selected case: `{CASE_ID}`

- `load_multiplier_mv_bus_2 = {LOAD_MULTIPLIER_MV_BUS_2:.2f}`
- `g_poa_wm2 = {G_POA_WM2:.1f}`
- `t_amb_c = {T_AMB_C:.1f}`
- `wind_ms = {WIND_MS:.1f}`
- `pv_size_factor = {PV_SIZE_FACTOR:.1f}`
- `kappa = {KAPPA:.2f}`

## Objective

The free scalar `theta` is mapped to the physical curtailment factor with
`c(theta) = sigmoid(theta)`, so `0 <= c <= 1` throughout the optimization.

The Exp. 5d objective is only:

```text
J(theta) = ((p_export_proxy(theta) - 7.0) / p_scale_mw) ** 2
```

with `p_scale_mw = {P_SCALE_MW:.1f}` and
`p_export_proxy = -p_slack_mw`. The reporting value
`p_export_mw = max(0, -p_slack_mw)` is not used to form the differentiable
objective.

There is no softplus export penalty, no `6.99 MW` target, no curtailment
regularization, and no `lambda_curtailment` term. Therefore Exp. 5d should be
interpreted as a pure target search for `7.0 MW`, not as a hard
one-sided inequality optimization. A final result can legitimately lie just
above or just below `7.0 MW`. {soft_note}

## Final Result

- Full-PV export: `{diagnostics.p_export_full_pv_mw:.6f} MW`
- Zero-PV export: `{diagnostics.p_export_zero_pv_mw:.6f} MW`
- Final curtailment factor: `{final.final_curtailment_factor:.6f}`
- Final PV utilization: `{final.final_pv_utilization_pct:.3f} %`
- Final PV curtailment: `{final.final_curtailment_pct:.3f} %`
- Final export proxy: `{final.final_p_export_proxy_mw:.6f} MW`
- Final reported export: `{final.final_p_export_mw:.6f} MW`
- Final export margin: `{final.final_export_margin_mw:.6f} MW`
- Final hard violation: `{final.final_hard_export_violation_mw:.6f} MW`
- Final objective: `{final.final_objective:.10g}`
- Grid best objective c: `{final.grid_best_objective_curtailment_factor:.6f}`
- Grid best feasible c: `{final.grid_best_feasible_curtailment_factor:.6f}`
- Optimizer vs. grid-best-objective distance:
  `{final.abs_c_difference_optimizer_vs_grid_objective:.6f}`

The objective-best grid reference minimizes the absolute target error around
`7.0 MW`. The feasible grid reference is different: it is the largest grid
point with `p_export_mw <= 7.0 MW`. These references can differ slightly because
one is a symmetric target search and the other is a one-sided boundary
reference.

## Artifacts

- `selected_case_baseline.csv/json`
- `optimization_trace.csv/json`
- `final_solution.csv/json`
- `grid_reference.csv/json`
- `constraint_diagnostics.csv/json`
- `run_summary.csv/json`
- `metadata.json`
- `README.md`

## Model Limits

PV remains a PQ injection. There is no PV-bus voltage regulation, no Q limits,
no PV-PQ switching, no controller logic, and no normative thermal equipment
assessment. The `7.0 MW` value is demonstrator-internal and not a normative
grid-code limit.
"""
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "README.md").write_text(text, encoding="utf-8")


def export_all(
    baseline_rows: list[SelectedCaseBaselineRow],
    trace_rows: list[OptimizationTraceRow],
    final_rows: list[FinalSolutionRow],
    grid_rows: list[GridReferenceRow],
    diagnostics_rows: list[ConstraintDiagnosticsRow],
    summary_rows: list[RunSummaryRow],
    results_dir: Path,
) -> None:
    validate_consistency(trace_rows, final_rows, summary_rows)
    _write_csv(results_dir / "selected_case_baseline.csv", baseline_rows, SELECTED_CASE_BASELINE_COLUMNS)
    _write_json(results_dir / "selected_case_baseline.json", baseline_rows)
    _write_csv(results_dir / "optimization_trace.csv", trace_rows, OPTIMIZATION_TRACE_COLUMNS)
    _write_json(results_dir / "optimization_trace.json", trace_rows)
    _write_csv(results_dir / "final_solution.csv", final_rows, FINAL_SOLUTION_COLUMNS)
    _write_json(results_dir / "final_solution.json", final_rows)
    _write_csv(results_dir / "grid_reference.csv", grid_rows, GRID_REFERENCE_COLUMNS)
    _write_json(results_dir / "grid_reference.json", grid_rows)
    _write_csv(
        results_dir / "constraint_diagnostics.csv",
        diagnostics_rows,
        CONSTRAINT_DIAGNOSTICS_COLUMNS,
    )
    _write_json(results_dir / "constraint_diagnostics.json", diagnostics_rows)
    _write_csv(results_dir / "run_summary.csv", summary_rows, RUN_SUMMARY_COLUMNS)
    _write_json(results_dir / "run_summary.json", summary_rows)
    write_metadata(results_dir, final_rows, grid_rows, diagnostics_rows)
    write_readme(results_dir, final_rows, diagnostics_rows)


def main() -> None:
    print("=" * 72)
    print("Experiment 5d: simple-objective PV curtailment optimization")
    print(f"Results directory: {RESULTS_DIR}")
    print("=" * 72)
    (
        baseline_rows,
        trace_rows,
        final_rows,
        grid_rows,
        diagnostics_rows,
        summary_rows,
    ) = run_experiment()
    export_all(
        baseline_rows,
        trace_rows,
        final_rows,
        grid_rows,
        diagnostics_rows,
        summary_rows,
        RESULTS_DIR,
    )
    final = final_rows[0]
    diagnostics = diagnostics_rows[0]
    print("\nConstraint diagnostics:")
    print(f"  full PV export: {diagnostics.p_export_full_pv_mw:.6f} MW")
    print(f"  zero PV export: {diagnostics.p_export_zero_pv_mw:.6f} MW")
    print(f"  feasible by zero PV: {diagnostics.feasible_by_zero_pv}")
    print("\nFinal solution:")
    print(f"  curtailment_factor: {final.final_curtailment_factor:.6f}")
    print(f"  PV utilization:     {final.final_pv_utilization_pct:.3f} %")
    print(f"  PV curtailment:     {final.final_curtailment_pct:.3f} %")
    print(f"  p_export_proxy:     {final.final_p_export_proxy_mw:.6f} MW")
    print(f"  p_export:           {final.final_p_export_mw:.6f} MW")
    print(f"  export margin:      {final.final_export_margin_mw:.6f} MW")
    print(
        "  grid best objective c: "
        f"{final.grid_best_objective_curtailment_factor:.6f}"
    )
    print(
        "  grid best feasible c:  "
        f"{final.grid_best_feasible_curtailment_factor:.6f}"
    )
    print("\nExported artifacts:")
    for name in REQUIRED_ARTIFACTS:
        print(f"  {name}")


if __name__ == "__main__":
    main()
