"""Experiment 5b - gradient-based PV curtailment for one selected case.

This experiment optimizes only the selected realistic high-PV operating point
from Experiment 5a. The AC power-flow core remains unchanged; the free
optimization variable is mapped through a sigmoid to keep the PV curtailment
factor in ``[0, 1]``.

Run:
    python experiments/exp05b_optimize_pv_curtailment.py
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


RESULTS_DIR = Path(__file__).resolve().parent / "results" / "exp05b_optimize_pv_curtailment"

CASE_ID = exp05a.SELECTED_REALISTIC_CASE_ID
LOAD_MULTIPLIER_MV_BUS_2 = exp05a.SELECTED_REALISTIC_LOAD_MULTIPLIER
G_POA_WM2 = exp05a.SELECTED_REALISTIC_G_POA_WM2
T_AMB_C = exp05a.SELECTED_REALISTIC_T_AMB_C
WIND_MS = exp05a.SELECTED_REALISTIC_WIND_MS
PV_SIZE_FACTOR = exp05a.PV_SIZE_FACTOR
KAPPA = exp05a.EXP5A_KAPPA
P_EXPORT_LIMIT_MW = 7.0
P_EXPORT_TARGET_MW = 6.99

C_MIN = 0.0
C_MAX = 1.0
C_INIT = 0.8
LEARNING_RATE = 0.05
MAX_ITER = 300
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8

BETA = 300.0
P_SCALE_MW = 1.0
LAMBDA_CURTAILMENT = 1e-4
GRID_POINTS = 1001

REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "selected_case_baseline.csv",
    "selected_case_baseline.json",
    "optimization_trace.csv",
    "optimization_trace.json",
    "final_solution.csv",
    "final_solution.json",
    "grid_reference.csv",
    "grid_reference.json",
    "constraint_diagnostics.csv",
    "constraint_diagnostics.json",
    "run_summary.csv",
    "run_summary.json",
    "metadata.json",
    "README.md",
)


@dataclass(frozen=True)
class SelectedCaseBaselineRow:
    baseline_type: str
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
    export_proxy_mw: float
    hard_export_violation_mw: float
    soft_export_violation_mw: float
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
    export_proxy_mw: float
    hard_export_violation_mw: float
    soft_export_violation_mw: float
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
    p_export_mw: float
    export_proxy_mw: float
    hard_export_violation_mw: float
    soft_export_violation_mw: float
    export_margin_mw: float
    feasible: bool
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
    p_export_limit_mw: float
    p_export_full_pv_mw: float
    p_export_zero_pv_mw: float
    full_pv_violates_limit: bool
    feasible_by_zero_pv: bool
    constraint_satisfied: bool
    notes: str


@dataclass(frozen=True)
class FinalSolutionRow:
    case_id: str
    final_theta: float
    final_curtailment_factor: float
    final_pv_utilization_pct: float
    final_curtailment_pct: float
    final_p_export_mw: float
    final_hard_export_violation_mw: float
    final_soft_export_violation_mw: float
    final_export_margin_mw: float
    final_p_slack_mw: float
    final_p_pv_mw: float
    final_q_pv_mvar: float
    final_vm_mv_bus_2_pu: float
    final_total_p_loss_mw: float
    final_s_trafo_hv_mva: float
    objective_initial: float
    objective_final: float
    objective_reduction_pct: float
    grid_best_curtailment_factor: float
    abs_c_difference_optimizer_vs_grid: float
    p_export_limit_mw: float
    p_export_target_mw: float
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

CASE_METRIC_NAMES: tuple[str, ...] = (
    "curtailment_factor",
    "p_slack_mw",
    "q_slack_mvar",
    "export_proxy_mw",
    "p_export_mw",
    "p_pv_mw",
    "q_pv_mvar",
    "vm_mv_bus_2_pu",
    "total_p_loss_mw",
    "total_q_loss_mvar",
    "s_trafo_hv_mva",
    "converged",
    "iterations",
    "residual_norm",
)


def sigmoid(theta: float | jnp.ndarray) -> jnp.ndarray:
    """Map a free scalar to ``(0, 1)``."""

    return jax.nn.sigmoid(theta)


def logit(curtailment_factor: float, eps: float = 1e-12) -> float:
    """Map a bounded curtailment factor to the free scalar theta."""

    clipped = min(max(curtailment_factor, eps), 1.0 - eps)
    return math.log(clipped / (1.0 - clipped))


def curtailment_from_theta(theta: float | jnp.ndarray) -> jnp.ndarray:
    """Bounded curtailment factor used by the optimizer."""

    return C_MIN + (C_MAX - C_MIN) * sigmoid(theta)


def selected_case_config() -> dict:
    """Return the fixed Exp. 5b operating point."""

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
    }


def build_selected_scenario() -> exp05a.ScenarioBase:
    """Build the scope-matched example_simple scenario for the selected load."""

    return exp05a.build_scenario_base(LOAD_MULTIPLIER_MV_BUS_2)


def objective_from_theta(theta: jnp.ndarray, scenario: exp05a.ScenarioBase) -> jnp.ndarray:
    """Smooth export-limit objective for implicit AD."""

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
    export_proxy_mw = -p_slack_mw
    target_error_mw = export_proxy_mw - P_EXPORT_TARGET_MW
    soft_violation_mw = jax.nn.softplus(
        BETA * (export_proxy_mw - P_EXPORT_LIMIT_MW)
    ) / BETA
    return (
        (target_error_mw / P_SCALE_MW) ** 2
        + (soft_violation_mw / P_SCALE_MW) ** 2
        + LAMBDA_CURTAILMENT * (1.0 - curtailment) ** 2
    )


def hard_export_violation_mw(export_proxy_mw: float) -> float:
    """Non-smooth reporting violation against the export limit."""

    return max(0.0, export_proxy_mw - P_EXPORT_LIMIT_MW)


def soft_export_violation_mw(export_proxy_mw: float) -> float:
    """Smooth softplus violation used in the differentiable objective."""

    return float(
        jax.nn.softplus(
            BETA * jnp.asarray(export_proxy_mw - P_EXPORT_LIMIT_MW, dtype=jnp.float64)
        )
        / BETA
    )


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
    out = dict(zip(CASE_METRIC_NAMES, values, strict=True))
    converted: dict = {}
    for key, value in out.items():
        if key == "converged":
            converted[key] = bool(value)
        elif key == "iterations":
            converted[key] = int(value)
        else:
            converted[key] = float(value)
    return converted


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
        }
    )
    return metrics


def _objective_components(
    theta: float,
    scenario: exp05a.ScenarioBase,
    objective_grad=None,
    evaluator=None,
) -> dict:
    objective_grad = objective_grad or _make_objective_grad(scenario)
    evaluator = evaluator or _make_case_evaluator(scenario)
    theta_jnp = jnp.asarray(theta, dtype=jnp.float64)
    objective, grad = objective_grad(theta_jnp)
    curtailment = float(curtailment_from_theta(theta_jnp))
    row = _solve_at_curtailment(scenario, curtailment, evaluator=evaluator)
    export_proxy = -row["p_slack_mw"]
    hard_violation = hard_export_violation_mw(float(export_proxy))
    soft_violation = soft_export_violation_mw(float(export_proxy))
    return {
        "theta": theta,
        "curtailment_factor": curtailment,
        "objective": float(objective),
        "grad_theta": float(grad),
        "export_proxy_mw": float(export_proxy),
        "p_export_mw": row["p_export_mw"],
        "hard_export_violation_mw": hard_violation,
        "soft_export_violation_mw": soft_violation,
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
        rows.append(
            SelectedCaseBaselineRow(
                baseline_type=baseline_type,
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
                export_proxy_mw=-row["p_slack_mw"],
                hard_export_violation_mw=hard_export_violation_mw(-row["p_slack_mw"]),
                soft_export_violation_mw=soft_export_violation_mw(-row["p_slack_mw"]),
                p_export_mw=row["p_export_mw"],
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
    """Summarize whether the selected one-dimensional limit problem is feasible."""

    by_type = {row.baseline_type: row for row in baseline_rows}
    full = by_type["full_pv"]
    zero = by_type["zero_pv"]
    full_violates = full.p_export_mw > P_EXPORT_LIMIT_MW
    feasible_by_zero = zero.p_export_mw <= P_EXPORT_LIMIT_MW
    notes = (
        "Expected feasible one-dimensional curtailment case."
        if full_violates and feasible_by_zero
        else "Check selected case: limit is not bracketed by c=1 and c=0."
    )
    return [
        ConstraintDiagnosticsRow(
            case_id=CASE_ID,
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
    """Run a small Adam optimizer over theta."""

    theta = logit(c_init)
    m = 0.0
    v = 0.0
    objective_grad = _make_objective_grad(scenario)
    evaluator = _make_case_evaluator(scenario)
    trace: list[OptimizationTraceRow] = []
    for iteration in range(max_iter + 1):
        values = _objective_components(theta, scenario, objective_grad, evaluator)
        trace.append(
            OptimizationTraceRow(
                iteration=iteration,
                theta=values["theta"],
                curtailment_factor=values["curtailment_factor"],
                objective=values["objective"],
                grad_theta=values["grad_theta"],
                p_export_limit_mw=P_EXPORT_LIMIT_MW,
                p_export_target_mw=P_EXPORT_TARGET_MW,
                export_proxy_mw=values["export_proxy_mw"],
                hard_export_violation_mw=values["hard_export_violation_mw"],
                soft_export_violation_mw=values["soft_export_violation_mw"],
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
) -> tuple[list[GridReferenceRow], GridReferenceRow]:
    """Evaluate a one-dimensional curtailment grid and select the largest feasible c."""

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
        export_proxy = -row["p_slack_mw"]
        raw_rows.append(
            {
                "grid_index": idx,
                "curtailment_factor": curtailment,
                "p_export_limit_mw": P_EXPORT_LIMIT_MW,
                "p_export_target_mw": P_EXPORT_TARGET_MW,
                "p_export_mw": p_export,
                "export_proxy_mw": export_proxy,
                "hard_export_violation_mw": hard_export_violation_mw(float(export_proxy)),
                "soft_export_violation_mw": soft_export_violation_mw(float(export_proxy)),
                "export_margin_mw": P_EXPORT_LIMIT_MW - p_export,
                "feasible": p_export <= P_EXPORT_LIMIT_MW,
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
    feasible = [row for row in raw_rows if row["feasible"]]
    best_raw = max(feasible, key=lambda row: row["curtailment_factor"]) if feasible else raw_rows[0]
    for row in raw_rows:
        row["is_grid_best"] = row["grid_index"] == best_raw["grid_index"]
    rows = [GridReferenceRow(**row) for row in raw_rows]
    best = rows[best_raw["grid_index"]]
    return rows, best


def build_final_solution(
    trace_rows: list[OptimizationTraceRow],
    grid_best: GridReferenceRow,
) -> list[FinalSolutionRow]:
    """Create the compact final-solution artifact."""

    initial = trace_rows[0]
    final = trace_rows[-1]
    if math.isfinite(initial.objective) and abs(initial.objective) > 0.0:
        objective_reduction = 100.0 * (initial.objective - final.objective) / abs(initial.objective)
    else:
        objective_reduction = float("nan")
    return [
        FinalSolutionRow(
            case_id=CASE_ID,
            final_theta=final.theta,
            final_curtailment_factor=final.curtailment_factor,
            final_pv_utilization_pct=100.0 * final.curtailment_factor,
            final_curtailment_pct=100.0 * (1.0 - final.curtailment_factor),
            final_p_export_mw=final.p_export_mw,
            final_hard_export_violation_mw=final.hard_export_violation_mw,
            final_soft_export_violation_mw=final.soft_export_violation_mw,
            final_export_margin_mw=P_EXPORT_LIMIT_MW - final.p_export_mw,
            final_p_slack_mw=final.p_slack_mw,
            final_p_pv_mw=final.p_pv_mw,
            final_q_pv_mvar=final.q_pv_mvar,
            final_vm_mv_bus_2_pu=final.vm_mv_bus_2_pu,
            final_total_p_loss_mw=final.total_p_loss_mw,
            final_s_trafo_hv_mva=final.s_trafo_hv_mva,
            objective_initial=initial.objective,
            objective_final=final.objective,
            objective_reduction_pct=objective_reduction,
            grid_best_curtailment_factor=grid_best.curtailment_factor,
            abs_c_difference_optimizer_vs_grid=abs(
                final.curtailment_factor - grid_best.curtailment_factor
            ),
            p_export_limit_mw=P_EXPORT_LIMIT_MW,
            p_export_target_mw=P_EXPORT_TARGET_MW,
            constraint_satisfied=final.p_export_mw <= P_EXPORT_LIMIT_MW,
        )
    ]


def build_run_summary(
    baseline_rows: list[SelectedCaseBaselineRow],
    trace_rows: list[OptimizationTraceRow],
    final_rows: list[FinalSolutionRow],
    grid_best: GridReferenceRow,
) -> list[RunSummaryRow]:
    by_type = {row.baseline_type: row for row in baseline_rows}
    final = final_rows[0]
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
            "p_export_mw = max(0, -p_slack_mw).",
        ),
        RunSummaryRow(
            "final_hard_export_violation_mw",
            final.final_hard_export_violation_mw,
            "MW",
            "max(0, export_proxy_mw - 7.0).",
        ),
        RunSummaryRow(
            "final_soft_export_violation_mw",
            final.final_soft_export_violation_mw,
            "MW",
            "softplus(beta * (export_proxy_mw - 7.0)) / beta.",
        ),
        RunSummaryRow(
            "final_export_margin_mw",
            final.final_export_margin_mw,
            "MW",
            "Positive means the 7.0 MW target is met.",
        ),
        RunSummaryRow(
            "grid_best_curtailment_factor",
            grid_best.curtailment_factor,
            "dimensionless",
            "Largest grid c with p_export_mw <= 7.0 MW.",
        ),
        RunSummaryRow(
            "abs_c_difference_optimizer_vs_grid",
            final.abs_c_difference_optimizer_vs_grid,
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
    """Run the full Exp. 5b workflow."""

    scenario = build_selected_scenario()
    baseline_rows = build_baseline_rows(scenario)
    diagnostics_rows = build_constraint_diagnostics(baseline_rows)
    trace_rows = run_optimizer(scenario, max_iter=max_iter)
    grid_rows, grid_best = build_grid_reference(scenario, n_points=grid_points)
    final_rows = build_final_solution(trace_rows, grid_best)
    summary_rows = build_run_summary(baseline_rows, trace_rows, final_rows, grid_best)
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


def write_metadata(results_dir: Path) -> None:
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_hash": _git_hash(),
        "python_version": sys.version,
        "experiment": "exp05b_optimize_pv_curtailment",
        "purpose": (
            "Gradient-based optimization of one PV curtailment factor for the "
            "selected realistic high-PV case from Experiment 5a."
        ),
        "network": "pandapower.networks.example_simple(), scope_matched",
        "coupling_bus": PV_COUPLING_BUS_NAME,
        "replaced_element": PV_COUPLING_SGEN_NAME,
        "selected_case": selected_case_config(),
        "optimization": {
            "variable": "theta",
            "bounded_parameter": "curtailment_factor = sigmoid(theta)",
            "c_min": C_MIN,
            "c_max": C_MAX,
            "c_init": C_INIT,
            "learning_rate": LEARNING_RATE,
            "max_iter": MAX_ITER,
            "optimizer": "Adam implemented locally, no Optax dependency",
            "export_proxy_mw": "-p_slack_mw",
            "p_export_mw": "max(0, -p_slack_mw)",
            "p_export_target_mw": P_EXPORT_TARGET_MW,
            "objective": (
                "((export_proxy_mw - target)/p_scale_mw)^2 + "
                "(softplus(beta*(export_proxy_mw - limit))/beta / p_scale_mw)^2 "
                "+ lambda_curtailment*(1-c)^2"
            ),
            "beta": BETA,
            "p_scale_mw": P_SCALE_MW,
            "lambda_curtailment": LAMBDA_CURTAILMENT,
            "tuning_note": (
                "The stale 150-iteration artifact used the older limit-penalty "
                "objective and stopped at an unnecessarily conservative c. The "
                "current target-tracking objective uses p_export_target_mw=6.99 "
                "MW and a sharper beta=300 soft limit to keep the optimizer "
                "close to the 1D grid reference while satisfying the 7.0 MW limit."
            ),
        },
        "grid_reference": {
            "n_points": GRID_POINTS,
            "selection": "largest curtailment_factor with p_export_mw <= 7.0 MW",
        },
        "known_simplifications": [
            "PV plant is modelled as weather-dependent PQ injection, not as voltage-regulating PV bus.",
            "No Q limits, no PV-PQ switching, no controller logic.",
            "The 7.0 MW export limit is a demonstrator-internal target, not a normative grid-code limit.",
            "No thermal equipment-limit assessment is claimed.",
            "Optimization covers only one selected operating point.",
        ],
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def write_readme(results_dir: Path) -> None:
    text = """# Experiment 5b - PV Curtailment Optimization

Experiment 5b performs the optimization that Experiment 5a intentionally does
not do. Experiment 5a screens the network and selects operating points; this
script optimizes only one selected realistic summer high-PV case:
`selected_realistic_load0p4_g1200_t30`.

The chosen case uses low local load (`load_multiplier = 0.4`), high plane-of-
array irradiance (`G = 1200 W/m2`), warm ambient temperature (`T_amb = 30 degC`),
and weak cooling wind (`wind = 2 m/s`). This keeps the high-export character of
the mathematical stress cases while avoiding the less realistic main narrative
of combining `G = 1200 W/m2` with `T_amb = -10 degC`.

## Optimization target

The target is a demonstrator-internal export limit of `7.0 MW`. The sign
convention is:

- negative `p_slack_mw` means export into the upstream network,
- `p_export_mw = max(0, -p_slack_mw)` is used for reporting,
- `export_proxy_mw = -p_slack_mw` is used in the differentiable objective.

The free scalar `theta` is mapped to the physical curtailment factor by
`c(theta) = sigmoid(theta)`, so `0 <= c <= 1` always holds. `c = 1` means full
available PV power and `c = 0` means full curtailment.

The objective is

`((export_proxy_mw - 6.99) / p_scale_mw)^2
 + (softplus(beta * (export_proxy_mw - 7.0)) / beta / p_scale_mw)^2
 + lambda_curtailment * (1 - c)^2`.

The first term tracks a target just below the hard limit. The second term
reports the smooth softplus barrier against the `7.0 MW` limit in the AD path.
The third term keeps curtailment minimal. The numerical settings use
`beta = 300`, `lambda_curtailment = 1e-4`, and 300 Adam iterations. This replaces
the older softer 150-iteration artifact, which converged to a feasible but
unnecessarily conservative curtailment factor. `hard_export_violation_mw` and
`soft_export_violation_mw` are exported so the difference between reporting and
the smooth penalty is visible. A one-dimensional grid reference is exported
alongside the optimizer trace; it is a plausibility check, not a scalable
optimizer.

## Artifacts

- `selected_case_baseline.csv/json`: full-PV and zero-PV endpoint solves.
- `optimization_trace.csv/json`: Adam iterations over `theta`.
- `final_solution.csv/json`: compact optimizer result and comparison to the
  grid reference.
- `grid_reference.csv/json`: one-dimensional curtailment sweep.
- `constraint_diagnostics.csv/json`: endpoint feasibility checks.
- `run_summary.csv/json`: compact key metrics.
- `metadata.json`: reproducibility metadata.

## Model limits

PV remains a PQ injection. There is no PV-bus voltage regulation, no Q limits,
no PV-PQ switching, no controller logic, and no normative thermal equipment
assessment. The 7.0 MW export limit is an internal demonstration target.
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
    write_metadata(results_dir)
    write_readme(results_dir)


def main() -> None:
    print("=" * 72)
    print("Experiment 5b: PV curtailment optimization")
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
    print(f"  p_export:           {final.final_p_export_mw:.6f} MW")
    print(f"  export margin:      {final.final_export_margin_mw:.6f} MW")
    print(f"  grid best c:        {final.grid_best_curtailment_factor:.6f}")
    print("\nExported artifacts:")
    for name in REQUIRED_ARTIFACTS:
        print(f"  {name}")


if __name__ == "__main__":
    main()
