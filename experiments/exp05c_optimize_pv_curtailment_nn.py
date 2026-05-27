"""Experiment 5c - PV curtailment optimization with the Exp. 4 NN surrogate.

This experiment solves the same one-dimensional curtailment problem as
Experiment 5b, but replaces the analytical PV/weather upstream block with the
trained P-only NN surrogate from Experiment 4. The AC power-flow core, selected
case, objective, Adam loop, sigmoid/logit parametrization, and grid reference
remain unchanged.

Run:
    python experiments/exp05c_optimize_pv_curtailment_nn.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass
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
from diffpf.models.pq_surrogate import (
    DEFAULT_WEATHER_NORMALIZATION,
    MLPParams,
    SurrogateTrainingConfig,
    WeatherInputNormalization,
    count_mlp_parameters,
    neural_pq_injection_from_weather,
)
from diffpf.models.pv import (
    PV_COUPLING_BUS_NAME,
    PV_COUPLING_SGEN_NAME,
    PVInjection,
    inject_pv_at_bus,
)
from diffpf.solver.implicit import solve_power_flow_implicit
from diffpf.solver.newton import solve_power_flow_result
from experiments import exp04_modular_upstream_nn_surrogate as exp04
from experiments import exp05a_network_screening as exp05a
from experiments import exp05b_optimize_pv_curtailment as exp05b


RESULTS_DIR = (
    Path(__file__).resolve().parent / "results" / "exp05c_optimize_pv_curtailment_nn"
)
EXP05B_RESULTS_DIR = (
    Path(__file__).resolve().parent / "results" / "exp05b_optimize_pv_curtailment"
)

EXPERIMENT_NAME = "exp05c_optimize_pv_curtailment_nn"
UPSTREAM_MODEL_NAME = "nn_p_only_fixed_kappa"
NN_PARAMETER_SOURCE_KIND = "reproduced_exp04_best_validation_checkpoint_in_process"

CASE_ID = exp05b.CASE_ID
LOAD_MULTIPLIER_MV_BUS_2 = exp05b.LOAD_MULTIPLIER_MV_BUS_2
G_POA_WM2 = exp05b.G_POA_WM2
T_AMB_C = exp05b.T_AMB_C
WIND_MS = exp05b.WIND_MS
PV_SIZE_FACTOR = exp05b.PV_SIZE_FACTOR
KAPPA = exp05b.KAPPA
P_EXPORT_LIMIT_MW = exp05b.P_EXPORT_LIMIT_MW
P_EXPORT_TARGET_MW = exp05b.P_EXPORT_TARGET_MW

C_MIN = exp05b.C_MIN
C_MAX = exp05b.C_MAX
C_INIT = exp05b.C_INIT
LEARNING_RATE = exp05b.LEARNING_RATE
MAX_ITER = exp05b.MAX_ITER
ADAM_BETA1 = exp05b.ADAM_BETA1
ADAM_BETA2 = exp05b.ADAM_BETA2
ADAM_EPS = exp05b.ADAM_EPS

BETA = exp05b.BETA
P_SCALE_MW = exp05b.P_SCALE_MW
LAMBDA_CURTAILMENT = exp05b.LAMBDA_CURTAILMENT
GRID_POINTS = exp05b.GRID_POINTS

REQUIRED_ARTIFACTS = exp05b.REQUIRED_ARTIFACTS

SelectedCaseBaselineRow = exp05b.SelectedCaseBaselineRow
OptimizationTraceRow = exp05b.OptimizationTraceRow
GridReferenceRow = exp05b.GridReferenceRow
ConstraintDiagnosticsRow = exp05b.ConstraintDiagnosticsRow
FinalSolutionRow = exp05b.FinalSolutionRow
RunSummaryRow = exp05b.RunSummaryRow

SELECTED_CASE_BASELINE_COLUMNS = exp05b.SELECTED_CASE_BASELINE_COLUMNS
OPTIMIZATION_TRACE_COLUMNS = exp05b.OPTIMIZATION_TRACE_COLUMNS
GRID_REFERENCE_COLUMNS = exp05b.GRID_REFERENCE_COLUMNS
CONSTRAINT_DIAGNOSTICS_COLUMNS = exp05b.CONSTRAINT_DIAGNOSTICS_COLUMNS
FINAL_SOLUTION_COLUMNS = exp05b.FINAL_SOLUTION_COLUMNS
RUN_SUMMARY_COLUMNS = exp05b.RUN_SUMMARY_COLUMNS
CASE_METRIC_NAMES = exp05b.CASE_METRIC_NAMES


@dataclass(frozen=True)
class NNSurrogateSource:
    """Document how the NN parameters used by Exp. 5c were obtained."""

    source_kind: str
    source_experiment: str
    persisted_parameter_artifact_used: bool
    note: str
    hidden_width: int
    hidden_layers: int
    activation: str
    parameter_count: int
    train_samples: int
    val_samples: int
    eval_samples: int
    best_phase: str
    best_cycle_id: int | None
    best_global_step: int
    best_val_mse: float
    best_val_mae_mw: float


def sigmoid(theta: float | jnp.ndarray) -> jnp.ndarray:
    return exp05b.sigmoid(theta)


def logit(curtailment_factor: float, eps: float = 1e-12) -> float:
    return exp05b.logit(curtailment_factor, eps=eps)


def curtailment_from_theta(theta: float | jnp.ndarray) -> jnp.ndarray:
    return exp05b.curtailment_from_theta(theta)


def selected_case_config() -> dict:
    """Return the fixed Exp. 5c operating point and NN upstream model."""

    config_dict = exp05b.selected_case_config()
    config_dict.update(
        {
            "upstream_model": UPSTREAM_MODEL_NAME,
            "nn_parameter_source": NN_PARAMETER_SOURCE_KIND,
        }
    )
    return config_dict


def build_selected_scenario() -> exp05a.ScenarioBase:
    return exp05b.build_selected_scenario()


def reproduce_exp04_best_nn_params(
    training_config: SurrogateTrainingConfig | None = None,
    norm: WeatherInputNormalization = DEFAULT_WEATHER_NORMALIZATION,
) -> tuple[MLPParams, WeatherInputNormalization, NNSurrogateSource]:
    """Reproduce the Exp. 4 best-validation NN checkpoint deterministically.

    Experiment 4 currently exports training diagnostics and error artifacts, but
    no standalone parameter checkpoint. The clean handoff used here is therefore
    an in-process deterministic rerun of the Exp. 4 training routine. That
    routine returns the best global validation parameters across Phase A and
    Phase B, not the final iterate unless it is also best.
    """

    config_obj = training_config or SurrogateTrainingConfig()
    params, _train_x, _val_x, _eval_x, _history, diagnostics = exp04.train_surrogate(
        config_obj,
        norm,
        return_diagnostics=True,
    )
    source = NNSurrogateSource(
        source_kind=NN_PARAMETER_SOURCE_KIND,
        source_experiment="exp04_modular_upstream_nn_surrogate",
        persisted_parameter_artifact_used=False,
        note=(
            "Experiment 4 does not currently persist a standalone parameter "
            "checkpoint. Exp. 5c reproduces the deterministic Exp. 4 training "
            "run in-process and uses the returned best global validation "
            "checkpoint. No random or untrained NN parameters are used."
        ),
        hidden_width=config_obj.hidden_width,
        hidden_layers=config_obj.hidden_layers,
        activation="tanh",
        parameter_count=count_mlp_parameters(params),
        train_samples=config_obj.train_samples,
        val_samples=config_obj.val_samples,
        eval_samples=config_obj.eval_samples,
        best_phase=diagnostics.best_phase,
        best_cycle_id=diagnostics.best_cycle_id,
        best_global_step=diagnostics.best_global_step,
        best_val_mse=diagnostics.best_val_mse,
        best_val_mae_mw=diagnostics.best_val_mae_mw,
    )
    return params, norm, source


def _nn_pv_injection(
    nn_params: MLPParams,
    norm: WeatherInputNormalization,
    g_poa_wm2,
    t_amb_c,
    wind_ms,
    curtailment_factor,
    pv_size_factor,
) -> PVInjection:
    available = neural_pq_injection_from_weather(
        nn_params,
        norm,
        g_poa_wm2,
        t_amb_c,
        wind_ms,
        kappa=KAPPA,
    )
    scale = jnp.asarray(curtailment_factor, dtype=jnp.float64) * jnp.asarray(
        pv_size_factor,
        dtype=jnp.float64,
    )
    p_pv_mw = scale * available.p_pv_mw
    q_pv_mvar = jnp.asarray(KAPPA, dtype=jnp.float64) * p_pv_mw
    return PVInjection(p_pv_mw=p_pv_mw, q_pv_mvar=q_pv_mvar)


def _params_for_nn_case(
    scenario: exp05a.ScenarioBase,
    nn_params: MLPParams,
    norm: WeatherInputNormalization,
    g_poa_wm2,
    t_amb_c,
    wind_ms,
    curtailment_factor,
    pv_size_factor,
):
    injection = _nn_pv_injection(
        nn_params,
        norm,
        g_poa_wm2,
        t_amb_c,
        wind_ms,
        curtailment_factor,
        pv_size_factor,
    )
    params = inject_pv_at_bus(
        scenario.params_base,
        scenario.pv_bus_internal_idx,
        injection,
        scenario.s_base_mva,
    )
    return params, injection


def objective_from_theta(
    theta: jnp.ndarray,
    scenario: exp05a.ScenarioBase,
    nn_params: MLPParams,
    norm: WeatherInputNormalization,
) -> jnp.ndarray:
    """Same smooth Exp. 5b objective, but driven by the NN PV block."""

    curtailment = curtailment_from_theta(theta)
    params, _ = _params_for_nn_case(
        scenario,
        nn_params,
        norm,
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
    return exp05b.hard_export_violation_mw(export_proxy_mw)


def soft_export_violation_mw(export_proxy_mw: float) -> float:
    return exp05b.soft_export_violation_mw(export_proxy_mw)


def _make_objective_grad(
    scenario: exp05a.ScenarioBase,
    nn_params: MLPParams,
    norm: WeatherInputNormalization,
):
    return jax.jit(
        jax.value_and_grad(lambda theta: objective_from_theta(theta, scenario, nn_params, norm))
    )


def _make_case_evaluator(
    scenario: exp05a.ScenarioBase,
    nn_params: MLPParams,
    norm: WeatherInputNormalization,
):
    @jax.jit
    def evaluate(curtailment_factor: jnp.ndarray) -> tuple:
        params, injection = _params_for_nn_case(
            scenario,
            nn_params,
            norm,
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
    nn_params: MLPParams,
    norm: WeatherInputNormalization,
    curtailment_factor: float,
    case_type: str = "optimization_eval",
    evaluator=None,
) -> dict:
    evaluator = evaluator or _make_case_evaluator(scenario, nn_params, norm)
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
    nn_params: MLPParams,
    norm: WeatherInputNormalization,
    objective_grad=None,
    evaluator=None,
) -> dict:
    objective_grad = objective_grad or _make_objective_grad(scenario, nn_params, norm)
    evaluator = evaluator or _make_case_evaluator(scenario, nn_params, norm)
    theta_jnp = jnp.asarray(theta, dtype=jnp.float64)
    objective, grad = objective_grad(theta_jnp)
    curtailment = float(curtailment_from_theta(theta_jnp))
    row = _solve_at_curtailment(
        scenario,
        nn_params,
        norm,
        curtailment,
        evaluator=evaluator,
    )
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


def build_baseline_rows(
    scenario: exp05a.ScenarioBase,
    nn_params: MLPParams,
    norm: WeatherInputNormalization,
) -> list[SelectedCaseBaselineRow]:
    rows: list[SelectedCaseBaselineRow] = []
    evaluator = _make_case_evaluator(scenario, nn_params, norm)
    for baseline_type, curtailment in (("full_pv", 1.0), ("zero_pv", 0.0)):
        row = _solve_at_curtailment(
            scenario,
            nn_params,
            norm,
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
    return exp05b.build_constraint_diagnostics(baseline_rows)


def run_optimizer(
    scenario: exp05a.ScenarioBase,
    nn_params: MLPParams,
    norm: WeatherInputNormalization,
    max_iter: int = MAX_ITER,
    learning_rate: float = LEARNING_RATE,
    c_init: float = C_INIT,
) -> list[OptimizationTraceRow]:
    theta = logit(c_init)
    m = 0.0
    v = 0.0
    objective_grad = _make_objective_grad(scenario, nn_params, norm)
    evaluator = _make_case_evaluator(scenario, nn_params, norm)
    trace: list[OptimizationTraceRow] = []
    for iteration in range(max_iter + 1):
        values = _objective_components(
            theta,
            scenario,
            nn_params,
            norm,
            objective_grad,
            evaluator,
        )
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
        theta -= learning_rate * m_hat / (jnp.sqrt(v_hat) + ADAM_EPS)
        theta = float(theta)
    return trace


def build_grid_reference(
    scenario: exp05a.ScenarioBase,
    nn_params: MLPParams,
    norm: WeatherInputNormalization,
    n_points: int = GRID_POINTS,
) -> tuple[list[GridReferenceRow], GridReferenceRow]:
    if n_points < 2:
        raise ValueError("n_points must be at least 2")
    raw_rows: list[dict] = []
    evaluator = _make_case_evaluator(scenario, nn_params, norm)
    for idx in range(n_points):
        curtailment = idx / (n_points - 1)
        row = _solve_at_curtailment(
            scenario,
            nn_params,
            norm,
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
    return exp05b.build_final_solution(trace_rows, grid_best)


def build_run_summary(
    baseline_rows: list[SelectedCaseBaselineRow],
    trace_rows: list[OptimizationTraceRow],
    final_rows: list[FinalSolutionRow],
    grid_best: GridReferenceRow,
) -> list[RunSummaryRow]:
    return exp05b.build_run_summary(baseline_rows, trace_rows, final_rows, grid_best)


def validate_consistency(
    trace_rows: list[OptimizationTraceRow],
    final_rows: list[FinalSolutionRow],
    summary_rows: list[RunSummaryRow],
    diagnostics_rows: list[ConstraintDiagnosticsRow],
) -> None:
    if not trace_rows:
        raise ValueError("optimization_trace must contain at least one row")
    if len(final_rows) != 1:
        raise ValueError("final_solution must contain exactly one row")
    final = final_rows[0]
    last = trace_rows[-1]
    summary = {row.metric: row.value for row in summary_rows}
    diagnostics = diagnostics_rows[0]
    checks = [
        (final.final_curtailment_factor, last.curtailment_factor, "curtailment_factor"),
        (final.final_p_export_mw, last.p_export_mw, "p_export_mw"),
        (final.final_hard_export_violation_mw, last.hard_export_violation_mw, "hard_violation"),
        (summary["final_curtailment_factor"], final.final_curtailment_factor, "summary c"),
        (summary["final_p_export_mw"], final.final_p_export_mw, "summary export"),
    ]
    for left, right, name in checks:
        if not jnp.isclose(left, right, rtol=1e-10, atol=1e-10):
            raise ValueError(f"Inconsistent final run value for {name}: {left} != {right}")
    if not 0.0 <= final.final_curtailment_factor <= 1.0:
        raise ValueError("final_curtailment_factor must be in [0, 1]")
    if diagnostics.feasible_by_zero_pv and final.final_hard_export_violation_mw > 1e-6:
        raise ValueError("Final solution violates the hard export limit although c=0 is feasible")


def run_experiment(
    max_iter: int = MAX_ITER,
    grid_points: int = GRID_POINTS,
    nn_params: MLPParams | None = None,
    norm: WeatherInputNormalization = DEFAULT_WEATHER_NORMALIZATION,
    nn_training_config: SurrogateTrainingConfig | None = None,
) -> tuple[
    list[SelectedCaseBaselineRow],
    list[OptimizationTraceRow],
    list[FinalSolutionRow],
    list[GridReferenceRow],
    list[ConstraintDiagnosticsRow],
    list[RunSummaryRow],
    NNSurrogateSource,
]:
    """Run the full Exp. 5c workflow."""

    if nn_params is None:
        nn_params, norm, nn_source = reproduce_exp04_best_nn_params(nn_training_config, norm)
    else:
        config_obj = nn_training_config or SurrogateTrainingConfig()
        nn_source = NNSurrogateSource(
            source_kind="provided_trained_params",
            source_experiment="external_test_or_caller",
            persisted_parameter_artifact_used=False,
            note="Trained NN parameters were supplied by the caller.",
            hidden_width=config_obj.hidden_width,
            hidden_layers=config_obj.hidden_layers,
            activation="tanh",
            parameter_count=count_mlp_parameters(nn_params),
            train_samples=config_obj.train_samples,
            val_samples=config_obj.val_samples,
            eval_samples=config_obj.eval_samples,
            best_phase="provided",
            best_cycle_id=None,
            best_global_step=-1,
            best_val_mse=float("nan"),
            best_val_mae_mw=float("nan"),
        )

    scenario = build_selected_scenario()
    baseline_rows = build_baseline_rows(scenario, nn_params, norm)
    diagnostics_rows = build_constraint_diagnostics(baseline_rows)
    trace_rows = run_optimizer(scenario, nn_params, norm, max_iter=max_iter)
    grid_rows, grid_best = build_grid_reference(
        scenario,
        nn_params,
        norm,
        n_points=grid_points,
    )
    final_rows = build_final_solution(trace_rows, grid_best)
    summary_rows = build_run_summary(baseline_rows, trace_rows, final_rows, grid_best)
    validate_consistency(trace_rows, final_rows, summary_rows, diagnostics_rows)
    return (
        baseline_rows,
        trace_rows,
        final_rows,
        grid_rows,
        diagnostics_rows,
        summary_rows,
        nn_source,
    )


def _to_native(obj):
    return exp05b._to_native(obj)


def _write_csv(path: Path, rows: list, columns: tuple[str, ...]) -> None:
    exp05b._write_csv(path, rows, columns)


def _write_json(path: Path, rows: list) -> None:
    exp05b._write_json(path, rows)


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
    nn_source: NNSurrogateSource,
) -> None:
    final = final_rows[0]
    grid_best = next(row for row in grid_rows if row.is_grid_best)
    diagnostics = diagnostics_rows[0]
    exp05b_final = _read_exp05b_final_if_available()
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_hash": _git_hash(),
        "python_version": sys.version,
        "experiment": EXPERIMENT_NAME,
        "purpose": (
            "Gradient-based optimization of one PV curtailment factor for the "
            "selected realistic high-PV case, using the Exp. 4 NN PV surrogate "
            "as the upstream PV model."
        ),
        "network": "pandapower.networks.example_simple(), scope_matched",
        "coupling_bus": PV_COUPLING_BUS_NAME,
        "replaced_element": PV_COUPLING_SGEN_NAME,
        "selected_case": selected_case_config(),
        "weather": {
            "g_poa_wm2": G_POA_WM2,
            "t_amb_c": T_AMB_C,
            "wind_ms": WIND_MS,
        },
        "upstream_model": {
            "model_type": UPSTREAM_MODEL_NAME,
            "p_curtailment": "P_pv_mw(c) = c * P_NN_mw",
            "q_coupling": "Q_pv_mvar(c) = -0.25 * P_pv_mw(c)",
            "input_normalization": {
                "g_norm": "(g_poa_wm2 - 600.0) / 600.0",
                "t_norm": "(t_amb_c - 17.5) / 27.5",
                "w_norm": "(wind_ms - 5.25) / 4.75",
            },
            "architecture": {
                "hidden_width": nn_source.hidden_width,
                "hidden_layers": nn_source.hidden_layers,
                "activation": nn_source.activation,
                "parameter_count": nn_source.parameter_count,
            },
            "parameter_source": asdict(nn_source),
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
            "export_proxy_mw": "-p_slack_mw",
            "p_export_mw": "max(0, -p_slack_mw)",
            "p_export_limit_mw": P_EXPORT_LIMIT_MW,
            "p_export_target_mw": P_EXPORT_TARGET_MW,
            "beta": BETA,
            "p_scale_mw": P_SCALE_MW,
            "lambda_curtailment": LAMBDA_CURTAILMENT,
            "objective": (
                "((export_proxy_mw - target)/p_scale_mw)^2 + "
                "(softplus(beta*(export_proxy_mw - limit))/beta / p_scale_mw)^2 "
                "+ lambda_curtailment*(1-c)^2"
            ),
        },
        "grid_reference": {
            "n_points": GRID_POINTS,
            "selection": "largest curtailment_factor with p_export_mw <= 7.0 MW",
            "best_curtailment_factor": grid_best.curtailment_factor,
        },
        "final_results": _to_native(asdict(final)),
        "constraint_diagnostics": _to_native(asdict(diagnostics)),
        "optional_exp05b_reference_final_solution": exp05b_final,
        "known_simplifications": [
            "NN surrogate is a synthetic distillation model, not a measured-data PV forecast.",
            "NN is P-only; Q remains deterministically coupled as Q = -0.25 * P.",
            "PV plant is modelled as weather-dependent PQ injection, not as voltage-regulating PV bus.",
            "No Q limits, no PV-PQ switching, no controller logic.",
            "The 7.0 MW export limit is a demonstrator-internal target, not a normative grid-code limit.",
            "Optimization covers only one selected operating point.",
        ],
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def write_readme(
    results_dir: Path,
    final_rows: list[FinalSolutionRow],
    diagnostics_rows: list[ConstraintDiagnosticsRow],
    nn_source: NNSurrogateSource,
) -> None:
    final = final_rows[0]
    diagnostics = diagnostics_rows[0]
    text = f"""# Experiment 5c - NN PV Curtailment Optimization

Experiment 5c solves the same one-dimensional curtailment problem as
Experiment 5b, but replaces the analytical PV/weather upstream block with the
trained Experiment 4 NN surrogate `{UPSTREAM_MODEL_NAME}`.

The selected case is `{CASE_ID}` with `G = {G_POA_WM2:.1f} W/m2`,
`T_amb = {T_AMB_C:.1f} degC`, `wind = {WIND_MS:.1f} m/s`, and
`load_multiplier = {LOAD_MULTIPLIER_MV_BUS_2:.2f}`. The demonstrator-internal
export limit remains `{P_EXPORT_LIMIT_MW:.2f} MW` and the target-tracking value
is `{P_EXPORT_TARGET_MW:.2f} MW`.

## NN upstream model

The NN is the Exp. 4 P-only JAX MLP with three weather inputs, two hidden
layers of width `{nn_source.hidden_width}`, `tanh` activations, and
`{nn_source.parameter_count}` parameters. It predicts active power only. The
curtailment coupling is:

```text
P_pv_mw(c) = c * P_NN_mw
Q_pv_mvar(c) = -0.25 * P_pv_mw(c)
```

Parameter source: `{nn_source.source_kind}`. Experiment 4 currently does not
persist a standalone parameter checkpoint, so Exp. 5c reproduces the
deterministic Exp. 4 training run in-process and uses the returned best global
validation checkpoint. No random or untrained NN parameters are used.

Best reproduced Exp. 4 checkpoint: phase `{nn_source.best_phase}`, cycle
`{nn_source.best_cycle_id}`, global step `{nn_source.best_global_step}`,
Val-MSE `{nn_source.best_val_mse:.8g}`, Val-MAE
`{nn_source.best_val_mae_mw:.8g} MW`.

## Final result

- Full-PV export: `{diagnostics.p_export_full_pv_mw:.6f} MW`
- Zero-PV export: `{diagnostics.p_export_zero_pv_mw:.6f} MW`
- Final curtailment factor: `{final.final_curtailment_factor:.6f}`
- Final PV utilization: `{final.final_pv_utilization_pct:.3f} %`
- Final PV curtailment: `{final.final_curtailment_pct:.3f} %`
- Final export: `{final.final_p_export_mw:.6f} MW`
- Final export margin: `{final.final_export_margin_mw:.6f} MW`
- Final hard violation: `{final.final_hard_export_violation_mw:.6f} MW`
- Final soft violation: `{final.final_soft_export_violation_mw:.6f} MW`
- Grid best curtailment factor: `{final.grid_best_curtailment_factor:.6f}`
- Optimizer-grid distance: `{final.abs_c_difference_optimizer_vs_grid:.6f}`

Differences to Experiment 5b are expected because the NN approximates the
analytical PV/weather model. The grid reference is recomputed for the NN case;
it is not copied from Experiment 5b.

## Artifacts

- `selected_case_baseline.csv/json`
- `optimization_trace.csv/json`
- `final_solution.csv/json`
- `grid_reference.csv/json`
- `constraint_diagnostics.csv/json`
- `run_summary.csv/json`
- `metadata.json`
- `README.md`

## Model limits

The NN is a synthetic distillation surrogate and not a measured-data forecast
model. PV remains a PQ injection. There is no PV-bus voltage regulation, no Q
limits, no PV-PQ switching, no controller logic, and no normative thermal
equipment-limit assessment. The 7.0 MW export limit is an internal
demonstration target.
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
    nn_source: NNSurrogateSource,
    results_dir: Path,
) -> None:
    validate_consistency(trace_rows, final_rows, summary_rows, diagnostics_rows)
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
    write_metadata(results_dir, final_rows, grid_rows, diagnostics_rows, nn_source)
    write_readme(results_dir, final_rows, diagnostics_rows, nn_source)


def main() -> None:
    print("=" * 72)
    print("Experiment 5c: NN PV curtailment optimization")
    print(f"Results directory: {RESULTS_DIR}")
    print("=" * 72)
    (
        baseline_rows,
        trace_rows,
        final_rows,
        grid_rows,
        diagnostics_rows,
        summary_rows,
        nn_source,
    ) = run_experiment()
    export_all(
        baseline_rows,
        trace_rows,
        final_rows,
        grid_rows,
        diagnostics_rows,
        summary_rows,
        nn_source,
        RESULTS_DIR,
    )
    final = final_rows[0]
    diagnostics = diagnostics_rows[0]
    print("\nNN parameter source:")
    print(f"  {nn_source.source_kind}")
    print(f"  best phase:         {nn_source.best_phase}")
    print(f"  best global step:   {nn_source.best_global_step}")
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
