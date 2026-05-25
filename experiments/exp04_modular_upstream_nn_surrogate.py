"""Experiment 4 - modular upstream coupling with a small neural PQ surrogate.

This experiment distills the analytical PV weather model into a tiny JAX MLP
and couples three upstream models to the unchanged AC power-flow core:

``analytic_pv_weather``
    Existing analytical PV/weather model from Experiment 3.
``nn_p_only_fixed_kappa``
    New P-only neural surrogate with ``Q = -0.25 * P``.
``direct_pq_scale_baseline``
    Minimal differentiable direct P/Q baseline.

Run:
    python experiments/exp04_modular_upstream_nn_surrogate.py
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
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffpf.core.residuals import calc_power_injection, state_to_voltage
from diffpf.core.types import NetworkParams, PFState
from diffpf.core.ybus import build_ybus
from diffpf.models.pq_surrogate import (
    DEFAULT_EVAL_SAMPLES,
    DEFAULT_HIDDEN_WIDTH,
    DEFAULT_LEARNING_RATE_END,
    DEFAULT_LEARNING_RATE_START,
    DEFAULT_LR_SCHEDULE,
    DEFAULT_MAX_TRAIN_STEPS,
    DEFAULT_TRAIN_SAMPLES,
    DEFAULT_VAL_SAMPLES,
    DEFAULT_WARM_RESTART_SCHEDULE,
    DEFAULT_WEATHER_NORMALIZATION,
    MLPParams,
    SurrogateTrainingConfig,
    WeatherInputNormalization,
    count_mlp_parameters,
    init_mlp_params,
    mlp_apply,
    neural_pq_injection_from_weather,
    normalize_weather_inputs,
)
from diffpf.models.pv import (
    PV_BASE_P_MW,
    PV_COUPLING_BUS_NAME,
    PV_COUPLING_SGEN_NAME,
    PV_Q_OVER_P,
    PVInjection,
    inject_pv_at_bus,
    pv_pq_injection_from_weather,
)
from diffpf.solver.implicit import solve_power_flow_implicit
from diffpf.solver.newton import NewtonOptions, solve_power_flow_result
from experiments.exp03_cross_domain_pv_weather import (
    ScenarioBase,
    _robust_rel_error,
    build_scenario_base,
)


RESULTS_DIR = (
    Path(__file__).resolve().parent / "results" / "exp04_modular_upstream_nn_surrogate"
)

NEWTON_OPTIONS = NewtonOptions(max_iters=50, tolerance=1e-10, damping=0.7)
EXP4_KAPPA = PV_Q_OVER_P

UPSTREAM_MODELS = (
    "analytic_pv_weather",
    "nn_p_only_fixed_kappa",
    "direct_pq_scale_baseline",
)
ELECTRICAL_SCENARIOS: tuple[tuple[str, float], ...] = (
    ("base", 1.0),
    ("load_low", 0.75),
    ("load_high", 1.25),
)
WEATHER_INPUT_SPECS: tuple[tuple[str, str], ...] = (
    ("g_poa_wm2", "W/m^2"),
    ("t_amb_c", "degC"),
    ("wind_ms", "m/s"),
)
OBSERVABLE_SPECS: tuple[tuple[str, str], ...] = (
    ("vm_mv_bus_2_pu", "p.u."),
    ("va_mv_bus_2_deg", "deg"),
    ("p_slack_mw", "MW"),
    ("q_slack_mvar", "MVAr"),
    ("total_p_loss_mw", "MW"),
    ("p_trafo_hv_mw", "MW"),
)
PATTERN_OBSERVABLES = (
    "vm_mv_bus_2_pu",
    "p_slack_mw",
    "total_p_loss_mw",
    "p_trafo_hv_mw",
)
WEATHER_CASES: tuple[dict, ...] = (
    {"weather_case_id": "low_sun_cool", "g_poa_wm2": 200.0, "t_amb_c": 5.0, "wind_ms": 2.0},
    {"weather_case_id": "moderate_cool", "g_poa_wm2": 600.0, "t_amb_c": 15.0, "wind_ms": 2.0},
    {"weather_case_id": "exp3_reference", "g_poa_wm2": 800.0, "t_amb_c": 25.0, "wind_ms": 2.0},
    {"weather_case_id": "hot_reference", "g_poa_wm2": 800.0, "t_amb_c": 45.0, "wind_ms": 2.0},
    {"weather_case_id": "high_sun_windy", "g_poa_wm2": 1000.0, "t_amb_c": 25.0, "wind_ms": 6.0},
    {"weather_case_id": "edge_training_high", "g_poa_wm2": 1200.0, "t_amb_c": 35.0, "wind_ms": 1.0},
)
FD_STEPS = {"g_poa_wm2": 1.0, "t_amb_c": 0.05, "wind_ms": 0.01}
REL_ERROR_FLOOR = 1e-12
REFERENCE_VAL_MSE = 2.565834826e-04
REFERENCE_VAL_MAE_MW = 0.024622580
REFERENCE_EVAL_P_MAE_MW = 0.024815085
REFERENCE_EVAL_P_RMSE_MW = 0.032617826
REFERENCE_MAX_P_ERROR_MW = 0.185804774
WIDTH8_REFERENCE_HIDDEN_WIDTH = 8
WIDTH8_REFERENCE_PARAMETER_COUNT = 113
WIDTH8_REFERENCE_VAL_MSE = 2.3189919622e-04
WIDTH8_REFERENCE_VAL_MAE_MW = 0.023338907
WIDTH8_REFERENCE_EVAL_P_MAE_MW = 0.023523218
WIDTH8_REFERENCE_EVAL_P_RMSE_MW = 0.031040266
WIDTH8_REFERENCE_MAX_P_ERROR_MW = 0.177669209

TRAINING_DATASET_SUMMARY_COLUMNS: tuple[str, ...]
TRAINING_HISTORY_COLUMNS: tuple[str, ...]
SURROGATE_ERROR_COLUMNS: tuple[str, ...]
MODEL_COMPARISON_COLUMNS: tuple[str, ...]
COUPLING_SUMMARY_COLUMNS: tuple[str, ...]
GRADIENT_SUCCESS_COLUMNS: tuple[str, ...]
SENSITIVITY_PATTERN_COLUMNS: tuple[str, ...]
RUN_SUMMARY_COLUMNS: tuple[str, ...]


@dataclass(frozen=True)
class TrainingDatasetSummaryRow:
    split: str
    n_samples: int
    min_g_poa_wm2: float
    max_g_poa_wm2: float
    min_t_amb_c: float
    max_t_amb_c: float
    min_wind_ms: float
    max_wind_ms: float
    min_p_ref_mw: float
    max_p_ref_mw: float


@dataclass(frozen=True)
class TrainingHistoryRow:
    step: int
    global_step: int
    phase: str
    phase_step: int
    cycle_id: int | None
    hidden_width: int
    parameter_count: int
    train_mse: float
    val_mse: float
    train_mae_mw: float
    val_mae_mw: float
    learning_rate: float
    is_best_checkpoint: bool


@dataclass(frozen=True)
class SurrogateErrorRow:
    split: str
    case_id: str
    g_poa_wm2: float
    t_amb_c: float
    wind_ms: float
    p_ref_mw: float
    p_nn_mw: float
    q_ref_mvar: float
    q_nn_mvar: float
    p_abs_error_mw: float
    p_rel_error: float
    q_abs_error_mvar: float
    q_rel_error: float
    p_rel_error_floor: float
    q_rel_error_floor: float


@dataclass(frozen=True)
class ModelComparisonRow:
    model_name: str
    network_scenario: str
    weather_case_id: str
    g_poa_wm2: float
    t_amb_c: float
    wind_ms: float
    p_inj_mw: float
    q_inj_mvar: float
    observable: str
    value: float
    unit: str
    converged: bool
    iterations: int
    residual_norm: float


@dataclass(frozen=True)
class CouplingSummaryRow:
    model_name: str
    upstream_type: str
    input_names: str
    output_mode: str
    coupling_bus_name: str
    replaced_element_name: str
    uses_network_params_p_spec: bool
    uses_network_params_q_spec: bool
    uses_same_injection_adapter: bool
    uses_same_pf_core: bool
    requires_core_change: bool
    has_controller_logic: bool
    has_q_limits: bool
    has_pv_pq_switching: bool
    notes: str


@dataclass(frozen=True)
class GradientSuccessRow:
    model_name: str
    network_scenario: str
    weather_case_id: str
    input_parameter: str
    observable: str
    ad_gradient: float
    fd_gradient: float
    abs_error: float
    rel_error: float
    ad_fd_abs_error: float
    ad_fd_rel_error_floor: float
    is_finite_ad: bool
    is_finite_fd: bool
    passes_fd_check: bool
    fd_step: float
    unit: str


@dataclass(frozen=True)
class SensitivityPatternSummaryRow:
    comparison_pair: str
    network_scenario: str
    observable: str
    input_parameter: str
    n_cases: int
    mean_abs_grad_ref: float
    mean_abs_grad_other: float
    mean_abs_diff: float
    median_rel_diff: float
    max_rel_diff: float
    sign_agreement_rate: float
    cosine_similarity: float


@dataclass(frozen=True)
class PFObservableErrorRow:
    candidate_model: str
    reference_model: str
    network_scenario: str
    weather_case_id: str
    observable: str
    unit: str
    reference_value: float
    candidate_value: float
    signed_error: float
    abs_error: float
    rel_error_floor: float
    reference_abs_floor: float


@dataclass(frozen=True)
class PFObservableErrorSummaryRow:
    candidate_model: str
    observable: str
    n: int
    mean_abs_error: float
    median_abs_error: float
    max_abs_error: float
    rmse: float
    mean_rel_error_floor: float
    max_rel_error_floor: float


@dataclass(frozen=True)
class SensitivityErrorRow:
    candidate_model: str
    reference_model: str
    network_scenario: str
    weather_case_id: str
    observable: str
    input_parameter: str
    reference_gradient: float
    candidate_gradient: float
    signed_gradient_error: float
    abs_gradient_error: float
    rel_gradient_error_floor: float
    sign_match: bool
    abs_reference_gradient: float
    abs_candidate_gradient: float
    magnitude_ratio_abs: float


@dataclass(frozen=True)
class SensitivityErrorSummaryRow:
    candidate_model: str
    network_scenario: str
    observable: str
    input_parameter: str
    n: int
    mean_abs_gradient_error: float
    median_abs_gradient_error: float
    max_abs_gradient_error: float
    rmse_gradient_error: float
    mean_rel_gradient_error_floor: float
    median_rel_gradient_error_floor: float
    max_rel_gradient_error_floor: float
    sign_match_rate: float
    median_magnitude_ratio_abs: float
    mean_cosine_similarity: float


@dataclass(frozen=True)
class TrainingDiagnostics:
    best_phase: str
    best_cycle_id: int | None
    best_global_step: int
    best_step: int
    best_val_mse: float
    best_val_mae_mw: float
    final_phase: str
    final_global_step: int
    final_step: int
    final_train_mse: float
    final_val_mse: float
    final_train_mae_mw: float
    final_val_mae_mw: float


@dataclass(frozen=True)
class TrainingImprovementSummaryRow:
    reference_val_mse: float
    reference_val_mae_mw: float
    reference_eval_p_mae_mw: float
    reference_eval_p_rmse_mw: float
    reference_max_p_error_mw: float
    new_best_val_mse: float
    new_best_val_mae_mw: float
    new_eval_p_mae_mw: float
    new_eval_p_rmse_mw: float
    new_max_p_error_mw: float
    relative_improvement_val_mse: float
    relative_improvement_val_mae: float
    relative_improvement_eval_p_mae: float
    relative_improvement_eval_p_rmse: float
    relative_improvement_max_p_error: float
    improved_over_reference: bool
    best_phase: str
    best_cycle_id: int | None
    best_global_step: int
    best_step: int
    final_phase: str
    final_global_step: int
    final_step: int
    train_points: int
    val_points: int
    eval_points: int


@dataclass(frozen=True)
class ArchitectureComparisonSummaryRow:
    reference_hidden_width: int
    candidate_hidden_width: int
    reference_parameter_count: int
    candidate_parameter_count: int
    reference_val_mse: float
    reference_val_mae_mw: float
    reference_eval_p_mae_mw: float
    reference_eval_p_rmse_mw: float
    reference_max_p_error_mw: float
    candidate_best_val_mse: float
    candidate_best_val_mae_mw: float
    candidate_eval_p_mae_mw: float
    candidate_eval_p_rmse_mw: float
    candidate_max_p_error_mw: float
    relative_improvement_val_mse: float
    relative_improvement_val_mae: float
    relative_improvement_eval_p_mae: float
    relative_improvement_eval_p_rmse: float
    relative_improvement_max_p_error: float
    improved_over_width8: bool
    best_phase: str
    best_cycle_id: int | None
    best_global_step: int
    train_points: int
    val_points: int
    eval_points: int


@dataclass(frozen=True)
class RunSummaryRow:
    model_name: str
    train_points: int
    val_points: int
    eval_points: int
    best_phase: str
    best_cycle_id: int | None
    best_global_step: int
    best_step: int
    best_val_mse: float
    best_val_mae_mw: float
    final_phase: str
    final_global_step: int
    final_step: int
    final_val_mse: float
    final_val_mae_mw: float
    convergence_rate: float
    max_residual_norm: float
    max_iterations: int
    n_failed_solves: int
    n_total_solves: int


TRAINING_DATASET_SUMMARY_COLUMNS = tuple(f.name for f in fields(TrainingDatasetSummaryRow))
TRAINING_HISTORY_COLUMNS = tuple(f.name for f in fields(TrainingHistoryRow))
SURROGATE_ERROR_COLUMNS = tuple(f.name for f in fields(SurrogateErrorRow))
MODEL_COMPARISON_COLUMNS = tuple(f.name for f in fields(ModelComparisonRow))
COUPLING_SUMMARY_COLUMNS = tuple(f.name for f in fields(CouplingSummaryRow))
GRADIENT_SUCCESS_COLUMNS = tuple(f.name for f in fields(GradientSuccessRow))
SENSITIVITY_PATTERN_COLUMNS = tuple(f.name for f in fields(SensitivityPatternSummaryRow))
PF_OBSERVABLE_ERROR_COLUMNS = tuple(f.name for f in fields(PFObservableErrorRow))
PF_OBSERVABLE_ERROR_SUMMARY_COLUMNS = tuple(f.name for f in fields(PFObservableErrorSummaryRow))
SENSITIVITY_ERROR_COLUMNS = tuple(f.name for f in fields(SensitivityErrorRow))
SENSITIVITY_ERROR_SUMMARY_COLUMNS = tuple(f.name for f in fields(SensitivityErrorSummaryRow))
TRAINING_IMPROVEMENT_SUMMARY_COLUMNS = tuple(
    f.name for f in fields(TrainingImprovementSummaryRow)
)
ARCHITECTURE_COMPARISON_SUMMARY_COLUMNS = tuple(
    f.name for f in fields(ArchitectureComparisonSummaryRow)
)
RUN_SUMMARY_COLUMNS = tuple(f.name for f in fields(RunSummaryRow))

REQUIRED_ARTIFACTS = (
    "metadata.json",
    "README.md",
    "training_dataset_summary.csv",
    "training_dataset_summary.json",
    "training_history.csv",
    "training_history.json",
    "surrogate_error_table.csv",
    "surrogate_error_table.json",
    "model_comparison.csv",
    "model_comparison.json",
    "coupling_summary.csv",
    "coupling_summary.json",
    "gradient_success_table.csv",
    "gradient_success_table.json",
    "sensitivity_pattern_summary.csv",
    "sensitivity_pattern_summary.json",
    "pf_observable_error_table.csv",
    "pf_observable_error_table.json",
    "pf_observable_error_summary.csv",
    "pf_observable_error_summary.json",
    "sensitivity_error_table.csv",
    "sensitivity_error_table.json",
    "sensitivity_error_summary.csv",
    "sensitivity_error_summary.json",
    "training_improvement_summary.csv",
    "training_improvement_summary.json",
    "architecture_comparison_summary.csv",
    "architecture_comparison_summary.json",
    "run_summary.csv",
    "run_summary.json",
)


def _weather_array(rows: list[dict]) -> jnp.ndarray:
    return jnp.asarray(
        [[row["g_poa_wm2"], row["t_amb_c"], row["wind_ms"]] for row in rows],
        dtype=jnp.float64,
    )


def _target_p_mw(weather: jnp.ndarray) -> jnp.ndarray:
    injection = pv_pq_injection_from_weather(
        weather[:, 0],
        weather[:, 1],
        weather[:, 2],
        kappa=EXP4_KAPPA,
    )
    return injection.p_pv_mw


def make_weather_dataset(key: jax.Array, n_samples: int) -> jnp.ndarray:
    """Create deterministic random weather samples plus anchor points."""

    anchors = jnp.asarray(
        [
            [0.0, -10.0, 0.5],
            [200.0, 5.0, 2.0],
            [600.0, 15.0, 2.0],
            [800.0, 25.0, 2.0],
            [1000.0, 25.0, 6.0],
            [1200.0, 45.0, 10.0],
        ],
        dtype=jnp.float64,
    )
    if n_samples <= anchors.shape[0]:
        return anchors[:n_samples]

    random_n = n_samples - anchors.shape[0]
    keys = jax.random.split(key, 3)
    g = jax.random.uniform(keys[0], (random_n,), minval=0.0, maxval=1200.0)
    t = jax.random.uniform(keys[1], (random_n,), minval=-10.0, maxval=45.0)
    w = jax.random.uniform(keys[2], (random_n,), minval=0.5, maxval=10.0)
    random_weather = jnp.stack([g, t, w], axis=1).astype(jnp.float64)
    return jnp.concatenate([anchors, random_weather], axis=0)


def _predict_p_mw(
    params: MLPParams,
    norm: WeatherInputNormalization,
    weather: jnp.ndarray,
) -> jnp.ndarray:
    y = mlp_apply(params, normalize_weather_inputs(weather, norm))
    return PV_BASE_P_MW * jnp.squeeze(y, axis=-1)


def _loss(params: MLPParams, norm: WeatherInputNormalization, x, y_norm) -> jnp.ndarray:
    pred_norm = jnp.squeeze(mlp_apply(params, normalize_weather_inputs(x, norm)), axis=-1)
    return jnp.mean((pred_norm - y_norm) ** 2)


def _cosine_decay_lr(step: int, total_steps: int, start_lr: float, end_lr: float) -> float:
    if total_steps <= 0:
        progress = 1.0
    else:
        progress = min(max(float(step) / float(total_steps), 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(end_lr) + (float(start_lr) - float(end_lr)) * cosine


def _base_start_lr(config: SurrogateTrainingConfig) -> float:
    start_lr = float(config.initial_learning_rate_start)
    if (
        float(config.learning_rate) != DEFAULT_LEARNING_RATE_START
        and float(config.initial_learning_rate_start) == DEFAULT_LEARNING_RATE_START
    ):
        start_lr = float(config.learning_rate)
    return start_lr


def base_learning_rate_at_step(step: int, config: SurrogateTrainingConfig) -> float:
    """Return the Phase-A learning rate."""

    start_lr = _base_start_lr(config)
    end_lr = float(config.initial_learning_rate_end)
    if config.initial_schedule == "constant":
        return start_lr
    if config.initial_schedule != "cosine_decay":
        raise ValueError(
            f"Unsupported initial learning-rate schedule: {config.initial_schedule!r}"
        )
    return _cosine_decay_lr(step, config.base_train_steps, start_lr, end_lr)


def warm_restart_cycle_for_step(
    phase_step: int,
    config: SurrogateTrainingConfig,
) -> tuple[int, int, int]:
    """Return ``(cycle_id, local_step, cycle_steps)`` for a finetune step."""

    cycle_steps = tuple(int(item) for item in config.restart_cycle_steps)
    if not cycle_steps:
        raise ValueError("restart_cycle_steps must contain at least one cycle.")
    if len(cycle_steps) != len(config.restart_lr_max) or len(cycle_steps) != len(
        config.restart_lr_min
    ):
        raise ValueError(
            "restart_cycle_steps, restart_lr_max, and restart_lr_min must have "
            "the same length."
        )
    if any(item <= 0 for item in cycle_steps):
        raise ValueError("All restart cycle lengths must be positive.")

    clamped_step = min(max(int(phase_step), 0), sum(cycle_steps))
    start = 0
    for idx, steps in enumerate(cycle_steps):
        end = start + steps
        is_last = idx == len(cycle_steps) - 1
        if clamped_step < end or is_last:
            return idx + 1, clamped_step - start, steps
        start = end
    raise AssertionError("unreachable restart-cycle lookup state")


def warm_restart_learning_rate_at_step(
    phase_step: int,
    config: SurrogateTrainingConfig,
) -> float:
    """Return the Phase-B cosine-warm-restart learning rate."""

    cycle_id, local_step, cycle_steps = warm_restart_cycle_for_step(phase_step, config)
    lr_max = float(config.restart_lr_max[cycle_id - 1])
    lr_min = float(config.restart_lr_min[cycle_id - 1])
    return _cosine_decay_lr(local_step, cycle_steps, lr_max, lr_min)


def learning_rate_for_phase_step(
    phase: str,
    phase_step: int,
    config: SurrogateTrainingConfig,
) -> float:
    """Return the configured learning rate for a phase-local step."""

    if phase == "base":
        return base_learning_rate_at_step(phase_step, config)
    if phase == "warm_restart_finetune":
        if config.lr_schedule != DEFAULT_WARM_RESTART_SCHEDULE:
            raise ValueError(
                "Warm-restart finetune requires lr_schedule="
                f"{DEFAULT_WARM_RESTART_SCHEDULE!r}."
            )
        return warm_restart_learning_rate_at_step(phase_step, config)
    raise ValueError(f"Unsupported training phase: {phase!r}")


def learning_rate_at_step(step: int, config: SurrogateTrainingConfig) -> float:
    """Return the learning rate for legacy callers and schedule unit tests."""

    if config.lr_schedule == "constant":
        return _base_start_lr(config)
    if config.lr_schedule == "cosine_decay":
        return _cosine_decay_lr(
            step,
            config.max_train_steps,
            float(config.learning_rate_start),
            float(config.learning_rate_end),
        )
    if config.lr_schedule == DEFAULT_WARM_RESTART_SCHEDULE:
        return warm_restart_learning_rate_at_step(step, config)
    raise ValueError(f"Unsupported learning-rate schedule: {config.lr_schedule!r}")


def _mae_mw(
    params: MLPParams,
    norm: WeatherInputNormalization,
    x: jnp.ndarray,
    y_norm: jnp.ndarray,
) -> jnp.ndarray:
    return jnp.mean(jnp.abs(_predict_p_mw(params, norm, x) - y_norm * PV_BASE_P_MW))


def train_surrogate(
    config: SurrogateTrainingConfig = SurrogateTrainingConfig(),
    norm: WeatherInputNormalization = DEFAULT_WEATHER_NORMALIZATION,
    *,
    return_diagnostics: bool = False,
):
    """Distill the analytical PV weather model into the small MLP.

    The current full Experiment 4 run is two-phase: Phase A reproduces the
    non-cyclic cosine-decay baseline, and Phase B warm-restart-finetunes from
    the best Phase-A checkpoint. The returned parameters are the best global
    validation checkpoint across both phases.
    """

    key = jax.random.PRNGKey(config.seed)
    key_params, key_train, key_val, key_eval = jax.random.split(key, 4)
    train_x = make_weather_dataset(key_train, config.train_samples)
    val_x = make_weather_dataset(key_val, config.val_samples)
    eval_x = make_weather_dataset(key_eval, config.eval_samples)
    train_y = _target_p_mw(train_x) / PV_BASE_P_MW
    val_y = _target_p_mw(val_x) / PV_BASE_P_MW
    params = init_mlp_params(
        key_params,
        hidden_width=config.hidden_width,
        hidden_layers=config.hidden_layers,
    )
    value_and_grad = jax.jit(jax.value_and_grad(_loss))
    parameter_count = count_mlp_parameters(params)

    history: list[TrainingHistoryRow] = []
    best_params = params
    best_phase = "base"
    best_cycle_id: int | None = None
    best_global_step = 0
    best_step = 0
    best_val_mse = float("inf")
    best_val_mae = float("inf")

    def run_phase(
        start_params: MLPParams,
        *,
        phase: str,
        n_steps: int,
        global_offset: int,
    ) -> MLPParams:
        nonlocal best_params
        nonlocal best_phase
        nonlocal best_cycle_id
        nonlocal best_global_step
        nonlocal best_step
        nonlocal best_val_mse
        nonlocal best_val_mae

        params_phase = start_params
        for phase_step in range(n_steps + 1):
            global_step = global_offset + phase_step
            cycle_id = (
                warm_restart_cycle_for_step(phase_step, config)[0]
                if phase == "warm_restart_finetune"
                else None
            )
            train_mse, grads = value_and_grad(params_phase, norm, train_x, train_y)
            val_mse = _loss(params_phase, norm, val_x, val_y)
            val_mse_float = float(val_mse)
            if not math.isfinite(float(train_mse)) or not math.isfinite(val_mse_float):
                raise RuntimeError(
                    "Non-finite surrogate training loss encountered in "
                    f"{phase} at phase_step={phase_step}, "
                    f"global_step={global_step}: train_mse={float(train_mse)!r}, "
                    f"val_mse={val_mse_float!r}."
                )

            is_best = val_mse_float < best_val_mse
            if is_best:
                best_params = params_phase
                best_phase = phase
                best_cycle_id = cycle_id
                best_global_step = global_step
                best_step = global_step
                best_val_mse = val_mse_float
                best_val_mae = float(_mae_mw(params_phase, norm, val_x, val_y))

            learning_rate = learning_rate_for_phase_step(phase, phase_step, config)
            if phase_step % config.log_every == 0 or phase_step == n_steps:
                train_mae = _mae_mw(params_phase, norm, train_x, train_y)
                val_mae = _mae_mw(params_phase, norm, val_x, val_y)
                history.append(
                    TrainingHistoryRow(
                        step=global_step,
                        global_step=global_step,
                        phase=phase,
                        phase_step=phase_step,
                        cycle_id=cycle_id,
                        hidden_width=config.hidden_width,
                        parameter_count=parameter_count,
                        train_mse=float(train_mse),
                        val_mse=val_mse_float,
                        train_mae_mw=float(train_mae),
                        val_mae_mw=float(val_mae),
                        learning_rate=learning_rate,
                        is_best_checkpoint=is_best,
                    )
                )
            if phase_step == n_steps:
                break
            params_phase = jax.tree_util.tree_map(
                lambda p, g: p - learning_rate * g,
                params_phase,
                grads,
            )
        return params_phase

    base_steps = config.base_train_steps if config.warm_restart_enabled else config.max_train_steps
    phase_a_final_params = run_phase(
        params,
        phase="base",
        n_steps=base_steps,
        global_offset=0,
    )
    final_phase = "base"
    if config.warm_restart_enabled:
        _ = phase_a_final_params
        phase_b_final_params = run_phase(
            best_params,
            phase="warm_restart_finetune",
            n_steps=config.finetune_steps,
            global_offset=base_steps,
        )
        _ = phase_b_final_params
        final_phase = "warm_restart_finetune"

    final = history[-1]
    diagnostics = TrainingDiagnostics(
        best_phase=best_phase,
        best_cycle_id=best_cycle_id,
        best_global_step=best_global_step,
        best_step=best_step,
        best_val_mse=best_val_mse,
        best_val_mae_mw=best_val_mae,
        final_phase=final_phase,
        final_global_step=final.global_step,
        final_step=final.step,
        final_train_mse=final.train_mse,
        final_val_mse=final.val_mse,
        final_train_mae_mw=final.train_mae_mw,
        final_val_mae_mw=final.val_mae_mw,
    )
    if return_diagnostics:
        return best_params, train_x, val_x, eval_x, history, diagnostics
    return best_params, train_x, val_x, eval_x, history


def _direct_pq_baseline(g_poa_wm2, _t_amb_c, _wind_ms) -> PVInjection:
    p_mw = PV_BASE_P_MW * jnp.asarray(g_poa_wm2, dtype=jnp.float64) / 1000.0
    return PVInjection(p_pv_mw=p_mw, q_pv_mvar=EXP4_KAPPA * p_mw)


def upstream_injection(
    model_name: str,
    nn_params: MLPParams,
    norm: WeatherInputNormalization,
    g_poa_wm2,
    t_amb_c,
    wind_ms,
) -> PVInjection:
    """Evaluate one upstream model and return a shared PVInjection object."""

    if model_name == "analytic_pv_weather":
        return pv_pq_injection_from_weather(g_poa_wm2, t_amb_c, wind_ms, kappa=EXP4_KAPPA)
    if model_name == "nn_p_only_fixed_kappa":
        return neural_pq_injection_from_weather(
            nn_params,
            norm,
            g_poa_wm2,
            t_amb_c,
            wind_ms,
            kappa=EXP4_KAPPA,
        )
    if model_name == "direct_pq_scale_baseline":
        return _direct_pq_baseline(g_poa_wm2, t_amb_c, wind_ms)
    raise ValueError(f"Unknown upstream model: {model_name!r}")


def inject_upstream_model(
    scenario: ScenarioBase,
    model_name: str,
    nn_params: MLPParams,
    norm: WeatherInputNormalization,
    g_poa_wm2,
    t_amb_c,
    wind_ms,
) -> tuple[NetworkParams, PVInjection]:
    """Use the same bus-injection adapter for every upstream model."""

    injection = upstream_injection(model_name, nn_params, norm, g_poa_wm2, t_amb_c, wind_ms)
    params = inject_pv_at_bus(
        scenario.params_base,
        scenario.pv_bus_internal_idx,
        injection,
        scenario.s_base_mva,
    )
    return params, injection


def _trafo_hv_p_mw(scenario: ScenarioBase, params: NetworkParams, state: PFState) -> jnp.ndarray:
    voltage = state_to_voltage(scenario.topology, params, state)
    idx = scenario.trafo_idx
    hv = params.trafo_hv_bus[idx]
    lv = params.trafo_lv_bus[idx]
    y_t = params.trafo_g_series_pu[idx] + 1j * params.trafo_b_series_pu[idx]
    y_m = params.trafo_g_mag_pu[idx] + 1j * params.trafo_b_mag_pu[idx]
    a = params.trafo_tap_ratio[idx]
    phi = params.trafo_shift_rad[idx]
    tap = a * jnp.exp(1j * phi)
    current_hv = ((y_t + y_m) / (a * a)) * voltage[hv] + (-y_t / jnp.conj(tap)) * voltage[lv]
    return jnp.real(voltage[hv] * jnp.conj(current_hv)) * scenario.s_base_mva


def evaluate_solved_observable(
    scenario: ScenarioBase,
    params: NetworkParams,
    state: PFState,
    observable_name: str,
) -> jnp.ndarray:
    """Evaluate one scalar observable from a solved state."""

    voltage = state_to_voltage(scenario.topology, params, state)
    s_bus = calc_power_injection(build_ybus(scenario.topology, params), voltage)
    if observable_name == "vm_mv_bus_2_pu":
        return jnp.abs(voltage[scenario.pv_bus_internal_idx])
    if observable_name == "va_mv_bus_2_deg":
        return jnp.angle(voltage[scenario.pv_bus_internal_idx]) * 180.0 / jnp.pi
    if observable_name == "p_slack_mw":
        return jnp.real(s_bus[scenario.topology.slack_bus]) * scenario.s_base_mva
    if observable_name == "q_slack_mvar":
        return jnp.imag(s_bus[scenario.topology.slack_bus]) * scenario.s_base_mva
    if observable_name == "total_p_loss_mw":
        return jnp.sum(jnp.real(s_bus)) * scenario.s_base_mva
    if observable_name == "p_trafo_hv_mw":
        return _trafo_hv_p_mw(scenario, params, state)
    raise ValueError(f"Unknown observable: {observable_name!r}")


def make_observable_fn(
    scenario: ScenarioBase,
    model_name: str,
    nn_params: MLPParams,
    norm: WeatherInputNormalization,
    observable_name: str,
):
    """Return differentiable function ``f(g, t, w) -> observable``."""

    def fn(g, t, w):
        params, _ = inject_upstream_model(scenario, model_name, nn_params, norm, g, t, w)
        solution = solve_power_flow_implicit(
            scenario.topology,
            params,
            scenario.state0,
            NEWTON_OPTIONS,
        )
        return evaluate_solved_observable(scenario, params, solution, observable_name)

    return fn


def _solve_case(
    scenario: ScenarioBase,
    model_name: str,
    nn_params: MLPParams,
    norm: WeatherInputNormalization,
    weather: dict,
) -> tuple[bool, int, float, PVInjection, dict[str, float]]:
    params, injection = inject_upstream_model(
        scenario,
        model_name,
        nn_params,
        norm,
        jnp.asarray(weather["g_poa_wm2"], dtype=jnp.float64),
        jnp.asarray(weather["t_amb_c"], dtype=jnp.float64),
        jnp.asarray(weather["wind_ms"], dtype=jnp.float64),
    )
    try:
        result = solve_power_flow_result(scenario.topology, params, scenario.state0, NEWTON_OPTIONS)
        converged = bool(result.converged)
        iterations = int(result.iterations)
        residual_norm = float(result.residual_norm)
        observables = {
            name: float(evaluate_solved_observable(scenario, params, result.solution, name))
            for name, _ in OBSERVABLE_SPECS
        } if converged else {}
    except Exception:
        converged = False
        iterations = -1
        residual_norm = float("nan")
        observables = {}
    return converged, iterations, residual_norm, injection, observables


def build_surrogate_error_rows(
    params: MLPParams,
    norm: WeatherInputNormalization,
    train_x: jnp.ndarray,
    val_x: jnp.ndarray,
    eval_x: jnp.ndarray,
) -> list[SurrogateErrorRow]:
    """Evaluate NN-vs-analytical PV error on train/val/eval cases."""

    rows: list[SurrogateErrorRow] = []
    splits = [("train", train_x), ("val", val_x), ("eval", eval_x)]
    for split, data in splits:
        p_ref = _target_p_mw(data)
        p_nn = _predict_p_mw(params, norm, data)
        for idx in range(int(data.shape[0])):
            q_ref = EXP4_KAPPA * float(p_ref[idx])
            q_nn = EXP4_KAPPA * float(p_nn[idx])
            p_abs = abs(float(p_nn[idx]) - float(p_ref[idx]))
            q_abs = abs(q_nn - q_ref)
            rows.append(
                SurrogateErrorRow(
                    split=split,
                    case_id=f"{split}_{idx:04d}",
                    g_poa_wm2=float(data[idx, 0]),
                    t_amb_c=float(data[idx, 1]),
                    wind_ms=float(data[idx, 2]),
                    p_ref_mw=float(p_ref[idx]),
                    p_nn_mw=float(p_nn[idx]),
                    q_ref_mvar=q_ref,
                    q_nn_mvar=q_nn,
                    p_abs_error_mw=p_abs,
                    p_rel_error=p_abs / max(abs(float(p_ref[idx])), REL_ERROR_FLOOR),
                    q_abs_error_mvar=q_abs,
                    q_rel_error=q_abs / max(abs(q_ref), REL_ERROR_FLOOR),
                    p_rel_error_floor=p_abs / max(abs(float(p_ref[idx])), REL_ERROR_FLOOR),
                    q_rel_error_floor=q_abs / max(abs(q_ref), REL_ERROR_FLOOR),
                )
            )
    return rows


def _relative_improvement(previous: float, new: float) -> float:
    return (previous - new) / previous if previous != 0.0 else float("nan")


def build_training_improvement_summary(
    error_rows: list[SurrogateErrorRow],
    config: SurrogateTrainingConfig,
    training_diagnostics: TrainingDiagnostics,
) -> list[TrainingImprovementSummaryRow]:
    """Compare the new main run against the documented previous Exp. 4 run."""

    eval_abs_errors = np.asarray(
        [row.p_abs_error_mw for row in error_rows if row.split == "eval"],
        dtype=float,
    )
    if eval_abs_errors.size == 0:
        eval_mae = float("nan")
        eval_rmse = float("nan")
        max_error = float("nan")
    else:
        eval_mae = float(np.mean(eval_abs_errors))
        eval_rmse = float(np.sqrt(np.mean(eval_abs_errors**2)))
        max_error = float(np.max(eval_abs_errors))

    return [
        TrainingImprovementSummaryRow(
            reference_val_mse=REFERENCE_VAL_MSE,
            reference_val_mae_mw=REFERENCE_VAL_MAE_MW,
            reference_eval_p_mae_mw=REFERENCE_EVAL_P_MAE_MW,
            reference_eval_p_rmse_mw=REFERENCE_EVAL_P_RMSE_MW,
            reference_max_p_error_mw=REFERENCE_MAX_P_ERROR_MW,
            new_best_val_mse=training_diagnostics.best_val_mse,
            new_best_val_mae_mw=training_diagnostics.best_val_mae_mw,
            new_eval_p_mae_mw=eval_mae,
            new_eval_p_rmse_mw=eval_rmse,
            new_max_p_error_mw=max_error,
            relative_improvement_val_mse=_relative_improvement(
                REFERENCE_VAL_MSE,
                training_diagnostics.best_val_mse,
            ),
            relative_improvement_val_mae=_relative_improvement(
                REFERENCE_VAL_MAE_MW,
                training_diagnostics.best_val_mae_mw,
            ),
            relative_improvement_eval_p_mae=_relative_improvement(
                REFERENCE_EVAL_P_MAE_MW,
                eval_mae,
            ),
            relative_improvement_eval_p_rmse=_relative_improvement(
                REFERENCE_EVAL_P_RMSE_MW,
                eval_rmse,
            ),
            relative_improvement_max_p_error=_relative_improvement(
                REFERENCE_MAX_P_ERROR_MW,
                max_error,
            ),
            improved_over_reference=training_diagnostics.best_val_mse < REFERENCE_VAL_MSE,
            best_phase=training_diagnostics.best_phase,
            best_cycle_id=training_diagnostics.best_cycle_id,
            best_global_step=training_diagnostics.best_global_step,
            best_step=training_diagnostics.best_step,
            final_phase=training_diagnostics.final_phase,
            final_global_step=training_diagnostics.final_global_step,
            final_step=training_diagnostics.final_step,
            train_points=config.train_samples,
            val_points=config.val_samples,
            eval_points=config.eval_samples,
        )
    ]


def build_architecture_comparison_summary(
    error_rows: list[SurrogateErrorRow],
    config: SurrogateTrainingConfig,
    params: MLPParams,
    training_diagnostics: TrainingDiagnostics,
) -> list[ArchitectureComparisonSummaryRow]:
    """Compare the current capacity run against the best width-8 reference."""

    eval_abs_errors = np.asarray(
        [row.p_abs_error_mw for row in error_rows if row.split == "eval"],
        dtype=float,
    )
    if eval_abs_errors.size == 0:
        eval_mae = float("nan")
        eval_rmse = float("nan")
        max_error = float("nan")
    else:
        eval_mae = float(np.mean(eval_abs_errors))
        eval_rmse = float(np.sqrt(np.mean(eval_abs_errors**2)))
        max_error = float(np.max(eval_abs_errors))

    return [
        ArchitectureComparisonSummaryRow(
            reference_hidden_width=WIDTH8_REFERENCE_HIDDEN_WIDTH,
            candidate_hidden_width=config.hidden_width,
            reference_parameter_count=WIDTH8_REFERENCE_PARAMETER_COUNT,
            candidate_parameter_count=count_mlp_parameters(params),
            reference_val_mse=WIDTH8_REFERENCE_VAL_MSE,
            reference_val_mae_mw=WIDTH8_REFERENCE_VAL_MAE_MW,
            reference_eval_p_mae_mw=WIDTH8_REFERENCE_EVAL_P_MAE_MW,
            reference_eval_p_rmse_mw=WIDTH8_REFERENCE_EVAL_P_RMSE_MW,
            reference_max_p_error_mw=WIDTH8_REFERENCE_MAX_P_ERROR_MW,
            candidate_best_val_mse=training_diagnostics.best_val_mse,
            candidate_best_val_mae_mw=training_diagnostics.best_val_mae_mw,
            candidate_eval_p_mae_mw=eval_mae,
            candidate_eval_p_rmse_mw=eval_rmse,
            candidate_max_p_error_mw=max_error,
            relative_improvement_val_mse=_relative_improvement(
                WIDTH8_REFERENCE_VAL_MSE,
                training_diagnostics.best_val_mse,
            ),
            relative_improvement_val_mae=_relative_improvement(
                WIDTH8_REFERENCE_VAL_MAE_MW,
                training_diagnostics.best_val_mae_mw,
            ),
            relative_improvement_eval_p_mae=_relative_improvement(
                WIDTH8_REFERENCE_EVAL_P_MAE_MW,
                eval_mae,
            ),
            relative_improvement_eval_p_rmse=_relative_improvement(
                WIDTH8_REFERENCE_EVAL_P_RMSE_MW,
                eval_rmse,
            ),
            relative_improvement_max_p_error=_relative_improvement(
                WIDTH8_REFERENCE_MAX_P_ERROR_MW,
                max_error,
            ),
            improved_over_width8=training_diagnostics.best_val_mse
            < WIDTH8_REFERENCE_VAL_MSE,
            best_phase=training_diagnostics.best_phase,
            best_cycle_id=training_diagnostics.best_cycle_id,
            best_global_step=training_diagnostics.best_global_step,
            train_points=config.train_samples,
            val_points=config.val_samples,
            eval_points=config.eval_samples,
        )
    ]


def summarize_dataset(split: str, data: jnp.ndarray) -> TrainingDatasetSummaryRow:
    p_ref = _target_p_mw(data)
    return TrainingDatasetSummaryRow(
        split=split,
        n_samples=int(data.shape[0]),
        min_g_poa_wm2=float(jnp.min(data[:, 0])),
        max_g_poa_wm2=float(jnp.max(data[:, 0])),
        min_t_amb_c=float(jnp.min(data[:, 1])),
        max_t_amb_c=float(jnp.max(data[:, 1])),
        min_wind_ms=float(jnp.min(data[:, 2])),
        max_wind_ms=float(jnp.max(data[:, 2])),
        min_p_ref_mw=float(jnp.min(p_ref)),
        max_p_ref_mw=float(jnp.max(p_ref)),
    )


def build_coupling_summary() -> list[CouplingSummaryRow]:
    rows = []
    notes = {
        "analytic_pv_weather": "Reference analytical NOCT-SAM plus PV P/Q model.",
        "nn_p_only_fixed_kappa": "Tiny JAX MLP predicts P only; Q follows fixed kappa.",
        "direct_pq_scale_baseline": "Direct differentiable irradiance scaling baseline.",
    }
    types = {
        "analytic_pv_weather": "analytical_weather_model",
        "nn_p_only_fixed_kappa": "jax_mlp_surrogate",
        "direct_pq_scale_baseline": "direct_differentiable_baseline",
    }
    for model_name in UPSTREAM_MODELS:
        rows.append(
            CouplingSummaryRow(
                model_name=model_name,
                upstream_type=types[model_name],
                input_names="g_poa_wm2,t_amb_c,wind_ms",
                output_mode="PQ injection in MW/MVAr",
                coupling_bus_name=PV_COUPLING_BUS_NAME,
                replaced_element_name=PV_COUPLING_SGEN_NAME,
                uses_network_params_p_spec=True,
                uses_network_params_q_spec=True,
                uses_same_injection_adapter=True,
                uses_same_pf_core=True,
                requires_core_change=False,
                has_controller_logic=False,
                has_q_limits=False,
                has_pv_pq_switching=False,
                notes=notes[model_name],
            )
        )
    return rows


def build_model_comparison(
    params: MLPParams,
    norm: WeatherInputNormalization,
    config: SurrogateTrainingConfig,
    training_diagnostics: TrainingDiagnostics,
) -> tuple[list[ModelComparisonRow], list[RunSummaryRow]]:
    scenarios = {
        name: build_scenario_base(name, load_factor)
        for name, load_factor in ELECTRICAL_SCENARIOS
    }
    rows: list[ModelComparisonRow] = []
    diagnostics: dict[str, list[tuple[bool, int, float]]] = {name: [] for name in UPSTREAM_MODELS}

    for model_name in UPSTREAM_MODELS:
        for scenario_name, load_factor in ELECTRICAL_SCENARIOS:
            scenario = scenarios[scenario_name]
            for weather in WEATHER_CASES:
                conv, iters, residual, injection, obs = _solve_case(
                    scenario,
                    model_name,
                    params,
                    norm,
                    weather,
                )
                diagnostics[model_name].append((conv, iters, residual))
                for obs_name, unit in OBSERVABLE_SPECS:
                    rows.append(
                        ModelComparisonRow(
                            model_name=model_name,
                            network_scenario=scenario_name,
                            weather_case_id=weather["weather_case_id"],
                            g_poa_wm2=weather["g_poa_wm2"],
                            t_amb_c=weather["t_amb_c"],
                            wind_ms=weather["wind_ms"],
                            p_inj_mw=float(injection.p_pv_mw),
                            q_inj_mvar=float(injection.q_pv_mvar),
                            observable=obs_name,
                            value=obs.get(obs_name, float("nan")),
                            unit=unit,
                            converged=conv,
                            iterations=iters,
                            residual_norm=residual,
                        )
                    )

    summary_rows: list[RunSummaryRow] = []
    for model_name, values in diagnostics.items():
        converged = [item for item in values if item[0]]
        finite_residuals = [item[2] for item in converged if math.isfinite(item[2])]
        summary_rows.append(
            RunSummaryRow(
                model_name=model_name,
                train_points=config.train_samples,
                val_points=config.val_samples,
                eval_points=config.eval_samples,
                best_phase=training_diagnostics.best_phase,
                best_cycle_id=training_diagnostics.best_cycle_id,
                best_global_step=training_diagnostics.best_global_step,
                best_step=training_diagnostics.best_step,
                best_val_mse=training_diagnostics.best_val_mse,
                best_val_mae_mw=training_diagnostics.best_val_mae_mw,
                final_phase=training_diagnostics.final_phase,
                final_global_step=training_diagnostics.final_global_step,
                final_step=training_diagnostics.final_step,
                final_val_mse=training_diagnostics.final_val_mse,
                final_val_mae_mw=training_diagnostics.final_val_mae_mw,
                convergence_rate=len(converged) / max(len(values), 1),
                max_residual_norm=max(finite_residuals) if finite_residuals else float("nan"),
                max_iterations=max((item[1] for item in converged), default=-1),
                n_failed_solves=len(values) - len(converged),
                n_total_solves=len(values),
            )
        )
    return rows, summary_rows


def build_pf_observable_error_tables(
    comparison_rows: list[ModelComparisonRow],
) -> tuple[list[PFObservableErrorRow], list[PFObservableErrorSummaryRow]]:
    """Compare candidate PF observables against the analytical PV reference."""

    reference_model = "analytic_pv_weather"
    candidates = ("nn_p_only_fixed_kappa", "direct_pq_scale_baseline")
    by_key = {
        (
            row.model_name,
            row.network_scenario,
            row.weather_case_id,
            row.observable,
        ): row
        for row in comparison_rows
    }
    rows: list[PFObservableErrorRow] = []
    for candidate in candidates:
        for ref in comparison_rows:
            if ref.model_name != reference_model:
                continue
            candidate_row = by_key.get(
                (candidate, ref.network_scenario, ref.weather_case_id, ref.observable)
            )
            if candidate_row is None:
                continue
            signed_error = candidate_row.value - ref.value
            abs_error = abs(signed_error)
            reference_abs_floor = max(abs(ref.value), REL_ERROR_FLOOR)
            rows.append(
                PFObservableErrorRow(
                    candidate_model=candidate,
                    reference_model=reference_model,
                    network_scenario=ref.network_scenario,
                    weather_case_id=ref.weather_case_id,
                    observable=ref.observable,
                    unit=ref.unit,
                    reference_value=ref.value,
                    candidate_value=candidate_row.value,
                    signed_error=signed_error,
                    abs_error=abs_error,
                    rel_error_floor=abs_error / reference_abs_floor,
                    reference_abs_floor=reference_abs_floor,
                )
            )

    summary_rows: list[PFObservableErrorSummaryRow] = []
    groups: dict[tuple[str, str], list[PFObservableErrorRow]] = {}
    for row in rows:
        groups.setdefault((row.candidate_model, row.observable), []).append(row)
    for (candidate, observable), group in groups.items():
        abs_errors = np.asarray([row.abs_error for row in group], dtype=float)
        signed_errors = np.asarray([row.signed_error for row in group], dtype=float)
        rel_errors = np.asarray([row.rel_error_floor for row in group], dtype=float)
        mask = np.isfinite(abs_errors) & np.isfinite(signed_errors) & np.isfinite(rel_errors)
        abs_errors = abs_errors[mask]
        signed_errors = signed_errors[mask]
        rel_errors = rel_errors[mask]
        summary_rows.append(
            PFObservableErrorSummaryRow(
                candidate_model=candidate,
                observable=observable,
                n=int(abs_errors.size),
                mean_abs_error=float(np.mean(abs_errors)) if abs_errors.size else float("nan"),
                median_abs_error=float(np.median(abs_errors)) if abs_errors.size else float("nan"),
                max_abs_error=float(np.max(abs_errors)) if abs_errors.size else float("nan"),
                rmse=(
                    float(np.sqrt(np.mean(signed_errors**2)))
                    if signed_errors.size
                    else float("nan")
                ),
                mean_rel_error_floor=(
                    float(np.mean(rel_errors)) if rel_errors.size else float("nan")
                ),
                max_rel_error_floor=(
                    float(np.max(rel_errors)) if rel_errors.size else float("nan")
                ),
            )
        )
    return rows, summary_rows


def _fd_value(fn, input_name: str, g: float, t: float, w: float, step: float) -> float:
    g0 = jnp.asarray(g, dtype=jnp.float64)
    t0 = jnp.asarray(t, dtype=jnp.float64)
    w0 = jnp.asarray(w, dtype=jnp.float64)
    h = jnp.asarray(step, dtype=jnp.float64)
    if input_name == "g_poa_wm2":
        return float((fn(g0 + h, t0, w0) - fn(g0 - h, t0, w0)) / (2.0 * h))
    if input_name == "t_amb_c":
        return float((fn(g0, t0 + h, w0) - fn(g0, t0 - h, w0)) / (2.0 * h))
    if input_name == "wind_ms":
        return float((fn(g0, t0, w0 + h) - fn(g0, t0, w0 - h)) / (2.0 * h))
    raise ValueError(input_name)


def _ad_value(fn, input_name: str, g: float, t: float, w: float) -> float:
    g0 = jnp.asarray(g, dtype=jnp.float64)
    t0 = jnp.asarray(t, dtype=jnp.float64)
    w0 = jnp.asarray(w, dtype=jnp.float64)
    if input_name == "g_poa_wm2":
        return float(jax.grad(lambda g_arg: fn(g_arg, t0, w0))(g0))
    if input_name == "t_amb_c":
        return float(jax.grad(lambda t_arg: fn(g0, t_arg, w0))(t0))
    if input_name == "wind_ms":
        return float(jax.grad(lambda w_arg: fn(g0, t0, w_arg))(w0))
    raise ValueError(input_name)


def build_gradient_success_table(
    params: MLPParams,
    norm: WeatherInputNormalization,
) -> list[GradientSuccessRow]:
    scenario = build_scenario_base("base", 1.0)
    weather = next(row for row in WEATHER_CASES if row["weather_case_id"] == "exp3_reference")
    rows: list[GradientSuccessRow] = []
    for model_name in UPSTREAM_MODELS:
        for input_name, input_unit in WEATHER_INPUT_SPECS:
            fn = make_observable_fn(scenario, model_name, params, norm, "p_slack_mw")
            step = FD_STEPS[input_name]
            try:
                ad_grad = _ad_value(
                    fn,
                    input_name,
                    weather["g_poa_wm2"],
                    weather["t_amb_c"],
                    weather["wind_ms"],
                )
            except Exception:
                ad_grad = float("nan")
            try:
                fd_grad = _fd_value(
                    fn,
                    input_name,
                    weather["g_poa_wm2"],
                    weather["t_amb_c"],
                    weather["wind_ms"],
                    step,
                )
            except Exception:
                fd_grad = float("nan")
            is_ad = math.isfinite(ad_grad)
            is_fd = math.isfinite(fd_grad)
            abs_error = abs(ad_grad - fd_grad) if is_ad and is_fd else float("nan")
            rel_error = _robust_rel_error(ad_grad, fd_grad) if is_ad and is_fd else float("nan")
            rel_error_floor = (
                abs_error / max(abs(fd_grad), REL_ERROR_FLOOR) if is_ad and is_fd else float("nan")
            )
            rows.append(
                GradientSuccessRow(
                    model_name=model_name,
                    network_scenario="base",
                    weather_case_id=weather["weather_case_id"],
                    input_parameter=input_name,
                    observable="p_slack_mw",
                    ad_gradient=ad_grad,
                    fd_gradient=fd_grad,
                    abs_error=abs_error,
                    rel_error=rel_error,
                    ad_fd_abs_error=abs_error,
                    ad_fd_rel_error_floor=rel_error_floor,
                    is_finite_ad=is_ad,
                    is_finite_fd=is_fd,
                    passes_fd_check=bool(is_ad and is_fd and rel_error < 5e-3),
                    fd_step=step,
                    unit=f"MW/{input_unit}",
                )
            )
    return rows


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0.0 else float("nan")


def build_sensitivity_pattern_summary(
    params: MLPParams,
    norm: WeatherInputNormalization,
) -> list[SensitivityPatternSummaryRow]:
    scenario = build_scenario_base("base", 1.0)
    gradients: dict[tuple[str, str, str], list[float]] = {}
    for model_name in UPSTREAM_MODELS:
        for obs_name in PATTERN_OBSERVABLES:
            fn = make_observable_fn(scenario, model_name, params, norm, obs_name)
            for weather in WEATHER_CASES:
                try:
                    d_g, d_t, d_w = jax.grad(fn, argnums=(0, 1, 2))(
                        jnp.asarray(weather["g_poa_wm2"], dtype=jnp.float64),
                        jnp.asarray(weather["t_amb_c"], dtype=jnp.float64),
                        jnp.asarray(weather["wind_ms"], dtype=jnp.float64),
                    )
                    values = {
                        "g_poa_wm2": float(d_g),
                        "t_amb_c": float(d_t),
                        "wind_ms": float(d_w),
                    }
                except Exception:
                    values = {name: float("nan") for name, _ in WEATHER_INPUT_SPECS}
                for input_name, _ in WEATHER_INPUT_SPECS:
                    gradients.setdefault((model_name, obs_name, input_name), []).append(
                        values[input_name]
                    )

    rows: list[SensitivityPatternSummaryRow] = []
    for other in ("nn_p_only_fixed_kappa", "direct_pq_scale_baseline"):
        for obs_name in PATTERN_OBSERVABLES:
            for input_name, _ in WEATHER_INPUT_SPECS:
                ref = np.asarray(gradients[("analytic_pv_weather", obs_name, input_name)])
                val = np.asarray(gradients[(other, obs_name, input_name)])
                mask = np.isfinite(ref) & np.isfinite(val)
                ref = ref[mask]
                val = val[mask]
                diff = np.abs(ref - val)
                rel = diff / np.maximum(np.maximum(np.abs(ref), np.abs(val)), 1e-12)
                rows.append(
                    SensitivityPatternSummaryRow(
                        comparison_pair=f"analytic_pv_weather vs {other}",
                        network_scenario="base",
                        observable=obs_name,
                        input_parameter=input_name,
                        n_cases=int(ref.size),
                        mean_abs_grad_ref=float(np.mean(np.abs(ref))) if ref.size else float("nan"),
                        mean_abs_grad_other=(
                            float(np.mean(np.abs(val))) if val.size else float("nan")
                        ),
                        mean_abs_diff=float(np.mean(diff)) if diff.size else float("nan"),
                        median_rel_diff=float(np.median(rel)) if rel.size else float("nan"),
                        max_rel_diff=float(np.max(rel)) if rel.size else float("nan"),
                        sign_agreement_rate=(
                            float(np.mean(np.sign(ref) == np.sign(val)))
                            if ref.size
                            else float("nan")
                        ),
                        cosine_similarity=(
                            _cosine_similarity(ref, val) if ref.size else float("nan")
                        ),
                    )
                )
    return rows


def build_sensitivity_error_tables(
    params: MLPParams,
    norm: WeatherInputNormalization,
) -> tuple[list[SensitivityErrorRow], list[SensitivityErrorSummaryRow]]:
    """Build detailed absolute sensitivity-error metrics against the reference."""

    scenario = build_scenario_base("base", 1.0)
    gradients: dict[tuple[str, str, str, str], float] = {}
    for model_name in UPSTREAM_MODELS:
        for obs_name in PATTERN_OBSERVABLES:
            fn = make_observable_fn(scenario, model_name, params, norm, obs_name)
            for weather in WEATHER_CASES:
                try:
                    d_g, d_t, d_w = jax.grad(fn, argnums=(0, 1, 2))(
                        jnp.asarray(weather["g_poa_wm2"], dtype=jnp.float64),
                        jnp.asarray(weather["t_amb_c"], dtype=jnp.float64),
                        jnp.asarray(weather["wind_ms"], dtype=jnp.float64),
                    )
                    values = {
                        "g_poa_wm2": float(d_g),
                        "t_amb_c": float(d_t),
                        "wind_ms": float(d_w),
                    }
                except Exception:
                    values = {name: float("nan") for name, _ in WEATHER_INPUT_SPECS}
                for input_name, _ in WEATHER_INPUT_SPECS:
                    gradients[
                        (model_name, weather["weather_case_id"], obs_name, input_name)
                    ] = values[input_name]

    reference_model = "analytic_pv_weather"
    candidates = ("nn_p_only_fixed_kappa", "direct_pq_scale_baseline")
    rows: list[SensitivityErrorRow] = []
    for candidate in candidates:
        for weather in WEATHER_CASES:
            for obs_name in PATTERN_OBSERVABLES:
                for input_name, _ in WEATHER_INPUT_SPECS:
                    ref_grad = gradients[
                        (reference_model, weather["weather_case_id"], obs_name, input_name)
                    ]
                    cand_grad = gradients[
                        (candidate, weather["weather_case_id"], obs_name, input_name)
                    ]
                    signed_error = cand_grad - ref_grad
                    abs_error = abs(signed_error)
                    abs_ref = abs(ref_grad)
                    abs_cand = abs(cand_grad)
                    denom = max(abs_ref, REL_ERROR_FLOOR)
                    rows.append(
                        SensitivityErrorRow(
                            candidate_model=candidate,
                            reference_model=reference_model,
                            network_scenario="base",
                            weather_case_id=weather["weather_case_id"],
                            observable=obs_name,
                            input_parameter=input_name,
                            reference_gradient=ref_grad,
                            candidate_gradient=cand_grad,
                            signed_gradient_error=signed_error,
                            abs_gradient_error=abs_error,
                            rel_gradient_error_floor=abs_error / denom,
                            sign_match=bool(np.sign(ref_grad) == np.sign(cand_grad)),
                            abs_reference_gradient=abs_ref,
                            abs_candidate_gradient=abs_cand,
                            magnitude_ratio_abs=abs_cand / denom,
                        )
                    )

    return rows, summarize_sensitivity_error_rows(rows)


def summarize_sensitivity_error_rows(
    rows: list[SensitivityErrorRow],
) -> list[SensitivityErrorSummaryRow]:
    """Aggregate detailed sensitivity-error rows into the exported summary."""

    summary_rows: list[SensitivityErrorSummaryRow] = []
    groups: dict[tuple[str, str, str, str], list[SensitivityErrorRow]] = {}
    for row in rows:
        groups.setdefault(
            (row.candidate_model, row.network_scenario, row.observable, row.input_parameter),
            [],
        ).append(row)
    for (candidate, scenario_name, observable, input_name), group in groups.items():
        signed = np.asarray([row.signed_gradient_error for row in group], dtype=float)
        abs_errors = np.asarray([row.abs_gradient_error for row in group], dtype=float)
        rel_errors = np.asarray([row.rel_gradient_error_floor for row in group], dtype=float)
        sign_matches = np.asarray([row.sign_match for row in group], dtype=bool)
        ratios = np.asarray([row.magnitude_ratio_abs for row in group], dtype=float)
        ref = np.asarray([row.reference_gradient for row in group], dtype=float)
        cand = np.asarray([row.candidate_gradient for row in group], dtype=float)
        mask = (
            np.isfinite(signed)
            & np.isfinite(abs_errors)
            & np.isfinite(rel_errors)
            & np.isfinite(ratios)
            & np.isfinite(ref)
            & np.isfinite(cand)
        )
        signed = signed[mask]
        abs_errors = abs_errors[mask]
        rel_errors = rel_errors[mask]
        sign_matches = sign_matches[mask]
        ratios = ratios[mask]
        ref = ref[mask]
        cand = cand[mask]
        summary_rows.append(
            SensitivityErrorSummaryRow(
                candidate_model=candidate,
                network_scenario=scenario_name,
                observable=observable,
                input_parameter=input_name,
                n=int(abs_errors.size),
                mean_abs_gradient_error=(
                    float(np.mean(abs_errors)) if abs_errors.size else float("nan")
                ),
                median_abs_gradient_error=(
                    float(np.median(abs_errors)) if abs_errors.size else float("nan")
                ),
                max_abs_gradient_error=(
                    float(np.max(abs_errors)) if abs_errors.size else float("nan")
                ),
                rmse_gradient_error=(
                    float(np.sqrt(np.mean(signed**2))) if signed.size else float("nan")
                ),
                mean_rel_gradient_error_floor=(
                    float(np.mean(rel_errors)) if rel_errors.size else float("nan")
                ),
                median_rel_gradient_error_floor=(
                    float(np.median(rel_errors)) if rel_errors.size else float("nan")
                ),
                max_rel_gradient_error_floor=(
                    float(np.max(rel_errors)) if rel_errors.size else float("nan")
                ),
                sign_match_rate=(
                    float(np.mean(sign_matches)) if sign_matches.size else float("nan")
                ),
                median_magnitude_ratio_abs=(
                    float(np.median(ratios)) if ratios.size else float("nan")
                ),
                mean_cosine_similarity=(
                    _cosine_similarity(ref, cand) if ref.size else float("nan")
                ),
            )
        )
    return summary_rows


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


def write_metadata(
    results_dir: Path,
    config: SurrogateTrainingConfig,
    params: MLPParams,
    training_diagnostics: TrainingDiagnostics,
    training_improvement_rows: list[TrainingImprovementSummaryRow],
    architecture_comparison_rows: list[ArchitectureComparisonSummaryRow],
) -> None:
    improvement = training_improvement_rows[0] if training_improvement_rows else None
    architecture_comparison = (
        architecture_comparison_rows[0] if architecture_comparison_rows else None
    )
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_hash": _git_hash(),
        "python_version": sys.version,
        "experiment": "exp04_modular_upstream_nn_surrogate",
        "network": "pandapower.networks.example_simple(), scope_matched",
        "coupling_bus": PV_COUPLING_BUS_NAME,
        "replaced_element": PV_COUPLING_SGEN_NAME,
        "upstream_models": list(UPSTREAM_MODELS),
        "training_config": config._asdict(),
        "training_variant": "two_phase_base_plus_warm_restart_finetune",
        "checkpoint_resume_source": (
            "No persisted width-16 parameter checkpoint is loaded. The capacity "
            "run is initialized from scratch; Phase B starts from the in-process "
            "best Phase-A validation checkpoint. Width-8 checkpoints are not "
            "architecture-compatible and are not reused."
        ),
        "architecture_capacity_run": {
            "reference_hidden_width": WIDTH8_REFERENCE_HIDDEN_WIDTH,
            "candidate_hidden_width": config.hidden_width,
            "hidden_layers": config.hidden_layers,
            "activation": "tanh",
            "width8_checkpoint_reused": False,
            "candidate_initialized_from_scratch": True,
            "architecture_comparison_summary": (
                _to_native(asdict(architecture_comparison))
                if architecture_comparison is not None
                else {}
            ),
        },
        "learning_rate_schedule": {
            "schedule": config.lr_schedule,
            "cyclic": bool(config.warm_restart_enabled),
            "warm_restart_enabled": bool(config.warm_restart_enabled),
            "initial_schedule": config.initial_schedule,
            "initial_start": base_learning_rate_at_step(0, config),
            "initial_end": base_learning_rate_at_step(config.base_train_steps, config),
            "initial_learning_rate_start": config.initial_learning_rate_start,
            "initial_learning_rate_end": config.initial_learning_rate_end,
            "base_train_steps": config.base_train_steps,
            "finetune_schedule": config.lr_schedule,
            "finetune_steps": config.finetune_steps,
            "restart_cycle_steps": list(config.restart_cycle_steps),
            "restart_lr_max": list(config.restart_lr_max),
            "restart_lr_min": list(config.restart_lr_min),
            "restart_start_values": [
                warm_restart_learning_rate_at_step(sum(config.restart_cycle_steps[:idx]), config)
                for idx in range(len(config.restart_cycle_steps))
            ],
            "restart_end": warm_restart_learning_rate_at_step(config.finetune_steps, config),
        },
        "best_validation_checkpoint": asdict(training_diagnostics),
        "reference_metrics": {
            "reference_val_mse": REFERENCE_VAL_MSE,
            "reference_val_mae_mw": REFERENCE_VAL_MAE_MW,
            "reference_eval_p_mae_mw": REFERENCE_EVAL_P_MAE_MW,
            "reference_eval_p_rmse_mw": REFERENCE_EVAL_P_RMSE_MW,
            "reference_max_p_error_mw": REFERENCE_MAX_P_ERROR_MW,
        },
        "training_improvement_summary": (
            _to_native(asdict(improvement)) if improvement is not None else {}
        ),
        "dataset_sizes": {
            "train_points": config.train_samples,
            "val_points": config.val_samples,
            "eval_points": config.eval_samples,
        },
        "default_dataset_sizes": {
            "train_points": DEFAULT_TRAIN_SAMPLES,
            "val_points": DEFAULT_VAL_SAMPLES,
            "eval_points": DEFAULT_EVAL_SAMPLES,
        },
        "weather_ranges": {
            "g_poa_wm2": [0.0, 1200.0],
            "t_amb_c": [-10.0, 45.0],
            "wind_ms": [0.5, 10.0],
        },
        "dataset_seed_policy": {
            "base_seed": config.seed,
            "split_mechanism": "jax.random.split(seed, 4): params, train, val, eval",
            "anchor_points_per_split": 6,
        },
        "training_method": "full-batch gradient descent in JAX",
        "training_target": "P_ref_mw / 2.0 from analytical pv_pq_injection_from_weather",
        "model_architecture": {
            "input_names": ["g_poa_wm2", "t_amb_c", "wind_ms"],
            "hidden_width": config.hidden_width,
            "hidden_layers": config.hidden_layers,
            "activation": "tanh",
            "output": "normalized P_nn / 2.0",
            "q_coupling": "Q = -0.25 * P",
        },
        "evaluation_split_note": (
            "The eval split is an 8192-point synthetic surrogate-error dataset. "
            "The compact WEATHER_CASES list is kept for power-flow, AD-vs-FD, "
            "and sensitivity-pattern comparisons."
        ),
        "normalization": {
            "center": [float(x) for x in DEFAULT_WEATHER_NORMALIZATION.center],
            "scale": [float(x) for x in DEFAULT_WEATHER_NORMALIZATION.scale],
        },
        "mlp_parameter_count": count_mlp_parameters(params),
        "kappa": EXP4_KAPPA,
        "weather_cases": list(WEATHER_CASES),
        "observables": [{"name": name, "unit": unit} for name, unit in OBSERVABLE_SPECS],
        "known_simplifications": [
            "P-only MLP; Q is fixed by kappa = -0.25.",
            "The neural model is a distillation surrogate, not a measured-data forecast model.",
            "No controller logic, no Q limits, no PV-to-PQ switching.",
            "No gradient-matching loss and no architecture change in the warm-restart finetune.",
            "All upstream models use the same inject_pv_at_bus adapter and unchanged PF core.",
        ],
        "primary_metrics": [
            "PF observable absolute error and floor-relative error vs analytic_pv_weather.",
            "Sensitivity absolute gradient error, floor-relative gradient error, sign match, and magnitude ratio vs analytic_pv_weather.",
            "Cosine similarity is retained only as an auxiliary direction metric.",
        ],
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def write_readme(
    results_dir: Path,
    config: SurrogateTrainingConfig,
    training_diagnostics: TrainingDiagnostics,
    training_improvement_rows: list[TrainingImprovementSummaryRow],
    architecture_comparison_rows: list[ArchitectureComparisonSummaryRow],
) -> None:
    improvement = training_improvement_rows[0] if training_improvement_rows else None
    architecture = (
        architecture_comparison_rows[0] if architecture_comparison_rows else None
    )
    if improvement is None:
        comparison_text = "Training-improvement metrics were not available."
    else:
        comparison_text = f"""Against the 8000-step cosine-decay reference run
(Val-MSE `{REFERENCE_VAL_MSE:.8g}`, Val-MAE
`{REFERENCE_VAL_MAE_MW:.8g} MW`, Eval P-MAE
`{REFERENCE_EVAL_P_MAE_MW:.8g} MW`, Eval P-RMSE
`{REFERENCE_EVAL_P_RMSE_MW:.8g} MW`, Max P-error
`{REFERENCE_MAX_P_ERROR_MW:.8g} MW`), the new best checkpoint is in phase
`{improvement.best_phase}` at global step `{improvement.best_global_step}` with
Val-MSE
`{improvement.new_best_val_mse:.8g}` and Val-MAE
`{improvement.new_best_val_mae_mw:.8g} MW`. The new eval split has P-MAE
`{improvement.new_eval_p_mae_mw:.8g} MW`, P-RMSE
`{improvement.new_eval_p_rmse_mw:.8g} MW`, and max P-error
`{improvement.new_max_p_error_mw:.8g} MW`.

Relative improvements are Val-MSE
`{improvement.relative_improvement_val_mse:.6g}`, Val-MAE
`{improvement.relative_improvement_val_mae:.6g}`, Eval P-MAE
`{improvement.relative_improvement_eval_p_mae:.6g}`, Eval P-RMSE
`{improvement.relative_improvement_eval_p_rmse:.6g}`, and Max P-error
`{improvement.relative_improvement_max_p_error:.6g}`. Improved over reference:
`{improvement.improved_over_reference}`."""

    if architecture is None:
        architecture_text = "Architecture-comparison metrics were not available."
    else:
        architecture_text = f"""This run increases only the MLP hidden width from
`{architecture.reference_hidden_width}` to `{architecture.candidate_hidden_width}`.
The width-8 reference has `{architecture.reference_parameter_count}` parameters;
the width-16 candidate has `{architecture.candidate_parameter_count}` parameters.
No width-8 checkpoint is reused because the parameter shapes are incompatible;
the width-16 model is initialized and trained from scratch.

Against the width-8 warm-restart reference (Val-MSE
`{WIDTH8_REFERENCE_VAL_MSE:.8g}`, Val-MAE `{WIDTH8_REFERENCE_VAL_MAE_MW:.8g} MW`,
Eval P-MAE `{WIDTH8_REFERENCE_EVAL_P_MAE_MW:.8g} MW`, Eval P-RMSE
`{WIDTH8_REFERENCE_EVAL_P_RMSE_MW:.8g} MW`, Max P-error
`{WIDTH8_REFERENCE_MAX_P_ERROR_MW:.8g} MW`), the width-16 candidate reaches
Val-MSE `{architecture.candidate_best_val_mse:.8g}`, Val-MAE
`{architecture.candidate_best_val_mae_mw:.8g} MW`, Eval P-MAE
`{architecture.candidate_eval_p_mae_mw:.8g} MW`, Eval P-RMSE
`{architecture.candidate_eval_p_rmse_mw:.8g} MW`, and max P-error
`{architecture.candidate_max_p_error_mw:.8g} MW`.

Relative improvements are Val-MSE
`{architecture.relative_improvement_val_mse:.6g}`, Val-MAE
`{architecture.relative_improvement_val_mae:.6g}`, Eval P-MAE
`{architecture.relative_improvement_eval_p_mae:.6g}`, Eval P-RMSE
`{architecture.relative_improvement_eval_p_rmse:.6g}`, and Max P-error
`{architecture.relative_improvement_max_p_error:.6g}`. Improved over width 8:
`{architecture.improved_over_width8}`."""

    text = f"""# Experiment 4 - Modular Upstream Coupling with a Neural PQ Surrogate

This experiment demonstrates that the differentiable AC power-flow core can be
coupled to different upstream models through the same P/Q injection interface.
It compares the analytical PV/weather model, a small JAX-only P-only MLP
surrogate, and a direct P/Q irradiance baseline.

No Equinox, Flax, Optax, PyTorch, TensorFlow, controller logic, Q limits, or
PV-to-PQ switching are used. The numerical power-flow core is unchanged.

## Distillation dataset

The default full Experiment 4 run trains the NN surrogate on 32768 synthetic
weather points, validates on 8192 points, and evaluates surrogate error on a
separate 8192-point eval split. The fixed weather ranges are:

- `g_poa_wm2`: 0 to 1200 W/m^2
- `t_amb_c`: -10 to 45 degC
- `wind_ms`: 0.5 to 10 m/s

The splits are generated reproducibly from the configured JAX seed using
separate PRNG subkeys. Each split includes the same six anchor points and then
independent random synthetic points. The training target is the analytical
PV-weather teacher output `P_ref_mw / 2.0`; the NN remains a distillation
surrogate, not a measured-data PV forecast model.

The MLP has three inputs, two hidden layers of width {config.hidden_width} with `tanh`
activations, and one scalar normalized active-power output. Reactive power is
not learned: it remains deterministically coupled as `Q = -0.25 * P`.

## Capacity run

{architecture_text}

Training uses full-batch gradient descent in JAX. Because no persisted
parameter checkpoint existed for the previous run, this script uses a
two-phase training variant: Phase A reproduces the 8000-step non-cyclic
`cosine_decay` run from `{config.initial_learning_rate_start:.1e}` to
`{config.initial_learning_rate_end:.1e}`. Phase B starts from the best Phase-A
validation checkpoint and runs 8000 additional warm-restart finetune steps with
four cosine cycles:

- cycle 1: `2e-2 -> 5e-4` over 2000 steps
- cycle 2: `1e-2 -> 2e-4` over 2000 steps
- cycle 3: `5e-3 -> 1e-4` over 2000 steps
- cycle 4: `2e-3 -> 5e-5` over 2000 steps

The exported comparison uses the best global validation checkpoint over both
phases rather than blindly using the final parameter vector. In this run the
best checkpoint is phase `{training_diagnostics.best_phase}` at global step
`{training_diagnostics.best_global_step}`.

## Comparison to previous training run

{comparison_text}

The compact weather-case list used for `model_comparison`,
`gradient_success_table`, and `sensitivity_pattern_summary` remains separate
from the 8192-point eval split. This keeps the power-flow and gradient
comparisons intentionally small.

## Main artifacts

- `training_dataset_summary.csv/json`: synthetic distillation dataset ranges.
- `training_history.csv/json`: compact MLP training log.
- `surrogate_error_table.csv/json`: NN-vs-analytical-PV P/Q errors.
- `model_comparison.csv/json`: solved observables for all upstream models.
- `coupling_summary.csv/json`: interface evidence for the modularity claim.
- `gradient_success_table.csv/json`: representative AD-vs-FD spot checks.
- `pf_observable_error_table.csv/json`: absolute PF observable errors versus
  `analytic_pv_weather`.
- `pf_observable_error_summary.csv/json`: PF error aggregates by model and
  observable.
- `sensitivity_pattern_summary.csv/json`: gradient-pattern comparisons with
  cosine similarity retained as an auxiliary metric.
- `sensitivity_error_table.csv/json`: absolute and floor-relative sensitivity
  errors, sign matches, and magnitude ratios versus `analytic_pv_weather`.
- `sensitivity_error_summary.csv/json`: sensitivity error aggregates by model,
  observable, and weather input.
- `training_improvement_summary.csv/json`: comparison against the 8000-step
  cosine-decay reference NN-surrogate run.
- `architecture_comparison_summary.csv/json`: width-8 versus width-16 capacity
  comparison.
- `run_summary.csv/json`: convergence summary by upstream model.
"""
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "README.md").write_text(text, encoding="utf-8")


def export_all(
    results_dir: Path,
    dataset_rows: list[TrainingDatasetSummaryRow],
    history_rows: list[TrainingHistoryRow],
    error_rows: list[SurrogateErrorRow],
    comparison_rows: list[ModelComparisonRow],
    coupling_rows: list[CouplingSummaryRow],
    gradient_rows: list[GradientSuccessRow],
    pattern_rows: list[SensitivityPatternSummaryRow],
    pf_error_rows: list[PFObservableErrorRow],
    pf_error_summary_rows: list[PFObservableErrorSummaryRow],
    sensitivity_error_rows: list[SensitivityErrorRow],
    sensitivity_error_summary_rows: list[SensitivityErrorSummaryRow],
    training_improvement_rows: list[TrainingImprovementSummaryRow],
    architecture_comparison_rows: list[ArchitectureComparisonSummaryRow],
    summary_rows: list[RunSummaryRow],
    config: SurrogateTrainingConfig,
    params: MLPParams,
    training_diagnostics: TrainingDiagnostics,
) -> None:
    _write_csv(
        results_dir / "training_dataset_summary.csv",
        dataset_rows,
        TRAINING_DATASET_SUMMARY_COLUMNS,
    )
    _write_json(results_dir / "training_dataset_summary.json", dataset_rows)
    _write_csv(results_dir / "training_history.csv", history_rows, TRAINING_HISTORY_COLUMNS)
    _write_json(results_dir / "training_history.json", history_rows)
    _write_csv(results_dir / "surrogate_error_table.csv", error_rows, SURROGATE_ERROR_COLUMNS)
    _write_json(results_dir / "surrogate_error_table.json", error_rows)
    _write_csv(results_dir / "model_comparison.csv", comparison_rows, MODEL_COMPARISON_COLUMNS)
    _write_json(results_dir / "model_comparison.json", comparison_rows)
    _write_csv(results_dir / "coupling_summary.csv", coupling_rows, COUPLING_SUMMARY_COLUMNS)
    _write_json(results_dir / "coupling_summary.json", coupling_rows)
    _write_csv(results_dir / "gradient_success_table.csv", gradient_rows, GRADIENT_SUCCESS_COLUMNS)
    _write_json(results_dir / "gradient_success_table.json", gradient_rows)
    _write_csv(
        results_dir / "sensitivity_pattern_summary.csv",
        pattern_rows,
        SENSITIVITY_PATTERN_COLUMNS,
    )
    _write_json(results_dir / "sensitivity_pattern_summary.json", pattern_rows)
    _write_csv(
        results_dir / "pf_observable_error_table.csv",
        pf_error_rows,
        PF_OBSERVABLE_ERROR_COLUMNS,
    )
    _write_json(results_dir / "pf_observable_error_table.json", pf_error_rows)
    _write_csv(
        results_dir / "pf_observable_error_summary.csv",
        pf_error_summary_rows,
        PF_OBSERVABLE_ERROR_SUMMARY_COLUMNS,
    )
    _write_json(results_dir / "pf_observable_error_summary.json", pf_error_summary_rows)
    _write_csv(
        results_dir / "sensitivity_error_table.csv",
        sensitivity_error_rows,
        SENSITIVITY_ERROR_COLUMNS,
    )
    _write_json(results_dir / "sensitivity_error_table.json", sensitivity_error_rows)
    _write_csv(
        results_dir / "sensitivity_error_summary.csv",
        sensitivity_error_summary_rows,
        SENSITIVITY_ERROR_SUMMARY_COLUMNS,
    )
    _write_json(results_dir / "sensitivity_error_summary.json", sensitivity_error_summary_rows)
    _write_csv(
        results_dir / "training_improvement_summary.csv",
        training_improvement_rows,
        TRAINING_IMPROVEMENT_SUMMARY_COLUMNS,
    )
    _write_json(results_dir / "training_improvement_summary.json", training_improvement_rows)
    _write_csv(
        results_dir / "architecture_comparison_summary.csv",
        architecture_comparison_rows,
        ARCHITECTURE_COMPARISON_SUMMARY_COLUMNS,
    )
    _write_json(
        results_dir / "architecture_comparison_summary.json",
        architecture_comparison_rows,
    )
    _write_csv(results_dir / "run_summary.csv", summary_rows, RUN_SUMMARY_COLUMNS)
    _write_json(results_dir / "run_summary.json", summary_rows)
    write_metadata(
        results_dir,
        config,
        params,
        training_diagnostics,
        training_improvement_rows,
        architecture_comparison_rows,
    )
    write_readme(
        results_dir,
        config,
        training_diagnostics,
        training_improvement_rows,
        architecture_comparison_rows,
    )


def run_experiment(
    config: SurrogateTrainingConfig = SurrogateTrainingConfig(),
) -> tuple[
    list[TrainingDatasetSummaryRow],
    list[TrainingHistoryRow],
    list[SurrogateErrorRow],
    list[ModelComparisonRow],
    list[CouplingSummaryRow],
    list[GradientSuccessRow],
    list[SensitivityPatternSummaryRow],
    list[PFObservableErrorRow],
    list[PFObservableErrorSummaryRow],
    list[SensitivityErrorRow],
    list[SensitivityErrorSummaryRow],
    list[TrainingImprovementSummaryRow],
    list[ArchitectureComparisonSummaryRow],
    list[RunSummaryRow],
    MLPParams,
    TrainingDiagnostics,
]:
    norm = DEFAULT_WEATHER_NORMALIZATION
    params, train_x, val_x, eval_x, history_rows, training_diagnostics = train_surrogate(
        config,
        norm,
        return_diagnostics=True,
    )
    dataset_rows = [
        summarize_dataset("train", train_x),
        summarize_dataset("val", val_x),
        summarize_dataset("eval", eval_x),
    ]
    error_rows = build_surrogate_error_rows(params, norm, train_x, val_x, eval_x)
    comparison_rows, summary_rows = build_model_comparison(
        params,
        norm,
        config,
        training_diagnostics,
    )
    pf_error_rows, pf_error_summary_rows = build_pf_observable_error_tables(comparison_rows)
    coupling_rows = build_coupling_summary()
    gradient_rows = build_gradient_success_table(params, norm)
    pattern_rows = build_sensitivity_pattern_summary(params, norm)
    sensitivity_error_rows, sensitivity_error_summary_rows = build_sensitivity_error_tables(
        params,
        norm,
    )
    training_improvement_rows = build_training_improvement_summary(
        error_rows,
        config,
        training_diagnostics,
    )
    architecture_comparison_rows = build_architecture_comparison_summary(
        error_rows,
        config,
        params,
        training_diagnostics,
    )
    return (
        dataset_rows,
        history_rows,
        error_rows,
        comparison_rows,
        coupling_rows,
        gradient_rows,
        pattern_rows,
        pf_error_rows,
        pf_error_summary_rows,
        sensitivity_error_rows,
        sensitivity_error_summary_rows,
        training_improvement_rows,
        architecture_comparison_rows,
        summary_rows,
        params,
        training_diagnostics,
    )


def main() -> None:
    config = SurrogateTrainingConfig()
    print("=" * 72)
    print("Experiment 4: modular upstream NN surrogate")
    print(f"Results directory: {RESULTS_DIR}")
    print("=" * 72)
    (
        dataset_rows,
        history_rows,
        error_rows,
        comparison_rows,
        coupling_rows,
        gradient_rows,
        pattern_rows,
        pf_error_rows,
        pf_error_summary_rows,
        sensitivity_error_rows,
        sensitivity_error_summary_rows,
        training_improvement_rows,
        architecture_comparison_rows,
        summary_rows,
        params,
        training_diagnostics,
    ) = run_experiment(config)
    export_all(
        RESULTS_DIR,
        dataset_rows,
        history_rows,
        error_rows,
        comparison_rows,
        coupling_rows,
        gradient_rows,
        pattern_rows,
        pf_error_rows,
        pf_error_summary_rows,
        sensitivity_error_rows,
        sensitivity_error_summary_rows,
        training_improvement_rows,
        architecture_comparison_rows,
        summary_rows,
        config,
        params,
        training_diagnostics,
    )
    print("\nRun summary:")
    for row in summary_rows:
        print(
            f"  {row.model_name:<26} convergence={row.convergence_rate:.3f} "
            f"failed={row.n_failed_solves}/{row.n_total_solves}"
        )
    print("\nExported artifacts:")
    for name in REQUIRED_ARTIFACTS:
        print(f"  {name}")


if __name__ == "__main__":
    main()
