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
    train_mse: float
    val_mse: float
    train_mae_mw: float
    val_mae_mw: float
    learning_rate: float


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
class RunSummaryRow:
    model_name: str
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


def train_surrogate(
    config: SurrogateTrainingConfig = SurrogateTrainingConfig(),
    norm: WeatherInputNormalization = DEFAULT_WEATHER_NORMALIZATION,
) -> tuple[MLPParams, jnp.ndarray, jnp.ndarray, list[TrainingHistoryRow]]:
    """Distill the analytical PV weather model into the small MLP."""

    key = jax.random.PRNGKey(config.seed)
    key_params, key_train, key_val = jax.random.split(key, 3)
    train_x = make_weather_dataset(key_train, config.train_samples)
    val_x = make_weather_dataset(key_val, config.val_samples)
    train_y = _target_p_mw(train_x) / PV_BASE_P_MW
    val_y = _target_p_mw(val_x) / PV_BASE_P_MW
    params = init_mlp_params(
        key_params,
        hidden_width=config.hidden_width,
        hidden_layers=config.hidden_layers,
    )
    value_and_grad = jax.jit(jax.value_and_grad(_loss))

    history: list[TrainingHistoryRow] = []
    for step in range(config.max_train_steps + 1):
        train_mse, grads = value_and_grad(params, norm, train_x, train_y)
        if step % config.log_every == 0 or step == config.max_train_steps:
            val_mse = _loss(params, norm, val_x, val_y)
            train_mae = jnp.mean(
                jnp.abs(_predict_p_mw(params, norm, train_x) - train_y * PV_BASE_P_MW)
            )
            val_mae = jnp.mean(jnp.abs(_predict_p_mw(params, norm, val_x) - val_y * PV_BASE_P_MW))
            history.append(
                TrainingHistoryRow(
                    step=step,
                    train_mse=float(train_mse),
                    val_mse=float(val_mse),
                    train_mae_mw=float(train_mae),
                    val_mae_mw=float(val_mae),
                    learning_rate=config.learning_rate,
                )
            )
        params = jax.tree_util.tree_map(
            lambda p, g: p - config.learning_rate * g,
            params,
            grads,
        )
    return params, train_x, val_x, history


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
) -> list[SurrogateErrorRow]:
    """Evaluate NN-vs-analytical PV error on train/val/eval cases."""

    rows: list[SurrogateErrorRow] = []
    eval_x = _weather_array(list(WEATHER_CASES))
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
                    p_rel_error=p_abs / max(abs(float(p_ref[idx])), 1e-12),
                    q_abs_error_mvar=q_abs,
                    q_rel_error=q_abs / max(abs(q_ref), 1e-12),
                )
            )
    return rows


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
                convergence_rate=len(converged) / max(len(values), 1),
                max_residual_norm=max(finite_residuals) if finite_residuals else float("nan"),
                max_iterations=max((item[1] for item in converged), default=-1),
                n_failed_solves=len(values) - len(converged),
                n_total_solves=len(values),
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
) -> None:
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
            "All upstream models use the same inject_pv_at_bus adapter and unchanged PF core.",
        ],
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def write_readme(results_dir: Path) -> None:
    text = """# Experiment 4 - Modular Upstream Coupling with a Neural PQ Surrogate

This experiment demonstrates that the differentiable AC power-flow core can be
coupled to different upstream models through the same P/Q injection interface.
It compares the analytical PV/weather model, a tiny JAX-only P-only MLP
surrogate, and a direct P/Q irradiance baseline.

No Equinox, Flax, Optax, PyTorch, TensorFlow, controller logic, Q limits, or
PV-to-PQ switching are used. The numerical power-flow core is unchanged.

## Main artifacts

- `training_dataset_summary.csv/json`: synthetic distillation dataset ranges.
- `training_history.csv/json`: compact MLP training log.
- `surrogate_error_table.csv/json`: NN-vs-analytical-PV P/Q errors.
- `model_comparison.csv/json`: solved observables for all upstream models.
- `coupling_summary.csv/json`: interface evidence for the modularity claim.
- `gradient_success_table.csv/json`: representative AD-vs-FD spot checks.
- `sensitivity_pattern_summary.csv/json`: gradient-pattern comparisons.
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
    summary_rows: list[RunSummaryRow],
    config: SurrogateTrainingConfig,
    params: MLPParams,
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
    _write_csv(results_dir / "run_summary.csv", summary_rows, RUN_SUMMARY_COLUMNS)
    _write_json(results_dir / "run_summary.json", summary_rows)
    write_metadata(results_dir, config, params)
    write_readme(results_dir)


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
    list[RunSummaryRow],
    MLPParams,
]:
    norm = DEFAULT_WEATHER_NORMALIZATION
    params, train_x, val_x, history_rows = train_surrogate(config, norm)
    dataset_rows = [
        summarize_dataset("train", train_x),
        summarize_dataset("val", val_x),
        summarize_dataset("eval", _weather_array(list(WEATHER_CASES))),
    ]
    error_rows = build_surrogate_error_rows(params, norm, train_x, val_x)
    comparison_rows, summary_rows = build_model_comparison(params, norm)
    coupling_rows = build_coupling_summary()
    gradient_rows = build_gradient_success_table(params, norm)
    pattern_rows = build_sensitivity_pattern_summary(params, norm)
    return (
        dataset_rows,
        history_rows,
        error_rows,
        comparison_rows,
        coupling_rows,
        gradient_rows,
        pattern_rows,
        summary_rows,
        params,
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
        summary_rows,
        params,
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
        summary_rows,
        config,
        params,
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
