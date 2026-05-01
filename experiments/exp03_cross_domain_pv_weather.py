"""Experiment 3 – Cross-Domain PV Weather Sensitivity on example_simple().

Demonstrates the key USP of diffpf: end-to-end differentiability from
meteorological weather inputs through a PV upstream model into an AC power
flow, yielding sensitivities of electrical observables w.r.t. weather inputs.

Chain:
    g_poa_wm2, t_amb_c, wind_ms
        -> cell_temperature_noct_sam(...)
        -> pv_pq_injection_from_weather(...)
        -> P_pv, Q_pv at bus "MV Bus 2"
        -> NetworkParams
        -> AC Power Flow (implicit Newton)
        -> electrical observables

Electrical operating points: base, load_low, load_high.
Weather design:
  A) 2D grid: g_poa_wm2 x t_amb_c at fixed wind (5x5 = 25 cases).
  B) 1D sweep: t_amb_c at fixed G and wind (6 cases).

alpha and kappa are fixed constants in Exp. 3 (not varied).

Run:
    python experiments/exp03_cross_domain_pv_weather.py
"""

from __future__ import annotations

import copy
import csv
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

from jax import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import numpy as np
import pandapower as pp
import pandapower.networks as pn

from diffpf.compile.network import compile_network
from diffpf.core.residuals import calc_power_injection, state_to_voltage
from diffpf.core.types import CompiledTopology, NetworkParams, PFState
from diffpf.core.ybus import build_ybus
from diffpf.io.pandapower_adapter import from_pandapower
from diffpf.io.topology_utils import merge_buses
from diffpf.models.pv import (
    PV_COUPLING_BUS_NAME,
    PV_COUPLING_SGEN_NAME,
    cell_temperature_noct_sam,
    inject_pv_at_bus,
    pv_pq_injection_from_weather,
)
from diffpf.solver.implicit import solve_power_flow_implicit
from diffpf.solver.newton import NewtonOptions, solve_power_flow_result

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

RESULTS_DIR = (
    Path(__file__).resolve().parent / "results" / "exp03_cross_domain_pv_weather"
)

NEWTON_OPTIONS = NewtonOptions(max_iters=50, tolerance=1e-10, damping=0.7)

# ---------------------------------------------------------------------------
# Fixed model constants – not varied in Experiment 3.
# ---------------------------------------------------------------------------

EXP3_ALPHA: float = 1.0
EXP3_KAPPA: float = -0.25

# ---------------------------------------------------------------------------
# Weather scenario design
# ---------------------------------------------------------------------------

# A) 2D grid at fixed wind
G_LEVELS_WM2: tuple[float, ...] = (200.0, 400.0, 600.0, 800.0, 1000.0)
T_LEVELS_C: tuple[float, ...] = (5.0, 15.0, 25.0, 35.0, 45.0)
WIND_REF_MS: float = 2.0

# B) 1D temperature sweep at fixed G and wind
G_REF_WM2: float = 800.0
T_SWEEP_C: tuple[float, ...] = (5.0, 15.0, 25.0, 35.0, 45.0, 55.0)

# FD step sizes for spot-check validation
FD_STEPS: dict[str, float] = {
    "g_poa_wm2": 5.0,
    "t_amb_c": 0.1,
    "wind_ms": 0.1,
}

# ---------------------------------------------------------------------------
# Electrical operating points
# ---------------------------------------------------------------------------

ELECTRICAL_SCENARIOS: tuple[tuple[str, float], ...] = (
    ("base", 1.00),
    ("load_low", 0.75),
    ("load_high", 1.25),
)

# ---------------------------------------------------------------------------
# Observable and input specifications
# ---------------------------------------------------------------------------

OBSERVABLE_SPECS: tuple[tuple[str, str], ...] = (
    ("vm_mv_bus_2_pu", "p.u."),
    ("p_slack_mw", "MW"),
    ("total_p_loss_mw", "MW"),
    ("p_trafo_hv_mw", "MW"),
)

WEATHER_INPUT_SPECS: tuple[tuple[str, str], ...] = (
    ("g_poa_wm2", "W/m^2"),
    ("t_amb_c", "degC"),
    ("wind_ms", "m/s"),
)

# ---------------------------------------------------------------------------
# Spot-check cases: (scenario, observable, input, g_poa, t_amb, wind)
# ---------------------------------------------------------------------------

SPOTCHECK_CASES: tuple[tuple, ...] = (
    ("base", "vm_mv_bus_2_pu", "g_poa_wm2", 800.0, 25.0, 2.0),
    ("base", "p_slack_mw", "t_amb_c", 800.0, 25.0, 2.0),
    ("load_high", "total_p_loss_mw", "g_poa_wm2", 800.0, 25.0, 2.0),
    ("base", "vm_mv_bus_2_pu", "wind_ms", 800.0, 25.0, 2.0),
)

# ---------------------------------------------------------------------------
# Result row dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScenarioGridRow:
    network_scenario: str
    load_factor: float
    weather_case_id: str
    weather_case_type: str
    g_poa_wm2: float
    t_amb_c: float
    wind_ms: float
    cell_temp_c: float
    p_pv_mw: float
    q_pv_mvar: float
    observable: str
    value: float
    unit: str
    converged: bool
    iterations: int
    residual_norm: float


@dataclass(frozen=True)
class SensitivityRow:
    network_scenario: str
    load_factor: float
    weather_case_id: str
    weather_case_type: str
    g_poa_wm2: float
    t_amb_c: float
    wind_ms: float
    observable: str
    observable_unit: str
    input_parameter: str
    input_unit: str
    value: float
    ad_converged: bool


@dataclass(frozen=True)
class SpotCheckRow:
    spotcheck_id: str
    network_scenario: str
    observable: str
    observable_unit: str
    input_parameter: str
    input_unit: str
    g_poa_wm2: float
    t_amb_c: float
    wind_ms: float
    fd_step: float
    ad_grad: float
    fd_grad: float
    abs_error: float
    rel_error: float
    ad_converged: bool
    fd_plus_converged: bool
    fd_minus_converged: bool


@dataclass(frozen=True)
class RunSummaryRow:
    network_scenario: str
    load_factor: float
    n_weather_cases: int
    n_converged: int
    n_failed: int
    min_vm_mv_bus_2_pu: float
    max_vm_mv_bus_2_pu: float
    min_p_pv_mw: float
    max_p_pv_mw: float


SCENARIO_GRID_COLUMNS = tuple(f.name for f in fields(ScenarioGridRow))
SENSITIVITY_COLUMNS = tuple(f.name for f in fields(SensitivityRow))
SPOTCHECK_COLUMNS = tuple(f.name for f in fields(SpotCheckRow))
RUN_SUMMARY_COLUMNS = tuple(f.name for f in fields(RunSummaryRow))


# ---------------------------------------------------------------------------
# Weather case generation
# ---------------------------------------------------------------------------


def _weather_cases_2d() -> list[dict]:
    return [
        {
            "weather_case_id": f"grid2d_g{int(g)}_t{int(t)}_w{WIND_REF_MS:.0f}",
            "weather_case_type": "grid_2d",
            "g_poa_wm2": g,
            "t_amb_c": t,
            "wind_ms": WIND_REF_MS,
        }
        for g in G_LEVELS_WM2
        for t in T_LEVELS_C
    ]


def _weather_cases_1d_sweep() -> list[dict]:
    return [
        {
            "weather_case_id": f"sweep1d_g{int(G_REF_WM2)}_t{int(t)}_w{WIND_REF_MS:.0f}",
            "weather_case_type": "sweep_1d",
            "g_poa_wm2": G_REF_WM2,
            "t_amb_c": t,
            "wind_ms": WIND_REF_MS,
        }
        for t in T_SWEEP_C
    ]


ALL_WEATHER_CASES: list[dict] = _weather_cases_2d() + _weather_cases_1d_sweep()


# ---------------------------------------------------------------------------
# Compiled scenario container
# ---------------------------------------------------------------------------


class ScenarioBase(NamedTuple):
    name: str
    load_factor: float
    topology: CompiledTopology
    params_base: NetworkParams
    state0: PFState
    pv_bus_internal_idx: int
    trafo_idx: int
    s_base_mva: float


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------


def _bus_to_repr(net) -> dict[int, int]:
    bb_pairs = [
        (int(sw["bus"]), int(sw["element"]))
        for _, sw in net.switch.iterrows()
        if sw["et"] == "b" and bool(sw["closed"])
    ]
    return merge_buses(list(net.bus.index), bb_pairs)


def _find_pv_bus_internal_idx(net, spec) -> int:
    matches = net.bus[net.bus["name"] == PV_COUPLING_BUS_NAME]
    if len(matches) != 1:
        raise ValueError(f"Expected one bus named {PV_COUPLING_BUS_NAME!r}, got {len(matches)}.")
    original_bus = int(matches.index[0])
    repr_bus = _bus_to_repr(net)[original_bus]
    spec_bus_names = [bus.name for bus in spec.buses]
    return spec_bus_names.index(str(repr_bus))


def _make_initial_state(net, spec) -> PFState:
    slack_ang_rad = float(net.ext_grid.iloc[0]["va_degree"]) * math.pi / 180.0
    trafo_shift_deg = float(net.trafo.iloc[0]["shift_degree"]) if len(net.trafo) else 0.0
    lv_ang_rad = (float(net.ext_grid.iloc[0]["va_degree"]) - trafo_shift_deg) * math.pi / 180.0

    vr_list: list[float] = []
    vi_list: list[float] = []
    for bus in spec.buses:
        if bus.is_slack:
            continue
        bus_id = int(bus.name)
        vn_kv = float(net.bus.loc[bus_id, "vn_kv"])
        angle = slack_ang_rad if vn_kv >= 100.0 else lv_ang_rad
        vr_list.append(math.cos(angle))
        vi_list.append(math.sin(angle))

    return PFState(
        vr_pu=jnp.asarray(vr_list, dtype=jnp.float64),
        vi_pu=jnp.asarray(vi_list, dtype=jnp.float64),
    )


def build_scenario_base(scenario_name: str, load_factor: float) -> ScenarioBase:
    """Compile example_simple() without the PV sgen, with load scaling."""
    net = pn.example_simple()

    # Scale load
    load_matches = net.load[net.load["name"] == "load"]
    if len(load_matches) == 1:
        load_idx = int(load_matches.index[0])
        net.load.at[load_idx, "scaling"] = (
            float(net.load.at[load_idx, "scaling"]) * load_factor
        )
    else:
        for idx in net.load.index:
            if bool(net.load.at[idx, "in_service"]):
                net.load.at[idx, "scaling"] = (
                    float(net.load.at[idx, "scaling"]) * load_factor
                )
                break

    # Disable the target sgen — PV model will inject in its place.
    target_matches = net.sgen[
        (net.sgen["name"] == PV_COUPLING_SGEN_NAME)
        & (net.sgen["in_service"] == True)  # noqa: E712
    ]
    if len(target_matches) != 1:
        raise ValueError(
            f"Expected one active sgen named {PV_COUPLING_SGEN_NAME!r}, got {len(target_matches)}."
        )
    net.sgen.at[int(target_matches.index[0]), "in_service"] = False

    # Scope-matched: convert active gen to sgen(P, Q=0), consistent with Exp. 1/2.
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

    spec = from_pandapower(net)
    topology, params = compile_network(spec)
    state0 = _make_initial_state(net, spec)
    pv_bus_idx = _find_pv_bus_internal_idx(net, spec)
    s_base_mva = float(net.sn_mva) if float(net.sn_mva) > 0 else 1.0

    return ScenarioBase(
        name=scenario_name,
        load_factor=load_factor,
        topology=topology,
        params_base=params,
        state0=state0,
        pv_bus_internal_idx=pv_bus_idx,
        trafo_idx=0,
        s_base_mva=s_base_mva,
    )


# ---------------------------------------------------------------------------
# Observable evaluation (JAX-differentiable)
# ---------------------------------------------------------------------------


def _trafo_hv_p_mw(
    topology: CompiledTopology,
    params: NetworkParams,
    state: PFState,
    s_base_mva: float,
    trafo_idx: int = 0,
) -> jnp.ndarray:
    voltage = state_to_voltage(topology, params, state)
    hv = params.trafo_hv_bus[trafo_idx]
    lv = params.trafo_lv_bus[trafo_idx]
    y_t = params.trafo_g_series_pu[trafo_idx] + 1j * params.trafo_b_series_pu[trafo_idx]
    y_m = params.trafo_g_mag_pu[trafo_idx] + 1j * params.trafo_b_mag_pu[trafo_idx]
    a = params.trafo_tap_ratio[trafo_idx]
    phi = params.trafo_shift_rad[trafo_idx]
    tap = a * jnp.exp(1j * phi)
    current_hv = (
        ((y_t + y_m) / (a * a)) * voltage[hv]
        + (-y_t / jnp.conj(tap)) * voltage[lv]
    )
    s_hv = voltage[hv] * jnp.conj(current_hv)
    return jnp.real(s_hv) * s_base_mva


def _evaluate_solved_observable(
    topology: CompiledTopology,
    params: NetworkParams,
    state: PFState,
    observable_name: str,
    scenario: ScenarioBase,
) -> jnp.ndarray:
    voltage = state_to_voltage(topology, params, state)
    y_bus = build_ybus(topology, params)
    s_bus = calc_power_injection(y_bus, voltage)

    if observable_name == "vm_mv_bus_2_pu":
        return jnp.abs(voltage[scenario.pv_bus_internal_idx])
    if observable_name == "p_slack_mw":
        return jnp.real(s_bus[topology.slack_bus]) * scenario.s_base_mva
    if observable_name == "total_p_loss_mw":
        return jnp.sum(jnp.real(s_bus)) * scenario.s_base_mva
    if observable_name == "p_trafo_hv_mw":
        return _trafo_hv_p_mw(
            topology, params, state, scenario.s_base_mva, scenario.trafo_idx
        )
    raise ValueError(f"Unknown observable: {observable_name!r}")


# ---------------------------------------------------------------------------
# Forward solve with diagnostics
# ---------------------------------------------------------------------------


def _solve_weather_case(
    scenario: ScenarioBase,
    g_poa: float,
    t_amb: float,
    wind: float,
) -> dict:
    g = jnp.asarray(g_poa, dtype=jnp.float64)
    t = jnp.asarray(t_amb, dtype=jnp.float64)
    w = jnp.asarray(wind, dtype=jnp.float64)

    injection = pv_pq_injection_from_weather(g, t, w, alpha=EXP3_ALPHA, kappa=EXP3_KAPPA)
    cell_temp = float(cell_temperature_noct_sam(g, t, w))
    p_pv_mw = float(injection.p_pv_mw)
    q_pv_mvar = float(injection.q_pv_mvar)

    params_pv = inject_pv_at_bus(
        scenario.params_base,
        scenario.pv_bus_internal_idx,
        injection,
        scenario.s_base_mva,
    )

    try:
        result = solve_power_flow_result(
            scenario.topology, params_pv, scenario.state0, NEWTON_OPTIONS
        )
        converged = bool(result.converged)
        iterations = int(result.iterations)
        residual_norm = float(result.residual_norm)
        solution = result.solution
    except Exception as exc:
        return {
            "converged": False,
            "iterations": -1,
            "residual_norm": float("nan"),
            "solution": None,
            "cell_temp_c": cell_temp,
            "p_pv_mw": p_pv_mw,
            "q_pv_mvar": q_pv_mvar,
            "observables": {},
            "error": str(exc),
        }

    observables: dict[str, float] = {}
    if converged:
        for obs_name, _ in OBSERVABLE_SPECS:
            try:
                val = _evaluate_solved_observable(
                    scenario.topology, params_pv, solution, obs_name, scenario
                )
                observables[obs_name] = float(val)
            except Exception:
                observables[obs_name] = float("nan")

    return {
        "converged": converged,
        "iterations": iterations,
        "residual_norm": residual_norm,
        "solution": solution,
        "cell_temp_c": cell_temp,
        "p_pv_mw": p_pv_mw,
        "q_pv_mvar": q_pv_mvar,
        "observables": observables,
        "error": "",
    }


# ---------------------------------------------------------------------------
# Sensitivity computation (end-to-end AD)
# ---------------------------------------------------------------------------


def _make_weather_fn(scenario: ScenarioBase, observable_name: str):
    """Return fn(g, t, w) -> scalar observable; differentiable via implicit AD."""

    def fn(g, t, w):
        params_pv = inject_pv_at_bus(
            scenario.params_base,
            scenario.pv_bus_internal_idx,
            pv_pq_injection_from_weather(g, t, w, alpha=EXP3_ALPHA, kappa=EXP3_KAPPA),
            scenario.s_base_mva,
        )
        solution = solve_power_flow_implicit(
            scenario.topology, params_pv, scenario.state0, NEWTON_OPTIONS
        )
        return _evaluate_solved_observable(
            scenario.topology, params_pv, solution, observable_name, scenario
        )

    return fn


def _compute_sensitivity(
    scenario: ScenarioBase,
    observable_name: str,
    g_poa: float,
    t_amb: float,
    wind: float,
) -> tuple[float, float, float, bool]:
    """Return (d_g, d_t, d_w, converged) via reverse-mode AD."""
    fn = _make_weather_fn(scenario, observable_name)
    g0 = jnp.asarray(g_poa, dtype=jnp.float64)
    t0 = jnp.asarray(t_amb, dtype=jnp.float64)
    w0 = jnp.asarray(wind, dtype=jnp.float64)

    try:
        d_g, d_t, d_w = jax.grad(fn, argnums=(0, 1, 2))(g0, t0, w0)
        d_g_f = float(d_g)
        d_t_f = float(d_t)
        d_w_f = float(d_w)
        ok = math.isfinite(d_g_f) and math.isfinite(d_t_f) and math.isfinite(d_w_f)
        return d_g_f, d_t_f, d_w_f, ok
    except Exception:
        return float("nan"), float("nan"), float("nan"), False


# ---------------------------------------------------------------------------
# Spot check: AD vs central finite difference
# ---------------------------------------------------------------------------


def _fd_solve_converged(
    scenario: ScenarioBase,
    g_poa: float,
    t_amb: float,
    wind: float,
) -> bool:
    try:
        injection = pv_pq_injection_from_weather(
            jnp.asarray(g_poa, dtype=jnp.float64),
            jnp.asarray(t_amb, dtype=jnp.float64),
            jnp.asarray(wind, dtype=jnp.float64),
            alpha=EXP3_ALPHA,
            kappa=EXP3_KAPPA,
        )
        params_pv = inject_pv_at_bus(
            scenario.params_base, scenario.pv_bus_internal_idx, injection, scenario.s_base_mva
        )
        result = solve_power_flow_result(
            scenario.topology, params_pv, scenario.state0, NEWTON_OPTIONS
        )
        return bool(result.converged)
    except Exception:
        return False


def _robust_rel_error(ad: float, fd: float, floor: float = 1e-12) -> float:
    if not math.isfinite(ad) or not math.isfinite(fd):
        return float("nan")
    return abs(ad - fd) / max(abs(ad), abs(fd), floor)


def _spotcheck_row(
    scenario: ScenarioBase,
    observable_name: str,
    input_name: str,
    g_poa: float,
    t_amb: float,
    wind: float,
) -> SpotCheckRow:
    obs_unit = dict(OBSERVABLE_SPECS)[observable_name]
    in_unit = dict(WEATHER_INPUT_SPECS)[input_name]
    fd_step = FD_STEPS[input_name]
    spotcheck_id = f"{scenario.name}:{observable_name}__d_{input_name}"

    fn = _make_weather_fn(scenario, observable_name)
    g0 = jnp.asarray(g_poa, dtype=jnp.float64)
    t0 = jnp.asarray(t_amb, dtype=jnp.float64)
    w0 = jnp.asarray(wind, dtype=jnp.float64)
    h = jnp.asarray(fd_step, dtype=jnp.float64)

    # AD gradient
    ad_grad = float("nan")
    ad_converged = False
    try:
        if input_name == "g_poa_wm2":
            ad_grad = float(jax.grad(lambda g: fn(g, t0, w0))(g0))
        elif input_name == "t_amb_c":
            ad_grad = float(jax.grad(lambda t: fn(g0, t, w0))(t0))
        elif input_name == "wind_ms":
            ad_grad = float(jax.grad(lambda w: fn(g0, t0, w))(w0))
        ad_converged = math.isfinite(ad_grad)
    except Exception:
        pass

    # FD convergence checks
    if input_name == "g_poa_wm2":
        fd_plus_converged = _fd_solve_converged(scenario, g_poa + fd_step, t_amb, wind)
        fd_minus_converged = _fd_solve_converged(scenario, g_poa - fd_step, t_amb, wind)
    elif input_name == "t_amb_c":
        fd_plus_converged = _fd_solve_converged(scenario, g_poa, t_amb + fd_step, wind)
        fd_minus_converged = _fd_solve_converged(scenario, g_poa, t_amb - fd_step, wind)
    elif input_name == "wind_ms":
        fd_plus_converged = _fd_solve_converged(scenario, g_poa, t_amb, wind + fd_step)
        fd_minus_converged = _fd_solve_converged(scenario, g_poa, t_amb, wind - fd_step)
    else:
        fd_plus_converged = fd_minus_converged = False

    # Central FD gradient
    fd_grad = float("nan")
    if fd_plus_converged and fd_minus_converged:
        try:
            if input_name == "g_poa_wm2":
                plus_val = fn(g0 + h, t0, w0)
                minus_val = fn(g0 - h, t0, w0)
            elif input_name == "t_amb_c":
                plus_val = fn(g0, t0 + h, w0)
                minus_val = fn(g0, t0 - h, w0)
            elif input_name == "wind_ms":
                plus_val = fn(g0, t0, w0 + h)
                minus_val = fn(g0, t0, w0 - h)
            else:
                plus_val = minus_val = jnp.asarray(float("nan"))
            fd_grad = float((plus_val - minus_val) / (2.0 * h))
        except Exception:
            pass

    abs_error = float("nan")
    rel_error = float("nan")
    if math.isfinite(ad_grad) and math.isfinite(fd_grad):
        abs_error = abs(ad_grad - fd_grad)
        rel_error = _robust_rel_error(ad_grad, fd_grad)

    return SpotCheckRow(
        spotcheck_id=spotcheck_id,
        network_scenario=scenario.name,
        observable=observable_name,
        observable_unit=obs_unit,
        input_parameter=input_name,
        input_unit=in_unit,
        g_poa_wm2=g_poa,
        t_amb_c=t_amb,
        wind_ms=wind,
        fd_step=fd_step,
        ad_grad=ad_grad,
        fd_grad=fd_grad,
        abs_error=abs_error,
        rel_error=rel_error,
        ad_converged=ad_converged,
        fd_plus_converged=fd_plus_converged,
        fd_minus_converged=fd_minus_converged,
    )


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------


def _to_native(obj):
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    try:
        return obj.item()
    except AttributeError:
        pass
    if isinstance(obj, bool):
        return obj
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
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def write_metadata(results_dir: Path) -> None:
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_hash": _git_hash(),
        "python_version": sys.version,
        "experiment": "exp03_cross_domain_pv_weather",
        "network": "pandapower.networks.example_simple(), scope_matched",
        "pv_coupling_bus": PV_COUPLING_BUS_NAME,
        "pv_coupling_sgen": PV_COUPLING_SGEN_NAME,
        "model_constants": {
            "alpha": EXP3_ALPHA,
            "kappa": EXP3_KAPPA,
            "note": "alpha and kappa are fixed constants in Exp. 3, not sweep variables.",
        },
        "electrical_scenarios": [
            {"name": name, "load_factor": lf} for name, lf in ELECTRICAL_SCENARIOS
        ],
        "weather_design": {
            "grid_2d": {
                "g_levels_wm2": list(G_LEVELS_WM2),
                "t_levels_c": list(T_LEVELS_C),
                "wind_ref_ms": WIND_REF_MS,
                "n_cases": len(G_LEVELS_WM2) * len(T_LEVELS_C),
            },
            "sweep_1d": {
                "g_ref_wm2": G_REF_WM2,
                "t_sweep_c": list(T_SWEEP_C),
                "wind_ref_ms": WIND_REF_MS,
                "n_cases": len(T_SWEEP_C),
            },
            "total_weather_cases_per_scenario": len(ALL_WEATHER_CASES),
        },
        "observables": [
            {"name": name, "unit": unit} for name, unit in OBSERVABLE_SPECS
        ],
        "weather_inputs": [
            {"name": name, "unit": unit} for name, unit in WEATHER_INPUT_SPECS
        ],
        "spotcheck_cases": [
            {
                "network_scenario": sc,
                "observable": obs,
                "input_parameter": inp,
                "g_poa_wm2": g,
                "t_amb_c": t,
                "wind_ms": w,
            }
            for sc, obs, inp, g, t, w in SPOTCHECK_CASES
        ],
        "fd_steps": FD_STEPS,
        "solver_options": {
            "max_iters": NEWTON_OPTIONS.max_iters,
            "tolerance": NEWTON_OPTIONS.tolerance,
            "damping": NEWTON_OPTIONS.damping,
            "initialization": "trafo_shift_aware",
        },
        "known_simplifications": [
            "PV plant modelled as weather-dependent PQ injection, not a voltage-regulating PV bus.",
            "No Q limits, no PV-to-PQ switching, no controller logic.",
            "Weather inputs remain in meteorological units; conversion to p.u. only for P_pv, Q_pv.",
            "Compact scenario space: 3 electrical x 31 weather = 93 forward solves.",
            "gen is converted to sgen(P, Q=0) (scope_matched, consistent with Exp. 1/2).",
            "alpha and kappa are fixed constants in Exp. 3 (EXP3_ALPHA=1.0, EXP3_KAPPA=-0.25).",
            "wind_adj = wind_ms in NOCT-SAM model (no height or mounting correction).",
        ],
    }
    path = results_dir / "metadata.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def write_readme(results_dir: Path) -> None:
    text = """# Experiment 3 – Cross-Domain PV Weather Sensitivity

Network: `pandapower.networks.example_simple()` (scope-matched: gen -> sgen).
PV coupling: `sgen "static generator"` at bus `"MV Bus 2"` replaced by the
weather-driven PV model (`pv_pq_injection_from_weather`).

## Weather design

- **2D grid** (`grid_2d`): 5 irradiance levels x 5 temperature levels at fixed
  wind (2 m/s) → 25 weather cases per electrical scenario.
- **1D sweep** (`sweep_1d`): 6 temperature levels at fixed G=800 W/m², wind=2 m/s.
- Total: 31 weather cases × 3 electrical scenarios = 93 forward solves.

## Fixed model constants

- `alpha = 1.0`, `kappa = -0.25` (not varied in this experiment).

## Artifacts

| File | Description |
|------|-------------|
| `scenario_grid.csv/json` | Forward solve results: one row per (scenario, weather_case, observable). |
| `sensitivity_table.csv/json` | AD sensitivities d_observable/d_weather_input for all cases. |
| `gradient_spotcheck.csv/json` | AD vs. central FD spot-check for 4 representative gradients. |
| `run_summary.csv/json` | Per-scenario summary: convergence counts, voltage/PV ranges. |
| `metadata.json` | Reproducibility metadata. |
| `README.md` | This file. |

## Known simplifications

- PQ injection, not PV-bus voltage regulation.
- No Q limits, no PV-to-PQ switching, no controller logic.
- Compact scenario space (no full 3D weather cube).

See `docs/context/experiment_03_plan.md` for the full scientific plan.
"""
    path = results_dir / "README.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def export_all(
    grid_rows: list[ScenarioGridRow],
    sensitivity_rows: list[SensitivityRow],
    spotcheck_rows: list[SpotCheckRow],
    summary_rows: list[RunSummaryRow],
    results_dir: Path,
) -> None:
    _write_csv(results_dir / "scenario_grid.csv", grid_rows, SCENARIO_GRID_COLUMNS)
    _write_json(results_dir / "scenario_grid.json", grid_rows)
    _write_csv(results_dir / "sensitivity_table.csv", sensitivity_rows, SENSITIVITY_COLUMNS)
    _write_json(results_dir / "sensitivity_table.json", sensitivity_rows)
    _write_csv(results_dir / "gradient_spotcheck.csv", spotcheck_rows, SPOTCHECK_COLUMNS)
    _write_json(results_dir / "gradient_spotcheck.json", spotcheck_rows)
    _write_csv(results_dir / "run_summary.csv", summary_rows, RUN_SUMMARY_COLUMNS)
    _write_json(results_dir / "run_summary.json", summary_rows)
    write_metadata(results_dir)
    write_readme(results_dir)


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------


def run_experiment() -> tuple[
    list[ScenarioGridRow],
    list[SensitivityRow],
    list[SpotCheckRow],
    list[RunSummaryRow],
]:
    """Run all Exp. 3 scenarios and return result rows."""

    print("=" * 72, flush=True)
    print("Experiment 3: Cross-Domain PV Weather Sensitivity", flush=True)
    print(f"Results directory: {RESULTS_DIR}", flush=True)
    print(f"Weather cases per scenario: {len(ALL_WEATHER_CASES)}", flush=True)
    print(f"  2D grid: {len(G_LEVELS_WM2) * len(T_LEVELS_C)} cases", flush=True)
    print(f"  1D sweep: {len(T_SWEEP_C)} cases", flush=True)
    print("=" * 72, flush=True)

    # Build scenario bases
    print("\nBuilding electrical scenario bases ...", flush=True)
    scenario_bases: dict[str, ScenarioBase] = {}
    for name, load_factor in ELECTRICAL_SCENARIOS:
        print(f"  {name} (load_factor={load_factor})", flush=True)
        scenario_bases[name] = build_scenario_base(name, load_factor)

    # Forward solves → scenario_grid
    print(f"\nRunning {len(ELECTRICAL_SCENARIOS)} x {len(ALL_WEATHER_CASES)} forward solves ...", flush=True)
    grid_rows: list[ScenarioGridRow] = []
    forward_results: dict[tuple[str, str], dict] = {}

    for scenario_name, load_factor in ELECTRICAL_SCENARIOS:
        scenario = scenario_bases[scenario_name]
        n_converged = 0
        for weather in ALL_WEATHER_CASES:
            g = weather["g_poa_wm2"]
            t = weather["t_amb_c"]
            w = weather["wind_ms"]
            wid = weather["weather_case_id"]
            wtype = weather["weather_case_type"]

            result = _solve_weather_case(scenario, g, t, w)
            forward_results[(scenario_name, wid)] = result
            if result["converged"]:
                n_converged += 1

            for obs_name, obs_unit in OBSERVABLE_SPECS:
                value = (
                    result["observables"].get(obs_name, float("nan"))
                    if result["converged"]
                    else float("nan")
                )
                grid_rows.append(
                    ScenarioGridRow(
                        network_scenario=scenario_name,
                        load_factor=load_factor,
                        weather_case_id=wid,
                        weather_case_type=wtype,
                        g_poa_wm2=g,
                        t_amb_c=t,
                        wind_ms=w,
                        cell_temp_c=result["cell_temp_c"],
                        p_pv_mw=result["p_pv_mw"],
                        q_pv_mvar=result["q_pv_mvar"],
                        observable=obs_name,
                        value=value,
                        unit=obs_unit,
                        converged=result["converged"],
                        iterations=result["iterations"],
                        residual_norm=result["residual_norm"],
                    )
                )
        print(f"  {scenario_name}: {n_converged}/{len(ALL_WEATHER_CASES)} converged", flush=True)

    print(f"  scenario_grid: {len(grid_rows)} rows", flush=True)

    # Sensitivity computation via end-to-end AD
    print("\nComputing weather sensitivities via implicit AD ...", flush=True)
    sensitivity_rows: list[SensitivityRow] = []
    input_unit_map = dict(WEATHER_INPUT_SPECS)
    total_cases = len(ELECTRICAL_SCENARIOS) * len(ALL_WEATHER_CASES)
    done = 0

    for scenario_name, load_factor in ELECTRICAL_SCENARIOS:
        scenario = scenario_bases[scenario_name]
        for weather in ALL_WEATHER_CASES:
            g = weather["g_poa_wm2"]
            t = weather["t_amb_c"]
            w = weather["wind_ms"]
            wid = weather["weather_case_id"]
            wtype = weather["weather_case_type"]
            fwd = forward_results.get((scenario_name, wid), {})
            done += 1
            if done % 10 == 0 or done == total_cases:
                print(f"  {done}/{total_cases} cases processed ...", flush=True)

            for obs_name, obs_unit in OBSERVABLE_SPECS:
                if not fwd.get("converged", False):
                    for input_name, input_unit in WEATHER_INPUT_SPECS:
                        sensitivity_rows.append(
                            SensitivityRow(
                                network_scenario=scenario_name,
                                load_factor=load_factor,
                                weather_case_id=wid,
                                weather_case_type=wtype,
                                g_poa_wm2=g,
                                t_amb_c=t,
                                wind_ms=w,
                                observable=obs_name,
                                observable_unit=obs_unit,
                                input_parameter=input_name,
                                input_unit=input_unit,
                                value=float("nan"),
                                ad_converged=False,
                            )
                        )
                    continue

                d_g, d_t, d_w, ok = _compute_sensitivity(scenario, obs_name, g, t, w)
                for input_name, grad_val in [
                    ("g_poa_wm2", d_g),
                    ("t_amb_c", d_t),
                    ("wind_ms", d_w),
                ]:
                    sensitivity_rows.append(
                        SensitivityRow(
                            network_scenario=scenario_name,
                            load_factor=load_factor,
                            weather_case_id=wid,
                            weather_case_type=wtype,
                            g_poa_wm2=g,
                            t_amb_c=t,
                            wind_ms=w,
                            observable=obs_name,
                            observable_unit=obs_unit,
                            input_parameter=input_name,
                            input_unit=input_unit_map[input_name],
                            value=grad_val,
                            ad_converged=ok,
                        )
                    )

    print(f"  sensitivity_table: {len(sensitivity_rows)} rows", flush=True)

    # Spot check: AD vs central FD
    print("\nRunning AD vs FD spot check ...", flush=True)
    spotcheck_rows: list[SpotCheckRow] = []
    for sc_name, obs_name, input_name, g, t, w in SPOTCHECK_CASES:
        print(f"  {sc_name}  {obs_name}  d/{input_name}", flush=True)
        row = _spotcheck_row(scenario_bases[sc_name], obs_name, input_name, g, t, w)
        spotcheck_rows.append(row)
        if row.ad_converged and math.isfinite(row.fd_grad):
            print(
                f"    AD={row.ad_grad:.6e}  FD={row.fd_grad:.6e}"
                f"  abs_err={row.abs_error:.3e}  rel_err={row.rel_error:.3e}",
                flush=True,
            )
        else:
            print(
                f"    AD converged={row.ad_converged}  FD+={row.fd_plus_converged}"
                f"  FD-={row.fd_minus_converged}",
                flush=True,
            )

    # Run summary
    summary_rows: list[RunSummaryRow] = []
    for scenario_name, load_factor in ELECTRICAL_SCENARIOS:
        n_total = len(ALL_WEATHER_CASES)
        n_conv = sum(
            1
            for wc in ALL_WEATHER_CASES
            if forward_results.get((scenario_name, wc["weather_case_id"]), {}).get(
                "converged", False
            )
        )
        vm_vals = [
            forward_results[(scenario_name, wc["weather_case_id"])]["observables"].get(
                "vm_mv_bus_2_pu", float("nan")
            )
            for wc in ALL_WEATHER_CASES
            if forward_results.get((scenario_name, wc["weather_case_id"]), {}).get(
                "converged", False
            )
        ]
        p_pv_vals = [
            forward_results[(scenario_name, wc["weather_case_id"])]["p_pv_mw"]
            for wc in ALL_WEATHER_CASES
        ]
        summary_rows.append(
            RunSummaryRow(
                network_scenario=scenario_name,
                load_factor=load_factor,
                n_weather_cases=n_total,
                n_converged=n_conv,
                n_failed=n_total - n_conv,
                min_vm_mv_bus_2_pu=min(vm_vals) if vm_vals else float("nan"),
                max_vm_mv_bus_2_pu=max(vm_vals) if vm_vals else float("nan"),
                min_p_pv_mw=min(p_pv_vals) if p_pv_vals else float("nan"),
                max_p_pv_mw=max(p_pv_vals) if p_pv_vals else float("nan"),
            )
        )

    return grid_rows, sensitivity_rows, spotcheck_rows, summary_rows


def main() -> None:
    grid_rows, sensitivity_rows, spotcheck_rows, summary_rows = run_experiment()
    export_all(grid_rows, sensitivity_rows, spotcheck_rows, summary_rows, RESULTS_DIR)

    print("\nRun summary:")
    for row in summary_rows:
        vm_range = f"{row.min_vm_mv_bus_2_pu:.4f} – {row.max_vm_mv_bus_2_pu:.4f}"
        p_range = f"{row.min_p_pv_mw:.3f} – {row.max_p_pv_mw:.3f}"
        print(
            f"  {row.network_scenario:<10} converged={row.n_converged}/{row.n_weather_cases}"
            f"  vm=[{vm_range}] pu  p_pv=[{p_range}] MW"
        )

    print("\nExported artifacts:")
    for path in sorted(RESULTS_DIR.iterdir()):
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
