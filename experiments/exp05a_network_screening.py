"""Experiment 5a - reduced network screening for PV curtailment preparation.

This experiment screens a compact scenario grid on the scope-matched
``pandapower.networks.example_simple()`` demonstrator. It replaces the
``sgen "static generator"`` at ``"MV Bus 2"`` by the existing JAX-compatible
PV/weather PQ model and identifies the most stressed operating points for a
later PV-curtailment experiment.

No optimization is performed here. The experiment only runs forward solves,
computes demonstrator-internal criticality indicators, selects the Top-20
screening cases, and computes local AD sensitivities with respect to the
``curtailment_factor`` for those Top-20 cases.

Run:
    python experiments/exp05a_network_screening.py
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
import pandapower as pp
import pandapower.networks as pn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffpf.compile.network import compile_network
from diffpf.core.residuals import calc_power_injection, state_to_voltage
from diffpf.core.types import CompiledTopology, NetworkParams, PFState
from diffpf.core.ybus import build_ybus
from diffpf.io.pandapower_adapter import from_pandapower
from diffpf.io.topology_utils import merge_buses
from diffpf.models.pv import (
    PV_COUPLING_BUS_NAME,
    PV_COUPLING_SGEN_NAME,
    PV_Q_OVER_P,
    cell_temperature_noct_sam,
    inject_pv_at_bus,
    pv_pq_injection_from_weather,
)
from diffpf.solver.implicit import solve_power_flow_implicit
from diffpf.solver.newton import NewtonOptions, solve_power_flow_result


RESULTS_DIR = Path(__file__).resolve().parent / "results" / "exp05a_network_screening"

NEWTON_OPTIONS = NewtonOptions(max_iters=50, tolerance=1e-10, damping=0.7)

LOAD_MULTIPLIERS_MV_BUS_2: tuple[float, ...] = (0.40, 0.70, 1.00, 1.30)
G_POA_LEVELS_WM2: tuple[float, ...] = (200.0, 600.0, 1200.0)
T_AMB_LEVELS_C: tuple[float, ...] = (-10.0, 5.0, 25.0, 45.0)
WIND_MS: float = 2.0
CURTAILMENT_FACTOR: float = 1.0
PV_SIZE_FACTOR: float = 1.0
EXP5A_KAPPA: float = PV_Q_OVER_P
TRAFO_RATING_PROXY_MVA: float = 25.0
TOP_N_CRITICAL_CASES: int = 20

SELECTED_REALISTIC_CASE_ID = "selected_realistic_load0p4_g1200_t30"
SELECTED_REALISTIC_LOAD_MULTIPLIER = 0.40
SELECTED_REALISTIC_G_POA_WM2 = 1200.0
SELECTED_REALISTIC_T_AMB_C = 30.0
SELECTED_REALISTIC_WIND_MS = 2.0

SENSITIVITY_OBSERVABLES: tuple[tuple[str, str], ...] = (
    ("p_pv_mw", "MW"),
    ("q_pv_mvar", "MVAr"),
    ("p_export_mw", "MW"),
    ("vm_mv_bus_2_pu", "p.u."),
    ("max_vm_pu", "p.u."),
    ("total_p_loss_mw", "MW"),
    ("s_trafo_hv_mva", "MVA"),
    ("criticality_score", "dimensionless"),
)

SELECTED_REALISTIC_SENSITIVITY_OBSERVABLES: tuple[tuple[str, str], ...] = (
    ("p_pv_mw", "MW"),
    ("q_pv_mvar", "MVAr"),
    ("p_slack_mw", "MW"),
    ("minus_p_slack_mw", "MW"),
    ("p_export_mw", "MW"),
    ("vm_mv_bus_2_pu", "p.u."),
    ("total_p_loss_mw", "MW"),
    ("s_trafo_hv_mva", "MVA"),
)

REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "screening_results.csv",
    "screening_results.json",
    "top_critical_cases.csv",
    "top_critical_cases.json",
    "sensitivity_top20.csv",
    "sensitivity_top20.json",
    "branch_flows.csv",
    "branch_flows.json",
    "run_summary.csv",
    "run_summary.json",
    "selected_realistic_case.csv",
    "selected_realistic_case.json",
    "selected_realistic_case_sensitivity.csv",
    "selected_realistic_case_sensitivity.json",
    "metadata.json",
    "README.md",
)


@dataclass(frozen=True)
class LineMeta:
    """Static metadata for one active line in compiled order."""

    compiled_line_idx: int
    pp_line_idx: int
    line_name: str
    from_bus_name: str
    to_bus_name: str
    from_bus_internal: int
    to_bus_internal: int
    is_mv_branch: bool


class ScenarioBase(NamedTuple):
    """Compiled scope-matched base network for one load multiplier."""

    load_multiplier: float
    topology: CompiledTopology
    params_base: NetworkParams
    state0: PFState
    pv_bus_internal_idx: int
    trafo_idx: int
    s_base_mva: float
    bus_names: tuple[str, ...]
    line_meta: tuple[LineMeta, ...]
    disabled_line_ids: tuple[int, ...]
    disabled_trafo_ids: tuple[int, ...]


@dataclass(frozen=True)
class ScreeningRow:
    case_id: str
    case_type: str
    selected_for_sensitivity: bool
    top20_rank: int
    load_multiplier_mv_bus_2: float
    g_poa_wm2: float
    t_amb_c: float
    wind_ms: float
    curtailment_factor: float
    pv_size_factor: float
    kappa: float
    t_cell_c: float
    p_pv_mw: float
    q_pv_mvar: float
    q_over_p: float
    converged: bool
    iterations: int
    residual_norm: float
    p_slack_mw: float
    q_slack_mvar: float
    p_export_mw: float
    vm_mv_bus_2_pu: float
    max_vm_pu: float
    max_vm_bus: str
    total_p_loss_mw: float
    total_q_loss_mvar: float
    p_trafo_hv_mw: float
    q_trafo_hv_mvar: float
    s_trafo_hv_mva: float
    trafo_loading_proxy: float
    max_line_s_mva: float
    max_line_id: int
    max_line_name: str
    disabled_lines_due_to_open_switches: str
    disabled_trafos_due_to_open_switches: str
    delta_p_slack_vs_no_pv_mw: float
    delta_p_export_vs_no_pv_mw: float
    delta_vm_mv_bus_2_vs_no_pv_pu: float
    delta_total_p_loss_vs_no_pv_mw: float
    loss_delta_pct_vs_no_pv: float
    delta_s_trafo_hv_vs_no_pv_mva: float
    numeric_critical: bool
    export_warning: bool
    export_critical: bool
    export_severe: bool
    voltage_warning: bool
    voltage_critical_demo: bool
    pv_voltage_impact: bool
    loss_delta_warning: bool
    trafo_delta_warning: bool
    export_score: float
    voltage_score: float
    loss_score: float
    trafo_score: float
    criticality_score: float
    notes: str


@dataclass(frozen=True)
class TopCriticalCaseRow:
    rank: int
    case_id: str
    load_multiplier_mv_bus_2: float
    g_poa_wm2: float
    t_amb_c: float
    wind_ms: float
    p_pv_mw: float
    p_export_mw: float
    max_vm_pu: float
    vm_mv_bus_2_pu: float
    total_p_loss_mw: float
    s_trafo_hv_mva: float
    delta_vm_mv_bus_2_vs_no_pv_pu: float
    delta_total_p_loss_vs_no_pv_mw: float
    loss_delta_pct_vs_no_pv: float
    delta_s_trafo_hv_vs_no_pv_mva: float
    numeric_critical: bool
    export_critical: bool
    voltage_critical_demo: bool
    criticality_score: float


@dataclass(frozen=True)
class SensitivityRow:
    case_id: str
    top20_rank: int
    input_parameter: str
    input_unit: str
    observable: str
    observable_unit: str
    value: float
    ad_converged: bool
    load_multiplier_mv_bus_2: float
    g_poa_wm2: float
    t_amb_c: float
    wind_ms: float
    curtailment_factor: float
    pv_size_factor: float
    kappa: float


@dataclass(frozen=True)
class BranchFlowRow:
    case_id: str
    case_type: str
    load_multiplier_mv_bus_2: float
    pp_line_idx: int
    compiled_line_idx: int
    line_name: str
    from_bus_name: str
    to_bus_name: str
    is_mv_branch: bool
    active: bool
    p_from_mw: float
    q_from_mvar: float
    s_from_mva: float
    p_to_mw: float
    q_to_mvar: float
    s_to_mva: float
    max_end_s_mva: float
    notes: str


@dataclass(frozen=True)
class RunSummaryRow:
    metric: str
    value: float
    unit: str
    notes: str


SCREENING_COLUMNS = tuple(f.name for f in fields(ScreeningRow))
TOP_CRITICAL_COLUMNS = tuple(f.name for f in fields(TopCriticalCaseRow))
SENSITIVITY_COLUMNS = tuple(f.name for f in fields(SensitivityRow))
BRANCH_FLOW_COLUMNS = tuple(f.name for f in fields(BranchFlowRow))
RUN_SUMMARY_COLUMNS = tuple(f.name for f in fields(RunSummaryRow))


def _safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _is_finite(value: float) -> bool:
    return math.isfinite(float(value))


def _format_level(value: float) -> str:
    text = f"{value:.2f}".replace("-", "neg").replace(".", "p")
    return text.rstrip("0").rstrip("p") if "p" in text else text


def screening_case_count() -> int:
    """Return the number of PV screening cases, excluding no-PV references."""

    return (
        len(LOAD_MULTIPLIERS_MV_BUS_2)
        * len(G_POA_LEVELS_WM2)
        * len(T_AMB_LEVELS_C)
    )


def no_pv_reference_count() -> int:
    """Return the number of no-PV reference cases."""

    return len(LOAD_MULTIPLIERS_MV_BUS_2)


def _collect_switch_info(net) -> tuple[dict[int, int], set[int], set[int]]:
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


def _make_initial_state(net, spec) -> PFState:
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


def _internal_idx_for_original_bus(spec, bus_to_repr: dict[int, int], original_bus: int) -> int:
    repr_bus = bus_to_repr[int(original_bus)]
    for idx, bus in enumerate(spec.buses):
        if int(bus.name) == repr_bus:
            return idx
    raise ValueError(f"Original bus {original_bus} / repr {repr_bus} not found.")


def _bus_names_from_spec(net, spec) -> tuple[str, ...]:
    names: list[str] = []
    for bus in spec.buses:
        bus_id = int(bus.name)
        names.append(str(net.bus.loc[bus_id, "name"]))
    return tuple(names)


def _line_metadata(net, spec, bus_to_repr: dict[int, int], disabled_lines: set[int]) -> tuple[LineMeta, ...]:
    repr_to_idx = {int(bus.name): idx for idx, bus in enumerate(spec.buses)}
    meta: list[LineMeta] = []
    compiled_idx = 0

    for line_idx, row in net.line.iterrows():
        if not bool(row["in_service"]):
            continue
        if int(line_idx) in disabled_lines:
            continue

        from_orig = int(row["from_bus"])
        to_orig = int(row["to_bus"])
        from_repr = bus_to_repr[from_orig]
        to_repr = bus_to_repr[to_orig]
        if from_repr == to_repr:
            continue

        from_kv = float(net.bus.loc[from_orig, "vn_kv"])
        to_kv = float(net.bus.loc[to_orig, "vn_kv"])
        line_name = str(row.get("name", f"line_{line_idx}"))
        meta.append(
            LineMeta(
                compiled_line_idx=compiled_idx,
                pp_line_idx=int(line_idx),
                line_name=line_name,
                from_bus_name=str(net.bus.loc[from_orig, "name"]),
                to_bus_name=str(net.bus.loc[to_orig, "name"]),
                from_bus_internal=repr_to_idx[from_repr],
                to_bus_internal=repr_to_idx[to_repr],
                is_mv_branch=from_kv < 100.0 and to_kv < 100.0,
            )
        )
        compiled_idx += 1

    return tuple(meta)


def build_scenario_base(load_multiplier: float) -> ScenarioBase:
    """Build one scope-matched ``example_simple`` base without target PV sgen."""

    net = pn.example_simple()

    bus_matches = net.bus[net.bus["name"] == PV_COUPLING_BUS_NAME]
    if len(bus_matches) != 1:
        raise ValueError(f"Expected one bus named {PV_COUPLING_BUS_NAME!r}.")
    pv_original_bus = int(bus_matches.index[0])

    load_matches = net.load[
        (net.load["bus"] == pv_original_bus)
        & (net.load["in_service"] == True)  # noqa: E712
    ]
    if len(load_matches) != 1:
        raise ValueError(
            f"Expected one active load at {PV_COUPLING_BUS_NAME!r}, got {len(load_matches)}."
        )
    load_idx = int(load_matches.index[0])
    net.load.at[load_idx, "scaling"] = (
        float(net.load.at[load_idx, "scaling"]) * load_multiplier
    )

    target_sgen = net.sgen[
        (net.sgen["name"] == PV_COUPLING_SGEN_NAME)
        & (net.sgen["in_service"] == True)  # noqa: E712
    ]
    if len(target_sgen) != 1:
        raise ValueError(
            f"Expected one active sgen named {PV_COUPLING_SGEN_NAME!r}, got {len(target_sgen)}."
        )
    net.sgen.at[int(target_sgen.index[0]), "in_service"] = False

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

    bus_to_repr, disabled_lines, disabled_trafos = _collect_switch_info(net)
    spec = from_pandapower(net)
    topology, params = compile_network(spec)
    state0 = _make_initial_state(net, spec)
    pv_bus_internal_idx = _internal_idx_for_original_bus(spec, bus_to_repr, pv_original_bus)
    bus_names = _bus_names_from_spec(net, spec)
    line_meta = _line_metadata(net, spec, bus_to_repr, disabled_lines)
    s_base_mva = float(net.sn_mva) if float(net.sn_mva) > 0.0 else 1.0

    return ScenarioBase(
        load_multiplier=load_multiplier,
        topology=topology,
        params_base=params,
        state0=state0,
        pv_bus_internal_idx=pv_bus_internal_idx,
        trafo_idx=0,
        s_base_mva=s_base_mva,
        bus_names=bus_names,
        line_meta=line_meta,
        disabled_line_ids=tuple(sorted(disabled_lines)),
        disabled_trafo_ids=tuple(sorted(disabled_trafos)),
    )


def _pv_injection(g_poa_wm2, t_amb_c, wind_ms, curtailment_factor, pv_size_factor):
    alpha = jnp.asarray(curtailment_factor, dtype=jnp.float64) * jnp.asarray(
        pv_size_factor, dtype=jnp.float64
    )
    return pv_pq_injection_from_weather(
        g_poa_wm2=jnp.asarray(g_poa_wm2, dtype=jnp.float64),
        t_amb_c=jnp.asarray(t_amb_c, dtype=jnp.float64),
        wind_ms=jnp.asarray(wind_ms, dtype=jnp.float64),
        alpha=alpha,
        kappa=jnp.asarray(EXP5A_KAPPA, dtype=jnp.float64),
    )


def _params_for_case(
    scenario: ScenarioBase,
    g_poa_wm2,
    t_amb_c,
    wind_ms,
    curtailment_factor,
    pv_size_factor,
) -> tuple[NetworkParams, object]:
    injection = _pv_injection(
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


def _trafo_hv_complex_power_mva(
    scenario: ScenarioBase,
    params: NetworkParams,
    state: PFState,
) -> jnp.ndarray:
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
    return voltage[hv] * jnp.conj(current_hv) * scenario.s_base_mva


def _line_flow_records(
    scenario: ScenarioBase,
    params: NetworkParams,
    state: PFState,
    case_id: str,
    case_type: str,
) -> list[BranchFlowRow]:
    voltage = state_to_voltage(scenario.topology, params, state)
    rows: list[BranchFlowRow] = []
    for meta in scenario.line_meta:
        idx = meta.compiled_line_idx
        y_series = params.g_series_pu[idx] + 1j * params.b_series_pu[idx]
        b_shunt = params.b_shunt_pu[idx]
        v_from = voltage[meta.from_bus_internal]
        v_to = voltage[meta.to_bus_internal]
        i_from = (v_from - v_to) * y_series + 1j * (b_shunt / 2.0) * v_from
        i_to = (v_to - v_from) * y_series + 1j * (b_shunt / 2.0) * v_to
        s_from = v_from * jnp.conj(i_from) * scenario.s_base_mva
        s_to = v_to * jnp.conj(i_to) * scenario.s_base_mva
        s_from_abs = float(jnp.abs(s_from))
        s_to_abs = float(jnp.abs(s_to))
        rows.append(
            BranchFlowRow(
                case_id=case_id,
                case_type=case_type,
                load_multiplier_mv_bus_2=scenario.load_multiplier,
                pp_line_idx=meta.pp_line_idx,
                compiled_line_idx=meta.compiled_line_idx,
                line_name=meta.line_name,
                from_bus_name=meta.from_bus_name,
                to_bus_name=meta.to_bus_name,
                is_mv_branch=meta.is_mv_branch,
                active=True,
                p_from_mw=float(jnp.real(s_from)),
                q_from_mvar=float(jnp.imag(s_from)),
                s_from_mva=s_from_abs,
                p_to_mw=float(jnp.real(s_to)),
                q_to_mvar=float(jnp.imag(s_to)),
                s_to_mva=s_to_abs,
                max_end_s_mva=max(s_from_abs, s_to_abs),
                notes="Flow magnitude only; no line rating or loading percent is claimed.",
            )
        )
    return rows


def _solved_metrics(
    scenario: ScenarioBase,
    params: NetworkParams,
    state: PFState,
    case_id: str,
    case_type: str,
) -> tuple[dict, list[BranchFlowRow]]:
    voltage = state_to_voltage(scenario.topology, params, state)
    y_bus = build_ybus(scenario.topology, params)
    s_bus = calc_power_injection(y_bus, voltage)
    vm = jnp.abs(voltage)
    max_idx = int(jnp.argmax(vm))
    s_slack = s_bus[scenario.topology.slack_bus] * scenario.s_base_mva
    s_trafo = _trafo_hv_complex_power_mva(scenario, params, state)
    branch_rows = _line_flow_records(scenario, params, state, case_id, case_type)
    if branch_rows:
        max_branch = max(branch_rows, key=lambda row: row.max_end_s_mva)
        max_line_s = max_branch.max_end_s_mva
        max_line_id = max_branch.pp_line_idx
        max_line_name = max_branch.line_name
    else:
        max_line_s = float("nan")
        max_line_id = -1
        max_line_name = ""

    p_slack = float(jnp.real(s_slack))
    q_slack = float(jnp.imag(s_slack))
    total_p_loss = float(jnp.sum(jnp.real(s_bus)) * scenario.s_base_mva)
    total_q_loss = float(jnp.sum(jnp.imag(s_bus)) * scenario.s_base_mva)
    s_trafo_abs = float(jnp.abs(s_trafo))
    return (
        {
            "p_slack_mw": p_slack,
            "q_slack_mvar": q_slack,
            "p_export_mw": max(0.0, -p_slack),
            "vm_mv_bus_2_pu": float(vm[scenario.pv_bus_internal_idx]),
            "max_vm_pu": float(vm[max_idx]),
            "max_vm_bus": scenario.bus_names[max_idx],
            "total_p_loss_mw": total_p_loss,
            "total_q_loss_mvar": total_q_loss,
            "p_trafo_hv_mw": float(jnp.real(s_trafo)),
            "q_trafo_hv_mvar": float(jnp.imag(s_trafo)),
            "s_trafo_hv_mva": s_trafo_abs,
            "trafo_loading_proxy": s_trafo_abs / TRAFO_RATING_PROXY_MVA,
            "max_line_s_mva": max_line_s,
            "max_line_id": max_line_id,
            "max_line_name": max_line_name,
        },
        branch_rows,
    )


def _solve_case(
    scenario: ScenarioBase,
    case_id: str,
    case_type: str,
    g_poa_wm2: float,
    t_amb_c: float,
    wind_ms: float,
    curtailment_factor: float,
    pv_size_factor: float,
) -> tuple[dict, list[BranchFlowRow]]:
    if case_type == "no_pv_reference":
        injection_p = 0.0
        injection_q = 0.0
        t_cell = float(cell_temperature_noct_sam(0.0, t_amb_c, wind_ms))
        params = scenario.params_base
    else:
        params, injection = _params_for_case(
            scenario,
            g_poa_wm2,
            t_amb_c,
            wind_ms,
            curtailment_factor,
            pv_size_factor,
        )
        injection_p = float(injection.p_pv_mw)
        injection_q = float(injection.q_pv_mvar)
        t_cell = float(cell_temperature_noct_sam(g_poa_wm2, t_amb_c, wind_ms))

    try:
        result = solve_power_flow_result(
            scenario.topology,
            params,
            scenario.state0,
            NEWTON_OPTIONS,
        )
        converged = bool(result.converged)
        iterations = int(result.iterations)
        residual_norm = float(result.residual_norm)
        if converged:
            metrics, branch_rows = _solved_metrics(
                scenario,
                params,
                result.solution,
                case_id,
                case_type,
            )
        else:
            metrics, branch_rows = {}, []
    except Exception as exc:
        converged = False
        iterations = -1
        residual_norm = float("nan")
        metrics = {}
        branch_rows = []
        error = f"solve failed: {type(exc).__name__}: {exc}"
    else:
        error = ""

    base = {
        "case_id": case_id,
        "case_type": case_type,
        "load_multiplier_mv_bus_2": scenario.load_multiplier,
        "g_poa_wm2": g_poa_wm2,
        "t_amb_c": t_amb_c,
        "wind_ms": wind_ms,
        "curtailment_factor": curtailment_factor,
        "pv_size_factor": pv_size_factor,
        "kappa": EXP5A_KAPPA,
        "t_cell_c": t_cell,
        "p_pv_mw": injection_p,
        "q_pv_mvar": injection_q,
        "q_over_p": EXP5A_KAPPA,
        "converged": converged,
        "iterations": iterations,
        "residual_norm": residual_norm,
        "disabled_lines_due_to_open_switches": ",".join(map(str, scenario.disabled_line_ids)),
        "disabled_trafos_due_to_open_switches": ",".join(map(str, scenario.disabled_trafo_ids)),
        "notes": error,
    }
    for name in [
        "p_slack_mw",
        "q_slack_mvar",
        "p_export_mw",
        "vm_mv_bus_2_pu",
        "max_vm_pu",
        "max_vm_bus",
        "total_p_loss_mw",
        "total_q_loss_mvar",
        "p_trafo_hv_mw",
        "q_trafo_hv_mvar",
        "s_trafo_hv_mva",
        "trafo_loading_proxy",
        "max_line_s_mva",
        "max_line_id",
        "max_line_name",
    ]:
        if name in ("max_vm_bus", "max_line_name"):
            base[name] = metrics.get(name, "")
        elif name == "max_line_id":
            base[name] = int(metrics.get(name, -1))
        else:
            base[name] = float(metrics.get(name, float("nan")))
    return base, branch_rows


def _score_components(
    p_export_mw: float,
    max_vm_pu: float,
    loss_delta_pct_vs_no_pv: float,
    delta_s_trafo_hv_vs_no_pv_mva: float,
) -> tuple[float, float, float, float, float]:
    export_score = max(0.0, (p_export_mw - 7.00) / 0.25) if _is_finite(p_export_mw) else 0.0
    voltage_score = max(0.0, (max_vm_pu - 1.0110) / 0.001) if _is_finite(max_vm_pu) else 0.0
    loss_score = (
        max(0.0, loss_delta_pct_vs_no_pv / 0.10)
        if _is_finite(loss_delta_pct_vs_no_pv)
        else 0.0
    )
    trafo_score = (
        max(0.0, delta_s_trafo_hv_vs_no_pv_mva / 0.5)
        if _is_finite(delta_s_trafo_hv_vs_no_pv_mva)
        else 0.0
    )
    score = 3.0 * export_score + 2.0 * voltage_score + loss_score + trafo_score
    return export_score, voltage_score, loss_score, trafo_score, score


def _apply_reference_deltas(row: dict, reference: dict | None) -> dict:
    out = dict(row)
    if reference is None:
        out.update(
            {
                "delta_p_slack_vs_no_pv_mw": 0.0,
                "delta_p_export_vs_no_pv_mw": 0.0,
                "delta_vm_mv_bus_2_vs_no_pv_pu": 0.0,
                "delta_total_p_loss_vs_no_pv_mw": 0.0,
                "loss_delta_pct_vs_no_pv": 0.0,
                "delta_s_trafo_hv_vs_no_pv_mva": 0.0,
            }
        )
    else:
        p_loss_ref = reference["total_p_loss_mw"]
        loss_delta = out["total_p_loss_mw"] - p_loss_ref
        loss_delta_pct = (
            loss_delta / abs(p_loss_ref)
            if _is_finite(loss_delta) and _is_finite(p_loss_ref) and abs(p_loss_ref) > 0.0
            else float("nan")
        )
        out.update(
            {
                "delta_p_slack_vs_no_pv_mw": out["p_slack_mw"] - reference["p_slack_mw"],
                "delta_p_export_vs_no_pv_mw": out["p_export_mw"] - reference["p_export_mw"],
                "delta_vm_mv_bus_2_vs_no_pv_pu": (
                    out["vm_mv_bus_2_pu"] - reference["vm_mv_bus_2_pu"]
                ),
                "delta_total_p_loss_vs_no_pv_mw": loss_delta,
                "loss_delta_pct_vs_no_pv": loss_delta_pct,
                "delta_s_trafo_hv_vs_no_pv_mva": (
                    out["s_trafo_hv_mva"] - reference["s_trafo_hv_mva"]
                ),
            }
        )

    numeric_critical = (not bool(out["converged"])) or (
        _is_finite(out["residual_norm"]) and out["residual_norm"] > 1e-8
    )
    if not _is_finite(out["residual_norm"]):
        numeric_critical = True

    export_score, voltage_score, loss_score, trafo_score, criticality_score = _score_components(
        out["p_export_mw"],
        out["max_vm_pu"],
        out["loss_delta_pct_vs_no_pv"],
        out["delta_s_trafo_hv_vs_no_pv_mva"],
    )
    if out["case_type"] != "screening":
        criticality_score = 0.0
        export_score = voltage_score = loss_score = trafo_score = 0.0

    out.update(
        {
            "numeric_critical": numeric_critical,
            "export_warning": out["p_export_mw"] >= 6.75 if _is_finite(out["p_export_mw"]) else False,
            "export_critical": out["p_export_mw"] >= 7.00 if _is_finite(out["p_export_mw"]) else False,
            "export_severe": out["p_export_mw"] >= 7.25 if _is_finite(out["p_export_mw"]) else False,
            "voltage_warning": out["max_vm_pu"] >= 1.0110 if _is_finite(out["max_vm_pu"]) else False,
            "voltage_critical_demo": (
                out["max_vm_pu"] >= 1.0120 if _is_finite(out["max_vm_pu"]) else False
            ),
            "pv_voltage_impact": (
                out["delta_vm_mv_bus_2_vs_no_pv_pu"] >= 0.0010
                if _is_finite(out["delta_vm_mv_bus_2_vs_no_pv_pu"])
                else False
            ),
            "loss_delta_warning": (
                out["loss_delta_pct_vs_no_pv"] >= 0.10
                if _is_finite(out["loss_delta_pct_vs_no_pv"])
                else False
            ),
            "trafo_delta_warning": (
                out["delta_s_trafo_hv_vs_no_pv_mva"] >= 0.5
                if _is_finite(out["delta_s_trafo_hv_vs_no_pv_mva"])
                else False
            ),
            "export_score": export_score,
            "voltage_score": voltage_score,
            "loss_score": loss_score,
            "trafo_score": trafo_score,
            "criticality_score": criticality_score,
            "selected_for_sensitivity": False,
            "top20_rank": 0,
        }
    )
    return out


def _screening_sort_key(row: dict) -> tuple[float, float, float, float]:
    return (
        row["criticality_score"],
        row["p_export_mw"] if _is_finite(row["p_export_mw"]) else -1.0,
        row["max_vm_pu"] if _is_finite(row["max_vm_pu"]) else -1.0,
        row["p_pv_mw"] if _is_finite(row["p_pv_mw"]) else -1.0,
    )


def select_top_critical_cases(rows: list[dict], n_top: int = TOP_N_CRITICAL_CASES) -> list[dict]:
    """Return the highest-ranked screening rows by demonstrator stress score."""

    screening = [row for row in rows if row["case_type"] == "screening"]
    return sorted(screening, key=_screening_sort_key, reverse=True)[:n_top]


def _jnp_score(
    p_export_mw,
    max_vm_pu,
    total_p_loss_mw,
    s_trafo_hv_mva,
    reference: dict,
):
    export_score = jnp.maximum(0.0, (p_export_mw - 7.00) / 0.25)
    voltage_score = jnp.maximum(0.0, (max_vm_pu - 1.0110) / 0.001)
    ref_loss = reference["total_p_loss_mw"]
    loss_delta_pct = jnp.where(
        abs(ref_loss) > 0.0,
        (total_p_loss_mw - ref_loss) / abs(ref_loss),
        0.0,
    )
    loss_score = jnp.maximum(0.0, loss_delta_pct / 0.10)
    trafo_delta = s_trafo_hv_mva - reference["s_trafo_hv_mva"]
    trafo_score = jnp.maximum(0.0, trafo_delta / 0.5)
    return 3.0 * export_score + 2.0 * voltage_score + loss_score + trafo_score


def _sensitivity_scalar(
    curtailment_factor,
    scenario: ScenarioBase,
    row: dict,
    reference: dict,
    observable: str,
) -> jnp.ndarray:
    params, injection = _params_for_case(
        scenario,
        row["g_poa_wm2"],
        row["t_amb_c"],
        row["wind_ms"],
        curtailment_factor,
        row["pv_size_factor"],
    )
    solution = solve_power_flow_implicit(
        scenario.topology,
        params,
        scenario.state0,
        NEWTON_OPTIONS,
    )
    voltage = state_to_voltage(scenario.topology, params, solution)
    s_bus = calc_power_injection(build_ybus(scenario.topology, params), voltage)
    vm = jnp.abs(voltage)
    s_slack = s_bus[scenario.topology.slack_bus] * scenario.s_base_mva
    p_slack = jnp.real(s_slack)
    p_export = jnp.maximum(0.0, -p_slack)
    total_p_loss = jnp.sum(jnp.real(s_bus)) * scenario.s_base_mva
    s_trafo = _trafo_hv_complex_power_mva(scenario, params, solution)
    s_trafo_abs = jnp.abs(s_trafo)
    max_vm = jnp.max(vm)

    if observable == "p_pv_mw":
        return injection.p_pv_mw
    if observable == "q_pv_mvar":
        return injection.q_pv_mvar
    if observable == "p_slack_mw":
        return p_slack
    if observable == "minus_p_slack_mw":
        return -p_slack
    if observable == "p_export_mw":
        return p_export
    if observable == "vm_mv_bus_2_pu":
        return vm[scenario.pv_bus_internal_idx]
    if observable == "max_vm_pu":
        return max_vm
    if observable == "total_p_loss_mw":
        return total_p_loss
    if observable == "s_trafo_hv_mva":
        return s_trafo_abs
    if observable == "criticality_score":
        return _jnp_score(p_export, max_vm, total_p_loss, s_trafo_abs, reference)
    raise ValueError(f"Unknown sensitivity observable: {observable!r}")


def compute_sensitivity_rows(
    top_rows: list[dict],
    scenario_by_load: dict[float, ScenarioBase],
    reference_by_load: dict[float, dict],
) -> list[SensitivityRow]:
    """Compute local AD sensitivities for Top-20 cases only."""

    rows: list[SensitivityRow] = []
    unit_map = dict(SENSITIVITY_OBSERVABLES)
    for top_row in top_rows:
        scenario = scenario_by_load[top_row["load_multiplier_mv_bus_2"]]
        reference = reference_by_load[top_row["load_multiplier_mv_bus_2"]]
        for observable, _ in SENSITIVITY_OBSERVABLES:
            try:
                grad_val = jax.grad(
                    lambda curtailment: _sensitivity_scalar(
                        curtailment,
                        scenario,
                        top_row,
                        reference,
                        observable,
                    )
                )(jnp.asarray(CURTAILMENT_FACTOR, dtype=jnp.float64))
                value = float(grad_val)
                ok = math.isfinite(value)
            except Exception:
                value = float("nan")
                ok = False
            rows.append(
                SensitivityRow(
                    case_id=top_row["case_id"],
                    top20_rank=int(top_row["top20_rank"]),
                    input_parameter="curtailment_factor",
                    input_unit="dimensionless",
                    observable=observable,
                    observable_unit=unit_map[observable],
                    value=value,
                    ad_converged=ok,
                    load_multiplier_mv_bus_2=top_row["load_multiplier_mv_bus_2"],
                    g_poa_wm2=top_row["g_poa_wm2"],
                    t_amb_c=top_row["t_amb_c"],
                    wind_ms=top_row["wind_ms"],
                    curtailment_factor=CURTAILMENT_FACTOR,
                    pv_size_factor=top_row["pv_size_factor"],
                    kappa=EXP5A_KAPPA,
                )
            )
    return rows


def selected_realistic_case_spec() -> dict:
    """Return the fixed realistic summer high-PV case used by Experiment 5b."""

    return {
        "case_id": SELECTED_REALISTIC_CASE_ID,
        "case_type": "selected_realistic_case",
        "load_multiplier_mv_bus_2": SELECTED_REALISTIC_LOAD_MULTIPLIER,
        "g_poa_wm2": SELECTED_REALISTIC_G_POA_WM2,
        "t_amb_c": SELECTED_REALISTIC_T_AMB_C,
        "wind_ms": SELECTED_REALISTIC_WIND_MS,
        "curtailment_factor": CURTAILMENT_FACTOR,
        "pv_size_factor": PV_SIZE_FACTOR,
        "kappa": EXP5A_KAPPA,
    }


def build_no_pv_reference(load_multiplier: float) -> tuple[dict, ScenarioBase, list[BranchFlowRow]]:
    """Solve the no-PV reference for one load multiplier."""

    scenario = build_scenario_base(load_multiplier)
    case_id = f"ref_load{_format_level(load_multiplier)}_no_pv"
    row, branches = _solve_case(
        scenario,
        case_id=case_id,
        case_type="no_pv_reference",
        g_poa_wm2=0.0,
        t_amb_c=25.0,
        wind_ms=WIND_MS,
        curtailment_factor=CURTAILMENT_FACTOR,
        pv_size_factor=PV_SIZE_FACTOR,
    )
    return _apply_reference_deltas(row, None), scenario, branches


def solve_selected_realistic_case() -> tuple[ScreeningRow, ScenarioBase, dict, list[BranchFlowRow]]:
    """Solve the selected 30 degC summer high-PV case with no-PV deltas."""

    spec = selected_realistic_case_spec()
    reference, scenario, _ = build_no_pv_reference(spec["load_multiplier_mv_bus_2"])
    row, branches = _solve_case(
        scenario,
        case_id=spec["case_id"],
        case_type=spec["case_type"],
        g_poa_wm2=spec["g_poa_wm2"],
        t_amb_c=spec["t_amb_c"],
        wind_ms=spec["wind_ms"],
        curtailment_factor=spec["curtailment_factor"],
        pv_size_factor=spec["pv_size_factor"],
    )
    row = _apply_reference_deltas(row, reference)
    row["selected_for_sensitivity"] = False
    row["top20_rank"] = 0
    row["notes"] = (
        (row["notes"] + "; ") if row["notes"] else ""
    ) + "Selected realistic summer high-PV case for Exp. 5b."
    return ScreeningRow(**row), scenario, reference, branches


def compute_selected_realistic_sensitivity_rows(
    selected_row: ScreeningRow,
    scenario: ScenarioBase,
    reference: dict,
) -> list[SensitivityRow]:
    """Compute local curtailment sensitivities for the selected realistic case."""

    row_dict = asdict(selected_row)
    rows: list[SensitivityRow] = []
    unit_map = dict(SELECTED_REALISTIC_SENSITIVITY_OBSERVABLES)
    for observable, _ in SELECTED_REALISTIC_SENSITIVITY_OBSERVABLES:
        try:
            grad_val = jax.grad(
                lambda curtailment: _sensitivity_scalar(
                    curtailment,
                    scenario,
                    row_dict,
                    reference,
                    observable,
                )
            )(jnp.asarray(CURTAILMENT_FACTOR, dtype=jnp.float64))
            value = float(grad_val)
            ok = math.isfinite(value)
        except Exception:
            value = float("nan")
            ok = False
        rows.append(
            SensitivityRow(
                case_id=selected_row.case_id,
                top20_rank=0,
                input_parameter="curtailment_factor",
                input_unit="dimensionless",
                observable=observable,
                observable_unit=unit_map[observable],
                value=value,
                ad_converged=ok,
                load_multiplier_mv_bus_2=selected_row.load_multiplier_mv_bus_2,
                g_poa_wm2=selected_row.g_poa_wm2,
                t_amb_c=selected_row.t_amb_c,
                wind_ms=selected_row.wind_ms,
                curtailment_factor=selected_row.curtailment_factor,
                pv_size_factor=selected_row.pv_size_factor,
                kappa=selected_row.kappa,
            )
        )
    return rows


def _top_case_rows(top_rows: list[dict]) -> list[TopCriticalCaseRow]:
    return [
        TopCriticalCaseRow(
            rank=int(row["top20_rank"]),
            case_id=row["case_id"],
            load_multiplier_mv_bus_2=row["load_multiplier_mv_bus_2"],
            g_poa_wm2=row["g_poa_wm2"],
            t_amb_c=row["t_amb_c"],
            wind_ms=row["wind_ms"],
            p_pv_mw=row["p_pv_mw"],
            p_export_mw=row["p_export_mw"],
            max_vm_pu=row["max_vm_pu"],
            vm_mv_bus_2_pu=row["vm_mv_bus_2_pu"],
            total_p_loss_mw=row["total_p_loss_mw"],
            s_trafo_hv_mva=row["s_trafo_hv_mva"],
            delta_vm_mv_bus_2_vs_no_pv_pu=row["delta_vm_mv_bus_2_vs_no_pv_pu"],
            delta_total_p_loss_vs_no_pv_mw=row["delta_total_p_loss_vs_no_pv_mw"],
            loss_delta_pct_vs_no_pv=row["loss_delta_pct_vs_no_pv"],
            delta_s_trafo_hv_vs_no_pv_mva=row["delta_s_trafo_hv_vs_no_pv_mva"],
            numeric_critical=row["numeric_critical"],
            export_critical=row["export_critical"],
            voltage_critical_demo=row["voltage_critical_demo"],
            criticality_score=row["criticality_score"],
        )
        for row in top_rows
    ]


def _summary_rows(screening_rows: list[ScreeningRow], sensitivity_rows: list[SensitivityRow]) -> list[RunSummaryRow]:
    screening = [row for row in screening_rows if row.case_type == "screening"]
    refs = [row for row in screening_rows if row.case_type == "no_pv_reference"]
    converged = [row for row in screening_rows if row.converged]
    top_selected = [row for row in screening_rows if row.selected_for_sensitivity]
    finite_scores = [row.criticality_score for row in screening if math.isfinite(row.criticality_score)]
    finite_export = [row.p_export_mw for row in screening if math.isfinite(row.p_export_mw)]
    finite_vm = [row.max_vm_pu for row in screening if math.isfinite(row.max_vm_pu)]
    return [
        RunSummaryRow("n_screening_cases", float(len(screening)), "count", "PV screening cases only."),
        RunSummaryRow("n_no_pv_reference_cases", float(len(refs)), "count", "One per load multiplier."),
        RunSummaryRow("n_total_forward_cases", float(len(screening_rows)), "count", "Screening plus references."),
        RunSummaryRow("n_converged_forward_cases", float(len(converged)), "count", "Converged Newton solves."),
        RunSummaryRow("n_top20_cases", float(len(top_selected)), "count", "Cases selected for AD sensitivities."),
        RunSummaryRow(
            "n_sensitivity_rows",
            float(len(sensitivity_rows)),
            "count",
            "Top-20 cases times sensitivity observables.",
        ),
        RunSummaryRow(
            "max_criticality_score",
            max(finite_scores) if finite_scores else float("nan"),
            "dimensionless",
            "Demonstrator stress score; not a grid-code violation metric.",
        ),
        RunSummaryRow(
            "max_p_export_mw",
            max(finite_export) if finite_export else float("nan"),
            "MW",
            "p_export_mw = max(0, -p_slack_mw).",
        ),
        RunSummaryRow(
            "max_vm_pu",
            max(finite_vm) if finite_vm else float("nan"),
            "p.u.",
            "Maximum bus-voltage magnitude in screening cases.",
        ),
    ]


def append_selected_realistic_summary(
    summary_rows: list[RunSummaryRow],
    selected_row: ScreeningRow,
    selected_sensitivity_rows: list[SensitivityRow],
) -> list[RunSummaryRow]:
    """Append compact summary metrics for the selected realistic add-on case."""

    return summary_rows + [
        RunSummaryRow(
            "n_selected_realistic_cases",
            1.0,
            "count",
            "Separate add-on case; not part of the 48-case screening grid.",
        ),
        RunSummaryRow(
            "selected_realistic_p_export_mw",
            selected_row.p_export_mw,
            "MW",
            "Full-PV export for the Exp. 5b demonstration case.",
        ),
        RunSummaryRow(
            "selected_realistic_sensitivity_rows",
            float(len(selected_sensitivity_rows)),
            "count",
            "Local curtailment sensitivities for selected realistic case.",
        ),
    ]


def run_experiment() -> tuple[
    list[ScreeningRow],
    list[TopCriticalCaseRow],
    list[SensitivityRow],
    list[BranchFlowRow],
    list[RunSummaryRow],
    ScreeningRow,
    list[SensitivityRow],
]:
    """Run Experiment 5a and return all artifact rows."""

    scenario_by_load = {
        load: build_scenario_base(load) for load in LOAD_MULTIPLIERS_MV_BUS_2
    }
    raw_rows: list[dict] = []
    branch_rows: list[BranchFlowRow] = []
    reference_by_load: dict[float, dict] = {}

    for load, scenario in scenario_by_load.items():
        case_id = f"ref_load{_format_level(load)}_no_pv"
        row, branches = _solve_case(
            scenario,
            case_id=case_id,
            case_type="no_pv_reference",
            g_poa_wm2=0.0,
            t_amb_c=25.0,
            wind_ms=WIND_MS,
            curtailment_factor=CURTAILMENT_FACTOR,
            pv_size_factor=PV_SIZE_FACTOR,
        )
        row = _apply_reference_deltas(row, None)
        reference_by_load[load] = row
        raw_rows.append(row)
        branch_rows.extend(branches)

    for load, scenario in scenario_by_load.items():
        for g_poa in G_POA_LEVELS_WM2:
            for t_amb in T_AMB_LEVELS_C:
                case_id = (
                    f"screen_load{_format_level(load)}"
                    f"_g{_format_level(g_poa)}_t{_format_level(t_amb)}"
                )
                row, branches = _solve_case(
                    scenario,
                    case_id=case_id,
                    case_type="screening",
                    g_poa_wm2=g_poa,
                    t_amb_c=t_amb,
                    wind_ms=WIND_MS,
                    curtailment_factor=CURTAILMENT_FACTOR,
                    pv_size_factor=PV_SIZE_FACTOR,
                )
                row = _apply_reference_deltas(row, reference_by_load[load])
                raw_rows.append(row)
                branch_rows.extend(branches)

    top_raw = select_top_critical_cases(raw_rows, TOP_N_CRITICAL_CASES)
    top_ids = {row["case_id"]: rank for rank, row in enumerate(top_raw, start=1)}
    ranked_rows: list[dict] = []
    for row in raw_rows:
        updated = dict(row)
        rank = top_ids.get(row["case_id"], 0)
        updated["top20_rank"] = rank
        updated["selected_for_sensitivity"] = rank > 0
        ranked_rows.append(updated)
    top_ranked = sorted(
        [row for row in ranked_rows if row["selected_for_sensitivity"]],
        key=lambda row: row["top20_rank"],
    )

    sensitivity_rows = compute_sensitivity_rows(
        top_ranked,
        scenario_by_load,
        reference_by_load,
    )
    screening_rows = [ScreeningRow(**row) for row in ranked_rows]
    top_rows = _top_case_rows(top_ranked)
    summary_rows = _summary_rows(screening_rows, sensitivity_rows)
    selected_row, selected_scenario, selected_reference, _ = solve_selected_realistic_case()
    selected_sensitivity_rows = compute_selected_realistic_sensitivity_rows(
        selected_row,
        selected_scenario,
        selected_reference,
    )
    summary_rows = append_selected_realistic_summary(
        summary_rows,
        selected_row,
        selected_sensitivity_rows,
    )
    return (
        screening_rows,
        top_rows,
        sensitivity_rows,
        branch_rows,
        summary_rows,
        selected_row,
        selected_sensitivity_rows,
    )


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
        "experiment": "exp05a_network_screening",
        "purpose": (
            "Reduced forward screening for later PV-curtailment optimization; "
            "no optimization is performed."
        ),
        "network": "pandapower.networks.example_simple(), scope_matched",
        "coupling_bus": PV_COUPLING_BUS_NAME,
        "replaced_element": PV_COUPLING_SGEN_NAME,
        "scenario_design": {
            "load_multiplier_mv_bus_2": list(LOAD_MULTIPLIERS_MV_BUS_2),
            "g_poa_wm2": list(G_POA_LEVELS_WM2),
            "t_amb_c": list(T_AMB_LEVELS_C),
            "wind_ms": WIND_MS,
            "curtailment_factor": CURTAILMENT_FACTOR,
            "pv_size_factor": PV_SIZE_FACTOR,
            "kappa": EXP5A_KAPPA,
            "n_screening_cases": screening_case_count(),
            "n_no_pv_reference_cases": no_pv_reference_count(),
            "n_forward_cases_total": screening_case_count() + no_pv_reference_count(),
            "n_top_cases_for_sensitivity": TOP_N_CRITICAL_CASES,
            "selected_realistic_case": selected_realistic_case_spec(),
            "selected_realistic_case_scope": (
                "Separate add-on case for Exp. 5b; not part of the 48-case "
                "screening grid or Top-20 ranking."
            ),
        },
        "criticality_definition": {
            "note": (
                "Indicators are demonstrator-internal stress indicators, not "
                "normative grid-code or equipment-limit violations."
            ),
            "numeric_critical": "(converged == False) OR (residual_norm > 1e-8)",
            "export_warning": "p_export_mw >= 6.75",
            "export_critical": "p_export_mw >= 7.00",
            "export_severe": "p_export_mw >= 7.25",
            "voltage_warning": "max_vm_pu >= 1.0110",
            "voltage_critical_demo": "max_vm_pu >= 1.0120",
            "pv_voltage_impact": "delta_vm_mv_bus_2_vs_no_pv_pu >= 0.0010",
            "loss_delta_warning": "loss_delta_pct_vs_no_pv >= 0.10 if finite",
            "trafo_delta_warning": "delta_s_trafo_hv_vs_no_pv_mva >= 0.5 if finite",
            "score": (
                "3*max(0,(p_export-7.00)/0.25) + "
                "2*max(0,(max_vm-1.0110)/0.001) + "
                "loss_score + trafo_score"
            ),
        },
        "sensitivity_scope": {
            "input_parameter": "curtailment_factor",
            "input_unit": "dimensionless",
            "computed_only_for": "Top-20 screening cases by criticality_score",
            "observables": [
                {"name": name, "unit": unit} for name, unit in SENSITIVITY_OBSERVABLES
            ],
            "selected_realistic_case_observables": [
                {"name": name, "unit": unit}
                for name, unit in SELECTED_REALISTIC_SENSITIVITY_OBSERVABLES
            ],
        },
        "solver_options": {
            "max_iters": NEWTON_OPTIONS.max_iters,
            "tolerance": NEWTON_OPTIONS.tolerance,
            "damping": NEWTON_OPTIONS.damping,
            "initialization": "trafo_shift_aware",
        },
        "known_simplifications": [
            "PV plant is modelled as weather-dependent PQ injection, not as voltage-regulating PV bus.",
            "No Q limits, no PV-PQ switching, no controller logic.",
            "No thermal branch ratings are claimed; line quantities are apparent-power flow proxies only.",
            "Transformer loading proxy uses s_trafo_hv_mva / 25.0 based on the example_simple transformer rating.",
            "The active gen is converted to sgen(P, Q=0) in the scope-matched model.",
            "Top-20 sensitivities are local derivatives at curtailment_factor = 1.0.",
        ],
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def write_readme(results_dir: Path) -> None:
    text = """# Experiment 5a - Network Screening for PV Curtailment Preparation

This directory contains a reduced network screening on
`pandapower.networks.example_simple()` in the existing scope-matched model.
The `sgen "static generator"` at `"MV Bus 2"` is disabled and replaced by the
JAX-compatible PV/weather PQ model.

The experiment does not run an optimization. It only identifies stressed
operating points for a later PV-curtailment experiment.

## Artifacts

- `screening_results.csv/json`: one row per forward case, including no-PV
  references, deltas, stress indicators, and criticality scores.
- `top_critical_cases.csv/json`: Top-20 screening cases by demonstrator stress
  score.
- `sensitivity_top20.csv/json`: local AD sensitivities with respect to
  `curtailment_factor`, computed only for the Top-20 cases.
- `selected_realistic_case.csv/json`: a separate summer high-PV add-on case
  with `load_multiplier = 0.4`, `G = 1200 W/m2`, and `T_amb = 30 degC`.
- `selected_realistic_case_sensitivity.csv/json`: local AD sensitivities for
  the selected add-on case; this is the hand-off point for Experiment 5b.
- `branch_flows.csv/json`: active line apparent-power flow proxies; no line
  loading percentages are claimed.
- `run_summary.csv/json`: compact counts and extrema.
- `metadata.json`: reproducibility metadata and criticality definitions.

## Selected realistic case

The original screening grid remains unchanged: 48 PV screening cases plus four
no-PV references. The `G = 1200 W/m2`, `T_amb = -10 degC` case remains useful
as a mathematical stress point, because cold PV modules yield high active power.
For the curtailment narrative in Experiment 5b, this add-on additionally solves
`selected_realistic_load0p4_g1200_t30`, a warmer summer high-PV case with low
local load. It is more plausible as a demonstrator for a real network-operator
export-management question while still preserving the stress behavior needed
for optimization.

## Important scope note

Criticality flags are demonstrator-internal stress indicators. They are not
normative voltage, export, or thermal-limit violations. The current model has
no controller logic, no Q limits, no PV-PQ switching, and no validated thermal
branch ratings.
"""
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "README.md").write_text(text, encoding="utf-8")


def export_all(
    screening_rows: list[ScreeningRow],
    top_rows: list[TopCriticalCaseRow],
    sensitivity_rows: list[SensitivityRow],
    branch_rows: list[BranchFlowRow],
    summary_rows: list[RunSummaryRow],
    results_dir: Path,
    selected_rows: list[ScreeningRow] | None = None,
    selected_sensitivity_rows: list[SensitivityRow] | None = None,
) -> None:
    selected_rows = selected_rows or []
    selected_sensitivity_rows = selected_sensitivity_rows or []
    _write_csv(results_dir / "screening_results.csv", screening_rows, SCREENING_COLUMNS)
    _write_json(results_dir / "screening_results.json", screening_rows)
    _write_csv(results_dir / "top_critical_cases.csv", top_rows, TOP_CRITICAL_COLUMNS)
    _write_json(results_dir / "top_critical_cases.json", top_rows)
    _write_csv(results_dir / "sensitivity_top20.csv", sensitivity_rows, SENSITIVITY_COLUMNS)
    _write_json(results_dir / "sensitivity_top20.json", sensitivity_rows)
    _write_csv(results_dir / "branch_flows.csv", branch_rows, BRANCH_FLOW_COLUMNS)
    _write_json(results_dir / "branch_flows.json", branch_rows)
    _write_csv(results_dir / "run_summary.csv", summary_rows, RUN_SUMMARY_COLUMNS)
    _write_json(results_dir / "run_summary.json", summary_rows)
    _write_csv(results_dir / "selected_realistic_case.csv", selected_rows, SCREENING_COLUMNS)
    _write_json(results_dir / "selected_realistic_case.json", selected_rows)
    _write_csv(
        results_dir / "selected_realistic_case_sensitivity.csv",
        selected_sensitivity_rows,
        SENSITIVITY_COLUMNS,
    )
    _write_json(
        results_dir / "selected_realistic_case_sensitivity.json",
        selected_sensitivity_rows,
    )
    write_metadata(results_dir)
    write_readme(results_dir)


def main() -> None:
    print("=" * 72)
    print("Experiment 5a: reduced network screening")
    print(f"Results directory: {RESULTS_DIR}")
    print("=" * 72)
    (
        screening_rows,
        top_rows,
        sensitivity_rows,
        branch_rows,
        summary_rows,
        selected_row,
        selected_sensitivity_rows,
    ) = run_experiment()
    export_all(
        screening_rows,
        top_rows,
        sensitivity_rows,
        branch_rows,
        summary_rows,
        RESULTS_DIR,
        [selected_row],
        selected_sensitivity_rows,
    )
    print("\nRun summary:")
    for row in summary_rows:
        print(f"  {row.metric:<30} {row.value:g} {row.unit}")
    print("\nTop critical cases:")
    for row in top_rows[:5]:
        print(
            f"  #{row.rank:02d} {row.case_id:<35} "
            f"score={row.criticality_score:.3f} "
            f"p_export={row.p_export_mw:.3f} MW max_vm={row.max_vm_pu:.5f}"
        )
    print("\nSelected realistic case for Experiment 5b:")
    print(
        f"  {selected_row.case_id:<35} "
        f"p_export={selected_row.p_export_mw:.3f} MW "
        f"p_pv={selected_row.p_pv_mw:.3f} MW "
        f"vm_mv_bus_2={selected_row.vm_mv_bus_2_pu:.5f} p.u."
    )
    print("\nExported artifacts:")
    for name in REQUIRED_ARTIFACTS:
        print(f"  {name}")


if __name__ == "__main__":
    main()
