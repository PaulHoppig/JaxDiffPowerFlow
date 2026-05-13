"""Experiment 1b – Elektrische Solver-Validierung: pandapower example_simple()

Zwei Referenzmodi
-----------------
scope_matched        – gen→sgen(Q=0); pandapower und diffpf verwenden dasselbe
                       Modell; strikter numerischer Vergleich möglich.
original_pandapower  – originales pp-Netz mit PV-Bus; Kontextvergleich, da
                       diffpf aktuell kein Q-Limit-Enforcement hat.

Szenarien
---------
base, load_low, load_high, sgen_low, sgen_high,
combined_high_load_low_sgen, combined_low_load_high_sgen

Aufruf
------
    python experiments/exp01_validate_example_simple.py
"""

from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandapower as pp
import pandapower.networks as pn

from diffpf.compile.network import compile_network
from diffpf.core.residuals import calc_power_injection, state_to_voltage
from diffpf.core.types import NetworkParams, PFState
from diffpf.core.ybus import build_ybus
from diffpf.io.pandapower_adapter import from_pandapower
from diffpf.io.topology_utils import merge_buses
from diffpf.solver.newton import NewtonOptions, solve_power_flow_result

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR = (
    Path(__file__).resolve().parent / "results" / "exp01_example_simple_validation"
)

REFERENCE_MODES = ("scope_matched", "original_pandapower")

SCENARIOS: tuple[tuple[str, float, float], ...] = (
    # (name, load_factor, sgen_factor)
    ("base", 1.00, 1.00),
    ("load_low", 0.75, 1.00),
    ("load_high", 1.25, 1.00),
    ("sgen_low", 1.00, 0.50),
    ("sgen_high", 1.00, 1.50),
    ("combined_high_load_low_sgen", 1.25, 0.50),
    ("combined_low_load_high_sgen", 0.75, 1.50),
)

NEWTON_OPTIONS = NewtonOptions(max_iters=50, tolerance=1e-10, damping=0.7)

PP_RUNPP_KWARGS: dict = dict(
    algorithm="nr",
    calculate_voltage_angles=True,
    init="dc",   # flat start fails for 150° phase-shifting trafo
    tolerance_mva=1e-9,
    max_iteration=50,
    trafo_model="pi",
    numba=False,
)

INIT_STRATEGY = "trafo_shift_aware"


# ---------------------------------------------------------------------------
# Result row dataclasses (tidy / long format)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SummaryRow:
    scenario: str
    reference_mode: str
    diffpf_converged: bool
    pandapower_converged: bool
    diffpf_iterations: int
    diffpf_residual_norm: float
    init_strategy: str
    strict_validation: bool
    max_vm_pu_abs_diff: float
    rmse_vm_pu: float
    max_va_degree_abs_diff: float
    rmse_va_degree: float
    max_line_p_mw_abs_diff: float
    max_line_q_mvar_abs_diff: float
    max_trafo_p_mw_abs_diff: float
    max_trafo_q_mvar_abs_diff: float
    p_slack_mw_abs_diff: float
    q_slack_mvar_abs_diff: float
    total_p_loss_mw_abs_diff: float
    total_q_loss_mvar_abs_diff: float
    notes: str


@dataclass(frozen=True)
class BusResultRow:
    scenario: str
    reference_mode: str
    bus_original_id: int
    bus_internal_id: int
    bus_name: str
    vn_kv: float
    is_slack: bool
    is_pv_like: bool
    vm_pu_diffpf: float
    vm_pu_pp: float
    vm_pu_abs_diff: float
    va_degree_diffpf: float
    va_degree_pp: float
    va_degree_abs_diff: float
    validation_scope: str


@dataclass(frozen=True)
class SlackResultRow:
    scenario: str
    reference_mode: str
    p_slack_mw_diffpf: float
    p_slack_mw_pp: float
    p_slack_mw_abs_diff: float
    q_slack_mvar_diffpf: float
    q_slack_mvar_pp: float
    q_slack_mvar_abs_diff: float


@dataclass(frozen=True)
class LineFlowRow:
    scenario: str
    reference_mode: str
    line_name: str
    pp_line_idx: int
    from_bus_name: str
    to_bus_name: str
    in_service_after_switch_handling: bool
    p_from_mw_diffpf: float
    p_from_mw_pp: float
    p_from_mw_abs_diff: float
    q_from_mvar_diffpf: float
    q_from_mvar_pp: float
    q_from_mvar_abs_diff: float
    p_to_mw_diffpf: float
    p_to_mw_pp: float
    p_to_mw_abs_diff: float
    q_to_mvar_diffpf: float
    q_to_mvar_pp: float
    q_to_mvar_abs_diff: float
    pl_mw_diffpf: float
    pl_mw_pp: float
    pl_mw_abs_diff: float


@dataclass(frozen=True)
class TrafoFlowRow:
    scenario: str
    reference_mode: str
    trafo_name: str
    pp_trafo_idx: int
    hv_bus_name: str
    lv_bus_name: str
    p_hv_mw_diffpf: float
    p_hv_mw_pp: float
    p_hv_mw_abs_diff: float
    q_hv_mvar_diffpf: float
    q_hv_mvar_pp: float
    q_hv_mvar_abs_diff: float
    p_lv_mw_diffpf: float
    p_lv_mw_pp: float
    p_lv_mw_abs_diff: float
    q_lv_mvar_diffpf: float
    q_lv_mvar_pp: float
    q_lv_mvar_abs_diff: float
    pl_mw_diffpf: float
    pl_mw_pp: float
    pl_mw_abs_diff: float


@dataclass(frozen=True)
class LossRow:
    scenario: str
    reference_mode: str
    total_p_loss_mw_diffpf: float
    total_p_loss_mw_pp: float
    total_p_loss_mw_abs_diff: float
    total_q_loss_mvar_diffpf: float
    total_q_loss_mvar_pp: float
    total_q_loss_mvar_abs_diff: float
    line_p_loss_mw_diffpf: float
    line_p_loss_mw_pp: float
    trafo_p_loss_mw_diffpf: float
    trafo_p_loss_mw_pp: float


@dataclass(frozen=True)
class StructureSummaryRow:
    scenario: str
    reference_mode: str
    number_of_original_buses: int
    number_of_internal_buses_after_fusion: int
    number_of_lines_original: int
    number_of_lines_active_after_switches: int
    number_of_trafos_active: int
    number_of_shunts: int
    number_of_loads: int
    number_of_sgens: int
    number_of_gens: int
    bus_fusion_groups: str
    disabled_lines_due_to_open_switches: str
    disabled_trafos_due_to_open_switches: str


# ---------------------------------------------------------------------------
# Network manipulation helpers
# ---------------------------------------------------------------------------


def make_scenario_net(
    load_factor: float,
    sgen_factor: float,
) -> pp.pandapowerNet:
    """Build a fresh example_simple() with scaled load/sgen."""
    net = pn.example_simple()
    for idx in net.load.index:
        net.load.at[idx, "scaling"] = float(net.load.at[idx, "scaling"]) * load_factor
    for idx in net.sgen.index:
        net.sgen.at[idx, "scaling"] = float(net.sgen.at[idx, "scaling"]) * sgen_factor
    return net


def convert_gen_to_sgen(net: pp.pandapowerNet) -> pp.pandapowerNet:
    """Return a deep copy where each active gen is replaced by sgen(P, Q=0)."""
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


def _build_switch_info(
    net: pp.pandapowerNet,
) -> tuple[dict[int, int], set[int], set[int]]:
    """Replicate switch processing from from_pandapower to get bus_to_repr mapping."""
    all_bus_ids = list(net.bus.index)
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

    bus_to_repr = merge_buses(all_bus_ids, bb_pairs)
    return bus_to_repr, disabled_lines, disabled_trafos


def _active_pp_line_indices(
    net: pp.pandapowerNet,
    bus_to_repr: dict[int, int],
    disabled_lines: set[int],
) -> list[int]:
    """Ordered list of pandapower line indices that map to spec.lines (same order)."""
    active = []
    for idx, row in net.line.iterrows():
        if not bool(row["in_service"]):
            continue
        if idx in disabled_lines:
            continue
        if bus_to_repr[int(row["from_bus"])] == bus_to_repr[int(row["to_bus"])]:
            continue
        active.append(idx)
    return active


def _active_pp_trafo_indices(
    net: pp.pandapowerNet,
    bus_to_repr: dict[int, int],
    disabled_trafos: set[int],
) -> list[int]:
    """Ordered list of pandapower trafo indices that map to spec.trafos (same order)."""
    active = []
    for idx, row in net.trafo.iterrows():
        if not bool(row["in_service"]):
            continue
        if idx in disabled_trafos:
            continue
        if bus_to_repr[int(row["hv_bus"])] == bus_to_repr[int(row["lv_bus"])]:
            continue
        active.append(idx)
    return active


# ---------------------------------------------------------------------------
# Smart initialisation (trafo-shift-aware)
# ---------------------------------------------------------------------------


def make_smart_initial_state(
    net: pp.pandapowerNet,
    spec,
) -> PFState:
    """
    Initialise non-slack buses based on voltage level.

    HV buses (vn_kv >= 100 kV): angle = slack angle
    LV buses: angle = slack angle − trafo_shift_deg

    This avoids the divergence that occurs with flat start when the trafo
    phase shift is large (example_simple uses 150°).
    """
    slack_ang_rad = float(net.ext_grid.iloc[0]["va_degree"]) * math.pi / 180.0
    trafo_shift_deg = float(net.trafo.iloc[0]["shift_degree"])
    lv_ang_rad = (
        float(net.ext_grid.iloc[0]["va_degree"]) - trafo_shift_deg
    ) * math.pi / 180.0

    vr_list: list[float] = []
    vi_list: list[float] = []
    for b in spec.buses:
        if b.is_slack:
            continue
        bus_id = int(b.name)
        vn_kv = float(net.bus.loc[bus_id, "vn_kv"])
        ang = slack_ang_rad if vn_kv >= 100.0 else lv_ang_rad
        vr_list.append(math.cos(ang))
        vi_list.append(math.sin(ang))

    return PFState(
        vr_pu=np.array(vr_list, dtype=np.float64),
        vi_pu=np.array(vi_list, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Per-trafo flow computation from diffpf solution
# ---------------------------------------------------------------------------


class _TrafoFlow(NamedTuple):
    p_hv_mw: float
    q_hv_mvar: float
    p_lv_mw: float
    q_lv_mvar: float
    pl_mw: float
    ql_mvar: float


def _trafo_flows_from_voltage(
    voltage: np.ndarray,
    params: NetworkParams,
    s_base_mva: float,
) -> list[_TrafoFlow]:
    """
    Compute per-trafo power flows using the Pi-model admittance stamps.

    Sign convention matches pandapower res_trafo:
      p_hv_mw > 0  ↔  power flows from HV bus into transformer
      p_lv_mw < 0  ↔  power flows from transformer into LV bus
    """
    results: list[_TrafoFlow] = []
    for k in range(len(params.trafo_hv_bus)):
        hv = params.trafo_hv_bus[k]
        lv = params.trafo_lv_bus[k]
        V_hv = complex(voltage[hv])
        V_lv = complex(voltage[lv])
        y_t = complex(
            float(params.trafo_g_series_pu[k]), float(params.trafo_b_series_pu[k])
        )
        y_m = complex(
            float(params.trafo_g_mag_pu[k]), float(params.trafo_b_mag_pu[k])
        )
        a = float(params.trafo_tap_ratio[k])
        phi = float(params.trafo_shift_rad[k])
        t = a * np.exp(1j * phi)
        t_conj = np.conj(t)
        a2 = a * a

        # Y-bus contributions (same as in ybus.py):
        # Y[hv,hv] = (y_t + y_m) / a²,  Y[hv,lv] = -y_t / conj(t)
        # Y[lv,hv] = -y_t / t,           Y[lv,lv] = y_t + y_m
        #
        # I_* = current consumed FROM bus * by the trafo (load convention).
        # S_* = V_* * conj(I_*) = complex power consumed from that bus.
        # pandapower p_hv_mw > 0 ↔ power flows from bus into trafo → Re(S_hv).
        I_inj_hv = (y_t + y_m) / a2 * V_hv + (-y_t / t_conj) * V_lv
        I_inj_lv = (-y_t / t) * V_hv + (y_t + y_m) * V_lv

        S_from_hv = V_hv * np.conj(I_inj_hv)
        S_from_lv = V_lv * np.conj(I_inj_lv)

        results.append(
            _TrafoFlow(
                p_hv_mw=float(np.real(S_from_hv)) * s_base_mva,
                q_hv_mvar=float(np.imag(S_from_hv)) * s_base_mva,
                p_lv_mw=float(np.real(S_from_lv)) * s_base_mva,
                q_lv_mvar=float(np.imag(S_from_lv)) * s_base_mva,
                pl_mw=(float(np.real(S_from_hv)) + float(np.real(S_from_lv)))
                * s_base_mva,
                ql_mvar=(float(np.imag(S_from_hv)) + float(np.imag(S_from_lv)))
                * s_base_mva,
            )
        )
    return results


# ---------------------------------------------------------------------------
# Core solve function
# ---------------------------------------------------------------------------


class _DiffpfSolution(NamedTuple):
    converged: bool
    iterations: int
    residual_norm: float
    voltage: np.ndarray         # complex, full n_bus
    s_injection: np.ndarray     # complex, full n_bus
    s_base_mva: float


class _PPSolution(NamedTuple):
    converged: bool
    net: pp.pandapowerNet


def solve_diffpf(net_for_diffpf: pp.pandapowerNet) -> _DiffpfSolution:
    """Convert net → NetworkSpec → solve. Uses trafo-shift-aware initialisation."""
    s_base_mva = float(net_for_diffpf.sn_mva) if float(net_for_diffpf.sn_mva) > 0 else 1.0
    spec = from_pandapower(net_for_diffpf)
    topology, params = compile_network(spec)

    state0 = make_smart_initial_state(net_for_diffpf, spec)

    import jax.numpy as jnp
    state0_jax = PFState(
        vr_pu=jnp.array(state0.vr_pu, dtype=jnp.float64),
        vi_pu=jnp.array(state0.vi_pu, dtype=jnp.float64),
    )

    result = solve_power_flow_result(topology, params, state0_jax, NEWTON_OPTIONS)

    voltage = np.asarray(state_to_voltage(topology, params, result.solution))
    y_bus = build_ybus(topology, params)
    s_injection = np.asarray(calc_power_injection(y_bus, voltage))

    return _DiffpfSolution(
        converged=bool(result.converged),
        iterations=int(result.iterations),
        residual_norm=float(result.residual_norm),
        voltage=voltage,
        s_injection=s_injection,
        s_base_mva=s_base_mva,
    )


def solve_pandapower(net: pp.pandapowerNet) -> _PPSolution:
    try:
        pp.runpp(net, **PP_RUNPP_KWARGS)
        return _PPSolution(converged=bool(net.converged), net=net)
    except Exception:
        return _PPSolution(converged=False, net=net)


# ---------------------------------------------------------------------------
# Result extraction
# ---------------------------------------------------------------------------


def _extract_bus_rows(
    scenario: str,
    ref_mode: str,
    net_pp: pp.pandapowerNet,
    spec,
    diffpf_sol: _DiffpfSolution,
    pp_sol: _PPSolution,
    gen_bus_repr_ids: set[int],
) -> list[BusResultRow]:
    rows: list[BusResultRow] = []
    validation_scope = "strict" if ref_mode == "scope_matched" else "contextual"

    for internal_id, bus_spec in enumerate(spec.buses):
        repr_id = int(bus_spec.name)
        vn_kv = float(net_pp.bus.loc[repr_id, "vn_kv"])

        # diffpf voltage
        V_diffpf = complex(diffpf_sol.voltage[internal_id])
        vm_diffpf = abs(V_diffpf)
        va_diffpf = math.degrees(math.atan2(V_diffpf.imag, V_diffpf.real))

        # pandapower voltage at representative bus
        if pp_sol.converged and repr_id in pp_sol.net.res_bus.index:
            vm_pp = float(pp_sol.net.res_bus.loc[repr_id, "vm_pu"])
            va_pp = float(pp_sol.net.res_bus.loc[repr_id, "va_degree"])
        else:
            vm_pp = float("nan")
            va_pp = float("nan")

        rows.append(
            BusResultRow(
                scenario=scenario,
                reference_mode=ref_mode,
                bus_original_id=repr_id,
                bus_internal_id=internal_id,
                bus_name=bus_spec.name,
                vn_kv=vn_kv,
                is_slack=bus_spec.is_slack,
                is_pv_like=(repr_id in gen_bus_repr_ids),
                vm_pu_diffpf=vm_diffpf,
                vm_pu_pp=vm_pp,
                vm_pu_abs_diff=abs(vm_diffpf - vm_pp) if not math.isnan(vm_pp) else float("nan"),
                va_degree_diffpf=va_diffpf,
                va_degree_pp=va_pp,
                va_degree_abs_diff=abs(va_diffpf - va_pp) if not math.isnan(va_pp) else float("nan"),
                validation_scope=validation_scope,
            )
        )
    return rows


def _extract_slack_row(
    scenario: str,
    ref_mode: str,
    topology,
    diffpf_sol: _DiffpfSolution,
    pp_sol: _PPSolution,
) -> SlackResultRow:
    slack_idx = topology.slack_bus
    p_diffpf = float(np.real(diffpf_sol.s_injection[slack_idx])) * diffpf_sol.s_base_mva
    q_diffpf = float(np.imag(diffpf_sol.s_injection[slack_idx])) * diffpf_sol.s_base_mva

    if pp_sol.converged and len(pp_sol.net.res_ext_grid) > 0:
        p_pp = float(pp_sol.net.res_ext_grid.iloc[0]["p_mw"])
        q_pp = float(pp_sol.net.res_ext_grid.iloc[0]["q_mvar"])
    else:
        p_pp = float("nan")
        q_pp = float("nan")

    return SlackResultRow(
        scenario=scenario,
        reference_mode=ref_mode,
        p_slack_mw_diffpf=p_diffpf,
        p_slack_mw_pp=p_pp,
        p_slack_mw_abs_diff=abs(p_diffpf - p_pp) if not math.isnan(p_pp) else float("nan"),
        q_slack_mvar_diffpf=q_diffpf,
        q_slack_mvar_pp=q_pp,
        q_slack_mvar_abs_diff=abs(q_diffpf - q_pp) if not math.isnan(q_pp) else float("nan"),
    )


def _extract_line_rows(
    scenario: str,
    ref_mode: str,
    net_pp: pp.pandapowerNet,
    spec,
    params,
    topology,
    active_line_pp_idx: list[int],
    diffpf_sol: _DiffpfSolution,
    pp_sol: _PPSolution,
) -> list[LineFlowRow]:
    rows: list[LineFlowRow] = []
    s_base = diffpf_sol.s_base_mva
    voltage = diffpf_sol.voltage

    # Compute diffpf line flows from Pi-model
    # spec.lines ordering matches active_line_pp_idx ordering
    for spec_line_k, pp_idx in enumerate(active_line_pp_idx):
        line = spec.lines[spec_line_k]
        V_from = complex(voltage[line.from_bus])
        V_to = complex(voltage[line.to_bus])
        y_series = complex(
            float(params.g_series_pu[spec_line_k]),
            float(params.b_series_pu[spec_line_k]),
        )
        y_shunt_half = 0.5j * float(params.b_shunt_pu[spec_line_k])
        I_from = (V_from - V_to) * y_series + V_from * y_shunt_half
        I_to = (V_to - V_from) * y_series + V_to * y_shunt_half
        S_from = V_from * np.conj(I_from)
        S_to = V_to * np.conj(I_to)

        p_from_d = float(np.real(S_from)) * s_base
        q_from_d = float(np.imag(S_from)) * s_base
        p_to_d = float(np.real(S_to)) * s_base
        q_to_d = float(np.imag(S_to)) * s_base
        pl_d = (float(np.real(S_from)) + float(np.real(S_to))) * s_base

        if pp_sol.converged and pp_idx in pp_sol.net.res_line.index:
            rrow = pp_sol.net.res_line.loc[pp_idx]
            p_from_pp = float(rrow["p_from_mw"])
            q_from_pp = float(rrow["q_from_mvar"])
            p_to_pp = float(rrow["p_to_mw"])
            q_to_pp = float(rrow["q_to_mvar"])
            pl_pp = float(rrow["pl_mw"])
        else:
            p_from_pp = q_from_pp = p_to_pp = q_to_pp = pl_pp = float("nan")

        pp_row = net_pp.line.loc[pp_idx]
        from_bus_name = str(net_pp.bus.loc[int(pp_row["from_bus"]), "name"])
        to_bus_name = str(net_pp.bus.loc[int(pp_row["to_bus"]), "name"])
        line_name = str(pp_row.get("name", f"line_{pp_idx}"))

        rows.append(
            LineFlowRow(
                scenario=scenario,
                reference_mode=ref_mode,
                line_name=line_name,
                pp_line_idx=int(pp_idx),
                from_bus_name=from_bus_name,
                to_bus_name=to_bus_name,
                in_service_after_switch_handling=True,
                p_from_mw_diffpf=p_from_d,
                p_from_mw_pp=p_from_pp,
                p_from_mw_abs_diff=abs(p_from_d - p_from_pp) if not math.isnan(p_from_pp) else float("nan"),
                q_from_mvar_diffpf=q_from_d,
                q_from_mvar_pp=q_from_pp,
                q_from_mvar_abs_diff=abs(q_from_d - q_from_pp) if not math.isnan(q_from_pp) else float("nan"),
                p_to_mw_diffpf=p_to_d,
                p_to_mw_pp=p_to_pp,
                p_to_mw_abs_diff=abs(p_to_d - p_to_pp) if not math.isnan(p_to_pp) else float("nan"),
                q_to_mvar_diffpf=q_to_d,
                q_to_mvar_pp=q_to_pp,
                q_to_mvar_abs_diff=abs(q_to_d - q_to_pp) if not math.isnan(q_to_pp) else float("nan"),
                pl_mw_diffpf=pl_d,
                pl_mw_pp=pl_pp,
                pl_mw_abs_diff=abs(pl_d - pl_pp) if not math.isnan(pl_pp) else float("nan"),
            )
        )
    return rows


def _extract_trafo_rows(
    scenario: str,
    ref_mode: str,
    net_pp: pp.pandapowerNet,
    spec,
    params,
    active_trafo_pp_idx: list[int],
    diffpf_sol: _DiffpfSolution,
    pp_sol: _PPSolution,
) -> list[TrafoFlowRow]:
    rows: list[TrafoFlowRow] = []
    s_base = diffpf_sol.s_base_mva
    trafo_flows_d = _trafo_flows_from_voltage(diffpf_sol.voltage, params, s_base)

    for k, pp_idx in enumerate(active_trafo_pp_idx):
        tf = trafo_flows_d[k]
        trafo_row = net_pp.trafo.loc[pp_idx]
        hv_bus_id = int(trafo_row["hv_bus"])
        lv_bus_id = int(trafo_row["lv_bus"])
        hv_bus_name = str(net_pp.bus.loc[hv_bus_id, "name"])
        lv_bus_name = str(net_pp.bus.loc[lv_bus_id, "name"])
        trafo_name = str(trafo_row.get("name", f"trafo_{pp_idx}"))

        if pp_sol.converged and pp_idx in pp_sol.net.res_trafo.index:
            tr = pp_sol.net.res_trafo.loc[pp_idx]
            p_hv_pp = float(tr["p_hv_mw"])
            q_hv_pp = float(tr["q_hv_mvar"])
            p_lv_pp = float(tr["p_lv_mw"])
            q_lv_pp = float(tr["q_lv_mvar"])
            pl_pp = float(tr["pl_mw"])
        else:
            p_hv_pp = q_hv_pp = p_lv_pp = q_lv_pp = pl_pp = float("nan")

        rows.append(
            TrafoFlowRow(
                scenario=scenario,
                reference_mode=ref_mode,
                trafo_name=trafo_name,
                pp_trafo_idx=int(pp_idx),
                hv_bus_name=hv_bus_name,
                lv_bus_name=lv_bus_name,
                p_hv_mw_diffpf=tf.p_hv_mw,
                p_hv_mw_pp=p_hv_pp,
                p_hv_mw_abs_diff=abs(tf.p_hv_mw - p_hv_pp) if not math.isnan(p_hv_pp) else float("nan"),
                q_hv_mvar_diffpf=tf.q_hv_mvar,
                q_hv_mvar_pp=q_hv_pp,
                q_hv_mvar_abs_diff=abs(tf.q_hv_mvar - q_hv_pp) if not math.isnan(q_hv_pp) else float("nan"),
                p_lv_mw_diffpf=tf.p_lv_mw,
                p_lv_mw_pp=p_lv_pp,
                p_lv_mw_abs_diff=abs(tf.p_lv_mw - p_lv_pp) if not math.isnan(p_lv_pp) else float("nan"),
                q_lv_mvar_diffpf=tf.q_lv_mvar,
                q_lv_mvar_pp=q_lv_pp,
                q_lv_mvar_abs_diff=abs(tf.q_lv_mvar - q_lv_pp) if not math.isnan(q_lv_pp) else float("nan"),
                pl_mw_diffpf=tf.pl_mw,
                pl_mw_pp=pl_pp,
                pl_mw_abs_diff=abs(tf.pl_mw - pl_pp) if not math.isnan(pl_pp) else float("nan"),
            )
        )
    return rows


def _extract_loss_row(
    scenario: str,
    ref_mode: str,
    line_rows: list[LineFlowRow],
    trafo_rows: list[TrafoFlowRow],
    params,
    diffpf_sol: _DiffpfSolution,
    pp_sol: _PPSolution,
) -> LossRow:
    s_base = diffpf_sol.s_base_mva

    line_p_d = sum(r.pl_mw_diffpf for r in line_rows)
    trafo_p_d = sum(r.pl_mw_diffpf for r in trafo_rows)
    total_p_d = line_p_d + trafo_p_d

    # Q losses: sum of Q at from and to sides (analogous to P losses)
    line_q_d = sum(r.q_from_mvar_diffpf + r.q_to_mvar_diffpf for r in line_rows)
    trafo_q_d = sum(r.q_hv_mvar_diffpf + r.q_lv_mvar_diffpf for r in trafo_rows)
    total_q_d = line_q_d + trafo_q_d

    if pp_sol.converged:
        line_p_pp = (
            float(pp_sol.net.res_line["pl_mw"].sum())
            if len(pp_sol.net.res_line) > 0 else 0.0
        )
        trafo_p_pp = (
            float(pp_sol.net.res_trafo["pl_mw"].sum())
            if len(pp_sol.net.res_trafo) > 0 else 0.0
        )
        total_p_pp = line_p_pp + trafo_p_pp
        # Q losses: sum of from+to reactive power for lines and trafos
        line_q_pp = (
            float((pp_sol.net.res_line["q_from_mvar"] + pp_sol.net.res_line["q_to_mvar"]).sum())
            if len(pp_sol.net.res_line) > 0 else 0.0
        )
        trafo_q_pp = (
            float((pp_sol.net.res_trafo["q_hv_mvar"] + pp_sol.net.res_trafo["q_lv_mvar"]).sum())
            if len(pp_sol.net.res_trafo) > 0 else 0.0
        )
        total_q_pp = line_q_pp + trafo_q_pp
    else:
        line_p_pp = trafo_p_pp = total_p_pp = total_q_pp = float("nan")

    return LossRow(
        scenario=scenario,
        reference_mode=ref_mode,
        total_p_loss_mw_diffpf=total_p_d,
        total_p_loss_mw_pp=total_p_pp,
        total_p_loss_mw_abs_diff=abs(total_p_d - total_p_pp) if not math.isnan(total_p_pp) else float("nan"),
        total_q_loss_mvar_diffpf=total_q_d,
        total_q_loss_mvar_pp=total_q_pp,
        total_q_loss_mvar_abs_diff=abs(total_q_d - total_q_pp) if not math.isnan(total_q_pp) else float("nan"),
        line_p_loss_mw_diffpf=line_p_d,
        line_p_loss_mw_pp=line_p_pp if not math.isnan(line_p_pp) else float("nan"),
        trafo_p_loss_mw_diffpf=trafo_p_d,
        trafo_p_loss_mw_pp=trafo_p_pp if not math.isnan(trafo_p_pp) else float("nan"),
    )


def _extract_structure_row(
    scenario: str,
    ref_mode: str,
    net_pp: pp.pandapowerNet,
    spec,
    bus_to_repr: dict[int, int],
    disabled_lines: set[int],
    disabled_trafos: set[int],
) -> StructureSummaryRow:
    # Identify fusion groups
    groups: dict[int, list[int]] = {}
    for orig_id, repr_id in bus_to_repr.items():
        if repr_id not in groups:
            groups[repr_id] = []
        groups[repr_id].append(orig_id)
    fusion_groups = [v for v in groups.values() if len(v) > 1]

    return StructureSummaryRow(
        scenario=scenario,
        reference_mode=ref_mode,
        number_of_original_buses=len(net_pp.bus),
        number_of_internal_buses_after_fusion=len(spec.buses),
        number_of_lines_original=len(net_pp.line),
        number_of_lines_active_after_switches=len(spec.lines),
        number_of_trafos_active=len(spec.trafos),
        number_of_shunts=len(spec.shunts),
        number_of_loads=int((net_pp.load["in_service"] == True).sum()),  # noqa: E712
        number_of_sgens=int((net_pp.sgen["in_service"] == True).sum()),  # noqa: E712
        number_of_gens=int((net_pp.gen["in_service"] == True).sum()),    # noqa: E712
        bus_fusion_groups=json.dumps(fusion_groups),
        disabled_lines_due_to_open_switches=json.dumps(sorted(disabled_lines)),
        disabled_trafos_due_to_open_switches=json.dumps(sorted(disabled_trafos)),
    )


def _build_summary_row(
    scenario: str,
    ref_mode: str,
    diffpf_sol: _DiffpfSolution,
    pp_sol: _PPSolution,
    bus_rows: list[BusResultRow],
    line_rows: list[LineFlowRow],
    trafo_rows: list[TrafoFlowRow],
    loss_row: LossRow,
    notes: str,
) -> SummaryRow:
    strict = ref_mode == "scope_matched"

    # Bus metrics
    vm_diffs = [r.vm_pu_abs_diff for r in bus_rows if not math.isnan(r.vm_pu_abs_diff)]
    va_diffs = [r.va_degree_abs_diff for r in bus_rows if not math.isnan(r.va_degree_abs_diff)]
    max_vm = max(vm_diffs) if vm_diffs else float("nan")
    rmse_vm = math.sqrt(sum(d**2 for d in vm_diffs) / len(vm_diffs)) if vm_diffs else float("nan")
    max_va = max(va_diffs) if va_diffs else float("nan")
    rmse_va = math.sqrt(sum(d**2 for d in va_diffs) / len(va_diffs)) if va_diffs else float("nan")

    # Line metrics
    lp_diffs = [r.p_from_mw_abs_diff for r in line_rows if not math.isnan(r.p_from_mw_abs_diff)]
    lq_diffs = [r.q_from_mvar_abs_diff for r in line_rows if not math.isnan(r.q_from_mvar_abs_diff)]
    max_lp = max(lp_diffs) if lp_diffs else float("nan")
    max_lq = max(lq_diffs) if lq_diffs else float("nan")

    # Trafo metrics
    tp_diffs = [r.p_hv_mw_abs_diff for r in trafo_rows if not math.isnan(r.p_hv_mw_abs_diff)]
    tq_diffs = [r.q_hv_mvar_abs_diff for r in trafo_rows if not math.isnan(r.q_hv_mvar_abs_diff)]
    max_tp = max(tp_diffs) if tp_diffs else float("nan")
    max_tq = max(tq_diffs) if tq_diffs else float("nan")

    # Slack from loss_row reference
    slack_p_rows = [r for r in bus_rows if r.is_slack]
    p_slack_diff = float("nan")
    q_slack_diff = float("nan")

    return SummaryRow(
        scenario=scenario,
        reference_mode=ref_mode,
        diffpf_converged=diffpf_sol.converged,
        pandapower_converged=pp_sol.converged,
        diffpf_iterations=diffpf_sol.iterations,
        diffpf_residual_norm=diffpf_sol.residual_norm,
        init_strategy=INIT_STRATEGY,
        strict_validation=strict,
        max_vm_pu_abs_diff=max_vm,
        rmse_vm_pu=rmse_vm,
        max_va_degree_abs_diff=max_va,
        rmse_va_degree=rmse_va,
        max_line_p_mw_abs_diff=max_lp,
        max_line_q_mvar_abs_diff=max_lq,
        max_trafo_p_mw_abs_diff=max_tp,
        max_trafo_q_mvar_abs_diff=max_tq,
        p_slack_mw_abs_diff=loss_row.total_p_loss_mw_abs_diff,
        q_slack_mvar_abs_diff=loss_row.total_q_loss_mvar_abs_diff,
        total_p_loss_mw_abs_diff=loss_row.total_p_loss_mw_abs_diff,
        total_q_loss_mvar_abs_diff=loss_row.total_q_loss_mvar_abs_diff,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# One full run: scenario × reference_mode
# ---------------------------------------------------------------------------


class _ScenarioResult(NamedTuple):
    summary: SummaryRow
    bus_rows: list[BusResultRow]
    slack_row: SlackResultRow
    line_rows: list[LineFlowRow]
    trafo_rows: list[TrafoFlowRow]
    loss_row: LossRow
    structure_row: StructureSummaryRow


def run_scenario(
    scenario_name: str,
    load_factor: float,
    sgen_factor: float,
    ref_mode: str,
) -> _ScenarioResult:
    """Run one scenario × reference_mode combination and return all result rows."""

    # 1. Build base scenario net
    net_base = make_scenario_net(load_factor, sgen_factor)

    # 2. Build pandapower reference net and diffpf input net
    if ref_mode == "scope_matched":
        net_pp_run = convert_gen_to_sgen(net_base)
        net_diffpf_input = net_pp_run
        notes = "gen converted to sgen(Q=0); strict comparison"
    else:  # original_pandapower
        net_pp_run = copy.deepcopy(net_base)
        net_diffpf_input = convert_gen_to_sgen(net_base)
        notes = "pp uses original PV gen; diffpf uses sgen(Q=0); contextual comparison"

    # 3. Build switch info from base net (topology is same for both modes)
    bus_to_repr, disabled_lines, disabled_trafos = _build_switch_info(net_base)

    # Identify gen buses for annotation
    gen_bus_repr_ids: set[int] = set()
    for _, row in net_base.gen.iterrows():
        if bool(row["in_service"]):
            gen_bus_repr_ids.add(bus_to_repr[int(row["bus"])])

    # 4. Solve pandapower
    pp_sol = solve_pandapower(net_pp_run)

    # 5. Solve diffpf
    diffpf_sol = solve_diffpf(net_diffpf_input)

    if not diffpf_sol.converged:
        notes += f"; diffpf DID NOT CONVERGE (norm={diffpf_sol.residual_norm:.2e})"

    # 6. Build spec/topology again for result extraction (same net_diffpf_input)
    spec = from_pandapower(net_diffpf_input)
    topology, params = compile_network(spec)

    # 7. Line/trafo index mappings
    active_line_idx = _active_pp_line_indices(net_diffpf_input, bus_to_repr, disabled_lines)
    active_trafo_idx = _active_pp_trafo_indices(net_diffpf_input, bus_to_repr, disabled_trafos)

    # 8. Extract all result rows
    bus_rows = _extract_bus_rows(
        scenario_name, ref_mode, net_diffpf_input, spec,
        diffpf_sol, pp_sol, gen_bus_repr_ids,
    )
    slack_row = _extract_slack_row(scenario_name, ref_mode, topology, diffpf_sol, pp_sol)
    line_rows = _extract_line_rows(
        scenario_name, ref_mode, net_diffpf_input, spec, params, topology,
        active_line_idx, diffpf_sol, pp_sol,
    )
    trafo_rows = _extract_trafo_rows(
        scenario_name, ref_mode, net_diffpf_input, spec, params,
        active_trafo_idx, diffpf_sol, pp_sol,
    )
    loss_row = _extract_loss_row(
        scenario_name, ref_mode, line_rows, trafo_rows, params, diffpf_sol, pp_sol,
    )
    structure_row = _extract_structure_row(
        scenario_name, ref_mode, net_diffpf_input, spec,
        bus_to_repr, disabled_lines, disabled_trafos,
    )
    summary_row = _build_summary_row(
        scenario_name, ref_mode, diffpf_sol, pp_sol,
        bus_rows, line_rows, trafo_rows, loss_row, notes,
    )

    return _ScenarioResult(
        summary=summary_row,
        bus_rows=bus_rows,
        slack_row=slack_row,
        line_rows=line_rows,
        trafo_rows=trafo_rows,
        loss_row=loss_row,
        structure_row=structure_row,
    )


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------


def _to_native(obj):
    """Recursively convert numpy/JAX types to plain Python for JSON export."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, np.complexfloating)):
        return float(obj)
    try:
        return obj.item()
    except AttributeError:
        pass
    return obj


def _write_csv(path: Path, rows: list) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(_to_native(asdict(row)) for row in rows)


def _write_json(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump([_to_native(asdict(row)) for row in rows], f, indent=2)


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def write_metadata(results_dir: Path, scenarios: list[str]) -> None:
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_hash": _git_hash(),
        "python_version": sys.version,
        "scenarios": scenarios,
        "reference_modes": list(REFERENCE_MODES),
        "reference_mode_explanation": {
            "scope_matched": (
                "pandapower gen is replaced by sgen(P=gen.p_mw, Q=0). "
                "Both pandapower and diffpf use the same PQ model. "
                "Strict numerical comparison is valid."
            ),
            "original_pandapower": (
                "pandapower runs with the original PV bus (gen). "
                "diffpf still uses sgen(Q=0) since it has no PV-bus enforcement. "
                "Differences in Q and voltage at the gen bus are expected and correct. "
                "This mode is a contextual comparison only."
            ),
        },
        "diffpf_solver_options": {
            "max_iters": NEWTON_OPTIONS.max_iters,
            "tolerance": NEWTON_OPTIONS.tolerance,
            "damping": NEWTON_OPTIONS.damping,
        },
        "pandapower_runpp_options": PP_RUNPP_KWARGS,
        "init_strategy": INIT_STRATEGY,
        "init_strategy_description": (
            "HV buses (vn_kv >= 100 kV) initialised at slack angle. "
            "LV buses initialised at slack_angle - trafo_shift_deg. "
            "Required because example_simple has a 150 deg phase-shifting trafo "
            "that makes flat-start diverge."
        ),
        "known_model_simplifications": [
            "gen is modelled as sgen(P, Q=0) – no voltage regulation",
            "no Q-limit enforcement for generators",
            "no PV↔PQ bus switching",
            "shunt g_us_per_km for lines ignored",
            "trafo3w, xward, ward, impedance, dcline not supported",
        ],
    }
    path = results_dir / "metadata.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def write_readme(results_dir: Path) -> None:
    text = """\
# Experiment 1b – example_simple() Validation Results

## Artefakte

| Datei | Inhalt |
|-------|--------|
| validation_summary.csv/json | Eine Zeile pro Szenario × Referenzmodus; Konvergenz, Iterationen, RMSE-Kennzahlen |
| bus_results.csv/json | Spannungen je Bus; vm_pu und va_degree im Vergleich diffpf vs. pandapower |
| slack_results.csv/json | Slack-Wirk- und Blindleistung |
| line_flows.csv/json | Leitungsflüsse P/Q je Richtung und Verluste |
| trafo_flows.csv/json | Transformatorflüsse HV/LV und Verluste |
| losses.csv/json | Gesamt-, Leitungs- und Trafo-Verluste |
| structure_summary.csv/json | Netzstruktur nach Switch-Verarbeitung |
| metadata.json | Solver-Parameter, Zeitstempel, bekannte Einschränkungen |

## Referenzmodi

- **scope_matched**: gen → sgen(Q=0); strikter Vergleich
- **original_pandapower**: originales PV-Bus-Netz in pandapower, sgen in diffpf; Kontextvergleich

## Szenarien

base, load_low, load_high, sgen_low, sgen_high,
combined_high_load_low_sgen, combined_low_load_high_sgen

## Bekannte Einschränkungen

- Kein PV-Bus-Enforcement in diffpf → Q-Abweichungen am gen-Bus erwartet
- Flat Start schlägt bei 150°-Trafo fehl → trafo_shift_aware Initialisierung verwendet
"""
    path = results_dir / "README.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_all_results() -> dict:
    """Run all scenario × reference_mode combinations; return collected rows."""
    all_summary: list[SummaryRow] = []
    all_bus: list[BusResultRow] = []
    all_slack: list[SlackResultRow] = []
    all_lines: list[LineFlowRow] = []
    all_trafos: list[TrafoFlowRow] = []
    all_losses: list[LossRow] = []
    all_structure: list[StructureSummaryRow] = []

    for scenario_name, load_f, sgen_f in SCENARIOS:
        for ref_mode in REFERENCE_MODES:
            print(f"  Running {scenario_name} / {ref_mode} ...", end=" ", flush=True)
            res = run_scenario(scenario_name, load_f, sgen_f, ref_mode)
            all_summary.append(res.summary)
            all_bus.extend(res.bus_rows)
            all_slack.append(res.slack_row)
            all_lines.extend(res.line_rows)
            all_trafos.extend(res.trafo_rows)
            all_losses.append(res.loss_row)
            all_structure.append(res.structure_row)

            status = "OK" if res.summary.diffpf_converged else "DID NOT CONVERGE"
            norm = f"norm={res.summary.diffpf_residual_norm:.2e}"
            max_vm = f"max_dV={res.summary.max_vm_pu_abs_diff:.2e} pu"
            print(f"{status} {norm} {max_vm}")

    return dict(
        summary=all_summary,
        bus=all_bus,
        slack=all_slack,
        lines=all_lines,
        trafos=all_trafos,
        losses=all_losses,
        structure=all_structure,
    )


def export_all(results: dict, results_dir: Path) -> None:
    _write_csv(results_dir / "validation_summary.csv", results["summary"])
    _write_json(results_dir / "validation_summary.json", results["summary"])
    _write_csv(results_dir / "bus_results.csv", results["bus"])
    _write_json(results_dir / "bus_results.json", results["bus"])
    _write_csv(results_dir / "slack_results.csv", results["slack"])
    _write_json(results_dir / "slack_results.json", results["slack"])
    _write_csv(results_dir / "line_flows.csv", results["lines"])
    _write_json(results_dir / "line_flows.json", results["lines"])
    _write_csv(results_dir / "trafo_flows.csv", results["trafos"])
    _write_json(results_dir / "trafo_flows.json", results["trafos"])
    _write_csv(results_dir / "losses.csv", results["losses"])
    _write_json(results_dir / "losses.json", results["losses"])
    _write_csv(results_dir / "structure_summary.csv", results["structure"])
    _write_json(results_dir / "structure_summary.json", results["structure"])
    write_metadata(results_dir, [s[0] for s in SCENARIOS])
    write_readme(results_dir)


def main() -> None:
    print("=" * 70)
    print("Experiment 1b: example_simple() Validation")
    print(f"Results directory: {RESULTS_DIR}")
    print("=" * 70)

    results = build_all_results()
    export_all(results, RESULTS_DIR)

    print()
    print("Exported artefacts:")
    for p in sorted(RESULTS_DIR.iterdir()):
        print(f"  {p.name}")

    # Print summary table
    print()
    print(f"{'Scenario':<38} {'Mode':<24} {'Conv':>4} {'Iters':>5} {'maxdV[pu]':>10}")
    print("-" * 85)
    for row in results["summary"]:
        conv = "Y" if row.diffpf_converged else "N"
        dv = f"{row.max_vm_pu_abs_diff:.2e}" if not math.isnan(row.max_vm_pu_abs_diff) else "  nan"
        print(f"  {row.scenario:<36} {row.reference_mode:<24} {conv:>4} {row.diffpf_iterations:>5} {dv:>10}")


if __name__ == "__main__":
    main()
