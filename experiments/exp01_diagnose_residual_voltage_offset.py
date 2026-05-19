"""Diagnose the remaining Exp.-1 scope-matched voltage offset.

The script is intentionally diagnostic-only: it does not modify the diffpf
core and writes all artifacts to a separate result directory.
"""

from __future__ import annotations

import copy
import json
import math
import sys
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import jax.numpy as jnp
import matplotlib
import numpy as np
import pandas as pd
import pandapower as pp
from pandapower.pypower.idx_brch import BR_B, BR_R, BR_X, F_BUS, SHIFT, TAP, T_BUS
from pandapower.pypower.idx_bus import BASE_KV, BS, BUS_I, GS, PD, QD
from pandapower.pypower.makeYbus import makeYbus

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import experiments.exp01_validate_example_simple as exp01
from diffpf.compile.network import compile_network
from diffpf.core.residuals import power_flow_residual
from diffpf.core.types import CompiledTopology, NetworkParams, PFState
from diffpf.core.ybus import build_ybus
from diffpf.io.pandapower_adapter import from_pandapower

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


MAIN_RESULTS_DIR = (
    Path(__file__).resolve().parent / "results" / "exp01_example_simple_validation"
)
RESULTS_DIR = (
    Path(__file__).resolve().parent
    / "results"
    / "exp01_residual_voltage_offset_diagnostic"
)
REFERENCE_MODE = "scope_matched"
PRE_OPEN_LINE_POLICY_BASELINE = {
    "max_vm_pu_abs_diff": 2.350559167396682e-05,
    "max_abs_complex_ybus_diff": 4.178332435860909e-03,
    "lines_only_ybus_diff": 4.178332435860909e-03,
    "residual_norm_at_pandapower_solution": 4.290310889485325e-03,
}


def _to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    return value


def _write_df(df: pd.DataFrame, stem: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / f"{stem}.csv", index=False)
    df.to_json(output_dir / f"{stem}.json", orient="records", indent=2)


def _write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_jsonable(payload), indent=2), encoding="utf-8")


def select_worst_case(
    validation_summary: pd.DataFrame,
    bus_results: pd.DataFrame,
) -> dict:
    """Return the scope-matched scenario and bus with largest voltage error."""

    scope = validation_summary[
        validation_summary["reference_mode"] == REFERENCE_MODE
    ].copy()
    if scope.empty:
        raise ValueError("No scope_matched rows in validation summary.")
    summary_row = scope.loc[scope["max_vm_pu_abs_diff"].idxmax()]

    buses = bus_results[
        (bus_results["reference_mode"] == REFERENCE_MODE)
        & (bus_results["scenario"] == summary_row["scenario"])
    ].copy()
    if buses.empty:
        raise ValueError(f"No bus rows for scenario {summary_row['scenario']!r}.")
    buses["vm_pu_signed_diff"] = buses["vm_pu_diffpf"] - buses["vm_pu_pp"]
    buses["va_degree_signed_diff"] = (
        buses["va_degree_diffpf"] - buses["va_degree_pp"]
    )
    bus_row = buses.loc[buses["vm_pu_abs_diff"].idxmax()]

    return {
        "scenario": str(summary_row["scenario"]),
        "max_vm_pu_abs_diff": float(summary_row["max_vm_pu_abs_diff"]),
        "max_va_degree_abs_diff": float(summary_row["max_va_degree_abs_diff"]),
        "bus_name": str(bus_row["bus_name"]),
        "bus_original_id": int(bus_row["bus_original_id"]),
        "bus_internal_id": int(bus_row["bus_internal_id"]),
        "vm_pu_diffpf": float(bus_row["vm_pu_diffpf"]),
        "vm_pu_pandapower": float(bus_row["vm_pu_pp"]),
        "vm_pu_diff": float(bus_row["vm_pu_signed_diff"]),
        "va_degree_diffpf": float(bus_row["va_degree_diffpf"]),
        "va_degree_pandapower": float(bus_row["va_degree_pp"]),
        "va_degree_diff": float(bus_row["va_degree_signed_diff"]),
    }


def bus_voltage_diagnostic(bus_results: pd.DataFrame, scenario: str) -> pd.DataFrame:
    rows = bus_results[
        (bus_results["reference_mode"] == REFERENCE_MODE)
        & (bus_results["scenario"] == scenario)
    ].copy()
    rows["vm_pu_diff"] = rows["vm_pu_diffpf"] - rows["vm_pu_pp"]
    rows["va_degree_diff"] = rows["va_degree_diffpf"] - rows["va_degree_pp"]
    rows = rows.rename(
        columns={
            "bus_original_id": "bus_index_pandapower",
            "bus_internal_id": "bus_index_diffpf",
            "vm_pu_pp": "vm_pu_pandapower",
            "va_degree_pp": "va_degree_pandapower",
        }
    )
    columns = [
        "bus_name",
        "bus_index_pandapower",
        "bus_index_diffpf",
        "vm_pu_diffpf",
        "vm_pu_pandapower",
        "vm_pu_diff",
        "va_degree_diffpf",
        "va_degree_pandapower",
        "va_degree_diff",
    ]
    return rows[columns].sort_values("vm_pu_diff", key=lambda s: s.abs(), ascending=False)


def build_scope_matched_case(scenario: str):
    scenarios = {name: (load, sgen) for name, load, sgen in exp01.SCENARIOS}
    if scenario not in scenarios:
        raise ValueError(f"Unknown scenario {scenario!r}.")
    load_factor, sgen_factor = scenarios[scenario]
    net = exp01.apply_scope_matched_open_line_policy(
        exp01.convert_gen_to_sgen(
            exp01.make_scenario_net(load_factor, sgen_factor)
        )
    )
    pp.runpp(net, **exp01.PP_RUNPP_KWARGS)
    spec = from_pandapower(net)
    topology, params = compile_network(spec)
    diffpf_solution = exp01.solve_diffpf(copy.deepcopy(net))
    return net, spec, topology, params, diffpf_solution


def _pp_internal_bus_lookup(net, spec) -> list[int]:
    lookup = net._pd2ppc_lookups["bus"]
    return [int(lookup[int(bus.name)]) for bus in spec.buses]


def _kron_reduce_to_retained(ybus: np.ndarray, retained: list[int]) -> np.ndarray:
    retained = list(retained)
    all_indices = list(range(ybus.shape[0]))
    eliminated = [idx for idx in all_indices if idx not in retained]
    y_aa = ybus[np.ix_(retained, retained)]
    if not eliminated:
        return y_aa
    y_ab = ybus[np.ix_(retained, eliminated)]
    y_ba = ybus[np.ix_(eliminated, retained)]
    y_bb = ybus[np.ix_(eliminated, eliminated)]
    try:
        correction = y_ab @ np.linalg.solve(y_bb, y_ba)
    except np.linalg.LinAlgError:
        correction = y_ab @ np.linalg.pinv(y_bb) @ y_ba
    return y_aa - correction


def pandapower_ybus_reduced(net, spec, ybus=None) -> np.ndarray:
    if ybus is None:
        ybus = net._ppc["internal"]["Ybus"]
    dense = np.asarray(ybus.toarray() if hasattr(ybus, "toarray") else ybus, dtype=complex)
    retained = _pp_internal_bus_lookup(net, spec)
    return _kron_reduce_to_retained(dense, retained)


def diffpf_ybus_dense(topology, params) -> np.ndarray:
    return np.asarray(build_ybus(topology, params), dtype=complex)


def calculate_ybus_difference_metrics(
    reference_ybus: np.ndarray,
    candidate_ybus: np.ndarray,
    bus_names: list[str],
) -> tuple[dict, pd.DataFrame]:
    diff = candidate_ybus - reference_ybus
    abs_diff = np.abs(diff)
    abs_reference = np.abs(reference_ybus)
    rel_diff = abs_diff / np.maximum(abs_reference, 1e-15)
    max_pos = np.unravel_index(int(np.argmax(abs_diff)), abs_diff.shape)
    entries = []
    for i in range(diff.shape[0]):
        for j in range(diff.shape[1]):
            entries.append(
                {
                    "from_bus": bus_names[i],
                    "to_bus": bus_names[j],
                    "row": i,
                    "col": j,
                    "pandapower_real": float(reference_ybus[i, j].real),
                    "pandapower_imag": float(reference_ybus[i, j].imag),
                    "diffpf_real": float(candidate_ybus[i, j].real),
                    "diffpf_imag": float(candidate_ybus[i, j].imag),
                    "real_diff": float(diff[i, j].real),
                    "imag_diff": float(diff[i, j].imag),
                    "abs_complex_diff": float(abs_diff[i, j]),
                    "relative_diff": float(rel_diff[i, j]),
                }
            )
    entries_df = pd.DataFrame(entries).sort_values(
        "abs_complex_diff", ascending=False
    )
    metrics = {
        "max_abs_complex_diff": float(abs_diff[max_pos]),
        "frobenius_norm_diff": float(np.linalg.norm(diff)),
        "max_abs_real_diff": float(np.max(np.abs(diff.real))),
        "max_abs_imag_diff": float(np.max(np.abs(diff.imag))),
        "largest_diff_from_bus": bus_names[max_pos[0]],
        "largest_diff_to_bus": bus_names[max_pos[1]],
        "largest_diff_row": int(max_pos[0]),
        "largest_diff_col": int(max_pos[1]),
        "n_entries_abs_diff_gt_1e_12": int(np.sum(abs_diff > 1e-12)),
        "n_entries_abs_diff_gt_1e_10": int(np.sum(abs_diff > 1e-10)),
        "n_entries_abs_diff_gt_1e_8": int(np.sum(abs_diff > 1e-8)),
    }
    return metrics, entries_df


def _empty_float_array():
    return jnp.asarray([], dtype=jnp.float64)


def _empty_int_array():
    return jnp.asarray([], dtype=jnp.int32)


def _component_diffpf_ybus(topology, params, include: set[str]) -> np.ndarray:
    include_lines = "lines" in include
    include_trafos = "trafos" in include
    include_shunts = "shunts" in include
    component_topology = replace(
        topology,
        from_bus=topology.from_bus if include_lines else _empty_int_array(),
        to_bus=topology.to_bus if include_lines else _empty_int_array(),
    )
    component_params = replace(
        params,
        g_series_pu=params.g_series_pu if include_lines else _empty_float_array(),
        b_series_pu=params.b_series_pu if include_lines else _empty_float_array(),
        b_shunt_pu=params.b_shunt_pu if include_lines else _empty_float_array(),
        trafo_g_series_pu=(
            params.trafo_g_series_pu if include_trafos else _empty_float_array()
        ),
        trafo_b_series_pu=(
            params.trafo_b_series_pu if include_trafos else _empty_float_array()
        ),
        trafo_g_mag_pu=(
            params.trafo_g_mag_pu if include_trafos else _empty_float_array()
        ),
        trafo_b_mag_pu=(
            params.trafo_b_mag_pu if include_trafos else _empty_float_array()
        ),
        trafo_tap_ratio=(
            params.trafo_tap_ratio if include_trafos else _empty_float_array()
        ),
        trafo_shift_rad=(
            params.trafo_shift_rad if include_trafos else _empty_float_array()
        ),
        trafo_hv_bus=params.trafo_hv_bus if include_trafos else (),
        trafo_lv_bus=params.trafo_lv_bus if include_trafos else (),
        shunt_g_pu=params.shunt_g_pu if include_shunts else _empty_float_array(),
        shunt_b_pu=params.shunt_b_pu if include_shunts else _empty_float_array(),
        shunt_bus=params.shunt_bus if include_shunts else (),
    )
    return diffpf_ybus_dense(component_topology, component_params)


def _component_pp_ybus(net, spec, include: set[str]) -> np.ndarray:
    ppc = net._ppc["internal"]
    base_mva = float(ppc["baseMVA"])
    bus = np.array(ppc["bus"], dtype=float, copy=True)
    branch = np.array(ppc["branch"], dtype=float, copy=True)
    if "shunts" not in include:
        bus[:, GS] = 0.0
        bus[:, BS] = 0.0

    ranges = net._pd2ppc_lookups["branch"]
    original_branch_indices: list[int] = []
    if "lines" in include and "line" in ranges:
        start, stop = ranges["line"]
        original_branch_indices.extend(range(start, stop))
    if "trafos" in include and "trafo" in ranges:
        start, stop = ranges["trafo"]
        original_branch_indices.extend(range(start, stop))

    branch_is = np.asarray(net._ppc["internal"].get("branch_is", []), dtype=bool)
    if branch_is.size:
        active_original = list(np.flatnonzero(branch_is))
        original_to_internal = {
            original_idx: internal_idx
            for internal_idx, original_idx in enumerate(active_original)
        }
        internal_indices = [
            original_to_internal[idx]
            for idx in original_branch_indices
            if idx in original_to_internal
        ]
    else:
        internal_indices = original_branch_indices
    branch = branch[internal_indices, :] if internal_indices else branch[:0, :]
    ybus, _, _ = makeYbus(base_mva, bus, branch)
    return pandapower_ybus_reduced(net, spec, ybus)


def component_ybus_diagnostics(net, spec, topology, params) -> pd.DataFrame:
    variants = [
        ("lines_only", {"lines"}),
        ("transformer_only", {"trafos"}),
        ("shunts_only", {"shunts"}),
        ("lines_plus_transformer", {"lines", "trafos"}),
        ("full_network", {"lines", "trafos", "shunts"}),
    ]
    bus_names = [bus.name for bus in spec.buses]
    rows = []
    for name, include in variants:
        pp_y = _component_pp_ybus(net, spec, include)
        diffpf_y = _component_diffpf_ybus(topology, params, include)
        metrics, _ = calculate_ybus_difference_metrics(pp_y, diffpf_y, bus_names)
        hint = "near_identical"
        if metrics["max_abs_complex_diff"] > 1e-8:
            if name == "lines_only":
                hint = "line_or_switch_model_difference"
            elif name == "transformer_only":
                hint = "transformer_parameter_or_stamp_difference"
            elif name == "shunts_only":
                hint = "shunt_model_difference"
            else:
                hint = "combined_difference"
        rows.append(
            {
                "component_variant": name,
                "max_abs_complex_diff": metrics["max_abs_complex_diff"],
                "max_abs_real_diff": metrics["max_abs_real_diff"],
                "max_abs_imag_diff": metrics["max_abs_imag_diff"],
                "frobenius_norm_diff": metrics["frobenius_norm_diff"],
                "largest_diff_from_bus": metrics["largest_diff_from_bus"],
                "largest_diff_to_bus": metrics["largest_diff_to_bus"],
                "interpretation_hint": hint,
            }
        )
    return pd.DataFrame(rows)


def algebraic_input_comparison(net, spec, topology, params) -> pd.DataFrame:
    lookup = _pp_internal_bus_lookup(net, spec)
    ppc_bus = net._ppc["internal"]["bus"]
    base_mva = float(net._ppc["internal"]["baseMVA"])
    rows = []

    slack_spec_idx = next(i for i, bus in enumerate(spec.buses) if bus.is_slack)
    slack_name = spec.buses[slack_spec_idx].name
    slack_vm = math.hypot(float(params.slack_vr_pu), float(params.slack_vi_pu))
    slack_va = math.degrees(math.atan2(float(params.slack_vi_pu), float(params.slack_vr_pu)))
    ext = net.ext_grid[net.ext_grid["in_service"] == True].iloc[0]  # noqa: E712
    rows.extend(
        [
            {
                "category": "base",
                "item": "sn_mva/baseMVA",
                "diffpf_value": float(net.sn_mva),
                "pandapower_value": base_mva,
                "abs_diff": abs(float(net.sn_mva) - base_mva),
                "match": abs(float(net.sn_mva) - base_mva) < 1e-12,
                "notes": "",
            },
            {
                "category": "slack",
                "item": "slack_bus",
                "diffpf_value": slack_name,
                "pandapower_value": int(ext["bus"]),
                "abs_diff": 0.0 if int(slack_name) == int(ext["bus"]) else 1.0,
                "match": int(slack_name) == int(ext["bus"]),
                "notes": f"diffpf internal slack index {topology.slack_bus}",
            },
            {
                "category": "slack",
                "item": "slack_vm_pu",
                "diffpf_value": slack_vm,
                "pandapower_value": float(ext["vm_pu"]),
                "abs_diff": abs(slack_vm - float(ext["vm_pu"])),
                "match": abs(slack_vm - float(ext["vm_pu"])) < 1e-12,
                "notes": "",
            },
            {
                "category": "slack",
                "item": "slack_va_degree",
                "diffpf_value": slack_va,
                "pandapower_value": float(ext["va_degree"]),
                "abs_diff": abs(slack_va - float(ext["va_degree"])),
                "match": abs(slack_va - float(ext["va_degree"])) < 1e-12,
                "notes": "",
            },
        ]
    )

    for diffpf_idx, bus in enumerate(spec.buses):
        pp_idx = lookup[diffpf_idx]
        bus_id = int(bus.name)
        pp_p_spec = -float(ppc_bus[pp_idx, PD]) / base_mva
        pp_q_spec = -float(ppc_bus[pp_idx, QD]) / base_mva
        pp_vn = float(ppc_bus[pp_idx, BASE_KV])
        diffpf_p = float(params.p_spec_pu[diffpf_idx])
        diffpf_q = float(params.q_spec_pu[diffpf_idx])
        diffpf_vn = float(net.bus.loc[bus_id, "vn_kv"])
        rows.extend(
            [
                {
                    "category": "bus_mapping",
                    "item": f"bus_{bus.name}_pp_internal_index",
                    "diffpf_value": diffpf_idx,
                    "pandapower_value": pp_idx,
                    "abs_diff": 0.0,
                    "match": True,
                    "notes": "indices differ by design; row records mapping",
                },
                {
                    "category": "p_spec_pu",
                    "item": f"bus_{bus.name}",
                    "diffpf_value": diffpf_p,
                    "pandapower_value": pp_p_spec,
                    "abs_diff": abs(diffpf_p - pp_p_spec),
                    "match": abs(diffpf_p - pp_p_spec) < 1e-12,
                    "notes": "",
                },
                {
                    "category": "q_spec_pu",
                    "item": f"bus_{bus.name}",
                    "diffpf_value": diffpf_q,
                    "pandapower_value": pp_q_spec,
                    "abs_diff": abs(diffpf_q - pp_q_spec),
                    "match": abs(diffpf_q - pp_q_spec) < 1e-12,
                    "notes": "",
                },
                {
                    "category": "vn_kv",
                    "item": f"bus_{bus.name}",
                    "diffpf_value": diffpf_vn,
                    "pandapower_value": pp_vn,
                    "abs_diff": abs(diffpf_vn - pp_vn),
                    "match": abs(diffpf_vn - pp_vn) < 1e-12,
                    "notes": "",
                },
            ]
        )

    bus_to_repr, disabled_lines, disabled_trafos = exp01._build_switch_info(net)
    diffpf_active_lines = len(
        exp01._active_pp_line_indices(net, bus_to_repr, disabled_lines)
    )
    pandapower_active_lines = int((net.line["in_service"] == True).sum())  # noqa: E712
    open_line_policy_applied = bool(exp01.open_line_switch_line_indices(net))
    rows.extend(
        [
            {
                "category": "topology",
                "item": "active_lines_after_diffpf_switch_handling",
                "diffpf_value": diffpf_active_lines,
                "pandapower_value": pandapower_active_lines,
                "abs_diff": abs(float(diffpf_active_lines - pandapower_active_lines)),
                "match": diffpf_active_lines == pandapower_active_lines,
                "notes": (
                    "open line switches disable lines in diffpf; scope_matched "
                    "pandapower reference applies the same line out-of-service "
                    f"policy for {sorted(disabled_lines)}"
                ),
            },
            {
                "category": "topology",
                "item": "scope_matched_open_line_policy_applied",
                "diffpf_value": "removes open-line-switch lines",
                "pandapower_value": open_line_policy_applied,
                "abs_diff": 0.0,
                "match": open_line_policy_applied,
                "notes": (
                    "pandapower reference lines affected by open line switches "
                    "are set out of service before runpp"
                ),
            },
            {
                "category": "topology",
                "item": "active_trafos_after_switch_handling",
                "diffpf_value": len(
                    exp01._active_pp_trafo_indices(net, bus_to_repr, disabled_trafos)
                ),
                "pandapower_value": len(net.trafo[net.trafo["in_service"] == True]),  # noqa: E712
                "abs_diff": float(len(disabled_trafos)),
                "match": len(disabled_trafos) == 0,
                "notes": f"open trafo switches disable trafos in diffpf: {sorted(disabled_trafos)}",
            },
            {
                "category": "topology",
                "item": "bus_fusion_groups",
                "diffpf_value": json.dumps({str(k): int(v) for k, v in bus_to_repr.items()}),
                "pandapower_value": json.dumps(net._pd2ppc_lookups["bus"].tolist()),
                "abs_diff": 0.0,
                "match": True,
                "notes": "both solvers reduce closed bus-bus switches; pandapower also creates aux buses for open line switches",
            },
        ]
    )
    return pd.DataFrame(rows)


def transformer_parameter_diagnostic(net, spec, params) -> pd.DataFrame:
    rows = []
    branch_lookup = net._pd2ppc_lookups["branch"]
    trafo_start, trafo_stop = branch_lookup.get("trafo", (0, 0))
    ppc_branch = net._ppc["internal"]["branch"]
    branch_is = np.asarray(net._ppc["internal"].get("branch_is", []), dtype=bool)
    ppc_row = None
    if trafo_stop > trafo_start:
        if branch_is.size:
            active_original = list(np.flatnonzero(branch_is))
            original_to_internal = {
                original_idx: internal_idx
                for internal_idx, original_idx in enumerate(active_original)
            }
            internal_idx = original_to_internal.get(trafo_start)
            if internal_idx is not None:
                ppc_row = ppc_branch[internal_idx]
        else:
            ppc_row = ppc_branch[trafo_start]

    for k, trafo in enumerate(spec.trafos):
        pp_idx = list(net.trafo.index)[k]
        pp_row = net.trafo.loc[pp_idx]
        hv_bus = int(pp_row["hv_bus"])
        lv_bus = int(pp_row["lv_bus"])
        tap_ratio = float(params.trafo_tap_ratio[k])
        shift_rad = float(params.trafo_shift_rad[k])
        complex_tap = tap_ratio * np.exp(1j * shift_rad)
        rows.append(
            {
                "pp_trafo_idx": int(pp_idx),
                "name": str(pp_row.get("name", "")),
                "hv_bus": hv_bus,
                "lv_bus": lv_bus,
                "diffpf_hv_bus": int(params.trafo_hv_bus[k]),
                "diffpf_lv_bus": int(params.trafo_lv_bus[k]),
                "hv_bus_vn_kv": float(net.bus.loc[hv_bus, "vn_kv"]),
                "lv_bus_vn_kv": float(net.bus.loc[lv_bus, "vn_kv"]),
                "trafo_vn_hv_kv": float(pp_row["vn_hv_kv"]),
                "trafo_vn_lv_kv": float(pp_row["vn_lv_kv"]),
                "sn_mva": float(pp_row["sn_mva"]),
                "vk_percent": float(pp_row["vk_percent"]),
                "vkr_percent": float(pp_row["vkr_percent"]),
                "pfe_kw": float(pp_row["pfe_kw"]),
                "i0_percent": float(pp_row["i0_percent"]),
                "tap_side": str(pp_row.get("tap_side", "")),
                "tap_pos": float(pp_row.get("tap_pos", 0.0)),
                "tap_neutral": float(pp_row.get("tap_neutral", 0.0)),
                "tap_step_percent": float(pp_row.get("tap_step_percent", 0.0)),
                "shift_degree": float(pp_row["shift_degree"]),
                "diffpf_r_pu": float(trafo.r_pu),
                "diffpf_x_pu": float(trafo.x_pu),
                "diffpf_g_series_pu": float(params.trafo_g_series_pu[k]),
                "diffpf_b_series_pu": float(params.trafo_b_series_pu[k]),
                "diffpf_g_mag_pu": float(params.trafo_g_mag_pu[k]),
                "diffpf_b_mag_pu_positive_magnitude": float(params.trafo_b_mag_pu[k]),
                "diffpf_tap_ratio": tap_ratio,
                "diffpf_shift_rad": shift_rad,
                "diffpf_complex_tap_real": float(complex_tap.real),
                "diffpf_complex_tap_imag": float(complex_tap.imag),
                "ppc_from_bus": float(ppc_row[F_BUS]) if ppc_row is not None else np.nan,
                "ppc_to_bus": float(ppc_row[T_BUS]) if ppc_row is not None else np.nan,
                "ppc_r_pu": float(ppc_row[BR_R]) if ppc_row is not None else np.nan,
                "ppc_x_pu": float(ppc_row[BR_X]) if ppc_row is not None else np.nan,
                "ppc_b_pu": float(ppc_row[BR_B]) if ppc_row is not None else np.nan,
                "ppc_tap_ratio": float(ppc_row[TAP]) if ppc_row is not None else np.nan,
                "ppc_shift_degree": float(ppc_row[SHIFT]) if ppc_row is not None else np.nan,
                "interpretation_hint": (
                    "transformer ppc branch matches diffpf closely if component "
                    "Ybus transformer_only is near zero"
                ),
            }
        )
    return pd.DataFrame(rows)


def _state_from_voltage(topology, voltage: np.ndarray) -> PFState:
    variable = np.asarray(topology.variable_buses, dtype=int)
    return PFState(
        vr_pu=jnp.asarray(np.real(voltage[variable]), dtype=jnp.float64),
        vi_pu=jnp.asarray(np.imag(voltage[variable]), dtype=jnp.float64),
    )


def _pandapower_voltage_in_diffpf_order(net, spec) -> np.ndarray:
    values = []
    for bus in spec.buses:
        bus_id = int(bus.name)
        vm = float(net.res_bus.loc[bus_id, "vm_pu"])
        va = math.radians(float(net.res_bus.loc[bus_id, "va_degree"]))
        values.append(vm * np.exp(1j * va))
    return np.asarray(values, dtype=complex)


def cross_residual_check(net, spec, topology, params, diffpf_solution) -> pd.DataFrame:
    voltage_diffpf = np.asarray(diffpf_solution.voltage, dtype=complex)
    voltage_pp = _pandapower_voltage_in_diffpf_order(net, spec)
    state_diffpf = _state_from_voltage(topology, voltage_diffpf)
    state_pp = _state_from_voltage(topology, voltage_pp)
    residual_diffpf = np.asarray(power_flow_residual(topology, params, state_diffpf))
    residual_pp = np.asarray(power_flow_residual(topology, params, state_pp))

    variable = np.asarray(topology.variable_buses, dtype=int)
    equation_names = []
    for equation_type in ("p", "q_or_v"):
        for bus_idx in variable:
            equation_names.append(f"{equation_type}_bus_{spec.buses[int(bus_idx)].name}")
    max_idx = int(np.argmax(np.abs(residual_pp)))
    return pd.DataFrame(
        [
            {
                "residual_norm_at_diffpf_solution": float(np.linalg.norm(residual_diffpf)),
                "residual_norm_at_pandapower_solution": float(np.linalg.norm(residual_pp)),
                "max_abs_residual_at_pandapower_solution": float(
                    np.max(np.abs(residual_pp))
                ),
                "largest_residual_equation": equation_names[max_idx],
                "largest_residual_value": float(residual_pp[max_idx]),
            }
        ]
    )


@contextmanager
def temporary_solver_options(newton_options, pp_kwargs: dict) -> Iterator[None]:
    old_newton = exp01.NEWTON_OPTIONS
    old_pp = dict(exp01.PP_RUNPP_KWARGS)
    exp01.NEWTON_OPTIONS = newton_options
    exp01.PP_RUNPP_KWARGS = dict(pp_kwargs)
    try:
        yield
    finally:
        exp01.NEWTON_OPTIONS = old_newton
        exp01.PP_RUNPP_KWARGS = old_pp


def tolerance_sensitivity(scenario: str) -> pd.DataFrame:
    scenarios = {name: (load, sgen) for name, load, sgen in exp01.SCENARIOS}
    load_factor, sgen_factor = scenarios[scenario]
    rows = []

    default_result = exp01.run_scenario(
        scenario, load_factor, sgen_factor, REFERENCE_MODE
    )
    rows.append(
        {
            "variant": "default_current_options",
            "diffpf_tolerance": exp01.NEWTON_OPTIONS.tolerance,
            "diffpf_max_iters": exp01.NEWTON_OPTIONS.max_iters,
            "pandapower_tolerance_mva": exp01.PP_RUNPP_KWARGS["tolerance_mva"],
            "diffpf_iterations": default_result.summary.diffpf_iterations,
            "diffpf_residual_norm": default_result.summary.diffpf_residual_norm,
            "max_vm_pu_abs_diff": default_result.summary.max_vm_pu_abs_diff,
            "max_va_degree_abs_diff": default_result.summary.max_va_degree_abs_diff,
        }
    )

    strict_newton = replace(exp01.NEWTON_OPTIONS, max_iters=100, tolerance=1e-12)
    strict_pp = dict(exp01.PP_RUNPP_KWARGS)
    strict_pp["tolerance_mva"] = 1e-12
    strict_pp["max_iteration"] = max(100, int(strict_pp.get("max_iteration", 50)))
    with temporary_solver_options(strict_newton, strict_pp):
        strict_result = exp01.run_scenario(
            scenario, load_factor, sgen_factor, REFERENCE_MODE
        )
        rows.append(
            {
                "variant": "stricter_tolerances",
                "diffpf_tolerance": strict_newton.tolerance,
                "diffpf_max_iters": strict_newton.max_iters,
                "pandapower_tolerance_mva": strict_pp["tolerance_mva"],
                "diffpf_iterations": strict_result.summary.diffpf_iterations,
                "diffpf_residual_norm": strict_result.summary.diffpf_residual_norm,
                "max_vm_pu_abs_diff": strict_result.summary.max_vm_pu_abs_diff,
                "max_va_degree_abs_diff": strict_result.summary.max_va_degree_abs_diff,
            }
        )
    return pd.DataFrame(rows)


def write_plots(bus_diag: pd.DataFrame, component_diag: pd.DataFrame, output_dir: Path) -> list[Path]:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.bar(bus_diag["bus_name"].astype(str), bus_diag["vm_pu_diff"].abs())
    ax.set_xlabel("Bus")
    ax.set_ylabel("|vm diff| [p.u.]")
    ax.set_title("Worst-case bus voltage magnitude differences")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        path = figures_dir / f"fig01_bus_voltage_diff_worst_case.{suffix}"
        fig.savefig(path, dpi=300 if suffix == "png" else None)
        outputs.append(path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.bar(
        component_diag["component_variant"],
        component_diag["max_abs_complex_diff"],
    )
    ax.set_yscale("log")
    ax.set_xlabel("Component variant")
    ax.set_ylabel("max |Ydiffpf - Ypp| [p.u.]")
    ax.set_title("Component-wise Ybus differences")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        path = figures_dir / f"fig02_ybus_component_differences.{suffix}"
        fig.savefig(path, dpi=300 if suffix == "png" else None)
        outputs.append(path)
    plt.close(fig)
    return outputs


def write_metadata(output_dir: Path, worst_case: dict) -> None:
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "exp01_residual_voltage_offset_diagnostic",
        "main_results_dir": str(MAIN_RESULTS_DIR),
        "reference_mode": REFERENCE_MODE,
        "worst_case": worst_case,
        "pandapower_runpp_kwargs": exp01.PP_RUNPP_KWARGS,
        "core_changed_by_diagnostic": False,
        "main_exp01_artifacts_overwritten": False,
    }
    _write_json(payload, output_dir / "metadata.json")


def write_readme(
    output_dir: Path,
    worst_case: dict,
    algebraic: pd.DataFrame,
    ybus_metrics: dict,
    component_diag: pd.DataFrame,
    trafo_diag: pd.DataFrame,
    cross_residual: pd.DataFrame,
    tolerance: pd.DataFrame,
) -> None:
    algebraic_mismatches = algebraic[algebraic["match"] == False]  # noqa: E712
    top_component = component_diag.sort_values(
        "max_abs_complex_diff", ascending=False
    ).iloc[0]
    residual_row = cross_residual.iloc[0]
    default_tol = tolerance[tolerance["variant"] == "default_current_options"].iloc[0]
    strict_tol = tolerance[tolerance["variant"] == "stricter_tolerances"].iloc[0]
    tol_change = (
        float(strict_tol["max_vm_pu_abs_diff"])
        - float(default_tol["max_vm_pu_abs_diff"])
    )

    ybus_near_identical = ybus_metrics["max_abs_complex_diff"] < 1e-10
    residual_near_identical = (
        float(residual_row["residual_norm_at_pandapower_solution"]) < 1e-8
    )
    if ybus_near_identical and residual_near_identical:
        likely = (
            "The previous cause is confirmed: after disabling open-line-switch "
            "lines in the scope-matched pandapower reference, the Ybus "
            "difference and cross-residual drop to numerical noise."
        )
    elif str(top_component["component_variant"]) == "lines_only":
        likely = (
            "The dominant Ybus difference is caused by line/switch modelling. "
            "pandapower keeps the line with the open line switch as an internal "
            "auxiliary-bus branch with a small charging effect, while diffpf's "
            "adapter removes that line from the active topology."
        )
    elif str(top_component["component_variant"]) == "transformer_only":
        likely = "The transformer remains the dominant Ybus-difference source."
    else:
        likely = (
            "The dominant Ybus-difference source is not isolated to the transformer; "
            "see component_ybus_diagnostics.csv for the ranking."
        )

    mismatch_text = (
        "No algebraic input mismatches remain after the scope-matched open-line "
        "policy."
        if len(algebraic_mismatches) == 0
        else (
            "The remaining flagged rows are listed in "
            "`algebraic_input_comparison.csv/json`."
        )
    )
    lines_only_diff = float(
        component_diag[
            component_diag["component_variant"] == "lines_only"
        ].iloc[0]["max_abs_complex_diff"]
    )

    text = f"""# Experiment 1 Diagnostic: Residual Voltage Offset

## Goal

This diagnostic localizes the remaining small `scope_matched` voltage mismatch
in Experiment 1. It does not modify the diffpf core and does not overwrite the
main Exp.-1 artifacts.

## Worst-Case Scenario

The largest voltage magnitude mismatch occurs in scenario
`{worst_case['scenario']}` at bus `{worst_case['bus_name']}`:

| Metric | Value |
|--------|------:|
| max `vm_pu_abs_diff` | {worst_case['max_vm_pu_abs_diff']:.8e} p.u. |
| max `va_degree_abs_diff` | {worst_case['max_va_degree_abs_diff']:.8e} deg |
| diffpf vm | {worst_case['vm_pu_diffpf']:.10f} p.u. |
| pandapower vm | {worst_case['vm_pu_pandapower']:.10f} p.u. |
| signed vm diff | {worst_case['vm_pu_diff']:.8e} p.u. |

## Algebraic Inputs

The bus mapping, base power, slack voltage, slack angle, voltage bases and
P/Q specifications are exported in `algebraic_input_comparison.csv/json`.
{mismatch_text} Number of flagged rows: {len(algebraic_mismatches)}.

## Ybus Comparison

The pandapower internal Ybus was Kron-reduced from pandapower's internal bus
set, including auxiliary buses, to the diffpf bus order. The full-network Ybus
metrics are:

| Metric | Value |
|--------|------:|
| max absolute complex difference | {ybus_metrics['max_abs_complex_diff']:.8e} |
| Frobenius norm difference | {ybus_metrics['frobenius_norm_diff']:.8e} |
| max absolute real difference | {ybus_metrics['max_abs_real_diff']:.8e} |
| max absolute imaginary difference | {ybus_metrics['max_abs_imag_diff']:.8e} |
| largest entry | {ybus_metrics['largest_diff_from_bus']} -> {ybus_metrics['largest_diff_to_bus']} |

Before applying the scope-matched open-line policy, the diagnostic found
`max_abs_complex_diff = {PRE_OPEN_LINE_POLICY_BASELINE['max_abs_complex_ybus_diff']:.8e}`
with the largest difference at bus `5 -> 5`.

## Component Diagnosis

The largest component-wise Ybus difference is
`{top_component['component_variant']}` with
`max_abs_complex_diff = {float(top_component['max_abs_complex_diff']):.8e}`.
The `lines_only` value is {lines_only_diff:.8e}; before the policy it was
{PRE_OPEN_LINE_POLICY_BASELINE['lines_only_ybus_diff']:.8e}.

{likely}

## Transformer Diagnosis

The transformer parameters are exported in
`transformer_parameter_diagnostic.csv/json`. The component-wise
`transformer_only` Ybus comparison indicates whether the transformer remains
numerically suspicious. In this run the transformer-only maximum difference is
{float(component_diag[component_diag['component_variant'] == 'transformer_only'].iloc[0]['max_abs_complex_diff']):.8e}.

## Cross-Residual Check

The diffpf residual norm at the diffpf solution is
{float(residual_row['residual_norm_at_diffpf_solution']):.8e}. The diffpf
residual norm at the pandapower voltage is
{float(residual_row['residual_norm_at_pandapower_solution']):.8e}, with maximum
single residual {float(residual_row['max_abs_residual_at_pandapower_solution']):.8e}
at `{residual_row['largest_residual_equation']}`.

Before applying the open-line policy, the pandapower-voltage residual norm was
{PRE_OPEN_LINE_POLICY_BASELINE['residual_norm_at_pandapower_solution']:.8e}.
After the policy, the residual is interpreted together with the Ybus metrics:
if it is near numerical noise, both solvers now solve the same simplified
equation system; otherwise the exported tables rank the remaining differences.

## Tolerance Sensitivity

The default max voltage error is
{float(default_tol['max_vm_pu_abs_diff']):.8e} p.u.; with stricter tolerances it
is {float(strict_tol['max_vm_pu_abs_diff']):.8e} p.u. The change is
{tol_change:.8e} p.u.
Before applying the open-line policy, the worst-case voltage error was
{PRE_OPEN_LINE_POLICY_BASELINE['max_vm_pu_abs_diff']:.8e} p.u.

## Most Likely Causes

1. Confirmed previous cause: open line-switch handling / auxiliary-bus line
   charging in pandapower versus complete line removal in diffpf.
2. After applying the scope-matched policy, remaining differences are numerical
   noise unless the exported component table reports a larger component.
3. Solver tolerance is unlikely to be the main cause when the stricter tolerance
   run does not materially change the voltage mismatch.

No core change was made by this diagnostic.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def run_diagnostic(output_dir: Path = RESULTS_DIR) -> dict:
    validation = pd.read_csv(MAIN_RESULTS_DIR / "validation_summary.csv")
    bus_results = pd.read_csv(MAIN_RESULTS_DIR / "bus_results.csv")
    worst_case = select_worst_case(validation, bus_results)
    scenario = worst_case["scenario"]

    net, spec, topology, params, diffpf_solution = build_scope_matched_case(scenario)
    bus_names = [bus.name for bus in spec.buses]

    bus_diag = bus_voltage_diagnostic(bus_results, scenario)
    algebraic = algebraic_input_comparison(net, spec, topology, params)

    pp_y = pandapower_ybus_reduced(net, spec)
    diffpf_y = diffpf_ybus_dense(topology, params)
    ybus_metrics, ybus_entries = calculate_ybus_difference_metrics(
        pp_y, diffpf_y, bus_names
    )
    component_diag = component_ybus_diagnostics(net, spec, topology, params)
    trafo_diag = transformer_parameter_diagnostic(net, spec, params)
    cross_residual = cross_residual_check(
        net, spec, topology, params, diffpf_solution
    )
    tolerance = tolerance_sensitivity(scenario)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_df(pd.DataFrame([worst_case]), "worst_case_scenario", output_dir)
    _write_df(bus_diag, "bus_voltage_diagnostic", output_dir)
    _write_df(algebraic, "algebraic_input_comparison", output_dir)
    _write_json(ybus_metrics, output_dir / "ybus_global_metrics.json")
    _write_df(ybus_entries, "ybus_entry_differences", output_dir)
    _write_df(component_diag, "component_ybus_diagnostics", output_dir)
    _write_df(trafo_diag, "transformer_parameter_diagnostic", output_dir)
    _write_df(cross_residual, "cross_residual_check", output_dir)
    _write_df(tolerance, "tolerance_sensitivity", output_dir)
    write_plots(bus_diag, component_diag, output_dir)
    write_metadata(output_dir, worst_case)
    write_readme(
        output_dir,
        worst_case,
        algebraic,
        ybus_metrics,
        component_diag,
        trafo_diag,
        cross_residual,
        tolerance,
    )

    return {
        "worst_case": worst_case,
        "ybus_metrics": ybus_metrics,
        "component_diagnostics": component_diag,
        "cross_residual": cross_residual,
        "tolerance": tolerance,
    }


def main() -> None:
    print("Experiment 1 diagnostic: residual voltage offset")
    print(f"Results directory: {RESULTS_DIR}")
    result = run_diagnostic(RESULTS_DIR)
    worst = result["worst_case"]
    component = result["component_diagnostics"].sort_values(
        "max_abs_complex_diff", ascending=False
    ).iloc[0]
    residual = result["cross_residual"].iloc[0]
    tolerance = result["tolerance"]
    print(
        "Worst case: "
        f"{worst['scenario']} bus {worst['bus_name']} "
        f"max_vm={worst['max_vm_pu_abs_diff']:.8e}"
    )
    print(
        "Ybus max abs diff: "
        f"{result['ybus_metrics']['max_abs_complex_diff']:.8e} "
        f"at {result['ybus_metrics']['largest_diff_from_bus']} -> "
        f"{result['ybus_metrics']['largest_diff_to_bus']}"
    )
    print(
        "Largest component difference: "
        f"{component['component_variant']} "
        f"({float(component['max_abs_complex_diff']):.8e})"
    )
    print(
        "Residual at pandapower voltage: "
        f"{float(residual['residual_norm_at_pandapower_solution']):.8e}"
    )
    print(
        "Tolerance max_vm: "
        f"default={float(tolerance.iloc[0]['max_vm_pu_abs_diff']):.8e}, "
        f"strict={float(tolerance.iloc[1]['max_vm_pu_abs_diff']):.8e}"
    )


if __name__ == "__main__":
    main()
