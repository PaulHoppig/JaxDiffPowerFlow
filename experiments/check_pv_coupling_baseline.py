"""Baseline reproduction check for the example_simple() PV coupling point."""

from __future__ import annotations

import copy
import csv
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np
import pandapower.networks as pn

from diffpf.compile.network import compile_network
from diffpf.core.observables import power_flow_observables
from diffpf.core.residuals import calc_power_injection, state_to_voltage
from diffpf.core.types import NetworkParams, PFState
from diffpf.core.ybus import build_ybus
from diffpf.io.pandapower_adapter import from_pandapower
from diffpf.io.topology_utils import merge_buses
from diffpf.models.pv import (
    PV_BASE_P_MW,
    PV_BASE_Q_MVAR,
    PV_COUPLING_BUS_NAME,
    PV_COUPLING_SGEN_NAME,
    PV_Q_OVER_P,
    inject_pv_at_bus,
    pv_pq_injection,
)
from diffpf.solver.newton import NewtonOptions, solve_power_flow_result

RESULTS_DIR = (
    Path(__file__).resolve().parent / "results" / "pv_coupling_baseline_check"
)

NEWTON_OPTIONS = NewtonOptions(max_iters=50, tolerance=1e-10, damping=0.7)
REFERENCE_IRRADIANCE_W_M2 = 1000.0
REFERENCE_CELL_TEMP_C = 25.0
STRICT_TOLERANCE = 1e-8


class _SolvedCase(NamedTuple):
    converged: bool
    iterations: int
    residual_norm: float
    voltage: np.ndarray
    s_bus: np.ndarray
    observables: Any
    s_base_mva: float
    slack_bus_idx: int
    target_bus_idx: int


@dataclass(frozen=True)
class _BaselineRow:
    observable: str
    original: float
    coupled: float
    delta: float
    abs_delta: float
    tolerance: float
    passed: bool


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _to_native(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_native(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if hasattr(value, "item"):
        return value.item()
    return value


def _find_target_sgen_idx(net) -> int:
    matches = net.sgen[
        (net.sgen["name"] == PV_COUPLING_SGEN_NAME)
        & (net.sgen["in_service"] == True)  # noqa: E712
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one active sgen named {PV_COUPLING_SGEN_NAME!r}, "
            f"found {len(matches)}."
        )

    idx = int(matches.index[0])
    p_mw = float(net.sgen.at[idx, "p_mw"])
    q_mvar = float(net.sgen.at[idx, "q_mvar"])
    if abs(p_mw - PV_BASE_P_MW) > 1e-12 or abs(q_mvar - PV_BASE_Q_MVAR) > 1e-12:
        raise ValueError(
            f"Unexpected target sgen values: P={p_mw} MW, Q={q_mvar} MVAr."
        )
    return idx


def _bus_to_repr(net) -> dict[int, int]:
    bb_pairs: list[tuple[int, int]] = []
    for _, sw in net.switch.iterrows():
        if sw["et"] == "b" and bool(sw["closed"]):
            bb_pairs.append((int(sw["bus"]), int(sw["element"])))
    return merge_buses(list(net.bus.index), bb_pairs)


def _target_bus_idx(net, spec) -> int:
    matches = net.bus[net.bus["name"] == PV_COUPLING_BUS_NAME]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one bus named {PV_COUPLING_BUS_NAME!r}, "
            f"found {len(matches)}."
        )

    original_bus = int(matches.index[0])
    repr_bus = _bus_to_repr(net)[original_bus]
    spec_bus_names = [bus.name for bus in spec.buses]
    return spec_bus_names.index(str(repr_bus))


def _make_shift_aware_initial_state(net, spec) -> PFState:
    slack_ang_rad = float(net.ext_grid.iloc[0]["va_degree"]) * math.pi / 180.0
    trafo_shift_deg = float(net.trafo.iloc[0]["shift_degree"])
    lv_ang_rad = (
        float(net.ext_grid.iloc[0]["va_degree"]) - trafo_shift_deg
    ) * math.pi / 180.0

    vr_list: list[float] = []
    vi_list: list[float] = []
    for bus in spec.buses:
        if bus.is_slack:
            continue
        bus_id = int(bus.name)
        vn_kv = float(net.bus.loc[bus_id, "vn_kv"])
        ang = slack_ang_rad if vn_kv >= 100.0 else lv_ang_rad
        vr_list.append(math.cos(ang))
        vi_list.append(math.sin(ang))

    return PFState(
        vr_pu=jnp.asarray(vr_list, dtype=jnp.float64),
        vi_pu=jnp.asarray(vi_list, dtype=jnp.float64),
    )


def _solve(net, params_override: NetworkParams | None = None) -> _SolvedCase:
    s_base_mva = float(net.sn_mva) if float(net.sn_mva) > 0 else 1.0
    spec = from_pandapower(net)
    topology, params = compile_network(spec)
    if params_override is not None:
        params = params_override

    state0 = _make_shift_aware_initial_state(net, spec)
    result = solve_power_flow_result(topology, params, state0, NEWTON_OPTIONS)
    voltage = np.asarray(state_to_voltage(topology, params, result.solution))
    s_bus = np.asarray(calc_power_injection(build_ybus(topology, params), voltage))
    observables = power_flow_observables(topology, params, result.solution)

    return _SolvedCase(
        converged=bool(result.converged),
        iterations=int(result.iterations),
        residual_norm=float(result.residual_norm),
        voltage=voltage,
        s_bus=s_bus,
        observables=observables,
        s_base_mva=s_base_mva,
        slack_bus_idx=topology.slack_bus,
        target_bus_idx=_target_bus_idx(net, spec),
    )


def _build_coupled_params(net_without_sgen) -> NetworkParams:
    spec = from_pandapower(net_without_sgen)
    _, params = compile_network(spec)
    bus_idx = _target_bus_idx(net_without_sgen, spec)
    s_base_mva = (
        float(net_without_sgen.sn_mva) if float(net_without_sgen.sn_mva) > 0 else 1.0
    )

    injection = pv_pq_injection(
        irradiance_w_m2=REFERENCE_IRRADIANCE_W_M2,
        cell_temp_c=REFERENCE_CELL_TEMP_C,
        alpha=1.0,
        kappa=PV_Q_OVER_P,
    )
    return inject_pv_at_bus(params, bus_idx, injection, s_base_mva)


def _compare(original: _SolvedCase, coupled: _SolvedCase) -> list[_BaselineRow]:
    bus_idx = original.target_bus_idx
    slack_idx = original.slack_bus_idx
    s_base = original.s_base_mva

    values = {
        "vm_bus_pu": (
            abs(complex(original.voltage[bus_idx])),
            abs(complex(coupled.voltage[bus_idx])),
        ),
        "p_slack_mw": (
            float(np.real(original.s_bus[slack_idx])) * s_base,
            float(np.real(coupled.s_bus[slack_idx])) * s_base,
        ),
        "q_slack_mvar": (
            float(np.imag(original.s_bus[slack_idx])) * s_base,
            float(np.imag(coupled.s_bus[slack_idx])) * s_base,
        ),
        "total_p_loss_mw": (
            float(np.sum(np.real(original.s_bus))) * s_base,
            float(np.sum(np.real(coupled.s_bus))) * s_base,
        ),
        "observable_line_p_loss_mw": (
            float(original.observables.total_p_loss_pu) * s_base,
            float(coupled.observables.total_p_loss_pu) * s_base,
        ),
    }

    rows: list[_BaselineRow] = []
    for name, (ref, alt) in values.items():
        delta = alt - ref
        abs_delta = abs(delta)
        rows.append(
            _BaselineRow(
                observable=name,
                original=ref,
                coupled=alt,
                delta=delta,
                abs_delta=abs_delta,
                tolerance=STRICT_TOLERANCE,
                passed=abs_delta < STRICT_TOLERANCE,
            )
        )
    return rows


def _write_outputs(
    rows: list[_BaselineRow],
    original: _SolvedCase,
    coupled: _SolvedCase,
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_hash": _git_hash(),
            "python_version": sys.version,
            "check": "pv_coupling_baseline_check",
        },
        "coupling": {
            "bus_name": PV_COUPLING_BUS_NAME,
            "sgen_name": PV_COUPLING_SGEN_NAME,
            "base_p_mw": PV_BASE_P_MW,
            "base_q_mvar": PV_BASE_Q_MVAR,
            "q_over_p": PV_Q_OVER_P,
            "reference_irradiance_w_m2": REFERENCE_IRRADIANCE_W_M2,
            "reference_cell_temp_c": REFERENCE_CELL_TEMP_C,
            "target_bus_idx": original.target_bus_idx,
        },
        "solver": {
            "max_iters": NEWTON_OPTIONS.max_iters,
            "tolerance": NEWTON_OPTIONS.tolerance,
            "damping": NEWTON_OPTIONS.damping,
            "original_converged": original.converged,
            "coupled_converged": coupled.converged,
            "original_iterations": original.iterations,
            "coupled_iterations": coupled.iterations,
            "original_residual_norm": original.residual_norm,
            "coupled_residual_norm": coupled.residual_norm,
        },
        "passed": (
            original.converged
            and coupled.converged
            and all(row.passed for row in rows)
        ),
        "comparisons": [asdict(row) for row in rows],
    }

    with (RESULTS_DIR / "baseline_check.json").open("w", encoding="utf-8") as f:
        json.dump(_to_native(payload), f, indent=2)

    with (RESULTS_DIR / "baseline_check.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(_to_native(asdict(row)) for row in rows)


def run_check() -> tuple[bool, list[_BaselineRow]]:
    net_original = pn.example_simple()
    target_sgen_idx = _find_target_sgen_idx(net_original)

    net_without_sgen = copy.deepcopy(net_original)
    net_without_sgen.sgen.at[target_sgen_idx, "in_service"] = False

    coupled_params = _build_coupled_params(net_without_sgen)
    original = _solve(net_original)
    coupled = _solve(net_without_sgen, params_override=coupled_params)
    rows = _compare(original, coupled)
    _write_outputs(rows, original, coupled)
    passed = (
        original.converged and coupled.converged and all(row.passed for row in rows)
    )
    return passed, rows


def main() -> None:
    passed, rows = run_check()
    print(f"Results directory: {RESULTS_DIR}")
    for row in rows:
        print(
            f"{row.observable}: abs_delta={row.abs_delta:.3e} "
            f"(tol={row.tolerance:.1e})"
        )
    if not passed:
        raise SystemExit("PV coupling baseline check failed.")


if __name__ == "__main__":
    main()
