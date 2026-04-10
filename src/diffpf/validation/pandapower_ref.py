"""pandapower reference validation for the 3-bus differentiable power-flow core."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandapower as pp

from diffpf.core import BaseValues, build_ybus, calc_power_injection, state_to_voltage
from diffpf.core.types import PFState
from diffpf.io import load_json
from diffpf.io.reader import RawNetwork
from diffpf.solver import NewtonOptions, solve_power_flow_result


@dataclass(frozen=True)
class LineFlowResult:
    """Directed line-flow result in physical units."""

    line_id: int
    from_bus: int
    to_bus: int
    p_from_mw: float
    q_from_mvar: float
    p_to_mw: float
    q_to_mvar: float
    p_loss_mw: float


@dataclass(frozen=True)
class PowerFlowValidationCase:
    """Single operating point for reference validation."""

    name: str
    p_pv_mw: float
    q_pv_mvar: float = -0.05


@dataclass(frozen=True)
class JaxPowerFlowResult:
    """Stationary power-flow outputs from the JAX solver."""

    converged: bool
    iterations: int
    residual_norm: float
    voltage_mag_pu: np.ndarray
    voltage_angle_deg: np.ndarray
    slack_p_mw: float
    slack_q_mvar: float
    total_loss_mw: float
    line_flows: tuple[LineFlowResult, ...]


@dataclass(frozen=True)
class PandapowerResult:
    """Stationary power-flow outputs from pandapower."""

    converged: bool
    iterations: int
    voltage_mag_pu: np.ndarray
    voltage_angle_deg: np.ndarray
    slack_p_mw: float
    slack_q_mvar: float
    total_loss_mw: float
    line_flows: tuple[LineFlowResult, ...]


@dataclass(frozen=True)
class ValidationMetrics:
    """Absolute comparison metrics between JAX and pandapower."""

    max_abs_voltage_mag_pu: float
    max_abs_voltage_angle_deg: float
    abs_total_loss_mw: float
    max_abs_line_flow_mw: float
    max_abs_line_flow_mvar: float


@dataclass(frozen=True)
class ValidationResult:
    """Combined validation payload for one operating point."""

    case: PowerFlowValidationCase
    jax: JaxPowerFlowResult
    pandapower: PandapowerResult
    metrics: ValidationMetrics


def _replace_bus(raw: RawNetwork, bus_id: int, **updates: float) -> RawNetwork:
    buses = [
        replace(bus, **updates) if bus.id == bus_id else bus
        for bus in raw.buses
    ]
    return replace(raw, buses=buses)


def make_operating_point(raw: RawNetwork, case: PowerFlowValidationCase) -> RawNetwork:
    """Return a RawNetwork with the PV bus injection modified for one test case."""

    pv_bus = next(bus for bus in raw.buses if bus.name == "pv_park")
    return _replace_bus(raw, pv_bus.id, p_mw=case.p_pv_mw, q_mvar=case.q_pv_mvar)


def default_validation_cases() -> tuple[PowerFlowValidationCase, ...]:
    """Three representative operating points for Experiment 1."""

    return (
        PowerFlowValidationCase(name="low_pv", p_pv_mw=0.15, q_pv_mvar=-0.02),
        PowerFlowValidationCase(name="medium_pv", p_pv_mw=0.70, q_pv_mvar=-0.05),
        PowerFlowValidationCase(name="high_pv", p_pv_mw=1.20, q_pv_mvar=-0.08),
    )


def _line_flows_from_voltage(
    raw: RawNetwork,
    voltage: np.ndarray,
) -> tuple[LineFlowResult, ...]:
    flows: list[LineFlowResult] = []
    base = BaseValues(s_mva=raw.base.s_mva, v_kv=raw.base.v_kv)
    bus_order = {bus.id: idx for idx, bus in enumerate(raw.buses)}

    for line in raw.lines:
        from_idx = bus_order[line.from_bus]
        to_idx = bus_order[line.to_bus]
        v_from = voltage[from_idx]
        v_to = voltage[to_idx]
        y_series = 1.0 / complex(line.r_pu, line.x_pu)
        y_shunt_half = 0.5j * line.b_shunt_pu

        i_from = (v_from - v_to) * y_series + v_from * y_shunt_half
        i_to = (v_to - v_from) * y_series + v_to * y_shunt_half
        s_from = v_from * np.conjugate(i_from)
        s_to = v_to * np.conjugate(i_to)

        flows.append(
            LineFlowResult(
                line_id=line.id,
                from_bus=line.from_bus,
                to_bus=line.to_bus,
                p_from_mw=base.pu_to_mw(float(np.real(s_from))),
                q_from_mvar=base.pu_to_mvar(float(np.imag(s_from))),
                p_to_mw=base.pu_to_mw(float(np.real(s_to))),
                q_to_mvar=base.pu_to_mvar(float(np.imag(s_to))),
                p_loss_mw=base.pu_to_mw(float(np.real(s_from + s_to))),
            )
        )

    return tuple(flows)


def solve_with_jax(
    raw: RawNetwork,
    options: NewtonOptions = NewtonOptions(max_iters=30, tolerance=1e-10, damping=1.0),
) -> JaxPowerFlowResult:
    """Solve one operating point with the JAX power-flow core."""

    topology, params, state = load_network_from_raw(raw)
    result = solve_power_flow_result(topology, params, state, options)
    voltage = np.asarray(state_to_voltage(topology, params, result.solution))
    y_bus = build_ybus(topology, params)
    s_injection = np.asarray(calc_power_injection(y_bus, voltage))
    base = BaseValues(s_mva=raw.base.s_mva, v_kv=raw.base.v_kv)

    slack_idx = topology.slack_bus
    return JaxPowerFlowResult(
        converged=result.converged,
        iterations=result.iterations,
        residual_norm=float(result.residual_norm),
        voltage_mag_pu=np.abs(voltage),
        voltage_angle_deg=np.degrees(np.angle(voltage)),
        slack_p_mw=base.pu_to_mw(float(np.real(s_injection[slack_idx]))),
        slack_q_mvar=base.pu_to_mvar(float(np.imag(s_injection[slack_idx]))),
        total_loss_mw=base.pu_to_mw(float(np.sum(np.real(s_injection)))),
        line_flows=_line_flows_from_voltage(raw, voltage),
    )


def solve_with_pandapower(raw: RawNetwork) -> PandapowerResult:
    """Solve one operating point with pandapower as reference."""

    net = pp.create_empty_network(sn_mva=raw.base.s_mva)
    bus_lookup: dict[int, int] = {}

    for bus in raw.buses:
        bus_lookup[bus.id] = pp.create_bus(
            net,
            vn_kv=raw.base.v_kv,
            name=bus.name,
        )

    for bus in raw.buses:
        pp_idx = bus_lookup[bus.id]
        if bus.type == "slack":
            pp.create_ext_grid(
                net,
                bus=pp_idx,
                vm_pu=bus.v_mag_pu,
                va_degree=bus.v_ang_deg,
                name=bus.name,
            )
        else:
            pp.create_load(
                net,
                bus=pp_idx,
                p_mw=-bus.p_mw,
                q_mvar=-bus.q_mvar,
                name=bus.name,
            )

    for line in raw.lines:
        pp.create_impedance(
            net,
            from_bus=bus_lookup[line.from_bus],
            to_bus=bus_lookup[line.to_bus],
            rft_pu=line.r_pu,
            xft_pu=line.x_pu,
            sn_mva=raw.base.s_mva,
            rtf_pu=line.r_pu,
            xtf_pu=line.x_pu,
            bf_pu=line.b_shunt_pu / 2.0,
            bt_pu=line.b_shunt_pu / 2.0,
            name=line.name,
        )

    pp.runpp(
        net,
        algorithm="nr",
        calculate_voltage_angles=True,
        init="flat",
        tolerance_mva=1e-10,
        max_iteration=30,
        trafo_model="pi",
        numba=False,
    )

    ordered_bus_indices = [bus_lookup[bus.id] for bus in raw.buses]
    voltage_mag = net.res_bus.loc[ordered_bus_indices, "vm_pu"].to_numpy(dtype=float)
    voltage_ang = net.res_bus.loc[ordered_bus_indices, "va_degree"].to_numpy(dtype=float)

    slack_bus_id = next(bus.id for bus in raw.buses if bus.type == "slack")
    slack_pp_idx = bus_lookup[slack_bus_id]
    ext_grid_row = net.res_ext_grid.iloc[0]

    flows: list[LineFlowResult] = []
    for line, row in zip(raw.lines, net.res_impedance.itertuples(index=False), strict=True):
        flows.append(
            LineFlowResult(
                line_id=line.id,
                from_bus=line.from_bus,
                to_bus=line.to_bus,
                p_from_mw=float(row.p_from_mw),
                q_from_mvar=float(row.q_from_mvar),
                p_to_mw=float(row.p_to_mw),
                q_to_mvar=float(row.q_to_mvar),
                p_loss_mw=float(row.pl_mw),
            )
        )

    return PandapowerResult(
        converged=bool(net.converged),
        iterations=int(net._ppc["iterations"]),
        voltage_mag_pu=voltage_mag,
        voltage_angle_deg=voltage_ang,
        slack_p_mw=float(ext_grid_row["p_mw"]),
        slack_q_mvar=float(ext_grid_row["q_mvar"]),
        total_loss_mw=float(net.res_impedance["pl_mw"].sum()),
        line_flows=tuple(flows),
    )


def compare_results(
    case: PowerFlowValidationCase,
    raw: RawNetwork,
    options: NewtonOptions = NewtonOptions(max_iters=30, tolerance=1e-10, damping=1.0),
) -> ValidationResult:
    """Solve one operating point with both solvers and compute comparison metrics."""

    jax_result = solve_with_jax(raw, options)
    pandapower_result = solve_with_pandapower(raw)

    max_flow_mw = max(
        max(abs(jf.p_from_mw - pf.p_from_mw), abs(jf.p_to_mw - pf.p_to_mw))
        for jf, pf in zip(jax_result.line_flows, pandapower_result.line_flows, strict=True)
    )
    max_flow_mvar = max(
        max(abs(jf.q_from_mvar - pf.q_from_mvar), abs(jf.q_to_mvar - pf.q_to_mvar))
        for jf, pf in zip(jax_result.line_flows, pandapower_result.line_flows, strict=True)
    )

    metrics = ValidationMetrics(
        max_abs_voltage_mag_pu=float(
            np.max(np.abs(jax_result.voltage_mag_pu - pandapower_result.voltage_mag_pu))
        ),
        max_abs_voltage_angle_deg=float(
            np.max(np.abs(jax_result.voltage_angle_deg - pandapower_result.voltage_angle_deg))
        ),
        abs_total_loss_mw=abs(jax_result.total_loss_mw - pandapower_result.total_loss_mw),
        max_abs_line_flow_mw=float(max_flow_mw),
        max_abs_line_flow_mvar=float(max_flow_mvar),
    )
    return ValidationResult(
        case=case,
        jax=jax_result,
        pandapower=pandapower_result,
        metrics=metrics,
    )


def run_validation_suite(
    network_path: str | Path,
    cases: tuple[PowerFlowValidationCase, ...] | None = None,
    options: NewtonOptions = NewtonOptions(max_iters=30, tolerance=1e-10, damping=1.0),
) -> tuple[ValidationResult, ...]:
    """Run the full Experiment 1 operating-point sweep."""

    raw = load_json(network_path)
    scenario_cases = cases if cases is not None else default_validation_cases()
    return tuple(
        compare_results(case, make_operating_point(raw, case), options)
        for case in scenario_cases
    )


def load_network_from_raw(raw: RawNetwork):
    """Compile a validated RawNetwork without re-reading from disk."""

    from diffpf.io.parser import parse

    return parse(raw)
