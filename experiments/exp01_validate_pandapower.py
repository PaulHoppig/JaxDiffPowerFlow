"""Experiment 1: Validate the JAX AC power-flow core against pandapower."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from diffpf.validation.pandapower_ref import ValidationResult, run_validation_suite


RESULTS_DIR = Path(__file__).resolve().parent / "results" / "exp01_pandapower_validation"


# ---------------------------------------------------------------------------
# Flat row types für CSV / JSON Export
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SummaryRow:
    """Eine Zeile pro Betriebspunkt: Konvergenz, Iterationen, Metriken."""

    case_name: str
    p_pv_mw: float
    q_pv_mvar: float
    jax_converged: bool
    jax_iterations: int
    jax_residual_norm: float
    pp_converged: bool
    pp_iterations: int
    jax_slack_p_mw: float
    jax_slack_q_mvar: float
    pp_slack_p_mw: float
    pp_slack_q_mvar: float
    jax_total_loss_mw: float
    pp_total_loss_mw: float
    max_abs_voltage_mag_pu: float
    max_abs_voltage_angle_deg: float
    abs_total_loss_mw: float
    max_abs_line_flow_mw: float
    max_abs_line_flow_mvar: float


@dataclass(frozen=True)
class _LineFlowRow:
    """Eine Zeile pro (Betriebspunkt, Leitung): JAX- vs. pandapower-Flüsse."""

    case_name: str
    line_id: int
    from_bus: int
    to_bus: int
    jax_p_from_mw: float
    jax_q_from_mvar: float
    jax_p_to_mw: float
    jax_q_to_mvar: float
    jax_p_loss_mw: float
    pp_p_from_mw: float
    pp_q_from_mvar: float
    pp_p_to_mw: float
    pp_q_to_mvar: float
    pp_p_loss_mw: float
    abs_diff_p_from_mw: float
    abs_diff_q_from_mvar: float
    abs_diff_p_to_mw: float
    abs_diff_q_to_mvar: float


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def _to_summary_row(result: ValidationResult) -> _SummaryRow:
    return _SummaryRow(
        case_name=result.case.name,
        p_pv_mw=result.case.p_pv_mw,
        q_pv_mvar=result.case.q_pv_mvar,
        jax_converged=result.jax.converged,
        jax_iterations=result.jax.iterations,
        jax_residual_norm=result.jax.residual_norm,
        pp_converged=result.pandapower.converged,
        pp_iterations=result.pandapower.iterations,
        jax_slack_p_mw=result.jax.slack_p_mw,
        jax_slack_q_mvar=result.jax.slack_q_mvar,
        pp_slack_p_mw=result.pandapower.slack_p_mw,
        pp_slack_q_mvar=result.pandapower.slack_q_mvar,
        jax_total_loss_mw=result.jax.total_loss_mw,
        pp_total_loss_mw=result.pandapower.total_loss_mw,
        max_abs_voltage_mag_pu=result.metrics.max_abs_voltage_mag_pu,
        max_abs_voltage_angle_deg=result.metrics.max_abs_voltage_angle_deg,
        abs_total_loss_mw=result.metrics.abs_total_loss_mw,
        max_abs_line_flow_mw=result.metrics.max_abs_line_flow_mw,
        max_abs_line_flow_mvar=result.metrics.max_abs_line_flow_mvar,
    )


def _to_line_flow_rows(result: ValidationResult) -> list[_LineFlowRow]:
    rows = []
    for jf, pf in zip(result.jax.line_flows, result.pandapower.line_flows, strict=True):
        rows.append(
            _LineFlowRow(
                case_name=result.case.name,
                line_id=jf.line_id,
                from_bus=jf.from_bus,
                to_bus=jf.to_bus,
                jax_p_from_mw=jf.p_from_mw,
                jax_q_from_mvar=jf.q_from_mvar,
                jax_p_to_mw=jf.p_to_mw,
                jax_q_to_mvar=jf.q_to_mvar,
                jax_p_loss_mw=jf.p_loss_mw,
                pp_p_from_mw=pf.p_from_mw,
                pp_q_from_mvar=pf.q_from_mvar,
                pp_p_to_mw=pf.p_to_mw,
                pp_q_to_mvar=pf.q_to_mvar,
                pp_p_loss_mw=pf.p_loss_mw,
                abs_diff_p_from_mw=abs(jf.p_from_mw - pf.p_from_mw),
                abs_diff_q_from_mvar=abs(jf.q_from_mvar - pf.q_from_mvar),
                abs_diff_p_to_mw=abs(jf.p_to_mw - pf.p_to_mw),
                abs_diff_q_to_mvar=abs(jf.q_to_mvar - pf.q_to_mvar),
            )
        )
    return rows


def _to_native(obj):
    """Konvertiert JAX-Arrays und numpy-Skalare rekursiv in native Python-Typen."""
    import numpy as np

    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, np.complexfloating)):
        return float(obj)
    # JAX ArrayImpl (0-d oder höherdimensional)
    try:
        return obj.item()
    except AttributeError:
        pass
    return obj


def _write_csv(path: Path, rows: tuple | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(_to_native(asdict(row)) for row in rows)


def _write_json(path: Path, rows: tuple | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump([_to_native(asdict(row)) for row in rows], handle, indent=2)


def _format_result_block(result: ValidationResult) -> str:
    return "\n".join(
        [
            f"[{result.case.name}] PV injection = {result.case.p_pv_mw:.3f} MW, {result.case.q_pv_mvar:.3f} MVAR",
            f"  Converged: JAX={result.jax.converged} in {result.jax.iterations} iterations | pandapower={result.pandapower.converged} in {result.pandapower.iterations} iterations",
            f"  Residual norm (JAX): {result.jax.residual_norm:.3e}",
            f"  Max |dV|: {result.metrics.max_abs_voltage_mag_pu:.3e} p.u.",
            f"  Max |dtheta|: {result.metrics.max_abs_voltage_angle_deg:.3e} deg",
            f"  |dP_loss|: {result.metrics.abs_total_loss_mw:.3e} MW",
            f"  Max |dP_line|: {result.metrics.max_abs_line_flow_mw:.3e} MW",
            f"  Max |dQ_line|: {result.metrics.max_abs_line_flow_mvar:.3e} MVAR",
            f"  Slack power: JAX=({result.jax.slack_p_mw:.6f} MW, {result.jax.slack_q_mvar:.6f} MVAR) | pandapower=({result.pandapower.slack_p_mw:.6f} MW, {result.pandapower.slack_q_mvar:.6f} MVAR)",
        ]
    )


# ---------------------------------------------------------------------------
# Einstiegspunkt
# ---------------------------------------------------------------------------


def main() -> None:
    case_path = Path(__file__).resolve().parents[1] / "cases" / "three_bus_poc.json"
    results = run_validation_suite(case_path)

    summary_rows = [_to_summary_row(r) for r in results]
    line_flow_rows = [row for r in results for row in _to_line_flow_rows(r)]

    _write_csv(RESULTS_DIR / "validation_summary.csv", summary_rows)
    _write_json(RESULTS_DIR / "validation_summary.json", summary_rows)
    _write_csv(RESULTS_DIR / "line_flows.csv", line_flow_rows)
    _write_json(RESULTS_DIR / "line_flows.json", line_flow_rows)

    print("Experiment 1: Validation of the JAX AC power-flow core")
    print(f"Case file: {case_path}")
    print(f"Results directory: {RESULTS_DIR}")
    print()
    for result in results:
        print(_format_result_block(result))
        print()
    print("Exported files:")
    print("  validation_summary.csv / validation_summary.json")
    print("  line_flows.csv / line_flows.json")


if __name__ == "__main__":
    main()
