"""Experiment 1: Validate the JAX AC power-flow core against pandapower."""

from __future__ import annotations

from pathlib import Path

from diffpf.validation.pandapower_ref import run_validation_suite


def _format_result_block(result) -> str:
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


def main() -> None:
    case_path = Path(__file__).resolve().parents[1] / "cases" / "three_bus_poc.json"
    results = run_validation_suite(case_path)

    print("Experiment 1: Validation of the JAX AC power-flow core")
    print(f"Case file: {case_path}")
    print()
    for result in results:
        print(_format_result_block(result))
        print()


if __name__ == "__main__":
    main()
