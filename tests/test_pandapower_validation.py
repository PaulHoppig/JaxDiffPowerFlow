"""Reference validation against pandapower for Experiment 1."""

from __future__ import annotations

from pathlib import Path

from diffpf.validation.pandapower_ref import run_validation_suite


CASES_DIR = Path(__file__).resolve().parents[1] / "cases"


def test_jax_matches_pandapower_across_validation_suite():
    results = run_validation_suite(CASES_DIR / "three_bus_poc.json")

    assert len(results) == 3
    for result in results:
        assert result.jax.converged
        assert result.pandapower.converged
        assert result.jax.residual_norm < 1e-8
        assert result.metrics.max_abs_voltage_mag_pu < 1e-8
        assert result.metrics.max_abs_voltage_angle_deg < 1e-6
        assert result.metrics.abs_total_loss_mw < 1e-8
        assert result.metrics.max_abs_line_flow_mw < 1e-8
        assert result.metrics.max_abs_line_flow_mvar < 1e-8
