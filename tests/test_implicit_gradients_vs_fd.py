"""Gradient validation tests for the implicit power-flow solver."""

from __future__ import annotations

from pathlib import Path

from diffpf.io import load_json
from diffpf.validation import (
    gradient_row,
    scenario_from_raw,
)
from diffpf.validation.pandapower_ref import default_validation_cases


CASES_DIR = Path(__file__).resolve().parents[1] / "cases"


def test_selected_implicit_gradients_match_finite_differences_across_scenarios():
    raw = load_json(CASES_DIR / "three_bus_poc.json")
    checks = (
        ("V1_mag", "P_load"),
        ("V2_mag", "P_pv"),
        ("theta1_rad", "Q_load"),
        ("P_loss_total", "P_load"),
        ("P_slack", "P_pv"),
    )

    for scenario in default_validation_cases():
        topology, params, state = scenario_from_raw(raw, scenario)
        for output_name, input_name in checks:
            row = gradient_row(
                scenario.name,
                output_name,
                input_name,
                topology,
                params,
                state,
                fd_step=1e-5,
            )
            assert row.abs_error < 2e-6
            assert row.rel_error < 2e-4
