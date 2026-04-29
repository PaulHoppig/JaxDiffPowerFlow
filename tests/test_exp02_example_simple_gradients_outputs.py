"""Schema tests for the example_simple gradient-validation experiment."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def exp_module():
    import experiments.exp02_validate_gradients_example_simple as module

    return module


def test_module_is_importable(exp_module):
    assert exp_module is not None


def test_scenarios_contain_required_three_operating_points(exp_module):
    names = [scenario.name for scenario in exp_module.SCENARIOS]
    assert "base" in names
    assert "load_high" in names
    assert "sgen_high" in names


def test_input_parameter_list_is_mandatory_set(exp_module):
    names = [spec.name for spec in exp_module.INPUT_PARAMETERS]
    assert names == [
        "load_scale_mv_bus_2",
        "sgen_scale_static_generator",
        "shunt_q_scale",
        "trafo_x_scale",
    ]


def test_output_observable_list_is_compact(exp_module):
    names = [spec.name for spec in exp_module.OUTPUT_OBSERVABLES]
    assert len(names) <= 4
    assert names == [
        "vm_mv_bus_2_pu",
        "p_slack_mw",
        "total_p_loss_mw",
        "p_trafo_hv_mw",
    ]


def test_mandatory_gradient_count_is_bounded(exp_module):
    n_gradients = (
        len(exp_module.SCENARIOS)
        * len(exp_module.INPUT_PARAMETERS)
        * len(exp_module.OUTPUT_OBSERVABLES)
    )
    assert n_gradients == 48


def test_gradient_table_schema_is_stable(exp_module):
    assert exp_module.GRADIENT_TABLE_COLUMNS == (
        "scenario",
        "input_parameter",
        "output_observable",
        "theta0",
        "fd_step",
        "ad_grad",
        "fd_grad",
        "abs_error",
        "rel_error",
        "ad_converged",
        "fd_plus_converged",
        "fd_minus_converged",
        "base_residual_norm",
        "plus_residual_norm",
        "minus_residual_norm",
        "base_iterations",
        "plus_iterations",
        "minus_iterations",
        "units_input",
        "units_output",
        "notes",
    )


def test_fd_step_study_schema_is_stable(exp_module):
    assert exp_module.FD_STEP_STUDY_COLUMNS == (
        "selected_gradient_id",
        "scenario",
        "input_parameter",
        "output_observable",
        "fd_step",
        "ad_grad",
        "fd_grad",
        "abs_error",
        "rel_error",
        "fd_plus_converged",
        "fd_minus_converged",
    )


def test_export_all_writes_expected_csv_json_artifacts(exp_module, tmp_path: Path):
    gradient_rows = [
        exp_module.GradientTableRow(
            scenario="base",
            input_parameter="load_scale_mv_bus_2",
            output_observable="vm_mv_bus_2_pu",
            theta0=1.0,
            fd_step=1e-4,
            ad_grad=0.1,
            fd_grad=0.1,
            abs_error=0.0,
            rel_error=0.0,
            ad_converged=True,
            fd_plus_converged=True,
            fd_minus_converged=True,
            base_residual_norm=1e-12,
            plus_residual_norm=1e-12,
            minus_residual_norm=1e-12,
            base_iterations=10,
            plus_iterations=10,
            minus_iterations=10,
            units_input="dimensionless",
            units_output="p.u.",
            notes="",
        )
    ]
    summary_rows = exp_module.summarize_errors(gradient_rows)
    step_rows = [
        exp_module.StepStudyRow(
            selected_gradient_id="base:load_scale_mv_bus_2->vm_mv_bus_2_pu",
            scenario="base",
            input_parameter="load_scale_mv_bus_2",
            output_observable="vm_mv_bus_2_pu",
            fd_step=1e-4,
            ad_grad=0.1,
            fd_grad=0.1,
            abs_error=0.0,
            rel_error=0.0,
            fd_plus_converged=True,
            fd_minus_converged=True,
        )
    ]

    exp_module.export_all(gradient_rows, summary_rows, step_rows, tmp_path)

    for name in [
        "gradient_table.csv",
        "gradient_table.json",
        "error_summary.csv",
        "error_summary.json",
        "fd_step_study.csv",
        "fd_step_study.json",
        "metadata.json",
        "README.md",
    ]:
        assert (tmp_path / name).exists(), f"{name} was not exported"

    with (tmp_path / "gradient_table.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == exp_module.GRADIENT_TABLE_COLUMNS
        assert len(list(reader)) == 1

    with (tmp_path / "fd_step_study.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == exp_module.FD_STEP_STUDY_COLUMNS
        assert len(list(reader)) == 1

    with (tmp_path / "gradient_table.json").open(encoding="utf-8") as handle:
        data = json.load(handle)
        assert data[0]["scenario"] == "base"
