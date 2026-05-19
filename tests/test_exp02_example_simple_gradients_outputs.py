"""Schema tests for the example_simple gradient-validation experiment."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

CANONICAL_RESULTS = (
    Path(__file__).resolve().parent.parent
    / "experiments"
    / "results"
    / "exp02_example_simple_gradients"
)
EXPECTED_DETAILED_STEPS = (
    1e+0, 5e-1, 2e-1,
    1e-1, 5e-2, 2e-2,
    1e-2, 5e-3, 2e-3,
    1e-3, 5e-4, 2e-4,
    1e-4, 5e-5, 2e-5,
    1e-5, 5e-6, 2e-6,
    1e-6, 5e-7, 2e-7,
    1e-7, 5e-8, 2e-8,
    1e-8, 5e-9, 2e-9,
    1e-9, 5e-10, 2e-10,
    1e-10,
)


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


def test_detailed_step_study_columns_are_correct(exp_module):
    assert exp_module.FD_STEP_STUDY_DETAILED_COLUMNS == (
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
        "fd_plus_iterations",
        "fd_minus_iterations",
        "fd_plus_residual_norm",
        "fd_minus_residual_norm",
    )


def test_fd_steps_detailed_constant_has_31_entries(exp_module):
    assert len(exp_module.FD_STEPS_DETAILED) == 31
    assert exp_module.FD_STEPS_DETAILED[0] == pytest.approx(1e0)
    assert exp_module.FD_STEPS_DETAILED[-1] == pytest.approx(1e-10)


def test_export_all_writes_detailed_artifacts_when_provided(exp_module, tmp_path: Path):
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
    detailed_rows = [
        exp_module.StepStudyDetailedRow(
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
            fd_plus_iterations=8,
            fd_minus_iterations=8,
            fd_plus_residual_norm=1e-12,
            fd_minus_residual_norm=1e-12,
        )
    ]
    summary_detailed = exp_module.build_fd_step_study_detailed_summary(detailed_rows)

    exp_module.export_all(
        gradient_rows,
        summary_rows,
        step_rows,
        tmp_path,
        step_detailed_rows=detailed_rows,
        step_detailed_summary_rows=summary_detailed,
    )

    for name in [
        "fd_step_study_detailed.csv",
        "fd_step_study_detailed.json",
        "fd_step_study_detailed_summary.csv",
        "fd_step_study_detailed_summary.json",
    ]:
        assert (tmp_path / name).exists(), f"{name} was not exported"

    with (tmp_path / "fd_step_study_detailed.csv").open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        assert tuple(reader.fieldnames or ()) == exp_module.FD_STEP_STUDY_DETAILED_COLUMNS
        assert len(list(reader)) == 1

    meta = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert "fd_step_detailed_steps" in meta
    assert len(meta["fd_step_detailed_steps"]) == 11


# ── Integration tests against the canonical results directory ─────────────────

def _skip_if_missing(path: Path):
    if not path.exists():
        pytest.skip(f"Canonical artefact not present: {path.name}")


def test_canonical_gradient_table_exists_and_has_48_rows():
    p = CANONICAL_RESULTS / "gradient_table.csv"
    _skip_if_missing(p)
    with p.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 48, f"Expected 48 gradient rows, got {len(rows)}"


def test_canonical_error_summary_artifacts_exist():
    for name in ("error_summary.csv", "error_summary.json"):
        p = CANONICAL_RESULTS / name
        _skip_if_missing(p)
        assert p.stat().st_size > 0


def test_canonical_metadata_has_fd_step_detailed_steps():
    p = CANONICAL_RESULTS / "metadata.json"
    _skip_if_missing(p)
    meta = json.loads(p.read_text(encoding="utf-8"))
    assert "fd_step_detailed_steps" in meta, "metadata.json must contain fd_step_detailed_steps"
    assert len(meta["fd_step_detailed_steps"]) == 11


def test_canonical_fd_step_study_detailed_exists():
    for name in ("fd_step_study_detailed.csv", "fd_step_study_detailed.json"):
        p = CANONICAL_RESULTS / name
        _skip_if_missing(p)
        assert p.stat().st_size > 0


def test_canonical_fd_step_study_detailed_has_93_rows():
    p = CANONICAL_RESULTS / "fd_step_study_detailed.csv"
    _skip_if_missing(p)
    with p.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 93, f"Expected 93 detailed step-study rows (3 gradients x 31 steps), got {len(rows)}"


def test_canonical_fd_step_study_detailed_step_values():
    p = CANONICAL_RESULTS / "fd_step_study_detailed.csv"
    _skip_if_missing(p)
    with p.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    observed = sorted({float(row["fd_step"]) for row in rows})
    expected = sorted(EXPECTED_DETAILED_STEPS)
    assert len(observed) == len(expected)
    for obs, exp in zip(observed, expected):
        assert obs == pytest.approx(exp, rel=1e-6), f"Unexpected fd_step value: {obs}"


def test_canonical_fd_step_study_detailed_columns():
    p = CANONICAL_RESULTS / "fd_step_study_detailed.csv"
    _skip_if_missing(p)
    with p.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or ()
    required = {
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
        "fd_plus_iterations",
        "fd_minus_iterations",
        "fd_plus_residual_norm",
        "fd_minus_residual_norm",
    }
    missing = required - set(fieldnames)
    assert not missing, f"Missing columns in fd_step_study_detailed.csv: {missing}"


def test_canonical_fd_step_study_detailed_rel_error_finite_when_converged():
    p = CANONICAL_RESULTS / "fd_step_study_detailed.csv"
    _skip_if_missing(p)
    with p.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        plus_ok = row["fd_plus_converged"].lower() in ("true", "1")
        minus_ok = row["fd_minus_converged"].lower() in ("true", "1")
        if plus_ok and minus_ok:
            rel = float(row["rel_error"])
            assert math.isfinite(rel), (
                f"rel_error should be finite when both FD directions converged: {row}"
            )


def test_canonical_fd_step_study_detailed_no_nested_lists():
    p = CANONICAL_RESULTS / "fd_step_study_detailed.csv"
    _skip_if_missing(p)
    with p.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        for key, val in row.items():
            assert "[" not in str(val), (
                f"CSV cell contains nested list in column '{key}': {val!r}"
            )
