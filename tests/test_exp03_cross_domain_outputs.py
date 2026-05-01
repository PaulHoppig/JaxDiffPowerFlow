"""Schema and smoke tests for Experiment 3 – Cross-Domain PV Weather Sensitivity.

These tests validate:
- The experiment module is importable.
- Constants and column definitions have the expected shape/content.
- The export pipeline writes all mandatory artifacts with the right schema.
- Mandatory observables and weather inputs appear in the exported data.
- metadata.json and README.md are created.

Tests are deliberately lightweight: they do NOT run the full forward/gradient
loop. Instead, they construct minimal stub rows and call `export_all` directly.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Module fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def exp_module():
    import experiments.exp03_cross_domain_pv_weather as module

    return module


# ---------------------------------------------------------------------------
# Import and constants
# ---------------------------------------------------------------------------


def test_module_is_importable(exp_module):
    assert exp_module is not None


def test_electrical_scenarios_contain_required_three_points(exp_module):
    names = [name for name, _ in exp_module.ELECTRICAL_SCENARIOS]
    assert "base" in names
    assert "load_low" in names
    assert "load_high" in names
    assert len(names) == 3


def test_mandatory_observables_are_present(exp_module):
    names = [name for name, _ in exp_module.OBSERVABLE_SPECS]
    assert "vm_mv_bus_2_pu" in names
    assert "p_slack_mw" in names
    assert "total_p_loss_mw" in names
    assert "p_trafo_hv_mw" in names


def test_mandatory_weather_inputs_are_present(exp_module):
    names = [name for name, _ in exp_module.WEATHER_INPUT_SPECS]
    assert "g_poa_wm2" in names
    assert "t_amb_c" in names
    assert "wind_ms" in names


def test_fixed_constants_match_specification(exp_module):
    assert exp_module.EXP3_ALPHA == 1.0
    assert exp_module.EXP3_KAPPA == -0.25


def test_weather_grid_size(exp_module):
    n_2d = len(exp_module.G_LEVELS_WM2) * len(exp_module.T_LEVELS_C)
    n_1d = len(exp_module.T_SWEEP_C)
    assert n_2d == 25
    assert n_1d == 6
    assert len(exp_module.ALL_WEATHER_CASES) == n_2d + n_1d


def test_weather_cases_have_required_keys(exp_module):
    for case in exp_module.ALL_WEATHER_CASES:
        assert "weather_case_id" in case
        assert "weather_case_type" in case
        assert "g_poa_wm2" in case
        assert "t_amb_c" in case
        assert "wind_ms" in case


def test_weather_case_types_are_valid(exp_module):
    types = {c["weather_case_type"] for c in exp_module.ALL_WEATHER_CASES}
    assert types == {"grid_2d", "sweep_1d"}


def test_spotcheck_cases_cover_all_three_weather_inputs(exp_module):
    inputs_covered = {case[2] for case in exp_module.SPOTCHECK_CASES}
    assert "g_poa_wm2" in inputs_covered
    assert "t_amb_c" in inputs_covered
    assert "wind_ms" in inputs_covered


# ---------------------------------------------------------------------------
# Column schema tests
# ---------------------------------------------------------------------------


def test_scenario_grid_columns_contain_required_fields(exp_module):
    cols = exp_module.SCENARIO_GRID_COLUMNS
    for required in [
        "network_scenario",
        "weather_case_id",
        "weather_case_type",
        "g_poa_wm2",
        "t_amb_c",
        "wind_ms",
        "cell_temp_c",
        "p_pv_mw",
        "q_pv_mvar",
        "observable",
        "value",
        "unit",
        "converged",
        "iterations",
        "residual_norm",
    ]:
        assert required in cols, f"Missing column: {required}"


def test_sensitivity_columns_contain_required_fields(exp_module):
    cols = exp_module.SENSITIVITY_COLUMNS
    for required in [
        "network_scenario",
        "weather_case_id",
        "observable",
        "observable_unit",
        "input_parameter",
        "input_unit",
        "value",
        "ad_converged",
    ]:
        assert required in cols, f"Missing column: {required}"


def test_spotcheck_columns_contain_required_fields(exp_module):
    cols = exp_module.SPOTCHECK_COLUMNS
    for required in [
        "spotcheck_id",
        "network_scenario",
        "observable",
        "input_parameter",
        "g_poa_wm2",
        "t_amb_c",
        "wind_ms",
        "fd_step",
        "ad_grad",
        "fd_grad",
        "abs_error",
        "rel_error",
        "ad_converged",
        "fd_plus_converged",
        "fd_minus_converged",
    ]:
        assert required in cols, f"Missing column: {required}"


def test_run_summary_columns_contain_required_fields(exp_module):
    cols = exp_module.RUN_SUMMARY_COLUMNS
    for required in [
        "network_scenario",
        "load_factor",
        "n_weather_cases",
        "n_converged",
        "n_failed",
    ]:
        assert required in cols, f"Missing column: {required}"


# ---------------------------------------------------------------------------
# Export pipeline smoke test (lightweight – uses stub rows)
# ---------------------------------------------------------------------------


def _make_stub_grid_row(exp_module):
    return exp_module.ScenarioGridRow(
        network_scenario="base",
        load_factor=1.0,
        weather_case_id="grid2d_g800_t25_w2",
        weather_case_type="grid_2d",
        g_poa_wm2=800.0,
        t_amb_c=25.0,
        wind_ms=2.0,
        cell_temp_c=40.0,
        p_pv_mw=1.85,
        q_pv_mvar=-0.4625,
        observable="vm_mv_bus_2_pu",
        value=0.9999,
        unit="p.u.",
        converged=True,
        iterations=5,
        residual_norm=1e-11,
    )


def _make_stub_sensitivity_row(exp_module):
    return exp_module.SensitivityRow(
        network_scenario="base",
        load_factor=1.0,
        weather_case_id="grid2d_g800_t25_w2",
        weather_case_type="grid_2d",
        g_poa_wm2=800.0,
        t_amb_c=25.0,
        wind_ms=2.0,
        observable="vm_mv_bus_2_pu",
        observable_unit="p.u.",
        input_parameter="g_poa_wm2",
        input_unit="W/m^2",
        value=1.23e-5,
        ad_converged=True,
    )


def _make_stub_spotcheck_row(exp_module):
    return exp_module.SpotCheckRow(
        spotcheck_id="base:vm_mv_bus_2_pu__d_g_poa_wm2",
        network_scenario="base",
        observable="vm_mv_bus_2_pu",
        observable_unit="p.u.",
        input_parameter="g_poa_wm2",
        input_unit="W/m^2",
        g_poa_wm2=800.0,
        t_amb_c=25.0,
        wind_ms=2.0,
        fd_step=5.0,
        ad_grad=1.23e-5,
        fd_grad=1.23e-5,
        abs_error=1e-10,
        rel_error=1e-5,
        ad_converged=True,
        fd_plus_converged=True,
        fd_minus_converged=True,
    )


def _make_stub_summary_row(exp_module):
    return exp_module.RunSummaryRow(
        network_scenario="base",
        load_factor=1.0,
        n_weather_cases=31,
        n_converged=31,
        n_failed=0,
        min_vm_mv_bus_2_pu=0.98,
        max_vm_mv_bus_2_pu=1.01,
        min_p_pv_mw=0.1,
        max_p_pv_mw=2.0,
    )


def test_export_all_writes_mandatory_files(exp_module, tmp_path: Path):
    grid_rows = [_make_stub_grid_row(exp_module)]
    sensitivity_rows = [_make_stub_sensitivity_row(exp_module)]
    spotcheck_rows = [_make_stub_spotcheck_row(exp_module)]
    summary_rows = [_make_stub_summary_row(exp_module)]

    exp_module.export_all(grid_rows, sensitivity_rows, spotcheck_rows, summary_rows, tmp_path)

    mandatory = [
        "scenario_grid.csv",
        "scenario_grid.json",
        "sensitivity_table.csv",
        "sensitivity_table.json",
        "gradient_spotcheck.csv",
        "gradient_spotcheck.json",
        "run_summary.csv",
        "run_summary.json",
        "metadata.json",
        "README.md",
    ]
    for name in mandatory:
        assert (tmp_path / name).exists(), f"Missing artifact: {name}"


def test_scenario_grid_csv_has_correct_schema(exp_module, tmp_path: Path):
    grid_rows = [_make_stub_grid_row(exp_module)]
    exp_module._write_csv(
        tmp_path / "scenario_grid.csv", grid_rows, exp_module.SCENARIO_GRID_COLUMNS
    )

    with (tmp_path / "scenario_grid.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == exp_module.SCENARIO_GRID_COLUMNS
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["observable"] == "vm_mv_bus_2_pu"
        assert rows[0]["network_scenario"] == "base"


def test_sensitivity_table_csv_has_correct_schema(exp_module, tmp_path: Path):
    sensitivity_rows = [_make_stub_sensitivity_row(exp_module)]
    exp_module._write_csv(
        tmp_path / "sensitivity_table.csv",
        sensitivity_rows,
        exp_module.SENSITIVITY_COLUMNS,
    )

    with (tmp_path / "sensitivity_table.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == exp_module.SENSITIVITY_COLUMNS
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["input_parameter"] == "g_poa_wm2"
        assert rows[0]["observable"] == "vm_mv_bus_2_pu"


def test_gradient_spotcheck_csv_has_correct_schema(exp_module, tmp_path: Path):
    spotcheck_rows = [_make_stub_spotcheck_row(exp_module)]
    exp_module._write_csv(
        tmp_path / "gradient_spotcheck.csv",
        spotcheck_rows,
        exp_module.SPOTCHECK_COLUMNS,
    )

    with (tmp_path / "gradient_spotcheck.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == exp_module.SPOTCHECK_COLUMNS
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["spotcheck_id"] == "base:vm_mv_bus_2_pu__d_g_poa_wm2"


def test_metadata_json_contains_required_keys(exp_module, tmp_path: Path):
    exp_module.write_metadata(tmp_path)
    with (tmp_path / "metadata.json").open(encoding="utf-8") as handle:
        meta = json.load(handle)

    assert "experiment" in meta
    assert "model_constants" in meta
    assert meta["model_constants"]["alpha"] == 1.0
    assert meta["model_constants"]["kappa"] == -0.25
    assert "weather_design" in meta
    assert "known_simplifications" in meta
    assert "electrical_scenarios" in meta
    assert "observables" in meta
    assert "weather_inputs" in meta


def test_readme_md_is_written(exp_module, tmp_path: Path):
    exp_module.write_readme(tmp_path)
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "Experiment 3" in readme
    assert "scenario_grid" in readme
    assert "sensitivity_table" in readme
    assert "gradient_spotcheck" in readme


def test_scenario_grid_json_is_valid(exp_module, tmp_path: Path):
    grid_rows = [_make_stub_grid_row(exp_module)]
    exp_module._write_json(tmp_path / "scenario_grid.json", grid_rows)

    with (tmp_path / "scenario_grid.json").open(encoding="utf-8") as handle:
        data = json.load(handle)
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["network_scenario"] == "base"
    assert data[0]["observable"] == "vm_mv_bus_2_pu"


def test_sensitivity_json_has_mandatory_observable_and_input(exp_module, tmp_path: Path):
    sensitivity_rows = [_make_stub_sensitivity_row(exp_module)]
    exp_module._write_json(tmp_path / "sensitivity_table.json", sensitivity_rows)

    with (tmp_path / "sensitivity_table.json").open(encoding="utf-8") as handle:
        data = json.load(handle)
    assert data[0]["observable"] == "vm_mv_bus_2_pu"
    assert data[0]["input_parameter"] == "g_poa_wm2"
    assert isinstance(data[0]["value"], float)
    assert isinstance(data[0]["ad_converged"], bool)


# ---------------------------------------------------------------------------
# Scenario total count sanity
# ---------------------------------------------------------------------------


def test_total_scenario_count_is_compact(exp_module):
    n_electrical = len(exp_module.ELECTRICAL_SCENARIOS)
    n_weather = len(exp_module.ALL_WEATHER_CASES)
    n_total = n_electrical * n_weather
    # Must be compact: no more than 200 total forward solves
    assert n_total <= 200, f"Too many scenarios: {n_total}"
    assert n_total >= 3, "At least one case per operating point"


def test_expected_sensitivity_row_count_per_case(exp_module):
    n_obs = len(exp_module.OBSERVABLE_SPECS)
    n_inputs = len(exp_module.WEATHER_INPUT_SPECS)
    expected_per_case = n_obs * n_inputs
    assert expected_per_case == 12  # 4 obs x 3 inputs


def test_spotcheck_has_at_least_three_cases(exp_module):
    assert len(exp_module.SPOTCHECK_CASES) >= 3
