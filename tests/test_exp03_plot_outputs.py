"""Lightweight tests for Experiment 3 plotting from existing artifacts."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def plot_module():
    import experiments.plot_exp03_figures as module

    return module


def test_plot_module_is_importable(plot_module):
    assert plot_module is not None


def test_load_artifacts_reads_existing_csv_files(plot_module):
    scenario_rows, sensitivity_rows = plot_module.load_artifacts()

    assert scenario_rows
    assert sensitivity_rows
    assert "weather_case_type" in scenario_rows[0]
    assert "input_parameter" in sensitivity_rows[0]


def test_sweep_1d_filter_returns_all_three_scenarios(plot_module):
    scenario_rows, _ = plot_module.load_artifacts()
    grouped = plot_module.select_fig01_rows(scenario_rows)

    assert set(grouped) == set(plot_module.SCENARIO_ORDER)
    assert all(len(rows) == 6 for rows in grouped.values())
    assert all(
        row["weather_case_type"] == "sweep_1d"
        and row["observable"] == "p_slack_mw"
        for rows in grouped.values()
        for row in rows
    )


def test_grid_2d_pivot_has_expected_shape(plot_module):
    scenario_rows, _ = plot_module.load_artifacts()
    g_values, t_values, matrix = plot_module.pivot_grid_2d_base_p_slack(
        scenario_rows
    )

    assert len(g_values) == 5
    assert len(t_values) == 5
    assert matrix.shape == (5, 5)


def test_sensitivity_filter_returns_temperature_gradient(plot_module):
    _, sensitivity_rows = plot_module.load_artifacts()
    grouped = plot_module.select_fig03_rows(sensitivity_rows)

    assert set(grouped) == set(plot_module.SCENARIO_ORDER)
    assert all(len(rows) == 6 for rows in grouped.values())
    assert all(
        row["weather_case_type"] == "sweep_1d"
        and row["observable"] == "p_slack_mw"
        and row["input_parameter"] == "t_amb_c"
        for rows in grouped.values()
        for row in rows
    )


def test_generate_figures_writes_expected_files(plot_module, tmp_path: Path):
    outputs = plot_module.generate_figures(
        results_dir=plot_module.RESULTS_DIR,
        figures_dir=tmp_path,
    )
    output_names = {path.name for path in outputs}

    expected = {
        "fig01_t_amb_sweep_p_slack.png",
        "fig01_t_amb_sweep_p_slack.pdf",
        "fig02_heatmap_g_t_p_slack_base.png",
        "fig02_heatmap_g_t_p_slack_base.pdf",
        "fig03_sensitivity_p_slack_vs_t_amb.png",
        "fig03_sensitivity_p_slack_vs_t_amb.pdf",
        "README.md",
    }
    assert expected <= output_names
    for name in expected:
        path = tmp_path / name
        assert path.exists()
        assert path.stat().st_size > 0
