"""Lightweight tests for Experiment 3 plotting from existing artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def plot_module():
    import experiments.plot_exp03_figures as module

    return module


def test_plot_module_is_importable(plot_module):
    assert plot_module is not None
    assert callable(plot_module.prepare_sensitivity_heatmap_grid)


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


def test_sensitivity_heatmap_grid_filters_and_pivots_g_gradient(plot_module):
    _, sensitivity_rows = plot_module.load_artifacts()
    grid = plot_module.prepare_sensitivity_heatmap_grid(
        sensitivity_rows,
        input_parameter="g_poa_wm2",
    )

    assert not grid.empty
    assert grid.shape == (5, 5)
    assert list(grid.index) == sorted(grid.index)
    assert list(grid.columns) == sorted(grid.columns)
    assert list(grid.columns) == [200.0, 400.0, 600.0, 800.0, 1000.0]
    assert list(grid.index) == [5.0, 15.0, 25.0, 35.0, 45.0]
    assert np.isfinite(grid.to_numpy(dtype=float)).all()

    df = pd.DataFrame(sensitivity_rows)
    expected = df[
        (df["weather_case_type"] == "grid_2d")
        & (df["network_scenario"] == "base")
        & (df["observable"] == "p_slack_mw")
        & (df["input_parameter"] == "g_poa_wm2")
    ]
    assert len(expected) == grid.size


def test_sensitivity_heatmap_grid_filters_and_pivots_temperature_gradient(
    plot_module,
):
    _, sensitivity_rows = plot_module.load_artifacts()
    grid = plot_module.prepare_sensitivity_heatmap_grid(
        sensitivity_rows,
        input_parameter="t_amb_c",
    )

    assert not grid.empty
    assert grid.shape == (5, 5)
    assert list(grid.index) == sorted(grid.index)
    assert list(grid.columns) == sorted(grid.columns)
    assert np.isfinite(grid.to_numpy(dtype=float)).all()

    df = pd.DataFrame(sensitivity_rows)
    expected = df[
        (df["weather_case_type"] == "grid_2d")
        & (df["network_scenario"] == "base")
        & (df["observable"] == "p_slack_mw")
        & (df["input_parameter"] == "t_amb_c")
    ]
    assert len(expected) == grid.size


def test_report_ticks_match_experiment_design(plot_module):
    assert plot_module.SWEEP_T_AMB_TICKS == (5, 15, 25, 35, 45, 55)
    assert plot_module.GRID_G_POA_TICKS == (200, 400, 600, 800, 1000)
    assert plot_module.GRID_T_AMB_TICKS == (5, 15, 25, 35, 45)
    assert plot_module.G_SWEEP_TICKS == (200, 400, 600, 800, 1000)


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


def test_sensitivity_values_are_converted_to_kw_per_degree(plot_module):
    rows = [{"value": "0.0065"}, {"value": "-0.001"}]

    assert plot_module.sensitivity_values_kw_per_c(rows) == [6.5, -1.0]


def test_sensitivity_values_are_converted_to_kw_per_wm2(plot_module):
    rows = [{"value": "0.00012"}, {"value": "-0.0005"}]

    assert plot_module.sensitivity_values_kw_per_wm2(rows) == pytest.approx([0.12, -0.5])


def test_sensitivity_heatmap_unit_transforms(plot_module):
    raw_grid = pd.DataFrame(
        [[-0.002, -0.001], [0.003, 0.004]],
        index=[5.0, 15.0],
        columns=[200.0, 400.0],
    )

    g_grid, g_label, g_unit = plot_module.sensitivity_heatmap_plot_grid(
        raw_grid,
        "g_poa_wm2",
    )
    t_grid, t_label, t_unit = plot_module.sensitivity_heatmap_plot_grid(
        raw_grid,
        "t_amb_c",
    )

    np.testing.assert_allclose(g_grid.to_numpy(), raw_grid.to_numpy() * 1000.0 * 100.0)
    np.testing.assert_allclose(t_grid.to_numpy(), raw_grid.to_numpy() * 1000.0)
    assert np.isfinite(g_grid.to_numpy(dtype=float)).all()
    assert np.isfinite(t_grid.to_numpy(dtype=float)).all()
    assert "kW per 100" in g_label
    assert g_unit == "kW per 100 W/m^2"
    assert "kW/" in t_label
    assert t_unit == "kW/degC"


def test_g_sweep_filter_returns_all_three_scenarios(plot_module):
    scenario_rows, _ = plot_module.load_artifacts()
    grouped = plot_module.select_fig04_rows(scenario_rows)

    assert set(grouped) == set(plot_module.SCENARIO_ORDER)
    assert all(len(rows) == 5 for rows in grouped.values())
    assert all(
        row["weather_case_type"] == "sweep_g_1d"
        and row["observable"] == "p_slack_mw"
        for rows in grouped.values()
        for row in rows
    )


def test_g_sweep_sensitivity_filter_returns_g_gradient(plot_module):
    _, sensitivity_rows = plot_module.load_artifacts()
    grouped = plot_module.select_fig05_rows(sensitivity_rows)

    assert set(grouped) == set(plot_module.SCENARIO_ORDER)
    assert all(len(rows) == 5 for rows in grouped.values())
    assert all(
        row["weather_case_type"] == "sweep_g_1d"
        and row["observable"] == "p_slack_mw"
        and row["input_parameter"] == "g_poa_wm2"
        for rows in grouped.values()
        for row in rows
    )


def test_padded_limits_zoom_around_gradient_values(plot_module):
    lower, upper = plot_module.padded_limits([6.3, 6.5, 6.7], pad_fraction=0.25)

    assert lower == pytest.approx(6.2)
    assert upper == pytest.approx(6.8)


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
        "fig04_g_sweep_p_slack.png",
        "fig04_g_sweep_p_slack.pdf",
        "fig05_sensitivity_p_slack_vs_g_poa.png",
        "fig05_sensitivity_p_slack_vs_g_poa.pdf",
        "fig06_heatmap_g_t_sensitivity_p_slack_wrt_g.png",
        "fig06_heatmap_g_t_sensitivity_p_slack_wrt_g.pdf",
        "fig07_heatmap_g_t_sensitivity_p_slack_wrt_t_amb.png",
        "fig07_heatmap_g_t_sensitivity_p_slack_wrt_t_amb.pdf",
        "README.md",
    }
    assert expected <= output_names
    for name in expected:
        path = tmp_path / name
        assert path.exists()
        assert path.stat().st_size > 0


def test_g_sweep_forward_plot_writes_files(plot_module, tmp_path: Path):
    scenario_rows, _ = plot_module.load_artifacts()
    outputs = plot_module.plot_g_sweep_p_slack(scenario_rows, tmp_path)

    names = {path.name for path in outputs}
    assert "fig04_g_sweep_p_slack.png" in names
    assert "fig04_g_sweep_p_slack.pdf" in names
    for path in outputs:
        assert path.exists()
        assert path.stat().st_size > 0


def test_g_sweep_sensitivity_plot_writes_files(plot_module, tmp_path: Path):
    _, sensitivity_rows = plot_module.load_artifacts()
    outputs = plot_module.plot_g_sweep_p_slack_sensitivity(
        sensitivity_rows,
        tmp_path,
    )

    names = {path.name for path in outputs}
    assert "fig05_sensitivity_p_slack_vs_g_poa.png" in names
    assert "fig05_sensitivity_p_slack_vs_g_poa.pdf" in names
    for path in outputs:
        assert path.exists()
        assert path.stat().st_size > 0


def test_sensitivity_heatmap_plots_write_files(plot_module, tmp_path: Path):
    _, sensitivity_rows = plot_module.load_artifacts()
    fig06 = plot_module.plot_fig06_heatmap_g_t_sensitivity_p_slack_wrt_g(
        sensitivity_rows,
        tmp_path,
    )
    fig07 = plot_module.plot_fig07_heatmap_g_t_sensitivity_p_slack_wrt_t_amb(
        sensitivity_rows,
        tmp_path,
    )

    names = {path.name for path in fig06 + fig07}
    assert "fig06_heatmap_g_t_sensitivity_p_slack_wrt_g.png" in names
    assert "fig06_heatmap_g_t_sensitivity_p_slack_wrt_g.pdf" in names
    assert "fig07_heatmap_g_t_sensitivity_p_slack_wrt_t_amb.png" in names
    assert "fig07_heatmap_g_t_sensitivity_p_slack_wrt_t_amb.pdf" in names
    for path in fig06 + fig07:
        assert path.exists()
        assert path.stat().st_size > 0


def test_g_sweep_plot_labels_contain_expected_terms(plot_module, tmp_path, monkeypatch):
    scenario_rows, sensitivity_rows = plot_module.load_artifacts()
    captured = []

    def fake_export(fig, stem, figures_dir):
        captured.append((fig, fig.axes[0], stem))
        png = tmp_path / f"{stem}.png"
        pdf = tmp_path / f"{stem}.pdf"
        png.write_bytes(b"png")
        pdf.write_bytes(b"pdf")
        return png, pdf

    monkeypatch.setattr(plot_module, "_plot_export", fake_export)
    plot_module.plot_g_sweep_p_slack(scenario_rows, tmp_path)
    plot_module.plot_g_sweep_p_slack_sensitivity(sensitivity_rows, tmp_path)

    try:
        forward_ax = captured[0][1]
        sensitivity_ax = captured[1][1]
        assert "G_poa" in forward_ax.get_xlabel()
        assert "P_slack" in forward_ax.get_ylabel()
        assert "G_poa" in sensitivity_ax.get_xlabel()
        assert "dP_slack/dG_poa" in sensitivity_ax.get_ylabel()
    finally:
        for fig, _, _ in captured:
            plot_module.plt.close(fig)
