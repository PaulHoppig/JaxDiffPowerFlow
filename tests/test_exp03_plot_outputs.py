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
