"""Lightweight tests for Experiment 2b plotting from existing artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def plot_module():
    import experiments.plot_exp02_gradient_figures as module

    return module


def _sample_gradient_df() -> pd.DataFrame:
    rows = []
    for scenario in ("base", "load_high", "sgen_high"):
        for input_parameter in (
            "load_scale_mv_bus_2",
            "sgen_scale_static_generator",
        ):
            for observable in ("vm_mv_bus_2_pu", "p_slack_mw"):
                ad = 1.0 if observable == "p_slack_mw" else -0.01
                fd = ad * (1.0 + 1e-8)
                rows.append(
                    {
                        "scenario": scenario,
                        "input_parameter": input_parameter,
                        "output_observable": observable,
                        "ad_grad": ad,
                        "fd_grad": fd,
                        "abs_error": abs(ad - fd),
                        "rel_error": abs(ad - fd) / max(abs(ad), abs(fd), 1e-12),
                    }
                )
    return pd.DataFrame(rows)


def _sample_fd_step_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "selected_gradient_id": [
                "base:load_scale_mv_bus_2->vm_mv_bus_2_pu",
                "base:load_scale_mv_bus_2->vm_mv_bus_2_pu",
                "base:sgen_scale_static_generator->p_slack_mw",
                "base:sgen_scale_static_generator->p_slack_mw",
            ],
            "scenario": ["base", "base", "base", "base"],
            "input_parameter": [
                "load_scale_mv_bus_2",
                "load_scale_mv_bus_2",
                "sgen_scale_static_generator",
                "sgen_scale_static_generator",
            ],
            "output_observable": [
                "vm_mv_bus_2_pu",
                "vm_mv_bus_2_pu",
                "p_slack_mw",
                "p_slack_mw",
            ],
            "fd_step": [1e-3, 1e-4, 1e-3, 1e-4],
            "ad_grad": [-0.01, -0.01, 1.0, 1.0],
            "fd_grad": [-0.01000001, -0.01, 1.000001, 1.0],
            "abs_error": [1e-8, 0.0, 1e-6, 0.0],
            "rel_error": [1e-6, 0.0, 1e-6, 0.0],
        }
    )


def _sample_error_summary_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scenario": ["base", "load_high", "sgen_high"],
            "input_parameter": [
                "load_scale_mv_bus_2",
                "load_scale_mv_bus_2",
                "load_scale_mv_bus_2",
            ],
            "max_rel_error": [1e-8, 2e-8, 3e-8],
            "median_rel_error": [1e-10, 2e-10, 3e-10],
        }
    )


def test_plot_module_is_importable(plot_module):
    assert plot_module is not None


def test_prettify_label_maps_known_names(plot_module):
    assert plot_module.prettify_label("load_scale_mv_bus_2") == "Load scale MV Bus 2"
    assert plot_module.prettify_label("sgen_scale_static_generator") == (
        "Static generator scale"
    )
    assert plot_module.prettify_label("vm_mv_bus_2_pu") == "|V| MV Bus 2"
    assert plot_module.prettify_label("p_trafo_hv_mw") == "Transformer HV P"


def test_safe_log10_handles_zero_values(plot_module):
    values = plot_module.safe_log10(np.asarray([0.0, 1e-8, -1e-4]))

    assert np.isfinite(values).all()
    np.testing.assert_allclose(values[0], np.log10(plot_module.EPS_LOG))
    np.testing.assert_allclose(values[1], -8.0)
    np.testing.assert_allclose(values[2], -4.0)


def test_heatmap_aggregation_works_with_small_dataframe(plot_module):
    observables, inputs, matrix = plot_module.aggregate_error_heatmap(
        _sample_gradient_df()
    )

    assert observables == ["vm_mv_bus_2_pu", "p_slack_mw"]
    assert inputs == ["load_scale_mv_bus_2", "sgen_scale_static_generator"]
    assert matrix.shape == (2, 2)
    assert np.isfinite(matrix).all()


def test_boxplot_grouping_by_observable(plot_module):
    observables, groups = plot_module.grouped_relative_errors_by_observable(
        _sample_gradient_df()
    )

    assert observables == ["vm_mv_bus_2_pu", "p_slack_mw"]
    assert len(groups) == 2
    assert all(len(group) == 6 for group in groups)


def test_fd_step_study_plot_writes_files(plot_module, tmp_path: Path):
    png_path, pdf_path = plot_module.plot_fd_step_study(
        _sample_fd_step_df(),
        tmp_path,
    )

    assert png_path.exists()
    assert pdf_path.exists()
    assert png_path.stat().st_size > 0
    assert pdf_path.stat().st_size > 0


def test_faceted_parity_plot_writes_files(plot_module, tmp_path: Path):
    png_path, pdf_path = plot_module.plot_parity(
        _sample_gradient_df(),
        tmp_path,
    )

    assert png_path.name == "fig01_ad_vs_fd_parity_by_observable.png"
    assert pdf_path.name == "fig01_ad_vs_fd_parity_by_observable.pdf"
    assert png_path.exists()
    assert pdf_path.exists()
    assert png_path.stat().st_size > 0
    assert pdf_path.stat().st_size > 0


def test_save_figure_writes_png_and_pdf(plot_module, tmp_path: Path):
    fig, ax = plot_module.plt.subplots()
    ax.plot([0, 1], [0, 1])

    png_path, pdf_path = plot_module.save_figure(fig, tmp_path, "sample")

    assert png_path.exists()
    assert pdf_path.exists()
    assert png_path.stat().st_size > 0
    assert pdf_path.stat().st_size > 0


def test_generate_figures_from_real_artifacts_when_present(plot_module, tmp_path: Path):
    required = [
        plot_module.RESULTS_DIR / "gradient_table.csv",
        plot_module.RESULTS_DIR / "error_summary.csv",
        plot_module.RESULTS_DIR / "fd_step_study.csv",
    ]
    if not all(path.exists() for path in required):
        pytest.skip("Exp. 2b artifacts are not present locally.")

    outputs = plot_module.generate_figures(plot_module.RESULTS_DIR, tmp_path)
    output_names = {path.name for path in outputs}
    expected = {
        "fig01_ad_vs_fd_parity_by_observable.png",
        "fig01_ad_vs_fd_parity_by_observable.pdf",
        "fig01a_ad_vs_fd_parity_global.png",
        "fig01a_ad_vs_fd_parity_global.pdf",
        "fig02_gradient_error_heatmap.png",
        "fig02_gradient_error_heatmap.pdf",
        "fig03_relative_error_boxplot.png",
        "fig03_relative_error_boxplot.pdf",
        "fig04_fd_step_study.png",
        "fig04_fd_step_study.pdf",
        "fig05_error_by_scenario.png",
        "fig05_error_by_scenario.pdf",
        "README.md",
    }
    assert expected <= output_names
    for name in expected:
        path = tmp_path / name
        assert path.exists()
        assert path.stat().st_size > 0


def test_error_by_scenario_plot_writes_files(plot_module, tmp_path: Path):
    png_path, pdf_path = plot_module.plot_error_by_scenario(
        _sample_error_summary_df(),
        tmp_path,
    )

    assert png_path.exists()
    assert pdf_path.exists()
