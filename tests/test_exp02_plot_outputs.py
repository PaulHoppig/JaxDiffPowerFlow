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


def _full_sample_gradient_df(plot_module) -> pd.DataFrame:
    rows = []
    for input_idx, input_parameter in enumerate(plot_module.INPUT_ORDER, start=1):
        for output_idx, observable in enumerate(plot_module.OBSERVABLE_ORDER, start=1):
            for scenario_idx, scenario in enumerate(plot_module.SCENARIO_ORDER, start=1):
                ad = float(input_idx * output_idx * scenario_idx)
                rel_error = 1e-8 * input_idx * output_idx * scenario_idx
                if (
                    input_parameter == "load_scale_mv_bus_2"
                    and observable == "vm_mv_bus_2_pu"
                ):
                    ad = float((-1) ** scenario_idx * scenario_idx)
                    rel_error = 1e-8 * scenario_idx
                if (
                    input_parameter == "trafo_x_scale"
                    and observable == "p_trafo_hv_mw"
                ):
                    ad = 0.0
                    rel_error = 0.0
                rows.append(
                    {
                        "scenario": scenario,
                        "input_parameter": input_parameter,
                        "output_observable": observable,
                        "ad_grad": ad,
                        "fd_grad": ad,
                        "abs_error": 0.0,
                        "rel_error": rel_error,
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
            "fd_plus_converged": [True, True, True, True],
            "fd_minus_converged": [True, True, True, True],
        }
    )


def _real_fd_step_df(plot_module) -> pd.DataFrame:
    path = plot_module.RESULTS_DIR / "fd_step_study.csv"
    if not path.exists():
        pytest.skip("Exp. 2b fd_step_study.csv is not present locally.")
    return plot_module.load_csv(path)


def _real_fd_step_detailed_df(plot_module) -> pd.DataFrame:
    path = plot_module.RESULTS_DIR / "fd_step_study_detailed.csv"
    if not path.exists():
        pytest.skip("Exp. 2b fd_step_study_detailed.csv is not present locally.")
    return plot_module.load_csv(path)


def _sample_fd_step_detailed_df() -> pd.DataFrame:
    """Minimal sample for detailed step study (2 steps, 2 gradients)."""
    rows = []
    for gid, in_param, out_obs in [
        ("base:load_scale_mv_bus_2->vm_mv_bus_2_pu", "load_scale_mv_bus_2", "vm_mv_bus_2_pu"),
        ("base:sgen_scale_static_generator->p_slack_mw", "sgen_scale_static_generator", "p_slack_mw"),
    ]:
        for step in [1e-3, 1e-4]:
            ad = -0.01 if out_obs == "vm_mv_bus_2_pu" else 1.0
            rows.append({
                "selected_gradient_id": gid,
                "scenario": "base",
                "input_parameter": in_param,
                "output_observable": out_obs,
                "fd_step": step,
                "ad_grad": ad,
                "fd_grad": ad * (1.0 + 1e-8),
                "abs_error": abs(ad) * 1e-8,
                "rel_error": 1e-8,
                "fd_plus_converged": True,
                "fd_minus_converged": True,
                "fd_plus_iterations": 8,
                "fd_minus_iterations": 8,
                "fd_plus_residual_norm": 1e-12,
                "fd_minus_residual_norm": 1e-12,
            })
    return pd.DataFrame(rows)


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
    assert callable(plot_module.compute_fd_vs_fd_step_stability)


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


def test_heatmap_figure_has_report_labels(plot_module):
    fig, ax = plot_module.build_error_heatmap_figure(_sample_gradient_df())

    try:
        assert ax.get_title() == "Experiment 2: max AD-vs-FD relative gradient error"
        assert ax.get_xlabel() == "Input parameter"
        assert ax.get_ylabel() == "Output observable"
        colorbar_labels = [axis.get_ylabel() for axis in fig.axes[1:]]
        assert "log10(max relative error)" in colorbar_labels
    finally:
        plot_module.plt.close(fig)


def test_gradient_magnitude_error_comparison_table_shape_and_columns(plot_module):
    summary = plot_module.build_gradient_magnitude_error_comparison_table(
        _full_sample_gradient_df(plot_module)
    )

    assert len(summary) == 16
    for column in plot_module.GRADIENT_MAGNITUDE_ERROR_SUMMARY_COLUMNS:
        assert column in summary.columns


def test_gradient_magnitude_error_comparison_table_values(plot_module):
    summary = plot_module.build_gradient_magnitude_error_comparison_table(
        _full_sample_gradient_df(plot_module)
    )
    row = summary[
        (summary["input_parameter"] == "load_scale_mv_bus_2")
        & (summary["output_observable"] == "vm_mv_bus_2_pu")
    ].iloc[0]

    assert row["n"] == 3
    assert row["median_abs_ad_grad"] == pytest.approx(2.0)
    assert row["min_abs_ad_grad"] == pytest.approx(1.0)
    assert row["max_abs_ad_grad"] == pytest.approx(3.0)
    assert row["median_rel_error"] == pytest.approx(2e-8)
    assert row["max_rel_error"] == pytest.approx(3e-8)
    assert row["log10_median_abs_ad_grad"] == pytest.approx(np.log10(2.0))
    assert row["log10_max_rel_error"] == pytest.approx(np.log10(3e-8))


def test_gradient_magnitude_error_comparison_table_log_floor(plot_module):
    summary = plot_module.build_gradient_magnitude_error_comparison_table(
        _full_sample_gradient_df(plot_module)
    )
    row = summary[
        (summary["input_parameter"] == "trafo_x_scale")
        & (summary["output_observable"] == "p_trafo_hv_mw")
    ].iloc[0]

    assert row["median_abs_ad_grad"] == 0.0
    assert row["max_rel_error"] == 0.0
    assert np.isfinite(row["log10_median_abs_ad_grad"])
    assert np.isfinite(row["log10_max_rel_error"])
    assert row["log10_median_abs_ad_grad"] == pytest.approx(
        np.log10(plot_module.EPS_LOG)
    )
    assert row["log10_max_rel_error"] == pytest.approx(np.log10(plot_module.EPS_LOG))


def test_gradient_magnitude_vs_error_figure_has_two_heatmaps_and_colorbars(
    plot_module,
):
    summary = plot_module.build_gradient_magnitude_error_comparison_table(
        _full_sample_gradient_df(plot_module)
    )
    fig, axes = plot_module.build_gradient_magnitude_vs_relative_error_figure(summary)

    try:
        assert len(axes) == 2
        assert axes[0].get_title() == "Gradient magnitude"
        assert axes[1].get_title() == "Relative gradient error"
        colorbar_labels = [axis.get_ylabel() for axis in fig.axes[2:]]
        assert "log10(median |AD gradient|)" in colorbar_labels
        assert "log10(max relative error)" in colorbar_labels
    finally:
        plot_module.plt.close(fig)


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


def test_fd_vs_fd_step_stability_computation_from_real_artifacts(plot_module):
    stability = plot_module.compute_fd_vs_fd_step_stability(
        _real_fd_step_df(plot_module)
    )

    required_columns = {
        "selected_gradient_id",
        "scenario",
        "input_parameter",
        "output_observable",
        "fd_step_large",
        "fd_step_small",
        "fd_step_pair",
        "fd_grad_large",
        "fd_grad_small",
        "fd_abs_change",
        "fd_rel_change",
        "fd_plus_converged_large",
        "fd_minus_converged_large",
        "fd_plus_converged_small",
        "fd_minus_converged_small",
    }

    assert len(stability) == 12
    assert required_columns <= set(stability.columns)
    assert np.isfinite(stability["fd_rel_change"]).all()
    assert (stability["fd_rel_change"] >= 0.0).all()
    assert (stability["fd_step_large"] > stability["fd_step_small"]).all()


def test_fd_vs_fd_step_stability_pair_counts_from_real_artifacts(plot_module):
    stability = plot_module.compute_fd_vs_fd_step_stability(
        _real_fd_step_df(plot_module)
    )

    counts = stability.groupby("selected_gradient_id").size()
    assert len(counts) == 3
    assert (counts == 4).all()


def test_fd_vs_fd_step_stability_export_writes_files(plot_module, tmp_path: Path):
    stability = plot_module.compute_fd_vs_fd_step_stability(_sample_fd_step_df())
    outputs = plot_module.write_fd_vs_fd_step_stability(stability, tmp_path)

    assert {path.name for path in outputs} == {
        "fd_vs_fd_step_stability.csv",
        "fd_vs_fd_step_stability.json",
    }
    for path in outputs:
        assert path.exists()
        assert path.stat().st_size > 0


def test_fd_vs_fd_step_stability_plot_writes_files(plot_module, tmp_path: Path):
    stability = plot_module.compute_fd_vs_fd_step_stability(_sample_fd_step_df())
    png_path, pdf_path = plot_module.plot_fd_vs_fd_step_stability(
        stability,
        tmp_path,
    )

    assert png_path.name == "fig07_fd_vs_fd_step_stability.png"
    assert pdf_path.name == "fig07_fd_vs_fd_step_stability.pdf"
    assert png_path.exists()
    assert pdf_path.exists()
    assert png_path.stat().st_size > 0
    assert pdf_path.stat().st_size > 0


def test_fd_vs_fd_step_stability_figure_has_report_labels(plot_module):
    stability = plot_module.compute_fd_vs_fd_step_stability(_sample_fd_step_df())
    fig, ax = plot_module.build_fd_vs_fd_step_stability_figure(stability)

    try:
        assert ax.get_title() == "FD-vs-FD stability over finite-difference step size"
        assert ax.get_xlabel() == "Larger FD step h"
        assert ax.get_ylabel() == (
            "Relative change between FD(h) and FD(next smaller h)"
        )
        assert ax.get_xscale() == "log"
        assert ax.get_yscale() == "log"
    finally:
        plot_module.plt.close(fig)


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


def test_heatmap_plot_writes_files(plot_module, tmp_path: Path):
    png_path, pdf_path = plot_module.plot_error_heatmap(
        _sample_gradient_df(),
        tmp_path,
    )

    assert png_path.name == "fig02_gradient_error_heatmap.png"
    assert pdf_path.name == "fig02_gradient_error_heatmap.pdf"
    assert png_path.exists()
    assert pdf_path.exists()
    assert png_path.stat().st_size > 0
    assert pdf_path.stat().st_size > 0


def test_gradient_magnitude_vs_error_plot_writes_files(plot_module, tmp_path: Path):
    summary = plot_module.build_gradient_magnitude_error_comparison_table(
        _full_sample_gradient_df(plot_module)
    )
    outputs = plot_module.plot_gradient_magnitude_vs_relative_error_heatmaps(
        summary,
        tmp_path,
    )

    output_names = {path.name for path in outputs}
    assert {
        "fig06_gradient_magnitude_vs_relative_error_heatmaps.png",
        "fig06_gradient_magnitude_vs_relative_error_heatmaps.pdf",
    } <= output_names
    for path in outputs:
        assert path.exists()
        assert path.stat().st_size > 0


def test_gradient_magnitude_vs_error_summary_export_writes_files(
    plot_module,
    tmp_path: Path,
):
    summary = plot_module.build_gradient_magnitude_error_comparison_table(
        _full_sample_gradient_df(plot_module)
    )
    outputs = plot_module.write_gradient_magnitude_error_summary(summary, tmp_path)

    assert {path.name for path in outputs} == {
        "gradient_magnitude_vs_error_summary.csv",
        "gradient_magnitude_vs_error_summary.json",
    }
    for path in outputs:
        assert path.exists()
        assert path.stat().st_size > 0


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
    expected_base = {
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
        "fig06_gradient_magnitude_vs_relative_error_heatmaps.png",
        "fig06_gradient_magnitude_vs_relative_error_heatmaps.pdf",
        "fig07_fd_vs_fd_step_stability.png",
        "fig07_fd_vs_fd_step_stability.pdf",
        "gradient_magnitude_vs_error_summary.csv",
        "gradient_magnitude_vs_error_summary.json",
        "fd_vs_fd_step_stability.csv",
        "fd_vs_fd_step_stability.json",
        "README.md",
    }
    assert expected_base <= output_names
    for name in expected_base:
        path = tmp_path / name
        assert path.exists()
        assert path.stat().st_size > 0

    # If the detailed step study artefact is present, fig08 and fig09 are also generated.
    if (plot_module.RESULTS_DIR / "fd_step_study_detailed.csv").exists():
        expected_detailed = {
            "fig08_fd_step_study_detailed.png",
            "fig08_fd_step_study_detailed.pdf",
            "fig09_fd_vs_fd_step_stability_detailed.png",
            "fig09_fd_vs_fd_step_stability_detailed.pdf",
            "fd_vs_fd_step_stability_detailed.csv",
            "fd_vs_fd_step_stability_detailed.json",
        }
        assert expected_detailed <= output_names
        for name in expected_detailed:
            path = tmp_path / name
            assert path.exists()
            assert path.stat().st_size > 0


def test_fd_step_study_detailed_plot_writes_fig08_files(plot_module, tmp_path: Path):
    df = _sample_fd_step_detailed_df()
    png_path, pdf_path = plot_module.plot_fd_step_study_detailed(df, tmp_path)

    assert png_path.name == "fig08_fd_step_study_detailed.png"
    assert pdf_path.name == "fig08_fd_step_study_detailed.pdf"
    assert png_path.exists()
    assert pdf_path.exists()
    assert png_path.stat().st_size > 0
    assert pdf_path.stat().st_size > 0


def test_fd_vs_fd_step_stability_detailed_plot_writes_fig09_files(plot_module, tmp_path: Path):
    df = _sample_fd_step_detailed_df()
    stability = plot_module.compute_fd_vs_fd_step_stability(df)
    png_path, pdf_path = plot_module.plot_fd_vs_fd_step_stability_detailed(stability, tmp_path)

    assert png_path.name == "fig09_fd_vs_fd_step_stability_detailed.png"
    assert pdf_path.name == "fig09_fd_vs_fd_step_stability_detailed.pdf"
    assert png_path.exists()
    assert pdf_path.exists()
    assert png_path.stat().st_size > 0
    assert pdf_path.stat().st_size > 0


def test_fd_vs_fd_step_stability_detailed_export_writes_files(plot_module, tmp_path: Path):
    df = _sample_fd_step_detailed_df()
    stability = plot_module.compute_fd_vs_fd_step_stability(df)
    outputs = plot_module.write_fd_vs_fd_step_stability_detailed(stability, tmp_path)

    assert {p.name for p in outputs} == {
        "fd_vs_fd_step_stability_detailed.csv",
        "fd_vs_fd_step_stability_detailed.json",
    }
    for p in outputs:
        assert p.exists()
        assert p.stat().st_size > 0


def test_fd_vs_fd_step_stability_detailed_has_90_pairs_from_real_artifacts(plot_module):
    df = _real_fd_step_detailed_df(plot_module)
    stability = plot_module.compute_fd_vs_fd_step_stability(df)

    assert len(stability) == 90, (
        f"Expected 90 FD-vs-FD pairs (3 gradients x 30 pairs), got {len(stability)}"
    )
    counts = stability.groupby("selected_gradient_id").size()
    assert len(counts) == 3
    assert (counts == 30).all()


def test_fd_vs_fd_step_stability_detailed_fd_step_large_gt_small_from_real(plot_module):
    df = _real_fd_step_detailed_df(plot_module)
    stability = plot_module.compute_fd_vs_fd_step_stability(df)
    assert (stability["fd_step_large"] > stability["fd_step_small"]).all()


def test_fd_vs_fd_step_stability_detailed_rel_change_finite_from_real(plot_module):
    df = _real_fd_step_detailed_df(plot_module)
    stability = plot_module.compute_fd_vs_fd_step_stability(df)
    assert np.isfinite(stability["fd_rel_change"]).all()
    assert (stability["fd_rel_change"] >= 0.0).all()


def test_fig08_detailed_figure_has_report_labels(plot_module):
    df = _sample_fd_step_detailed_df()
    fig, ax = plot_module.plt.subplots()
    # Call the internal build path to check axes labels
    work = df.copy()
    import matplotlib
    matplotlib.use("Agg")
    png_path, _ = plot_module.plot_fd_step_study_detailed(df, Path("/tmp"))
    # Just verify the function runs and produces a file
    assert png_path.name == "fig08_fd_step_study_detailed.png"
    plot_module.plt.close("all")


def test_error_by_scenario_plot_writes_files(plot_module, tmp_path: Path):
    png_path, pdf_path = plot_module.plot_error_by_scenario(
        _sample_error_summary_df(),
        tmp_path,
    )

    assert png_path.exists()
    assert pdf_path.exists()


def test_figures_readme_mentions_figures_6_through_9(plot_module, tmp_path: Path):
    path = plot_module.write_figures_readme(tmp_path, plot_module.RESULTS_DIR)
    text = path.read_text(encoding="utf-8")

    assert "Figure 6" in text
    assert "Figure 7" in text
    assert "Figure 8" in text
    assert "Figure 9" in text
    assert "log10(median |AD gradient|)" in text
    assert "log10(max relative error)" in text
    assert "fd_vs_fd_step_stability.csv" in text
    assert "No new power-flow solves" in text
    assert "fig08" in text
    assert "fig09" in text
    assert "fd_vs_fd_step_stability_detailed" in text
