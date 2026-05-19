"""Tests for the final Experiment 1 validation figure pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def plot_module():
    import experiments.plot_exp01_validation_figures as module

    return module


def _sample_errors(module) -> pd.DataFrame:
    rows = []
    for idx, scenario in enumerate(module.SCENARIO_ORDER, start=1):
        rows.append(
            {
                "scenario": scenario,
                "reference_mode": "scope_matched",
                "strict_validation": True,
                "max_vm_pu_abs_diff": 1e-14 * idx,
                "max_va_degree_abs_diff": 1e-12 * idx,
                "p_slack_mw_abs_diff": 1e-10 * idx,
                "q_slack_mvar_abs_diff": 2e-10 * idx,
                "total_p_loss_mw_abs_diff": 1e-12 * idx,
                "total_q_loss_mvar_abs_diff": 2e-12 * idx,
                "trafo_pl_mw_abs_diff": 1e-14 * idx,
                "trafo_ql_mvar_abs_diff": 2e-13 * idx,
            }
        )
    return pd.DataFrame(rows)


def _sample_results_dir(module, tmp_path: Path) -> Path:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    errors = _sample_errors(module)
    validation = errors.drop(
        columns=["trafo_pl_mw_abs_diff", "trafo_ql_mvar_abs_diff"]
    )
    validation.to_csv(results_dir / "validation_summary.csv", index=False)

    trafo_rows = []
    for _, row in errors.iterrows():
        q_diff = float(row["trafo_ql_mvar_abs_diff"])
        trafo_rows.append(
            {
                "scenario": row["scenario"],
                "reference_mode": "scope_matched",
                "pl_mw_abs_diff": row["trafo_pl_mw_abs_diff"],
                "q_hv_mvar_diffpf": q_diff,
                "q_lv_mvar_diffpf": 0.0,
                "q_hv_mvar_pp": 0.0,
                "q_lv_mvar_pp": 0.0,
            }
        )
    pd.DataFrame(trafo_rows).to_csv(results_dir / "trafo_flows.csv", index=False)
    return results_dir


def test_plot_module_is_importable(plot_module):
    assert plot_module is not None


def test_build_final_max_errors_table_uses_base_units(plot_module):
    table = plot_module.build_final_max_errors_table(_sample_errors(plot_module))

    assert set(table["unit"]) == {"p.u.", "deg", "MW", "MVAr"}
    assert "kW" not in set(table["unit"])
    assert "mdeg" not in set(table["unit"])
    assert "m.p.u." not in set(table["unit"])
    assert len(table) == 8
    assert table.loc[table["unit"] == "p.u.", "max_abs_error"].iloc[0] == pytest.approx(
        7e-14
    )


def test_export_final_max_errors_table_writes_csv_and_markdown(plot_module, tmp_path):
    table = plot_module.build_final_max_errors_table(_sample_errors(plot_module))

    outputs = plot_module.export_final_max_errors_table(table, tmp_path)
    names = {path.name for path in outputs}

    assert "final_max_errors_table.csv" in names
    assert "final_max_errors_table.md" in names
    csv_text = (tmp_path / "final_max_errors_table.csv").read_text(encoding="utf-8")
    md_text = (tmp_path / "final_max_errors_table.md").read_text(encoding="utf-8")
    assert "7.000e-14" in csv_text
    assert "7.000e-14" in md_text


def test_dotplot_writes_png_and_pdf(plot_module, tmp_path):
    png, pdf = plot_module.plot_scope_matched_error_dotplot(
        _sample_errors(plot_module),
        tmp_path,
    )

    assert png.name == "fig02_scope_matched_error_dotplot.png"
    assert pdf.name == "fig02_scope_matched_error_dotplot.pdf"
    assert png.exists() and png.stat().st_size > 0
    assert pdf.exists() and pdf.stat().st_size > 0


def test_heatmap_writes_png_and_pdf(plot_module, tmp_path):
    png, pdf = plot_module.plot_scope_matched_error_heatmap(
        _sample_errors(plot_module),
        tmp_path,
    )

    assert png.name == "fig03_scope_matched_error_heatmap_log10.png"
    assert pdf.name == "fig03_scope_matched_error_heatmap_log10.pdf"
    assert png.exists() and png.stat().st_size > 0
    assert pdf.exists() and pdf.stat().st_size > 0


def test_heatmap_uses_log10_and_handles_zero(plot_module):
    values = pd.DataFrame({"a": [1e-3, 1e-6, 0.0]})

    log_values = plot_module.compute_log10_errors(values, eps=1e-300)

    assert log_values.loc[0, "a"] == pytest.approx(-3.0)
    assert log_values.loc[1, "a"] == pytest.approx(-6.0)
    assert log_values.loc[2, "a"] == pytest.approx(-300.0)
    assert np.isfinite(log_values.to_numpy()).all()


def test_model_alignment_table_contains_expected_steps_and_base_units(plot_module):
    table = plot_module.build_model_alignment_error_reduction_table()

    assert set(table["model_step"]) == {
        "initial_scope_matched_before_trafo_fix",
        "after_trafo_magnetization_fix",
        "final_after_open_line_policy",
    }
    assert set(table["unit"]) <= {"p.u.", "deg", "MW", "-"}
    assert "kW" not in set(table["unit"])
    assert "mdeg" not in set(table["unit"])
    assert "m.p.u." not in set(table["unit"])

    initial = table[table["model_step"] == "initial_scope_matched_before_trafo_fix"]
    assert set(initial["metric"]) == {
        "p_slack_mw_abs_diff",
        "total_p_loss_mw_abs_diff",
        "trafo_pl_mw_abs_diff",
    }
    assert initial["source_note"].str.contains("representative offset").all()


def test_export_model_alignment_table_writes_csv_and_json(plot_module, tmp_path):
    table = plot_module.build_model_alignment_error_reduction_table()

    outputs = plot_module.export_model_alignment_error_reduction(table, tmp_path)
    names = {path.name for path in outputs}

    assert "model_alignment_error_reduction.csv" in names
    assert "model_alignment_error_reduction.json" in names
    csv_text = (tmp_path / "model_alignment_error_reduction.csv").read_text(
        encoding="utf-8"
    )
    json_text = (tmp_path / "model_alignment_error_reduction.json").read_text(
        encoding="utf-8"
    )
    assert "initial_scope_matched_before_trafo_fix" in csv_text
    assert "1.436e-02" in csv_text
    assert "0.000e+00" in json_text


def test_model_alignment_power_plot_writes_png_and_pdf(plot_module, tmp_path):
    table = plot_module.build_model_alignment_error_reduction_table()

    png, pdf = plot_module.plot_model_alignment_power_reduction(table, tmp_path)

    assert png.name == "fig04_model_alignment_error_reduction_power.png"
    assert pdf.name == "fig04_model_alignment_error_reduction_power.pdf"
    assert png.exists() and png.stat().st_size > 0
    assert pdf.exists() and pdf.stat().st_size > 0


def test_model_alignment_diagnostic_plot_handles_zero_and_writes_outputs(
    plot_module,
    tmp_path,
):
    table = plot_module.build_model_alignment_error_reduction_table()

    png, pdf = plot_module.plot_model_alignment_diagnostic_reduction(table, tmp_path)

    assert png.name == "fig05_model_alignment_diagnostic_reduction.png"
    assert pdf.name == "fig05_model_alignment_diagnostic_reduction.pdf"
    assert png.exists() and png.stat().st_size > 0
    assert pdf.exists() and pdf.stat().st_size > 0
    final_ybus = table[
        (table["model_step"] == "final_after_open_line_policy")
        & (table["metric"] == "ybus_max_abs_complex_diff")
    ]["abs_error"].iloc[0]
    assert final_ybus == pytest.approx(0.0)


def test_generate_figures_creates_expected_final_files(plot_module, tmp_path):
    results_dir = _sample_results_dir(plot_module, tmp_path)
    figures_dir = tmp_path / "figures"

    outputs = plot_module.generate_figures(results_dir, figures_dir)
    names = {path.name for path in outputs}

    assert set(plot_module.EXPECTED_OUTPUTS) <= names
    for name in plot_module.EXPECTED_OUTPUTS:
        path = figures_dir / name
        assert path.exists()
        assert path.stat().st_size > 0


def test_cleanup_removes_old_boxplot_outputs(plot_module, tmp_path):
    figures_dir = tmp_path / "figures"
    figures_dir.mkdir()
    old_file = figures_dir / "fig02_scope_matched_error_boxplots.png"
    old_file.write_text("old", encoding="utf-8")

    plot_module.cleanup_old_plot_outputs(figures_dir)

    assert not old_file.exists()


def test_real_artifacts_generate_final_outputs_when_present(plot_module, tmp_path):
    required = plot_module.RESULTS_DIR / "validation_summary.csv"
    if not required.exists():
        pytest.skip("Exp. 1 validation_summary.csv artifact is not present locally.")

    outputs = plot_module.generate_figures(plot_module.RESULTS_DIR, tmp_path)
    names = {path.name for path in outputs}
    assert set(plot_module.EXPECTED_OUTPUTS) <= names
