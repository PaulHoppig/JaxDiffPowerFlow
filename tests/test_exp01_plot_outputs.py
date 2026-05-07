"""Lightweight tests for Experiment 1 plotting from existing artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def plot_module():
    import experiments.plot_exp01_validation_figures as module

    return module


def _sample_summary_df(module) -> pd.DataFrame:
    rows = []
    for idx, scenario in enumerate(module.SCENARIO_ORDER, start=1):
        row = {
            "scenario": scenario,
            "reference_mode": "scope_matched",
            "strict_validation": True,
            "max_vm_pu_abs_diff": 1e-5 * idx,
            "max_va_degree_abs_diff": 2e-4 * idx,
            "p_slack_mw_abs_diff": 0.01 * idx,
            "q_slack_mvar_abs_diff": 0.02 * idx,
            "total_p_loss_mw_abs_diff": 0.03 * idx,
            "total_q_loss_mvar_abs_diff": 0.04 * idx,
        }
        rows.append(row)
        original = row.copy()
        original["reference_mode"] = "original_pandapower"
        original["strict_validation"] = False
        rows.append(original)
    return pd.DataFrame(rows)


def test_plot_module_is_importable(plot_module):
    assert plot_module is not None


def test_filter_scope_matched_keeps_only_strict_scope_matched(plot_module):
    filtered = plot_module.filter_scope_matched(_sample_summary_df(plot_module))

    assert len(filtered) == len(plot_module.SCENARIO_ORDER)
    assert set(filtered["reference_mode"]) == {"scope_matched"}
    assert filtered["strict_validation"].all()


def test_build_error_long_table_has_expected_shape_and_columns(plot_module):
    filtered = plot_module.filter_scope_matched(_sample_summary_df(plot_module))
    long_df = plot_module.build_error_long_table(filtered)

    assert len(long_df) == 7 * 6
    for column in plot_module.LONG_TABLE_COLUMNS:
        assert column in long_df.columns


def test_unit_conversions_work(plot_module):
    filtered = plot_module.filter_scope_matched(_sample_summary_df(plot_module))
    long_df = plot_module.build_error_long_table(filtered)
    base_rows = long_df[long_df["scenario"] == "base"]

    p_slack = base_rows[base_rows["metric_key"] == "p_slack_mw_abs_diff"].iloc[0]
    angle = base_rows[base_rows["metric_key"] == "max_va_degree_abs_diff"].iloc[0]
    voltage = base_rows[base_rows["metric_key"] == "max_vm_pu_abs_diff"].iloc[0]

    assert p_slack["display_value"] == pytest.approx(10.0)
    assert p_slack["display_unit"] == "kW"
    assert angle["display_value"] == pytest.approx(0.2)
    assert angle["display_unit"] == "mdeg"
    assert voltage["display_value"] == pytest.approx(0.01)
    assert voltage["display_unit"] == "m.p.u."


def test_build_error_stability_summary_has_one_row_per_metric(plot_module):
    filtered = plot_module.filter_scope_matched(_sample_summary_df(plot_module))
    long_df = plot_module.build_error_long_table(filtered)
    summary = plot_module.build_error_stability_summary(long_df)

    assert len(summary) == len(plot_module.ERROR_METRICS)
    assert set(summary["metric_key"]) == set(plot_module.ERROR_METRICS)
    assert (summary["n"] == len(plot_module.SCENARIO_ORDER)).all()


def test_coefficient_of_variation_is_nan_for_zero_mean(plot_module):
    long_df = pd.DataFrame(
        {
            "scenario": ["a", "b", "c"],
            "scenario_order": [1, 2, 3],
            "metric_key": ["zero_mean"] * 3,
            "metric_label": ["Zero mean"] * 3,
            "raw_value": [-1.0, 0.0, 1.0],
            "display_value": [-1000.0, 0.0, 1000.0],
            "display_unit": ["kW"] * 3,
            "reference_mode": ["scope_matched"] * 3,
        }
    )
    summary = plot_module.build_error_stability_summary(long_df)

    assert np.isnan(summary.loc[0, "coefficient_of_variation_raw"])
    assert np.isnan(summary.loc[0, "coefficient_of_variation_display"])


def test_plot_functions_write_png_and_pdf(plot_module, tmp_path: Path):
    filtered = plot_module.filter_scope_matched(_sample_summary_df(plot_module))
    long_df = plot_module.build_error_long_table(filtered)

    outputs = []
    outputs.extend(plot_module.plot_error_by_scenario(long_df, tmp_path))
    outputs.extend(plot_module.plot_error_boxplots(long_df, tmp_path))

    output_names = {path.name for path in outputs}
    assert {
        "fig01_scope_matched_error_by_scenario.png",
        "fig01_scope_matched_error_by_scenario.pdf",
        "fig02_scope_matched_error_boxplots.png",
        "fig02_scope_matched_error_boxplots.pdf",
    } <= output_names
    for path in outputs:
        assert path.exists()
        assert path.stat().st_size > 0


def test_readme_and_table_exports_work(plot_module, tmp_path: Path):
    filtered = plot_module.filter_scope_matched(_sample_summary_df(plot_module))
    long_df = plot_module.build_error_long_table(filtered)
    summary = plot_module.build_error_stability_summary(long_df)

    outputs = []
    outputs.extend(plot_module.write_long_table(long_df, tmp_path))
    outputs.extend(plot_module.write_summary_tables(summary, tmp_path))
    outputs.append(plot_module.write_figures_readme(tmp_path))

    expected = {
        "scope_matched_error_long_table.csv",
        "scope_matched_error_long_table.json",
        "scope_matched_error_stability_summary.csv",
        "scope_matched_error_stability_summary.json",
        "README.md",
    }
    assert expected <= {path.name for path in outputs}
    for path in outputs:
        assert path.exists()
        assert path.stat().st_size > 0

    with (tmp_path / "scope_matched_error_long_table.json").open(encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list)
    assert len(data) == 42


def test_generate_figures_from_real_artifacts_when_present(plot_module, tmp_path: Path):
    required = plot_module.RESULTS_DIR / "validation_summary.csv"
    if not required.exists():
        pytest.skip("Exp. 1 validation_summary.csv artifact is not present locally.")

    outputs = plot_module.generate_figures(plot_module.RESULTS_DIR, tmp_path)
    output_names = {path.name for path in outputs}
    expected = {
        "scope_matched_error_long_table.csv",
        "scope_matched_error_long_table.json",
        "scope_matched_error_stability_summary.csv",
        "scope_matched_error_stability_summary.json",
        "fig01_scope_matched_error_by_scenario.png",
        "fig01_scope_matched_error_by_scenario.pdf",
        "fig02_scope_matched_error_boxplots.png",
        "fig02_scope_matched_error_boxplots.pdf",
        "README.md",
    }
    assert expected <= output_names
    for name in expected:
        path = tmp_path / name
        assert path.exists()
        assert path.stat().st_size > 0
