from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

import experiments.exp01_diagnose_residual_voltage_offset as diagnostic


def test_select_worst_case_uses_scope_matched_maximum() -> None:
    validation = pd.DataFrame(
        [
            {
                "scenario": "base",
                "reference_mode": "scope_matched",
                "max_vm_pu_abs_diff": 1.0e-5,
                "max_va_degree_abs_diff": 2.0e-4,
            },
            {
                "scenario": "target",
                "reference_mode": "scope_matched",
                "max_vm_pu_abs_diff": 3.0e-5,
                "max_va_degree_abs_diff": 4.0e-4,
            },
            {
                "scenario": "ignored",
                "reference_mode": "original_pandapower",
                "max_vm_pu_abs_diff": 9.0,
                "max_va_degree_abs_diff": 9.0,
            },
        ]
    )
    buses = pd.DataFrame(
        [
            {
                "scenario": "target",
                "reference_mode": "scope_matched",
                "bus_name": "1",
                "bus_original_id": 1,
                "bus_internal_id": 0,
                "vm_pu_diffpf": 1.0,
                "vm_pu_pp": 1.0,
                "vm_pu_abs_diff": 0.0,
                "va_degree_diffpf": 0.0,
                "va_degree_pp": 0.0,
            },
            {
                "scenario": "target",
                "reference_mode": "scope_matched",
                "bus_name": "5",
                "bus_original_id": 5,
                "bus_internal_id": 3,
                "vm_pu_diffpf": 0.99,
                "vm_pu_pp": 1.01,
                "vm_pu_abs_diff": 0.02,
                "va_degree_diffpf": 1.0,
                "va_degree_pp": 1.5,
            },
        ]
    )

    worst = diagnostic.select_worst_case(validation, buses)

    assert worst["scenario"] == "target"
    assert worst["bus_name"] == "5"
    assert math.isclose(worst["vm_pu_diff"], -0.02)
    assert math.isclose(worst["va_degree_diff"], -0.5)


def test_calculate_ybus_difference_metrics_on_synthetic_matrices() -> None:
    reference = np.asarray([[1 + 2j, 0], [0, 3 + 4j]], dtype=complex)
    candidate = np.asarray([[1 + 2j, 0.1j], [0, 3.25 + 4j]], dtype=complex)

    metrics, entries = diagnostic.calculate_ybus_difference_metrics(
        reference,
        candidate,
        ["a", "b"],
    )

    assert math.isclose(metrics["max_abs_complex_diff"], 0.25)
    assert metrics["largest_diff_from_bus"] == "b"
    assert metrics["largest_diff_to_bus"] == "b"
    assert metrics["n_entries_abs_diff_gt_1e_12"] == 2
    assert list(entries.columns).count("abs_complex_diff") == 1


def test_diagnostic_artifact_writers_use_requested_output_dir(tmp_path) -> None:
    output_dir = tmp_path / "diagnostic"
    df = pd.DataFrame([{"a": 1.0, "b": "x"}])
    diagnostic._write_df(df, "sample", output_dir)
    diagnostic._write_json({"ok": True}, output_dir / "sample_payload.json")

    assert (output_dir / "sample.csv").exists()
    assert (output_dir / "sample.json").exists()
    assert json.loads((output_dir / "sample_payload.json").read_text()) == {"ok": True}
    assert diagnostic.RESULTS_DIR != diagnostic.MAIN_RESULTS_DIR


def test_write_readme_creates_summary_without_main_artifact_writes(tmp_path) -> None:
    worst_case = {
        "scenario": "target",
        "bus_name": "5",
        "max_vm_pu_abs_diff": 2.3e-5,
        "max_va_degree_abs_diff": 2.5e-4,
        "vm_pu_diffpf": 1.0,
        "vm_pu_pandapower": 1.000023,
        "vm_pu_diff": -2.3e-5,
    }
    algebraic = pd.DataFrame(
        [{"match": False}, {"match": True}],
    )
    ybus_metrics = {
        "max_abs_complex_diff": 4.0e-3,
        "frobenius_norm_diff": 4.0e-3,
        "max_abs_real_diff": 2.0e-8,
        "max_abs_imag_diff": 4.0e-3,
        "largest_diff_from_bus": "5",
        "largest_diff_to_bus": "5",
    }
    component = pd.DataFrame(
        [
            {
                "component_variant": "lines_only",
                "max_abs_complex_diff": 4.0e-3,
            },
            {
                "component_variant": "transformer_only",
                "max_abs_complex_diff": 1.0e-14,
            },
        ]
    )
    trafo = pd.DataFrame([{"name": "trafo"}])
    cross = pd.DataFrame(
        [
            {
                "residual_norm_at_diffpf_solution": 1.0e-11,
                "residual_norm_at_pandapower_solution": 4.0e-3,
                "max_abs_residual_at_pandapower_solution": 4.0e-3,
                "largest_residual_equation": "q_or_v_bus_5",
            }
        ]
    )
    tolerance = pd.DataFrame(
        [
            {"variant": "default_current_options", "max_vm_pu_abs_diff": 2.3e-5},
            {"variant": "stricter_tolerances", "max_vm_pu_abs_diff": 2.3e-5},
        ]
    )

    diagnostic.write_readme(
        tmp_path,
        worst_case,
        algebraic,
        ybus_metrics,
        component,
        trafo,
        cross,
        tolerance,
    )

    readme = tmp_path / "README.md"
    assert readme.exists()
    text = readme.read_text(encoding="utf-8")
    assert "open line-switch handling" in text
    assert "No core change was made" in text
