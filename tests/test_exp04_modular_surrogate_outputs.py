"""Lightweight schema tests for Experiment 4 neural surrogate artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def exp_module():
    import experiments.exp04_modular_upstream_nn_surrogate as module

    return module


def test_module_is_importable(exp_module):
    assert exp_module is not None


def test_required_artifact_names_are_defined(exp_module):
    required = set(exp_module.REQUIRED_ARTIFACTS)

    assert "metadata.json" in required
    assert "coupling_summary.csv" in required
    assert "model_comparison.csv" in required
    assert "gradient_success_table.csv" in required
    assert "sensitivity_pattern_summary.csv" in required


def test_coupling_summary_columns_answer_research_question(exp_module):
    cols = exp_module.COUPLING_SUMMARY_COLUMNS
    for name in [
        "model_name",
        "uses_network_params_p_spec",
        "uses_network_params_q_spec",
        "uses_same_injection_adapter",
        "uses_same_pf_core",
        "requires_core_change",
        "has_controller_logic",
        "has_q_limits",
        "has_pv_pq_switching",
    ]:
        assert name in cols


def test_coupling_summary_has_expected_flags(exp_module):
    rows = exp_module.build_coupling_summary()

    assert {row.model_name for row in rows} == set(exp_module.UPSTREAM_MODELS)
    assert all(row.uses_same_pf_core for row in rows)
    assert all(row.uses_same_injection_adapter for row in rows)
    assert all(not row.requires_core_change for row in rows)
    assert all(not row.has_controller_logic for row in rows)
    assert all(not row.has_q_limits for row in rows)
    assert all(not row.has_pv_pq_switching for row in rows)


def test_export_helpers_write_csv_and_json_with_schema(exp_module, tmp_path: Path):
    rows = exp_module.build_coupling_summary()

    exp_module._write_csv(
        tmp_path / "coupling_summary.csv",
        rows,
        exp_module.COUPLING_SUMMARY_COLUMNS,
    )
    exp_module._write_json(tmp_path / "coupling_summary.json", rows)

    with (tmp_path / "coupling_summary.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == exp_module.COUPLING_SUMMARY_COLUMNS
        assert len(list(reader)) == len(exp_module.UPSTREAM_MODELS)

    with (tmp_path / "coupling_summary.json").open(encoding="utf-8") as handle:
        data = json.load(handle)
    assert len(data) == len(exp_module.UPSTREAM_MODELS)
    assert data[0]["uses_same_pf_core"] is True


def test_training_dataset_summary_schema_exists(exp_module):
    for name in [
        "split",
        "n_samples",
        "min_g_poa_wm2",
        "max_g_poa_wm2",
        "min_p_ref_mw",
        "max_p_ref_mw",
    ]:
        assert name in exp_module.TRAINING_DATASET_SUMMARY_COLUMNS


def test_mini_training_run_is_possible(exp_module):
    config = exp_module.SurrogateTrainingConfig(
        seed=7,
        train_samples=12,
        val_samples=8,
        hidden_width=4,
        hidden_layers=1,
        learning_rate=0.02,
        max_train_steps=3,
        log_every=1,
    )

    params, train_x, val_x, history = exp_module.train_surrogate(config)

    assert exp_module.count_mlp_parameters(params) < 500
    assert train_x.shape == (12, 3)
    assert val_x.shape == (8, 3)
    assert len(history) >= 2
    assert all(row.train_mse >= 0.0 for row in history)


def test_export_all_writes_mandatory_stub_artifacts(exp_module, tmp_path: Path):
    config = exp_module.SurrogateTrainingConfig(max_train_steps=1, log_every=1)
    params = exp_module.init_mlp_params(exp_module.jax.random.PRNGKey(0))
    dataset_rows = [
        exp_module.TrainingDatasetSummaryRow(
            split="train",
            n_samples=1,
            min_g_poa_wm2=800.0,
            max_g_poa_wm2=800.0,
            min_t_amb_c=25.0,
            max_t_amb_c=25.0,
            min_wind_ms=2.0,
            max_wind_ms=2.0,
            min_p_ref_mw=1.8,
            max_p_ref_mw=1.8,
        )
    ]
    history_rows = [
        exp_module.TrainingHistoryRow(
            step=0,
            train_mse=1.0,
            val_mse=1.0,
            train_mae_mw=0.1,
            val_mae_mw=0.1,
            learning_rate=0.02,
        )
    ]
    error_rows = [
        exp_module.SurrogateErrorRow(
            split="eval",
            case_id="case",
            g_poa_wm2=800.0,
            t_amb_c=25.0,
            wind_ms=2.0,
            p_ref_mw=1.8,
            p_nn_mw=1.79,
            q_ref_mvar=-0.45,
            q_nn_mvar=-0.4475,
            p_abs_error_mw=0.01,
            p_rel_error=0.005,
            q_abs_error_mvar=0.0025,
            q_rel_error=0.005,
        )
    ]
    comparison_rows = [
        exp_module.ModelComparisonRow(
            model_name="analytic_pv_weather",
            network_scenario="base",
            weather_case_id="case",
            g_poa_wm2=800.0,
            t_amb_c=25.0,
            wind_ms=2.0,
            p_inj_mw=1.8,
            q_inj_mvar=-0.45,
            observable="p_slack_mw",
            value=-6.0,
            unit="MW",
            converged=True,
            iterations=5,
            residual_norm=1e-11,
        )
    ]
    coupling_rows = exp_module.build_coupling_summary()
    gradient_rows = [
        exp_module.GradientSuccessRow(
            model_name="analytic_pv_weather",
            network_scenario="base",
            weather_case_id="case",
            input_parameter="t_amb_c",
            observable="p_slack_mw",
            ad_gradient=1.0,
            fd_gradient=1.0,
            abs_error=0.0,
            rel_error=0.0,
            is_finite_ad=True,
            is_finite_fd=True,
            passes_fd_check=True,
            fd_step=0.05,
            unit="MW/degC",
        )
    ]
    pattern_rows = [
        exp_module.SensitivityPatternSummaryRow(
            comparison_pair="analytic_pv_weather vs nn_p_only_fixed_kappa",
            network_scenario="base",
            observable="p_slack_mw",
            input_parameter="t_amb_c",
            n_cases=1,
            mean_abs_grad_ref=1.0,
            mean_abs_grad_other=0.9,
            mean_abs_diff=0.1,
            median_rel_diff=0.1,
            max_rel_diff=0.1,
            sign_agreement_rate=1.0,
            cosine_similarity=1.0,
        )
    ]
    summary_rows = [
        exp_module.RunSummaryRow(
            model_name="analytic_pv_weather",
            convergence_rate=1.0,
            max_residual_norm=1e-11,
            max_iterations=5,
            n_failed_solves=0,
            n_total_solves=1,
        )
    ]

    exp_module.export_all(
        tmp_path,
        dataset_rows,
        history_rows,
        error_rows,
        comparison_rows,
        coupling_rows,
        gradient_rows,
        pattern_rows,
        summary_rows,
        config,
        params,
    )

    for name in exp_module.REQUIRED_ARTIFACTS:
        assert (tmp_path / name).exists(), f"Missing artifact: {name}"
