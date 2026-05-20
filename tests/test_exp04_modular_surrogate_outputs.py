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


def test_default_training_config_uses_full_exp04_dataset_sizes(exp_module):
    config = exp_module.SurrogateTrainingConfig()

    assert config.train_samples == 32768
    assert config.val_samples == 8192
    assert config.eval_samples == 8192
    assert config.max_train_steps == 4000
    assert config.lr_schedule == "cosine_decay"


def test_required_artifact_names_are_defined(exp_module):
    required = set(exp_module.REQUIRED_ARTIFACTS)

    assert "metadata.json" in required
    assert "coupling_summary.csv" in required
    assert "model_comparison.csv" in required
    assert "gradient_success_table.csv" in required
    assert "sensitivity_pattern_summary.csv" in required
    assert "pf_observable_error_table.csv" in required
    assert "pf_observable_error_summary.csv" in required
    assert "sensitivity_error_table.csv" in required
    assert "sensitivity_error_summary.csv" in required


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
        eval_samples=6,
        hidden_width=4,
        hidden_layers=1,
        learning_rate=0.02,
        max_train_steps=3,
        log_every=1,
    )

    params, train_x, val_x, eval_x, history = exp_module.train_surrogate(config)

    assert exp_module.count_mlp_parameters(params) < 500
    assert train_x.shape == (12, 3)
    assert val_x.shape == (8, 3)
    assert eval_x.shape == (6, 3)
    assert len(history) >= 2
    assert all(row.train_mse >= 0.0 for row in history)
    assert len({row.learning_rate for row in history}) > 1


def test_cosine_decay_learning_rate_has_expected_endpoints(exp_module):
    config = exp_module.SurrogateTrainingConfig(max_train_steps=10)

    first = exp_module.learning_rate_at_step(0, config)
    middle = exp_module.learning_rate_at_step(5, config)
    last = exp_module.learning_rate_at_step(10, config)

    assert first == pytest.approx(5e-2)
    assert last == pytest.approx(5e-4)
    assert first > middle > last


def test_new_error_table_columns_exist(exp_module):
    for name in [
        "candidate_model",
        "reference_model",
        "observable",
        "signed_error",
        "abs_error",
        "rel_error_floor",
    ]:
        assert name in exp_module.PF_OBSERVABLE_ERROR_COLUMNS

    for name in [
        "candidate_model",
        "reference_model",
        "reference_gradient",
        "candidate_gradient",
        "abs_gradient_error",
        "rel_gradient_error_floor",
        "sign_match",
        "magnitude_ratio_abs",
    ]:
        assert name in exp_module.SENSITIVITY_ERROR_COLUMNS


def test_pf_observable_error_table_computes_absolute_errors(exp_module):
    rows = [
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
        ),
        exp_module.ModelComparisonRow(
            model_name="nn_p_only_fixed_kappa",
            network_scenario="base",
            weather_case_id="case",
            g_poa_wm2=800.0,
            t_amb_c=25.0,
            wind_ms=2.0,
            p_inj_mw=1.79,
            q_inj_mvar=-0.4475,
            observable="p_slack_mw",
            value=-5.75,
            unit="MW",
            converged=True,
            iterations=5,
            residual_norm=1e-11,
        ),
    ]

    table, summary = exp_module.build_pf_observable_error_tables(rows)

    assert len(table) == 1
    assert table[0].signed_error == pytest.approx(0.25)
    assert table[0].abs_error == pytest.approx(0.25)
    assert table[0].rel_error_floor == pytest.approx(0.25 / 6.0)
    assert summary[0].rmse == pytest.approx(0.25)


def test_sensitivity_error_summary_computes_main_metrics(exp_module):
    rows = [
        exp_module.SensitivityErrorRow(
            candidate_model="nn_p_only_fixed_kappa",
            reference_model="analytic_pv_weather",
            network_scenario="base",
            weather_case_id="a",
            observable="p_slack_mw",
            input_parameter="t_amb_c",
            reference_gradient=2.0,
            candidate_gradient=1.0,
            signed_gradient_error=-1.0,
            abs_gradient_error=1.0,
            rel_gradient_error_floor=0.5,
            sign_match=True,
            abs_reference_gradient=2.0,
            abs_candidate_gradient=1.0,
            magnitude_ratio_abs=0.5,
        ),
        exp_module.SensitivityErrorRow(
            candidate_model="nn_p_only_fixed_kappa",
            reference_model="analytic_pv_weather",
            network_scenario="base",
            weather_case_id="b",
            observable="p_slack_mw",
            input_parameter="t_amb_c",
            reference_gradient=-4.0,
            candidate_gradient=2.0,
            signed_gradient_error=6.0,
            abs_gradient_error=6.0,
            rel_gradient_error_floor=1.5,
            sign_match=False,
            abs_reference_gradient=4.0,
            abs_candidate_gradient=2.0,
            magnitude_ratio_abs=0.5,
        ),
    ]

    summary = exp_module.summarize_sensitivity_error_rows(rows)

    assert len(summary) == 1
    assert summary[0].mean_abs_gradient_error == pytest.approx(3.5)
    assert summary[0].median_rel_gradient_error_floor == pytest.approx(1.0)
    assert summary[0].sign_match_rate == pytest.approx(0.5)
    assert summary[0].median_magnitude_ratio_abs == pytest.approx(0.5)


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
            p_rel_error_floor=0.005,
            q_rel_error_floor=0.005,
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
            ad_fd_abs_error=0.0,
            ad_fd_rel_error_floor=0.0,
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
            train_points=config.train_samples,
            val_points=config.val_samples,
            eval_points=config.eval_samples,
            best_step=1,
            best_val_mse=1.0,
            best_val_mae_mw=0.1,
            final_step=1,
            final_val_mse=1.0,
            final_val_mae_mw=0.1,
            convergence_rate=1.0,
            max_residual_norm=1e-11,
            max_iterations=5,
            n_failed_solves=0,
            n_total_solves=1,
        )
    ]
    pf_error_rows, pf_error_summary_rows = exp_module.build_pf_observable_error_tables(
        comparison_rows
        + [
            exp_module.ModelComparisonRow(
                model_name="nn_p_only_fixed_kappa",
                network_scenario="base",
                weather_case_id="case",
                g_poa_wm2=800.0,
                t_amb_c=25.0,
                wind_ms=2.0,
                p_inj_mw=1.79,
                q_inj_mvar=-0.4475,
                observable="p_slack_mw",
                value=-5.99,
                unit="MW",
                converged=True,
                iterations=5,
                residual_norm=1e-11,
            )
        ]
    )
    sensitivity_error_rows = [
        exp_module.SensitivityErrorRow(
            candidate_model="nn_p_only_fixed_kappa",
            reference_model="analytic_pv_weather",
            network_scenario="base",
            weather_case_id="case",
            observable="p_slack_mw",
            input_parameter="t_amb_c",
            reference_gradient=2.0,
            candidate_gradient=1.0,
            signed_gradient_error=-1.0,
            abs_gradient_error=1.0,
            rel_gradient_error_floor=0.5,
            sign_match=True,
            abs_reference_gradient=2.0,
            abs_candidate_gradient=1.0,
            magnitude_ratio_abs=0.5,
        )
    ]
    sensitivity_error_summary_rows = [
        exp_module.SensitivityErrorSummaryRow(
            candidate_model="nn_p_only_fixed_kappa",
            network_scenario="base",
            observable="p_slack_mw",
            input_parameter="t_amb_c",
            n=1,
            mean_abs_gradient_error=1.0,
            median_abs_gradient_error=1.0,
            max_abs_gradient_error=1.0,
            rmse_gradient_error=1.0,
            mean_rel_gradient_error_floor=0.5,
            median_rel_gradient_error_floor=0.5,
            max_rel_gradient_error_floor=0.5,
            sign_match_rate=1.0,
            median_magnitude_ratio_abs=0.5,
            mean_cosine_similarity=1.0,
        )
    ]
    diagnostics = exp_module.TrainingDiagnostics(
        best_step=1,
        best_val_mse=1.0,
        best_val_mae_mw=0.1,
        final_step=1,
        final_train_mse=1.0,
        final_val_mse=1.0,
        final_train_mae_mw=0.1,
        final_val_mae_mw=0.1,
    )

    exp_module.export_all(
        tmp_path,
        dataset_rows,
        history_rows,
        error_rows,
        comparison_rows,
        coupling_rows,
        gradient_rows,
        pattern_rows,
        pf_error_rows,
        pf_error_summary_rows,
        sensitivity_error_rows,
        sensitivity_error_summary_rows,
        summary_rows,
        config,
        params,
        diagnostics,
    )

    for name in exp_module.REQUIRED_ARTIFACTS:
        assert (tmp_path / name).exists(), f"Missing artifact: {name}"
