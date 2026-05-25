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
    assert config.learning_rate_start == pytest.approx(8e-2)
    assert config.learning_rate_end == pytest.approx(1e-4)
    assert config.max_train_steps == 8000
    assert config.lr_schedule == "cosine_warm_restarts_decay"
    assert config.initial_schedule == "cosine_decay"
    assert config.hidden_width == 16
    assert config.hidden_layers == 2
    assert config.warm_restart_enabled is True
    assert config.base_train_steps == 8000
    assert config.finetune_steps == 8000


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
    assert "training_improvement_summary.csv" in required
    assert "architecture_comparison_summary.csv" in required


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


def test_training_history_schema_documents_two_phase_training(exp_module):
    for name in [
        "step",
        "global_step",
        "phase",
        "phase_step",
        "cycle_id",
        "learning_rate",
        "is_best_checkpoint",
        "hidden_width",
        "parameter_count",
    ]:
        assert name in exp_module.TRAINING_HISTORY_COLUMNS


def test_mini_training_run_is_possible(exp_module):
    config = exp_module.SurrogateTrainingConfig(
        seed=7,
        train_samples=12,
        val_samples=8,
        eval_samples=6,
        hidden_width=4,
        hidden_layers=1,
        learning_rate=0.02,
        lr_schedule="cosine_decay",
        warm_restart_enabled=False,
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
    assert history[0].learning_rate > history[-1].learning_rate
    assert all(row.phase == "base" for row in history)
    assert all(row.global_step == row.step for row in history)


def test_cosine_decay_learning_rate_has_expected_endpoints(exp_module):
    config = exp_module.SurrogateTrainingConfig(
        lr_schedule="cosine_decay",
        max_train_steps=10,
    )

    first = exp_module.learning_rate_at_step(0, config)
    middle = exp_module.learning_rate_at_step(5, config)
    last = exp_module.learning_rate_at_step(10, config)

    assert first == pytest.approx(8e-2)
    assert last == pytest.approx(1e-4)
    assert first > middle > last


def test_warm_restart_learning_rate_has_expected_cycle_boundaries(exp_module):
    config = exp_module.SurrogateTrainingConfig()

    assert exp_module.learning_rate_at_step(0, config) == pytest.approx(2e-2)
    assert exp_module.learning_rate_at_step(2000, config) == pytest.approx(1e-2)
    assert exp_module.learning_rate_at_step(4000, config) == pytest.approx(5e-3)
    assert exp_module.learning_rate_at_step(6000, config) == pytest.approx(2e-3)
    assert exp_module.learning_rate_at_step(8000, config) == pytest.approx(5e-5)

    cycle_starts = [
        exp_module.learning_rate_at_step(step, config)
        for step in (0, 2000, 4000, 6000)
    ]
    assert cycle_starts == sorted(cycle_starts, reverse=True)
    assert exp_module.learning_rate_at_step(1999, config) < cycle_starts[0]
    assert exp_module.learning_rate_at_step(2000, config) > exp_module.learning_rate_at_step(
        1999,
        config,
    )


def test_training_improvement_summary_compares_current_reference(exp_module):
    config = exp_module.SurrogateTrainingConfig()
    diagnostics = exp_module.TrainingDiagnostics(
        best_phase="warm_restart_finetune",
        best_cycle_id=2,
        best_global_step=10000,
        best_step=8000,
        best_val_mse=2.0e-4,
        best_val_mae_mw=0.02,
        final_phase="warm_restart_finetune",
        final_global_step=16000,
        final_step=8000,
        final_train_mse=4.1e-4,
        final_val_mse=4.0e-4,
        final_train_mae_mw=0.021,
        final_val_mae_mw=0.02,
    )
    rows = [
        exp_module.SurrogateErrorRow(
            split="eval",
            case_id="eval_0000",
            g_poa_wm2=800.0,
            t_amb_c=25.0,
            wind_ms=2.0,
            p_ref_mw=1.8,
            p_nn_mw=1.7,
            q_ref_mvar=-0.45,
            q_nn_mvar=-0.425,
            p_abs_error_mw=0.1,
            p_rel_error=0.05,
            q_abs_error_mvar=0.025,
            q_rel_error=0.05,
            p_rel_error_floor=0.05,
            q_rel_error_floor=0.05,
        ),
        exp_module.SurrogateErrorRow(
            split="eval",
            case_id="eval_0001",
            g_poa_wm2=900.0,
            t_amb_c=25.0,
            wind_ms=2.0,
            p_ref_mw=2.0,
            p_nn_mw=1.8,
            q_ref_mvar=-0.5,
            q_nn_mvar=-0.45,
            p_abs_error_mw=0.2,
            p_rel_error=0.1,
            q_abs_error_mvar=0.05,
            q_rel_error=0.1,
            p_rel_error_floor=0.1,
            q_rel_error_floor=0.1,
        ),
    ]

    summary = exp_module.build_training_improvement_summary(rows, config, diagnostics)

    assert len(summary) == 1
    assert summary[0].reference_val_mse == pytest.approx(2.565834826e-04)
    assert summary[0].new_best_val_mse == pytest.approx(2.0e-4)
    assert summary[0].new_eval_p_mae_mw == pytest.approx(0.15)
    assert summary[0].new_eval_p_rmse_mw == pytest.approx((0.025) ** 0.5)
    assert summary[0].relative_improvement_val_mse > 0.0
    assert summary[0].improved_over_reference is True
    assert summary[0].best_phase == "warm_restart_finetune"
    assert summary[0].best_cycle_id == 2
    assert summary[0].best_global_step == 10000


def test_architecture_comparison_summary_compares_width16_against_width8(exp_module):
    config = exp_module.SurrogateTrainingConfig()
    params = exp_module.init_mlp_params(exp_module.jax.random.PRNGKey(0))
    diagnostics = exp_module.TrainingDiagnostics(
        best_phase="warm_restart_finetune",
        best_cycle_id=4,
        best_global_step=16000,
        best_step=8000,
        best_val_mse=2.0e-4,
        best_val_mae_mw=0.02,
        final_phase="warm_restart_finetune",
        final_global_step=16000,
        final_step=8000,
        final_train_mse=2.1e-4,
        final_val_mse=2.0e-4,
        final_train_mae_mw=0.021,
        final_val_mae_mw=0.02,
    )
    rows = [
        exp_module.SurrogateErrorRow(
            split="eval",
            case_id="eval_0000",
            g_poa_wm2=800.0,
            t_amb_c=25.0,
            wind_ms=2.0,
            p_ref_mw=1.8,
            p_nn_mw=1.7,
            q_ref_mvar=-0.45,
            q_nn_mvar=-0.425,
            p_abs_error_mw=0.1,
            p_rel_error=0.05,
            q_abs_error_mvar=0.025,
            q_rel_error=0.05,
            p_rel_error_floor=0.05,
            q_rel_error_floor=0.05,
        )
    ]

    summary = exp_module.build_architecture_comparison_summary(
        rows,
        config,
        params,
        diagnostics,
    )

    assert len(summary) == 1
    assert summary[0].reference_hidden_width == 8
    assert summary[0].candidate_hidden_width == 16
    assert summary[0].reference_parameter_count == 113
    assert summary[0].candidate_parameter_count == 353
    assert summary[0].reference_val_mse == pytest.approx(2.3189919622e-04)
    assert summary[0].candidate_best_val_mse == pytest.approx(2.0e-4)
    assert summary[0].candidate_eval_p_mae_mw == pytest.approx(0.1)
    assert summary[0].improved_over_width8 is True


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
            global_step=0,
            phase="base",
            phase_step=0,
            cycle_id=None,
            train_mse=1.0,
            val_mse=1.0,
            train_mae_mw=0.1,
            val_mae_mw=0.1,
            learning_rate=0.02,
            is_best_checkpoint=True,
            hidden_width=config.hidden_width,
            parameter_count=exp_module.count_mlp_parameters(params),
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
            best_phase="base",
            best_cycle_id=None,
            best_global_step=1,
            best_step=1,
            best_val_mse=1.0,
            best_val_mae_mw=0.1,
            final_phase="base",
            final_global_step=1,
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
        best_phase="base",
        best_cycle_id=None,
        best_global_step=1,
        best_step=1,
        best_val_mse=1.0,
        best_val_mae_mw=0.1,
        final_phase="base",
        final_global_step=1,
        final_step=1,
        final_train_mse=1.0,
        final_val_mse=1.0,
        final_train_mae_mw=0.1,
        final_val_mae_mw=0.1,
    )
    training_improvement_rows = exp_module.build_training_improvement_summary(
        error_rows,
        config,
        diagnostics,
    )
    architecture_comparison_rows = exp_module.build_architecture_comparison_summary(
        error_rows,
        config,
        params,
        diagnostics,
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
        training_improvement_rows,
        architecture_comparison_rows,
        summary_rows,
        config,
        params,
        diagnostics,
    )

    for name in exp_module.REQUIRED_ARTIFACTS:
        assert (tmp_path / name).exists(), f"Missing artifact: {name}"

    with (tmp_path / "metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    assert metadata["learning_rate_schedule"]["initial_learning_rate_start"] == pytest.approx(
        8e-2
    )
    assert metadata["learning_rate_schedule"]["initial_learning_rate_end"] == pytest.approx(
        1e-4
    )
    assert metadata["learning_rate_schedule"]["base_train_steps"] == 8000
    assert metadata["learning_rate_schedule"]["finetune_schedule"] == (
        "cosine_warm_restarts_decay"
    )
    assert metadata["best_validation_checkpoint"]["best_phase"] == "base"
    assert "training_improvement_summary" in metadata
    assert metadata["model_architecture"]["hidden_width"] == 16
    assert metadata["mlp_parameter_count"] == 353
    assert metadata["architecture_capacity_run"]["candidate_hidden_width"] == 16
    assert metadata["architecture_capacity_run"]["width8_checkpoint_reused"] is False
    assert "architecture_comparison_summary" in metadata["architecture_capacity_run"]

    readme_text = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "8.0e-02" in readme_text
    assert "1.0e-04" in readme_text
    assert "Comparison to previous training run" in readme_text
    assert "warm-restart finetune" in readme_text
    assert "Capacity run" in readme_text
    assert "hidden width" in readme_text
