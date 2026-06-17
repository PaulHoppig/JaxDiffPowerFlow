"""Schema and smoke tests for Experiment 5c NN curtailment optimization."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def exp_module():
    import experiments.exp05c_optimize_pv_curtailment_nn as module

    return module


@pytest.fixture(scope="module")
def small_nn(exp_module):
    config = exp_module.SurrogateTrainingConfig(
        seed=11,
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
    params, _train_x, _val_x, _eval_x, _history = exp_module.exp04.train_surrogate(config)
    return params, exp_module.DEFAULT_WEATHER_NORMALIZATION, config


@pytest.fixture(scope="module")
def small_results(exp_module, small_nn):
    params, norm, config = small_nn
    return exp_module.run_experiment(
        max_iter=2,
        grid_points=5,
        nn_params=params,
        norm=norm,
        nn_training_config=config,
    )


def _assert_finite_numeric_rows(rows: list) -> None:
    for row in rows:
        for value in row.__dict__.values():
            if isinstance(value, bool) or isinstance(value, str) or value is None:
                continue
            assert math.isfinite(float(value))


def test_module_is_importable(exp_module):
    assert exp_module is not None


def test_output_paths_are_exp05c_specific(exp_module):
    assert exp_module.RESULTS_DIR.name == "exp05c_optimize_pv_curtailment_nn"
    assert exp_module.RESULTS_DIR != exp_module.EXP05B_RESULTS_DIR
    assert exp_module.EXP05B_RESULTS_DIR.name == "exp05b_optimize_pv_curtailment"


def test_selected_case_and_optimizer_settings_match_exp05b(exp_module):
    config = exp_module.selected_case_config()

    assert config["case_id"] == "selected_realistic_load0p4_g1200_t30"
    assert config["g_poa_wm2"] == pytest.approx(1200.0)
    assert config["t_amb_c"] == pytest.approx(30.0)
    assert config["wind_ms"] == pytest.approx(2.0)
    assert config["upstream_model"] == "nn_p_only_fixed_kappa"
    assert config["p_export_limit_mw"] == pytest.approx(7.0)
    assert config["p_export_target_mw"] == pytest.approx(7.0)
    assert exp_module.BETA == pytest.approx(exp_module.exp05b.BETA)
    assert exp_module.LAMBDA_CURTAILMENT == pytest.approx(
        exp_module.exp05b.LAMBDA_CURTAILMENT
    )
    assert exp_module.MAX_ITER == exp_module.exp05b.MAX_ITER
    assert exp_module.GRID_POINTS == exp_module.exp05b.GRID_POINTS


def test_nn_pv_coupling_uses_fixed_q_over_p_ratio(exp_module, small_nn):
    params, norm, _config = small_nn
    injection = exp_module._nn_pv_injection(
        params,
        norm,
        1200.0,
        30.0,
        2.0,
        0.7,
        1.0,
    )

    if abs(float(injection.p_pv_mw)) > 1e-12:
        assert float(injection.q_pv_mvar / injection.p_pv_mw) == pytest.approx(-0.25)


def test_small_optimization_outputs_have_required_columns(exp_module, small_results):
    baseline_rows, trace_rows, final_rows, grid_rows, diagnostics_rows, summary_rows, _ = (
        small_results
    )

    assert len(baseline_rows) == 2
    assert len(trace_rows) == 3
    assert len(final_rows) == 1
    assert len(grid_rows) == 5
    assert len(diagnostics_rows) == 1
    assert summary_rows

    for name in [
        "baseline_type",
        "case_id",
        "p_export_mw",
        "p_pv_mw",
        "q_pv_mvar",
    ]:
        assert name in exp_module.SELECTED_CASE_BASELINE_COLUMNS
    for name in ["iteration", "curtailment_factor", "p_export_mw", "q_pv_mvar"]:
        assert name in exp_module.OPTIMIZATION_TRACE_COLUMNS
    for name in ["curtailment_factor", "is_grid_best", "p_export_mw"]:
        assert name in exp_module.GRID_REFERENCE_COLUMNS
    for name in ["final_curtailment_factor", "final_p_export_mw"]:
        assert name in exp_module.FINAL_SOLUTION_COLUMNS
    for name in ["feasible_by_zero_pv", "constraint_satisfied"]:
        assert name in exp_module.CONSTRAINT_DIAGNOSTICS_COLUMNS
    for name in ["metric", "value", "unit", "notes"]:
        assert name in exp_module.RUN_SUMMARY_COLUMNS


def test_final_solution_trace_and_summary_are_consistent(exp_module, small_results):
    _, trace_rows, final_rows, _, diagnostics_rows, summary_rows, _ = small_results
    final = final_rows[0]
    last_trace = trace_rows[-1]
    summary = {row.metric: row.value for row in summary_rows}

    assert 0.0 <= final.final_curtailment_factor <= 1.0
    assert final.final_curtailment_factor == pytest.approx(last_trace.curtailment_factor)
    assert final.final_p_export_mw == pytest.approx(last_trace.p_export_mw)
    assert summary["final_curtailment_factor"] == pytest.approx(
        final.final_curtailment_factor
    )
    assert summary["final_p_export_mw"] == pytest.approx(final.final_p_export_mw)
    exp_module.validate_consistency(trace_rows, final_rows, summary_rows, diagnostics_rows)


def test_export_all_writes_required_exp05c_artifacts(
    exp_module,
    small_results,
    tmp_path: Path,
):
    baseline_rows, trace_rows, final_rows, grid_rows, diagnostics_rows, summary_rows, nn_source = (
        small_results
    )

    exp_module.export_all(
        baseline_rows,
        trace_rows,
        final_rows,
        grid_rows,
        diagnostics_rows,
        summary_rows,
        nn_source,
        tmp_path,
    )

    for name in exp_module.REQUIRED_ARTIFACTS:
        assert (tmp_path / name).exists(), f"Missing artifact: {name}"

    with (tmp_path / "final_solution.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == exp_module.FINAL_SOLUTION_COLUMNS
        assert len(list(reader)) == 1

    with (tmp_path / "metadata.json").open(encoding="utf-8") as handle:
        meta = json.load(handle)
    assert meta["experiment"] == "exp05c_optimize_pv_curtailment_nn"
    assert meta["upstream_model"]["model_type"] == "nn_p_only_fixed_kappa"
    assert meta["upstream_model"]["q_coupling"] == "Q_pv_mvar(c) = -0.25 * P_pv_mw(c)"

    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "Experiment 5c" in readme
    assert "NN surrogate" in readme


def test_no_nan_or_inf_in_smoke_outputs(small_results):
    baseline_rows, trace_rows, final_rows, grid_rows, diagnostics_rows, summary_rows, _ = (
        small_results
    )

    _assert_finite_numeric_rows(baseline_rows)
    _assert_finite_numeric_rows(trace_rows)
    _assert_finite_numeric_rows(final_rows)
    _assert_finite_numeric_rows(grid_rows)
    _assert_finite_numeric_rows(diagnostics_rows)
    _assert_finite_numeric_rows(summary_rows)
