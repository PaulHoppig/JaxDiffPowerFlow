"""Schema and smoke tests for Experiment 5d simple-objective curtailment."""

from __future__ import annotations

import csv
import inspect
import json
import math
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def exp_module():
    import experiments.exp05d_optimize_pv_curtailment_simple_objective as module

    return module


@pytest.fixture(scope="module")
def small_results(exp_module):
    return exp_module.run_experiment(max_iter=2, grid_points=5)


def _assert_numeric_rows_are_finite_except_soft(rows: list) -> None:
    for row in rows:
        for key, value in row.__dict__.items():
            if isinstance(value, bool) or isinstance(value, str) or value is None:
                continue
            if "soft_export_violation_mw" in key:
                assert math.isnan(float(value))
            else:
                assert math.isfinite(float(value))


def test_module_is_importable(exp_module):
    assert exp_module is not None


def test_output_paths_are_exp05d_specific(exp_module):
    assert exp_module.RESULTS_DIR.name == "exp05d_optimize_pv_curtailment_simple_objective"
    assert exp_module.RESULTS_DIR != exp_module.EXP05B_RESULTS_DIR
    assert exp_module.EXP05B_RESULTS_DIR.name == "exp05b_optimize_pv_curtailment"


def test_selected_case_config_matches_exp05b_case_with_simple_target(exp_module):
    config = exp_module.selected_case_config()

    assert config["case_id"] == "selected_realistic_load0p4_g1200_t30"
    assert config["load_multiplier_mv_bus_2"] == pytest.approx(0.4)
    assert config["g_poa_wm2"] == pytest.approx(1200.0)
    assert config["t_amb_c"] == pytest.approx(30.0)
    assert config["wind_ms"] == pytest.approx(2.0)
    assert config["pv_size_factor"] == pytest.approx(1.0)
    assert config["kappa"] == pytest.approx(-0.25)
    assert config["p_export_limit_mw"] == pytest.approx(7.0)
    assert config["p_export_target_mw"] == pytest.approx(7.0)
    assert config["p_scale_mw"] == pytest.approx(1.0)
    assert config["objective_variant"] == "simple_target_7mw"
    assert config["upstream_model"] == "analytical_pv_weather"


def test_simple_objective_formula_is_exact(exp_module):
    value = exp_module.simple_objective_from_export_proxy(7.25, p_scale_mw=1.0)

    assert float(value) == pytest.approx(((7.25 - 7.0) / 1.0) ** 2)


def test_objective_source_has_no_softplus_or_curtailment_regularization(exp_module):
    source = inspect.getsource(exp_module.objective_from_theta)

    assert "softplus" not in source
    assert "LAMBDA_CURTAILMENT" not in source
    assert "lambda_curtailment" not in source
    assert "P_EXPORT_TARGET_MW" in source or "simple_objective_from_export_proxy" in source


def test_sigmoid_and_logit_keep_curtailment_bounded(exp_module):
    for c in [0.001, 0.2, 0.8, 0.999]:
        theta = exp_module.logit(c)
        recovered = float(exp_module.curtailment_from_theta(theta))
        assert 0.0 <= recovered <= 1.0
        assert recovered == pytest.approx(c)

    assert 0.0 <= float(exp_module.sigmoid(-100.0)) <= 1.0
    assert 0.0 <= float(exp_module.sigmoid(100.0)) <= 1.0


def test_objective_returns_finite_scalar(exp_module):
    scenario = exp_module.build_selected_scenario()
    theta = exp_module.jnp.asarray(exp_module.logit(exp_module.C_INIT))
    value = exp_module.objective_from_theta(theta, scenario)

    assert value.shape == ()
    assert math.isfinite(float(value))


def test_small_outputs_have_required_columns(exp_module, small_results):
    baseline_rows, trace_rows, final_rows, grid_rows, diagnostics_rows, summary_rows = (
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
        "objective_variant",
        "p_export_proxy_mw",
        "objective",
        "p_export_mw",
    ]:
        assert name in exp_module.SELECTED_CASE_BASELINE_COLUMNS
    for name in [
        "iteration",
        "curtailment_factor",
        "objective",
        "grad_theta",
        "p_export_proxy_mw",
        "p_export_mw",
    ]:
        assert name in exp_module.OPTIMIZATION_TRACE_COLUMNS
    for name in [
        "curtailment_factor",
        "objective",
        "is_grid_best_objective",
        "is_grid_best_feasible",
        "p_export_mw",
    ]:
        assert name in exp_module.GRID_REFERENCE_COLUMNS
    for name in [
        "experiment",
        "objective_variant",
        "selected_case_id",
        "final_iteration",
        "final_curtailment_factor",
        "final_objective",
        "final_grad_theta",
        "final_p_export_proxy_mw",
        "grid_best_objective_curtailment_factor",
        "grid_best_feasible_curtailment_factor",
        "abs_c_difference_optimizer_vs_grid_objective",
    ]:
        assert name in exp_module.FINAL_SOLUTION_COLUMNS
    for name in [
        "p_export_full_pv_mw",
        "p_export_zero_pv_mw",
        "feasible_by_zero_pv",
    ]:
        assert name in exp_module.CONSTRAINT_DIAGNOSTICS_COLUMNS


def test_grid_reference_marks_objective_and_feasible_rows(exp_module, small_results):
    _, _, _, grid_rows, _, _ = small_results

    assert all(0.0 <= row.curtailment_factor <= 1.0 for row in grid_rows)
    assert sum(row.is_grid_best_objective for row in grid_rows) == 1
    assert sum(row.is_grid_best for row in grid_rows) == 1
    assert sum(row.is_grid_best_feasible for row in grid_rows) <= 1


def test_final_solution_trace_and_summary_are_consistent(exp_module, small_results):
    _, trace_rows, final_rows, _, _, summary_rows = small_results
    final = final_rows[0]
    last_trace = trace_rows[-1]
    summary = {row.metric: row.value for row in summary_rows}

    assert final.objective_variant == "simple_target_7mw"
    assert 0.0 <= final.final_curtailment_factor <= 1.0
    assert final.final_iteration == last_trace.iteration
    assert final.final_curtailment_factor == pytest.approx(last_trace.curtailment_factor)
    assert final.final_objective == pytest.approx(last_trace.objective)
    assert final.final_grad_theta == pytest.approx(last_trace.grad_theta)
    assert final.final_p_export_mw == pytest.approx(last_trace.p_export_mw)
    assert final.final_p_export_proxy_mw == pytest.approx(last_trace.p_export_proxy_mw)
    assert summary["final_curtailment_factor"] == pytest.approx(
        final.final_curtailment_factor
    )
    assert summary["final_p_export_mw"] == pytest.approx(final.final_p_export_mw)
    exp_module.validate_consistency(trace_rows, final_rows, summary_rows)


def test_export_all_writes_required_exp05d_artifacts(
    exp_module,
    small_results,
    tmp_path: Path,
):
    baseline_rows, trace_rows, final_rows, grid_rows, diagnostics_rows, summary_rows = (
        small_results
    )

    exp_module.export_all(
        baseline_rows,
        trace_rows,
        final_rows,
        grid_rows,
        diagnostics_rows,
        summary_rows,
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
    assert meta["experiment"] == "exp05d_optimize_pv_curtailment_simple_objective"
    assert meta["objective_variant"] == "simple_target_7mw"
    assert meta["optimization"]["contains_softplus_penalty"] is False
    assert meta["optimization"]["contains_curtailment_regularization"] is False

    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "Experiment 5d" in readme
    assert "simple" in readme.lower()
    assert "6.99 MW" in readme
    assert "no softplus" in readme.lower()


def test_no_nan_or_inf_in_smoke_outputs_except_soft_diagnostic(small_results):
    baseline_rows, trace_rows, final_rows, grid_rows, diagnostics_rows, summary_rows = (
        small_results
    )

    _assert_numeric_rows_are_finite_except_soft(baseline_rows)
    _assert_numeric_rows_are_finite_except_soft(trace_rows)
    _assert_numeric_rows_are_finite_except_soft(final_rows)
    _assert_numeric_rows_are_finite_except_soft(grid_rows)
    _assert_numeric_rows_are_finite_except_soft(diagnostics_rows)
    _assert_numeric_rows_are_finite_except_soft(summary_rows)
