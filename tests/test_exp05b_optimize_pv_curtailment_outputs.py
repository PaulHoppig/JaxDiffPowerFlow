"""Lightweight schema and smoke tests for Experiment 5b."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def exp_module():
    import experiments.exp05b_optimize_pv_curtailment as module

    return module


@pytest.fixture(scope="module")
def small_results(exp_module):
    return exp_module.run_experiment(max_iter=2, grid_points=5)


def _assert_finite_numeric_rows(rows: list) -> None:
    for row in rows:
        for value in row.__dict__.values():
            if isinstance(value, bool) or isinstance(value, str):
                continue
            assert math.isfinite(float(value))


def test_module_is_importable(exp_module):
    assert exp_module is not None


def test_selected_case_config_matches_prompt(exp_module):
    config = exp_module.selected_case_config()

    assert config["case_id"] == "selected_realistic_load0p4_g1200_t30"
    assert config["load_multiplier_mv_bus_2"] == pytest.approx(0.4)
    assert config["g_poa_wm2"] == pytest.approx(1200.0)
    assert config["t_amb_c"] == pytest.approx(30.0)
    assert config["wind_ms"] == pytest.approx(2.0)
    assert config["pv_size_factor"] == pytest.approx(1.0)
    assert config["kappa"] == pytest.approx(-0.25)
    assert config["p_export_limit_mw"] == pytest.approx(7.0)
    assert config["p_export_target_mw"] == pytest.approx(6.99)


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


def test_small_optimization_trace_has_required_columns(exp_module, small_results):
    _, trace_rows, _, _, _, _ = small_results

    assert len(trace_rows) == 3
    for name in [
        "iteration",
        "theta",
        "curtailment_factor",
        "objective",
        "grad_theta",
        "p_export_limit_mw",
        "p_export_target_mw",
        "export_proxy_mw",
        "hard_export_violation_mw",
        "soft_export_violation_mw",
        "p_export_mw",
        "export_violation_mw",
        "p_slack_mw",
        "p_pv_mw",
        "q_pv_mvar",
        "vm_mv_bus_2_pu",
        "total_p_loss_mw",
        "s_trafo_hv_mva",
        "converged",
        "iterations",
        "residual_norm",
    ]:
        assert name in exp_module.OPTIMIZATION_TRACE_COLUMNS
    assert all(0.0 <= row.curtailment_factor <= 1.0 for row in trace_rows)


def test_grid_reference_values_are_bounded(exp_module, small_results):
    _, _, _, grid_rows, _, _ = small_results

    assert len(grid_rows) == 5
    assert all(0.0 <= row.curtailment_factor <= 1.0 for row in grid_rows)
    assert sum(row.is_grid_best for row in grid_rows) == 1


def test_final_solution_and_diagnostics_fields(exp_module, small_results):
    _, trace_rows, final_rows, _, diagnostics_rows, summary_rows = small_results
    final = final_rows[0]
    diagnostics = diagnostics_rows[0]
    last_trace = trace_rows[-1]

    for name in [
        "final_curtailment_factor",
        "final_pv_utilization_pct",
        "final_curtailment_pct",
        "final_p_export_mw",
        "final_hard_export_violation_mw",
        "final_soft_export_violation_mw",
        "final_export_margin_mw",
        "grid_best_curtailment_factor",
        "abs_c_difference_optimizer_vs_grid",
        "p_export_target_mw",
        "constraint_satisfied",
    ]:
        assert name in exp_module.FINAL_SOLUTION_COLUMNS
        assert hasattr(final, name)

    for name in [
        "p_export_limit_mw",
        "p_export_full_pv_mw",
        "p_export_zero_pv_mw",
        "constraint_satisfied",
        "feasible_by_zero_pv",
    ]:
        assert name in exp_module.CONSTRAINT_DIAGNOSTICS_COLUMNS
        assert hasattr(diagnostics, name)

    assert final.final_curtailment_factor == pytest.approx(last_trace.curtailment_factor)
    assert final.final_p_export_mw == pytest.approx(last_trace.p_export_mw)
    assert final.final_hard_export_violation_mw == pytest.approx(
        last_trace.hard_export_violation_mw
    )
    summary = {row.metric: row.value for row in summary_rows}
    assert summary["final_curtailment_factor"] == pytest.approx(
        final.final_curtailment_factor
    )
    assert summary["final_p_export_mw"] == pytest.approx(final.final_p_export_mw)


def test_export_all_writes_required_artifacts(exp_module, small_results, tmp_path: Path):
    baseline_rows, trace_rows, final_rows, grid_rows, diagnostics_rows, summary_rows = small_results

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
    assert meta["experiment"] == "exp05b_optimize_pv_curtailment"

    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "PV Curtailment Optimization" in readme
    assert "7.0 MW" in readme


def test_no_nan_or_inf_in_smoke_outputs(small_results):
    baseline_rows, trace_rows, final_rows, grid_rows, diagnostics_rows, summary_rows = small_results

    _assert_finite_numeric_rows(baseline_rows)
    _assert_finite_numeric_rows(trace_rows)
    _assert_finite_numeric_rows(final_rows)
    _assert_finite_numeric_rows(grid_rows)
    _assert_finite_numeric_rows(diagnostics_rows)
    _assert_finite_numeric_rows(summary_rows)
