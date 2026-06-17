"""Tests for Experiment 5d plotting from existing artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture(scope="module")
def plot_module():
    import experiments.plot_exp05d_figures as module

    return module


def test_module_is_importable(plot_module):
    assert plot_module is not None


def test_output_path_is_exp05d_specific(plot_module):
    assert plot_module.EXP05D_RESULTS_DIR.name == (
        "exp05d_optimize_pv_curtailment_simple_objective"
    )
    assert plot_module.FIGURES_DIR.name == "exp05d_figures"


def test_grid_best_objective_uses_minimum_objective(plot_module):
    grid = pd.DataFrame(
        {
            "curtailment_factor": [0.0, 0.5, 0.718, 0.719, 1.0],
            "p_export_mw": [5.4, 6.5, 6.999, 7.001, 7.6],
            "objective": [2.56, 0.25, 0.000001, 0.000004, 0.36],
        }
    )

    best = plot_module._grid_best_objective(grid)

    assert best["curtailment_factor"] == pytest.approx(0.718)


def _write_dummy_artifacts(exp05a_dir: Path, exp05d_dir: Path) -> None:
    exp05a_dir.mkdir(parents=True, exist_ok=True)
    exp05d_dir.mkdir(parents=True, exist_ok=True)

    screening = pd.DataFrame(
        [
            {
                "case_id": "ref_load0p4_no_pv",
                "case_type": "no_pv_reference",
                "load_multiplier_mv_bus_2": 0.4,
                "g_poa_wm2": 0.0,
                "p_export_mw": 5.4,
            },
            {
                "case_id": "screen_low",
                "case_type": "screening",
                "load_multiplier_mv_bus_2": 1.0,
                "g_poa_wm2": 200.0,
                "p_export_mw": 5.9,
            },
            {
                "case_id": "screen_high",
                "case_type": "screening",
                "load_multiplier_mv_bus_2": 0.4,
                "g_poa_wm2": 1200.0,
                "p_export_mw": 7.7,
            },
        ]
    )
    selected = pd.DataFrame(
        [
            {
                "case_id": "selected_realistic_load0p4_g1200_t30",
                "case_type": "selected_realistic_case",
                "load_multiplier_mv_bus_2": 0.4,
                "g_poa_wm2": 1200.0,
                "p_export_mw": 7.6,
            }
        ]
    )
    baseline = pd.DataFrame(
        [
            {
                "baseline_type": "full_pv",
                "curtailment_factor": 1.0,
                "p_export_mw": 7.6,
            },
            {
                "baseline_type": "zero_pv",
                "curtailment_factor": 0.0,
                "p_export_mw": 5.46,
            },
        ]
    )
    final = pd.DataFrame(
        [
            {
                "final_curtailment_factor": 0.719,
                "final_p_export_mw": 7.0001,
                "grid_best_objective_curtailment_factor": 0.719,
                "grid_best_feasible_curtailment_factor": 0.718,
                "grid_best_curtailment_factor": 0.719,
                "p_export_limit_mw": 7.0,
                "p_export_target_mw": 7.0,
            }
        ]
    )
    grid = pd.DataFrame(
        {
            "curtailment_factor": [0.0, 0.5, 0.718, 0.719, 0.75, 1.0],
            "p_export_mw": [5.46, 6.55, 6.998, 7.0001, 7.06, 7.6],
            "objective": [2.37, 0.20, 0.000004, 0.00000001, 0.0036, 0.36],
            "is_grid_best_objective": [False, False, False, True, False, False],
            "is_grid_best_feasible": [False, False, True, False, False, False],
        }
    )
    trace = pd.DataFrame(
        {
            "iteration": [0, 1, 2, 3],
            "p_export_mw": [7.17, 7.05, 7.005, 7.0001],
            "curtailment_factor": [0.8, 0.75, 0.72, 0.719],
        }
    )

    screening.to_csv(exp05a_dir / "screening_results.csv", index=False)
    selected.to_csv(exp05a_dir / "selected_realistic_case.csv", index=False)
    baseline.to_csv(exp05d_dir / "selected_case_baseline.csv", index=False)
    final.to_csv(exp05d_dir / "final_solution.csv", index=False)
    grid.to_csv(exp05d_dir / "grid_reference.csv", index=False)
    trace.to_csv(exp05d_dir / "optimization_trace.csv", index=False)


def test_generate_figures_from_dummy_exp05d_artifacts(plot_module, tmp_path: Path):
    exp05a_dir = tmp_path / "exp05a"
    exp05d_dir = tmp_path / "exp05d"
    output_dir = tmp_path / "figures"
    _write_dummy_artifacts(exp05a_dir, exp05d_dir)

    generated = plot_module.generate_figures(
        exp05a_dir=exp05a_dir,
        exp05d_dir=exp05d_dir,
        output_dir=output_dir,
    )

    for stem in plot_module.FIGURE_STEMS:
        assert (output_dir / f"{stem}.png").exists()
        assert (output_dir / f"{stem}.pdf").exists()
    assert (output_dir / "README.md").exists()
    assert (output_dir / "figure_metadata.json").exists()
    assert len(generated) == 10

    readme = (output_dir / "README.md").read_text(encoding="utf-8")
    assert "Experiment 5d" in readme
    assert "simple objective" in readme
    assert "6.99-MW" in readme
