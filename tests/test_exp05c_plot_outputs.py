"""Tests for Experiment 5c plotting from existing artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture(scope="module")
def plot_module():
    import experiments.plot_exp05c_figures as module

    return module


def test_module_is_importable(plot_module):
    assert plot_module is not None


def test_output_path_is_exp05c_specific(plot_module):
    assert plot_module.EXP05C_RESULTS_DIR.name == "exp05c_optimize_pv_curtailment_nn"
    assert plot_module.FIGURES_DIR.name == "exp05c_figures"


def _write_dummy_artifacts(exp05a_dir: Path, exp05c_dir: Path) -> None:
    exp05a_dir.mkdir(parents=True, exist_ok=True)
    exp05c_dir.mkdir(parents=True, exist_ok=True)

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
                "final_curtailment_factor": 0.714,
                "final_p_export_mw": 6.99,
                "grid_best_curtailment_factor": 0.718,
                "p_export_limit_mw": 7.0,
                "p_export_target_mw": 6.99,
            }
        ]
    )
    grid = pd.DataFrame(
        {
            "curtailment_factor": [0.0, 0.5, 0.714, 0.718, 0.75, 1.0],
            "p_export_mw": [5.46, 6.55, 6.99, 6.998, 7.06, 7.6],
        }
    )
    trace = pd.DataFrame(
        {
            "iteration": [0, 1, 2, 3],
            "p_export_mw": [7.17, 7.05, 6.995, 6.99],
            "curtailment_factor": [0.8, 0.75, 0.72, 0.714],
        }
    )

    screening.to_csv(exp05a_dir / "screening_results.csv", index=False)
    selected.to_csv(exp05a_dir / "selected_realistic_case.csv", index=False)
    baseline.to_csv(exp05c_dir / "selected_case_baseline.csv", index=False)
    final.to_csv(exp05c_dir / "final_solution.csv", index=False)
    grid.to_csv(exp05c_dir / "grid_reference.csv", index=False)
    trace.to_csv(exp05c_dir / "optimization_trace.csv", index=False)


def test_generate_figures_from_dummy_exp05c_artifacts(plot_module, tmp_path: Path):
    exp05a_dir = tmp_path / "exp05a"
    exp05c_dir = tmp_path / "exp05c"
    output_dir = tmp_path / "figures"
    _write_dummy_artifacts(exp05a_dir, exp05c_dir)

    generated = plot_module.generate_figures(
        exp05a_dir=exp05a_dir,
        exp05c_dir=exp05c_dir,
        output_dir=output_dir,
    )

    for stem in plot_module.FIGURE_STEMS:
        assert (output_dir / f"{stem}.png").exists()
        assert (output_dir / f"{stem}.pdf").exists()
    assert (output_dir / "README.md").exists()
    assert (output_dir / "figure_metadata.json").exists()
    assert len(generated) == 10

    readme = (output_dir / "README.md").read_text(encoding="utf-8")
    assert "Experiment 5c" in readme
    assert "NN upstream PV surrogate" in readme
