"""Create figures for Experiment 5c from existing CSV/JSON artifacts.

This script reads already exported artifacts from Experiment 5a and Experiment
5c. It does not run power-flow solves, does not retrain the NN surrogate, does
not run an optimization, and does not perform a grid search.

Run:
    python experiments/plot_exp05c_figures.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import plot_exp05_figures as exp05_plot


EXP05A_RESULTS_DIR = exp05_plot.EXP05A_RESULTS_DIR
EXP05C_RESULTS_DIR = (
    Path(__file__).resolve().parent / "results" / "exp05c_optimize_pv_curtailment_nn"
)
FIGURES_DIR = Path(__file__).resolve().parent / "results" / "exp05c_figures"

FIGURE_STEMS = exp05_plot.FIGURE_STEMS
SELECTED_CASE_ID = exp05_plot.SELECTED_CASE_ID
EXPORT_LIMIT_MW = exp05_plot.EXPORT_LIMIT_MW
EXPORT_TARGET_MW = 7.0


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def write_readme(output_dir: Path) -> Path:
    """Write the Exp. 5c figure directory README."""

    text = """# Experiment 5c Figures

This directory contains figures for Experiment 5c. The figures are created
only from existing CSV/JSON artifacts of Experiment 5a and Experiment 5c. The
plotting script does not run power-flow solves, does not retrain the NN
surrogate, does not run a grid search, and does not start a curtailment
optimization.

## Figures

- `fig51_screening_export_overview.png/pdf`: Existing Exp.-5a screening export
  context with the selected 30 degC case highlighted. The actual optimization
  shown in the other figures is Exp. 5c and uses the NN upstream PV surrogate.
- `fig53_export_before_after_reference.png/pdf`: Full PV, optimized, grid
  reference, and zero PV exports for Exp. 5c.
- `fig54_grid_reference_export_vs_curtailment.png/pdf`: Exp.-5c grid
  reference curve `p_export(c)` with optimizer and grid-reference points.
- `fig55_optimization_trace_export_and_curtailment.png/pdf`: Exp.-5c optimizer
  trace for export and curtailment factor.

## Data Sources

- `experiments/results/exp05a_network_screening/screening_results.csv`
- `experiments/results/exp05a_network_screening/selected_realistic_case.csv`
- `experiments/results/exp05c_optimize_pv_curtailment_nn/selected_case_baseline.csv`
- `experiments/results/exp05c_optimize_pv_curtailment_nn/final_solution.csv`
- `experiments/results/exp05c_optimize_pv_curtailment_nn/grid_reference.csv`
- `experiments/results/exp05c_optimize_pv_curtailment_nn/optimization_trace.csv`

## Scope Notes

Experiment 5c replaces only the upstream PV model with the trained Experiment 4
NN surrogate. The electrical core remains unchanged. The 7.0 MW line is a
demonstrator-internal target, not a normative grid-code limit. The NN is a
synthetic P-only distillation surrogate; reactive power remains
deterministically coupled as `Q = -0.25 * P`.
"""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "README.md"
    path.write_text(text, encoding="utf-8")
    return path


def write_metadata(
    output_dir: Path,
    inputs: exp05_plot.FigureInputs,
    figure_paths: list[Path],
    readme_path: Path,
) -> Path:
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_hash": _git_hash(),
        "script": "experiments/plot_exp05c_figures.py",
        "experiment": "exp05c_optimize_pv_curtailment_nn",
        "selected_case_id": SELECTED_CASE_ID,
        "export_limit_mw": EXPORT_LIMIT_MW,
        "export_target_mw": EXPORT_TARGET_MW,
        "input_artifacts": inputs.as_dict(),
        "output_files": [str(path) for path in figure_paths] + [str(readme_path)],
        "notes": [
            "Figures are generated only from existing Exp. 5a and Exp. 5c artifacts.",
            "No power-flow solves, NN training, grid search, or optimization are run.",
            "Exp. 5c uses the NN upstream PV surrogate from Experiment 4.",
        ],
    }
    path = output_dir / "figure_metadata.json"
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def generate_figures(
    exp05a_dir: Path = EXP05A_RESULTS_DIR,
    exp05c_dir: Path = EXP05C_RESULTS_DIR,
    output_dir: Path = FIGURES_DIR,
) -> list[Path]:
    """Generate all Experiment 5c figures from existing artifacts."""

    exp05_plot._configure_matplotlib()
    inputs = exp05_plot.FigureInputs(exp05a_dir=exp05a_dir, exp05b_dir=exp05c_dir)
    output_dir = exp05_plot.ensure_output_dir(output_dir)

    screening_df = exp05_plot.load_csv(inputs.screening_results)
    selected_df = exp05_plot.load_csv(inputs.selected_realistic_case)
    baseline_df = exp05_plot.load_csv(inputs.selected_case_baseline)
    final_df = exp05_plot.load_csv(inputs.final_solution)
    grid_df = exp05_plot.load_csv(inputs.grid_reference)
    trace_df = exp05_plot.load_csv(inputs.optimization_trace)

    limit_mw, target_mw = exp05_plot._artifact_limit_and_target(final_df)
    figure_paths: list[Path] = []
    for png_path, pdf_path in [
        exp05_plot.plot_fig51(screening_df, selected_df, output_dir, limit_mw),
        exp05_plot.plot_fig53(baseline_df, final_df, grid_df, output_dir),
        exp05_plot.plot_fig54(grid_df, final_df, output_dir),
        exp05_plot.plot_fig55(trace_df, output_dir, limit_mw=limit_mw, target_mw=target_mw),
    ]:
        figure_paths.extend([png_path, pdf_path])

    readme_path = write_readme(output_dir)
    metadata_path = write_metadata(output_dir, inputs, figure_paths, readme_path)
    return figure_paths + [readme_path, metadata_path]


def main() -> None:
    print("=" * 72)
    print("Experiment 5c: figure generation from existing artifacts")
    print(f"Output directory: {FIGURES_DIR}")
    print("=" * 72)
    paths = generate_figures()
    print("\nGenerated files:")
    for path in paths:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
