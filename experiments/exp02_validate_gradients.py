"""Experiment 2: Validate implicit power-flow gradients against finite differences."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

from diffpf.io import load_json
from diffpf.validation import (
    experiment2_scenarios,
    finite_difference_step_study,
    scenario_from_raw,
    summarize_errors,
    validate_scenario_gradients,
)


RESULTS_DIR = Path(__file__).resolve().parent / "results" / "exp02_gradient_validation"


def _write_csv(path: Path, rows: tuple) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def _write_json(path: Path, rows: tuple) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump([asdict(row) for row in rows], handle, indent=2)


def _format_summary(summary) -> str:
    return (
        f"{summary.scenario}: n={summary.n_gradients}, "
        f"max_abs={summary.max_abs_error:.3e}, "
        f"max_rel={summary.max_rel_error:.3e}, "
        f"mean_abs={summary.mean_abs_error:.3e}, "
        f"mean_rel={summary.mean_rel_error:.3e}"
    )


def main() -> None:
    case_path = Path(__file__).resolve().parents[1] / "cases" / "three_bus_poc.json"
    raw = load_json(case_path)

    gradient_rows = []
    step_rows = []
    for scenario in experiment2_scenarios():
        topology, params, state = scenario_from_raw(raw, scenario)
        gradient_rows.extend(
            validate_scenario_gradients(
                scenario.name,
                topology,
                params,
                state,
                fd_step=1e-5,
            )
        )
        if scenario.name == "medium_pv":
            step_rows.extend(
                finite_difference_step_study(
                    scenario.name,
                    topology,
                    params,
                    state,
                )
            )

    gradient_rows = tuple(gradient_rows)
    step_rows = tuple(step_rows)
    summaries = summarize_errors(gradient_rows)

    _write_csv(RESULTS_DIR / "gradient_table.csv", gradient_rows)
    _write_json(RESULTS_DIR / "gradient_table.json", gradient_rows)
    _write_csv(RESULTS_DIR / "error_summary.csv", summaries)
    _write_json(RESULTS_DIR / "error_summary.json", summaries)
    _write_csv(RESULTS_DIR / "fd_step_study.csv", step_rows)
    _write_json(RESULTS_DIR / "fd_step_study.json", step_rows)

    print("Experiment 2: Implicit-gradient validation against central finite differences")
    print(f"Case file: {case_path}")
    print(f"Results directory: {RESULTS_DIR}")
    print()
    print("Aggregated error summary:")
    for summary in summaries:
        print(f"  {_format_summary(summary)}")
    print()
    print("Exported files:")
    print("  gradient_table.csv / gradient_table.json")
    print("  error_summary.csv / error_summary.json")
    print("  fd_step_study.csv / fd_step_study.json")


if __name__ == "__main__":
    main()
