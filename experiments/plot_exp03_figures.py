"""Create publication-style figures from existing Experiment 3 artifacts.

This script reads only the already exported CSV files in
``experiments/results/exp03_cross_domain_pv_weather``. It does not run new power
flows, does not call pandapower, and does not modify the numerical JAX core.

Run:
    python experiments/plot_exp03_figures.py
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = (
    Path(__file__).resolve().parent / "results" / "exp03_cross_domain_pv_weather"
)
FIGURES_DIR = RESULTS_DIR / "figures"

SCENARIO_ORDER = ("base", "load_low", "load_high")

SCENARIO_GRID_REQUIRED_COLUMNS = {
    "network_scenario",
    "weather_case_type",
    "g_poa_wm2",
    "t_amb_c",
    "observable",
    "value",
    "unit",
}

SENSITIVITY_REQUIRED_COLUMNS = {
    "network_scenario",
    "weather_case_type",
    "g_poa_wm2",
    "t_amb_c",
    "observable",
    "observable_unit",
    "input_parameter",
    "input_unit",
    "value",
}


def _read_csv(path: Path, required_columns: set[str]) -> list[dict[str, str]]:
    """Read a CSV artifact and validate its basic schema."""

    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}")

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        missing = sorted(required_columns - fieldnames)
        if missing:
            raise ValueError(f"{path.name} is missing required columns: {missing}")
        return list(reader)


def load_artifacts(results_dir: Path = RESULTS_DIR) -> tuple[list[dict], list[dict]]:
    """Load the existing Exp. 3 scenario and sensitivity CSV artifacts."""

    scenario_rows = _read_csv(
        results_dir / "scenario_grid.csv",
        SCENARIO_GRID_REQUIRED_COLUMNS,
    )
    sensitivity_rows = _read_csv(
        results_dir / "sensitivity_table.csv",
        SENSITIVITY_REQUIRED_COLUMNS,
    )
    return scenario_rows, sensitivity_rows


def _as_float(row: dict, column: str) -> float:
    return float(row[column])


def _plot_export(fig, stem: str, figures_dir: Path) -> tuple[Path, Path]:
    """Save a figure as PNG and PDF and close it."""

    figures_dir.mkdir(parents=True, exist_ok=True)
    png_path = figures_dir / f"{stem}.png"
    pdf_path = figures_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def _rows_for_scenario(
    rows: Iterable[dict],
    scenario: str,
    weather_case_type: str,
    observable: str,
) -> list[dict]:
    selected = [
        row
        for row in rows
        if row["network_scenario"] == scenario
        and row["weather_case_type"] == weather_case_type
        and row["observable"] == observable
    ]
    return sorted(selected, key=lambda row: _as_float(row, "t_amb_c"))


def _sensitivity_rows_for_scenario(
    rows: Iterable[dict],
    scenario: str,
    weather_case_type: str,
    observable: str,
    input_parameter: str,
) -> list[dict]:
    selected = [
        row
        for row in rows
        if row["network_scenario"] == scenario
        and row["weather_case_type"] == weather_case_type
        and row["observable"] == observable
        and row["input_parameter"] == input_parameter
    ]
    return sorted(selected, key=lambda row: _as_float(row, "t_amb_c"))


def select_fig01_rows(scenario_rows: list[dict]) -> dict[str, list[dict]]:
    """Return sweep_1d p_slack_mw rows grouped by electrical scenario."""

    grouped = {
        scenario: _rows_for_scenario(
            scenario_rows,
            scenario=scenario,
            weather_case_type="sweep_1d",
            observable="p_slack_mw",
        )
        for scenario in SCENARIO_ORDER
    }
    _ensure_nonempty_groups(grouped, "fig01 sweep_1d p_slack_mw")
    return grouped


def select_fig03_rows(sensitivity_rows: list[dict]) -> dict[str, list[dict]]:
    """Return sweep_1d d(p_slack_mw)/d(t_amb_c) rows grouped by scenario."""

    grouped = {
        scenario: _sensitivity_rows_for_scenario(
            sensitivity_rows,
            scenario=scenario,
            weather_case_type="sweep_1d",
            observable="p_slack_mw",
            input_parameter="t_amb_c",
        )
        for scenario in SCENARIO_ORDER
    }
    _ensure_nonempty_groups(grouped, "fig03 sweep_1d d_p_slack/d_t_amb")
    return grouped


def _ensure_nonempty_groups(grouped: dict[str, list[dict]], label: str) -> None:
    empty = [name for name, rows in grouped.items() if not rows]
    if empty:
        raise ValueError(f"No rows for {label}: {empty}")


def _coordinate_edges(values: list[float]) -> np.ndarray:
    coords = np.asarray(values, dtype=float)
    if coords.size == 1:
        step = 1.0
        return np.asarray([coords[0] - 0.5 * step, coords[0] + 0.5 * step])
    mids = 0.5 * (coords[:-1] + coords[1:])
    first = coords[0] - (mids[0] - coords[0])
    last = coords[-1] + (coords[-1] - mids[-1])
    return np.concatenate([[first], mids, [last]])


def pivot_grid_2d_base_p_slack(
    scenario_rows: list[dict],
) -> tuple[list[float], list[float], np.ndarray]:
    """Pivot base/grid_2d p_slack_mw rows into a T x G matrix."""

    selected = [
        row
        for row in scenario_rows
        if row["network_scenario"] == "base"
        and row["weather_case_type"] == "grid_2d"
        and row["observable"] == "p_slack_mw"
    ]
    if not selected:
        raise ValueError("No rows found for base/grid_2d p_slack_mw heatmap.")

    g_values = sorted({_as_float(row, "g_poa_wm2") for row in selected})
    t_values = sorted({_as_float(row, "t_amb_c") for row in selected})
    value_by_coord = {
        (_as_float(row, "t_amb_c"), _as_float(row, "g_poa_wm2")): _as_float(
            row, "value"
        )
        for row in selected
    }

    matrix = np.empty((len(t_values), len(g_values)), dtype=float)
    for i, temp in enumerate(t_values):
        for j, irradiance in enumerate(g_values):
            key = (temp, irradiance)
            if key not in value_by_coord:
                raise ValueError(
                    "Missing heatmap value for "
                    f"t_amb_c={temp}, g_poa_wm2={irradiance}."
                )
            matrix[i, j] = value_by_coord[key]

    return g_values, t_values, matrix


def plot_fig01_t_amb_sweep_p_slack(
    scenario_rows: list[dict],
    figures_dir: Path = FIGURES_DIR,
) -> tuple[Path, Path]:
    """Figure 1: 1D temperature sweep of slack active power."""

    grouped = select_fig01_rows(scenario_rows)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for scenario in SCENARIO_ORDER:
        rows = grouped[scenario]
        x = [_as_float(row, "t_amb_c") for row in rows]
        y = [_as_float(row, "value") for row in rows]
        ax.plot(x, y, marker="o", linewidth=1.6, label=scenario)

    ax.set_xlabel("Ambient temperature $T_{amb}$ [degC]")
    ax.set_ylabel("Slack active power $P_{slack}$ [MW]")
    ax.legend(title="Scenario")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _plot_export(fig, "fig01_t_amb_sweep_p_slack", figures_dir)


def plot_fig02_heatmap_g_t_p_slack_base(
    scenario_rows: list[dict],
    figures_dir: Path = FIGURES_DIR,
) -> tuple[Path, Path]:
    """Figure 2: base-scenario weather heatmap for slack active power."""

    g_values, t_values, matrix = pivot_grid_2d_base_p_slack(scenario_rows)
    g_edges = _coordinate_edges(g_values)
    t_edges = _coordinate_edges(t_values)

    fig, ax = plt.subplots(figsize=(6.3, 4.5))
    mesh = ax.pcolormesh(g_edges, t_edges, matrix, shading="auto")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Slack active power $P_{slack}$ [MW]")
    ax.set_xlabel("Plane-of-array irradiance $G_{poa}$ [W/m^2]")
    ax.set_ylabel("Ambient temperature $T_{amb}$ [degC]")
    ax.set_title("Base scenario")
    fig.tight_layout()
    return _plot_export(fig, "fig02_heatmap_g_t_p_slack_base", figures_dir)


def plot_fig03_sensitivity_p_slack_vs_t_amb(
    sensitivity_rows: list[dict],
    figures_dir: Path = FIGURES_DIR,
) -> tuple[Path, Path]:
    """Figure 3: sensitivity d(P_slack)/d(T_amb) over the 1D sweep."""

    grouped = select_fig03_rows(sensitivity_rows)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for scenario in SCENARIO_ORDER:
        rows = grouped[scenario]
        x = [_as_float(row, "t_amb_c") for row in rows]
        y = [_as_float(row, "value") for row in rows]
        ax.plot(x, y, marker="o", linewidth=1.6, label=scenario)

    ax.set_xlabel("Ambient temperature $T_{amb}$ [degC]")
    ax.set_ylabel(r"$dP_{slack} / dT_{amb}$ [MW/degC]")
    ax.legend(title="Scenario")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _plot_export(fig, "fig03_sensitivity_p_slack_vs_t_amb", figures_dir)


def write_figures_readme(
    figures_dir: Path = FIGURES_DIR,
    results_dir: Path = RESULTS_DIR,
) -> Path:
    """Write a concise description of figure sources and filters."""

    figures_dir.mkdir(parents=True, exist_ok=True)
    text = f"""# Experiment 3 Figures

These figures are generated only from existing Experiment 3 artifacts in
`{results_dir.as_posix()}`. No power-flow solves or new experiment scenarios are
run by the plotting script.

## Figure 1

Files: `fig01_t_amb_sweep_p_slack.png` and `fig01_t_amb_sweep_p_slack.pdf`.
Data source: `scenario_grid.csv`. Filter:
`weather_case_type == "sweep_1d"` and `observable == "p_slack_mw"`.
Lines compare `base`, `load_low`, and `load_high` over `t_amb_c`.

## Figure 2

Files: `fig02_heatmap_g_t_p_slack_base.png` and
`fig02_heatmap_g_t_p_slack_base.pdf`. Data source: `scenario_grid.csv`.
Filter: `weather_case_type == "grid_2d"`, `network_scenario == "base"`, and
`observable == "p_slack_mw"`. Values are pivoted to a
`t_amb_c x g_poa_wm2` matrix.

## Figure 3

Files: `fig03_sensitivity_p_slack_vs_t_amb.png` and
`fig03_sensitivity_p_slack_vs_t_amb.pdf`. Data source:
`sensitivity_table.csv`. Filter: `weather_case_type == "sweep_1d"`,
`observable == "p_slack_mw"`, and `input_parameter == "t_amb_c"`.
Lines compare `base`, `load_low`, and `load_high`.
"""
    path = figures_dir / "README.md"
    path.write_text(text, encoding="utf-8")
    return path


def generate_figures(
    results_dir: Path = RESULTS_DIR,
    figures_dir: Path | None = None,
) -> list[Path]:
    """Generate all mandatory Exp. 3 figures from existing artifacts."""

    target_dir = figures_dir if figures_dir is not None else results_dir / "figures"
    scenario_rows, sensitivity_rows = load_artifacts(results_dir)

    outputs: list[Path] = []
    outputs.extend(plot_fig01_t_amb_sweep_p_slack(scenario_rows, target_dir))
    outputs.extend(plot_fig02_heatmap_g_t_p_slack_base(scenario_rows, target_dir))
    outputs.extend(plot_fig03_sensitivity_p_slack_vs_t_amb(sensitivity_rows, target_dir))
    outputs.append(write_figures_readme(target_dir, results_dir))
    return outputs


def main() -> None:
    outputs = generate_figures(RESULTS_DIR, FIGURES_DIR)
    print(f"Figures directory: {FIGURES_DIR}")
    for path in outputs:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
