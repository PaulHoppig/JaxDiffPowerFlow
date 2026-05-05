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
SWEEP_T_AMB_TICKS = (5, 15, 25, 35, 45, 55)
GRID_G_POA_TICKS = (200, 400, 600, 800, 1000)
GRID_T_AMB_TICKS = (5, 15, 25, 35, 45)
SLACK_SIGN_NOTE = "negative = export to upstream grid"
SWEEP_FIXED_WEATHER_LABEL = r"$G_{poa} = 800$ W/m$^2$, $v_{wind} = 2$ m/s"

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


def _scenario_label(scenario: str) -> str:
    labels = {
        "base": "Base",
        "load_low": "Low load",
        "load_high": "High load",
    }
    return labels.get(scenario, scenario)


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


def sensitivity_values_kw_per_c(rows: list[dict]) -> list[float]:
    """Return sensitivity values converted from MW/degC to kW/degC."""

    return [1000.0 * _as_float(row, "value") for row in rows]


def padded_limits(values: list[float], pad_fraction: float = 0.25) -> tuple[float, float]:
    """Return finite display limits with padding around the data range."""

    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return -1.0, 1.0

    lower = float(np.min(finite))
    upper = float(np.max(finite))
    if lower == upper:
        pad = max(abs(lower) * 0.05, 1e-6)
    else:
        pad = pad_fraction * (upper - lower)
    return lower - pad, upper + pad


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
        ax.plot(
            x,
            y,
            marker="o",
            markersize=4.2,
            linewidth=1.6,
            label=_scenario_label(scenario),
        )

    ax.set_xticks(SWEEP_T_AMB_TICKS)
    ax.set_xlabel(r"Ambient temperature $T_{amb}$ [$^\circ$C]")
    ax.set_ylabel("Slack active power $P_{slack}$ [MW]")
    ax.set_title(SWEEP_FIXED_WEATHER_LABEL, fontsize=10)
    ax.legend(title="Scenario", fontsize=8, title_fontsize=9, frameon=False)
    ax.grid(True, alpha=0.3)
    ax.text(
        0.01,
        -0.22,
        SLACK_SIGN_NOTE,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
    return _plot_export(fig, "fig01_t_amb_sweep_p_slack", figures_dir)


def plot_fig02_heatmap_g_t_p_slack_base(
    scenario_rows: list[dict],
    figures_dir: Path = FIGURES_DIR,
) -> tuple[Path, Path]:
    """Figure 2: base-scenario weather heatmap for slack active power."""

    g_values, t_values, matrix = pivot_grid_2d_base_p_slack(scenario_rows)
    g_edges = _coordinate_edges(g_values)
    t_edges = _coordinate_edges(t_values)

    fig, ax = plt.subplots(figsize=(6.6, 4.8), constrained_layout=True)
    mesh = ax.pcolormesh(
        g_edges,
        t_edges,
        matrix,
        shading="flat",
        edgecolors="white",
        linewidth=0.55,
    )
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Slack active power $P_{slack}$ [MW]")
    ax.set_xticks(GRID_G_POA_TICKS)
    ax.set_yticks(GRID_T_AMB_TICKS)
    ax.set_xlabel(r"Plane-of-array irradiance $G_{poa}$ [W/m$^2$]")
    ax.set_ylabel(r"Ambient temperature $T_{amb}$ [$^\circ$C]")
    ax.set_title("Base scenario: Slack active power over weather grid", fontsize=10)
    ax.text(
        0.01,
        -0.20,
        SLACK_SIGN_NOTE,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )
    return _plot_export(fig, "fig02_heatmap_g_t_p_slack_base", figures_dir)


def plot_fig03_sensitivity_p_slack_vs_t_amb(
    sensitivity_rows: list[dict],
    figures_dir: Path = FIGURES_DIR,
) -> tuple[Path, Path]:
    """Figure 3: sensitivity d(P_slack)/d(T_amb) over the 1D sweep."""

    grouped = select_fig03_rows(sensitivity_rows)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    all_y: list[float] = []
    for scenario in SCENARIO_ORDER:
        rows = grouped[scenario]
        x = [_as_float(row, "t_amb_c") for row in rows]
        y = sensitivity_values_kw_per_c(rows)
        all_y.extend(y)
        ax.plot(
            x,
            y,
            marker="o",
            markersize=4.2,
            linewidth=1.6,
            label=_scenario_label(scenario),
        )

    ax.set_ylim(*padded_limits(all_y, pad_fraction=0.30))

    ax.set_xticks(SWEEP_T_AMB_TICKS)
    ax.set_xlabel(r"Ambient temperature $T_{amb}$ [$^\circ$C]")
    ax.set_ylabel(
        r"Local sensitivity $\partial P_{slack} / \partial T_{amb}$ [kW/$^\circ$C]"
    )
    ax.set_title(
        r"Local AD sensitivity at $G_{poa} = 800$ W/m$^2$, $v_{wind} = 2$ m/s",
        fontsize=10,
    )
    ax.legend(title="Scenario", fontsize=8, title_fontsize=9, frameon=False)
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
Lines compare `base`, `load_low`, and `load_high` over `t_amb_c` at fixed
`g_poa_wm2 = 800` and `wind_ms = 2`.

## Figure 2

Files: `fig02_heatmap_g_t_p_slack_base.png` and
`fig02_heatmap_g_t_p_slack_base.pdf`. Data source: `scenario_grid.csv`.
Filter: `weather_case_type == "grid_2d"`, `network_scenario == "base"`, and
`observable == "p_slack_mw"`. Values are pivoted to a
`t_amb_c x g_poa_wm2` matrix and displayed as a discrete 5 x 5 weather grid.

## Figure 3

Files: `fig03_sensitivity_p_slack_vs_t_amb.png` and
`fig03_sensitivity_p_slack_vs_t_amb.pdf`. Data source:
`sensitivity_table.csv`. Filter: `weather_case_type == "sweep_1d"`,
`observable == "p_slack_mw"`, and `input_parameter == "t_amb_c"`.
Lines compare `base`, `load_low`, and `load_high` over `t_amb_c`. The raw
artifact stores sensitivities in MW/degC; the figure converts them to kW/degC
by multiplying `value` by 1000.

## Sign convention

For all slack active-power plots, negative `P_slack` values denote export to
the upstream grid.
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
