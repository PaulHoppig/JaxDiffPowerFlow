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
import pandas as pd


RESULTS_DIR = (
    Path(__file__).resolve().parent / "results" / "exp03_cross_domain_pv_weather"
)
FIGURES_DIR = RESULTS_DIR / "figures"

SCENARIO_ORDER = ("base", "load_low", "load_high")
SWEEP_T_AMB_TICKS = (5, 15, 25, 35, 45, 55)
GRID_G_POA_TICKS = (200, 400, 600, 800, 1000)
GRID_T_AMB_TICKS = (5, 15, 25, 35, 45)
G_SWEEP_TICKS = (200, 400, 600, 800, 1000)
SLACK_SIGN_NOTE = "Negative Werte bedeuten Export in das vorgelagerte Netz."
SWEEP_FIXED_WEATHER_LABEL = r"$G_{poa} = 800$ W/m$^2$, $v_{wind} = 2$ m/s"
G_SWEEP_FIXED_WEATHER_LABEL = r"$T_{amb} = 25^\circ$C, Wind = 2 m/s"

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

SENSITIVITY_HEATMAP_REQUIRED_COLUMNS = {
    "network_scenario",
    "weather_case_type",
    "g_poa_wm2",
    "t_amb_c",
    "observable",
    "input_parameter",
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
        "base": "Basis",
        "load_low": "Niedrige Last",
        "load_high": "Hohe Last",
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


def padded_limits(values: list[float], pad_fraction: float = 0.1) -> tuple[float, float]:
    """Return y-limits padded around finite values."""

    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return -1.0, 1.0
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if lo == hi:
        pad = max(abs(lo) * pad_fraction, 1e-9)
    else:
        pad = (hi - lo) * pad_fraction
    return lo - pad, hi + pad


def sensitivity_values_kw_per_c(rows: list[dict]) -> list[float]:
    """Return sensitivity values converted from MW/degC to kW/degC."""

    return [1000.0 * _as_float(row, "value") for row in rows]


def sensitivity_values_kw_per_wm2(rows: list[dict]) -> list[float]:
    """Return sensitivity values converted from MW/(W/m^2) to kW/(W/m^2)."""

    return [1000.0 * _as_float(row, "value") for row in rows]


def select_fig04_rows(scenario_rows: list[dict]) -> dict[str, list[dict]]:
    """Return sweep_g_1d p_slack_mw rows grouped by electrical scenario."""

    grouped = {
        scenario: sorted(
            [
                row
                for row in scenario_rows
                if row["network_scenario"] == scenario
                and row["weather_case_type"] == "sweep_g_1d"
                and row["observable"] == "p_slack_mw"
            ],
            key=lambda row: _as_float(row, "g_poa_wm2"),
        )
        for scenario in SCENARIO_ORDER
    }
    _ensure_nonempty_groups(grouped, "fig04 sweep_g_1d p_slack_mw")
    return grouped


def select_fig05_rows(sensitivity_rows: list[dict]) -> dict[str, list[dict]]:
    """Return sweep_g_1d d(p_slack_mw)/d(g_poa_wm2) rows grouped by scenario."""

    grouped = {
        scenario: sorted(
            [
                row
                for row in sensitivity_rows
                if row["network_scenario"] == scenario
                and row["weather_case_type"] == "sweep_g_1d"
                and row["observable"] == "p_slack_mw"
                and row["input_parameter"] == "g_poa_wm2"
            ],
            key=lambda row: _as_float(row, "g_poa_wm2"),
        )
        for scenario in SCENARIO_ORDER
    }
    _ensure_nonempty_groups(grouped, "fig05 sweep_g_1d d_p_slack/d_g_poa")
    return grouped


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


def _as_dataframe(data: pd.DataFrame | Iterable[dict]) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    return pd.DataFrame(list(data))


def _sensitivity_value_column(df: pd.DataFrame) -> str:
    for candidate in ("value", "sensitivity", "gradient", "ad_sensitivity"):
        if candidate in df.columns:
            return candidate
    raise ValueError(
        "sensitivity_table.csv must contain a sensitivity value column; "
        "expected one of: value, sensitivity, gradient, ad_sensitivity."
    )


def prepare_sensitivity_heatmap_grid(
    sensitivity_df: pd.DataFrame | Iterable[dict],
    input_parameter: str,
    network_scenario: str = "base",
    observable: str = "p_slack_mw",
) -> pd.DataFrame:
    """Filter and pivot grid_2d sensitivity rows to a T_amb x G_poa grid."""

    df = _as_dataframe(sensitivity_df)
    value_col = _sensitivity_value_column(df)
    required = SENSITIVITY_HEATMAP_REQUIRED_COLUMNS | {value_col}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"sensitivity_table.csv is missing required columns: {missing}")

    work = df.copy()
    for column in ("g_poa_wm2", "t_amb_c", value_col):
        work[column] = pd.to_numeric(work[column], errors="coerce")

    selected = work[
        (work["weather_case_type"] == "grid_2d")
        & (work["network_scenario"] == network_scenario)
        & (work["observable"] == observable)
        & (work["input_parameter"] == input_parameter)
    ].dropna(subset=["g_poa_wm2", "t_amb_c", value_col])
    if selected.empty:
        raise ValueError(
            "No sensitivity rows found for heatmap: "
            f"network_scenario={network_scenario!r}, observable={observable!r}, "
            f"input_parameter={input_parameter!r}."
        )

    try:
        grid = selected.pivot(
            index="t_amb_c",
            columns="g_poa_wm2",
            values=value_col,
        )
    except ValueError as exc:
        raise ValueError(
            "Duplicate sensitivity rows found for the requested G x T heatmap."
        ) from exc

    grid = grid.sort_index(axis=0).sort_index(axis=1)
    if grid.empty or grid.isna().any().any():
        raise ValueError(
            "Incomplete sensitivity heatmap grid after pivoting; "
            "check missing g_poa_wm2/t_amb_c combinations."
        )
    return grid


def sensitivity_heatmap_plot_grid(
    raw_grid: pd.DataFrame,
    input_parameter: str,
) -> tuple[pd.DataFrame, str, str]:
    """Convert raw MW/input-unit sensitivities to report units."""

    if input_parameter == "g_poa_wm2":
        return (
            raw_grid.astype(float) * 1000.0 * 100.0,
            r"$\partial P_\mathrm{slack} / \partial G_\mathrm{poa}$ [kW je 100 W/m$^2$]",
            "kW per 100 W/m^2",
        )
    if input_parameter == "t_amb_c":
        return (
            raw_grid.astype(float) * 1000.0,
            r"$\partial P_\mathrm{slack} / \partial T_\mathrm{amb}$ [kW/$^\circ$C]",
            "kW/degC",
        )
    raise ValueError(f"Unsupported sensitivity heatmap input_parameter: {input_parameter!r}")


def plot_sensitivity_heatmap(
    grid: pd.DataFrame,
    title: str,
    colorbar_label: str,
    output_png: Path,
    output_pdf: Path,
) -> tuple[Path, Path]:
    """Plot a discrete G x T sensitivity heatmap in the Fig. 2 style."""

    g_values = [float(value) for value in grid.columns]
    t_values = [float(value) for value in grid.index]
    g_edges = _coordinate_edges(g_values)
    t_edges = _coordinate_edges(t_values)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    mesh = ax.pcolormesh(
        g_edges,
        t_edges,
        grid.to_numpy(dtype=float),
        shading="flat",
        edgecolors="white",
        linewidth=0.55,
    )
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(colorbar_label)
    ax.set_xticks(g_values)
    ax.set_yticks(t_values)
    ax.set_xlabel(r"Einstrahlung in Modulebene $G_{poa}$ [W/m$^2$]")
    ax.set_ylabel(r"Umgebungstemperatur $T_{amb}$ [$^\circ$C]")
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 1.0))
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)
    return output_png, output_pdf


def plot_fig06_heatmap_g_t_sensitivity_p_slack_wrt_g(
    sensitivity_rows: list[dict],
    figures_dir: Path = FIGURES_DIR,
) -> tuple[Path, Path]:
    """Figure 6: grid_2d d(P_slack)/d(G_poa) heatmap for the base scenario."""

    raw_grid = prepare_sensitivity_heatmap_grid(sensitivity_rows, "g_poa_wm2")
    plot_grid, colorbar_label, _ = sensitivity_heatmap_plot_grid(
        raw_grid,
        "g_poa_wm2",
    )
    stem = "fig06_heatmap_g_t_sensitivity_p_slack_wrt_g"
    return plot_sensitivity_heatmap(
        plot_grid,
        "Slack-P sensitivity to irradiance over weather grid",
        colorbar_label,
        figures_dir / f"{stem}.png",
        figures_dir / f"{stem}.pdf",
    )


def plot_fig07_heatmap_g_t_sensitivity_p_slack_wrt_t_amb(
    sensitivity_rows: list[dict],
    figures_dir: Path = FIGURES_DIR,
) -> tuple[Path, Path]:
    """Figure 7: grid_2d d(P_slack)/d(T_amb) heatmap for the base scenario."""

    raw_grid = prepare_sensitivity_heatmap_grid(sensitivity_rows, "t_amb_c")
    plot_grid, colorbar_label, _ = sensitivity_heatmap_plot_grid(
        raw_grid,
        "t_amb_c",
    )
    stem = "fig07_heatmap_g_t_sensitivity_p_slack_wrt_t_amb"
    return plot_sensitivity_heatmap(
        plot_grid,
        "Slack-P sensitivity to ambient temperature over weather grid",
        colorbar_label,
        figures_dir / f"{stem}.png",
        figures_dir / f"{stem}.pdf",
    )


def plot_fig08_heatmap_g_t_sensitivity_p_slack_combined(
    sensitivity_rows: list[dict],
    figures_dir: Path = FIGURES_DIR,
) -> tuple[Path, Path]:
    """Figure 8: side-by-side grid_2d Slack-P sensitivity heatmaps."""

    raw_g = prepare_sensitivity_heatmap_grid(sensitivity_rows, "g_poa_wm2")
    raw_t = prepare_sensitivity_heatmap_grid(sensitivity_rows, "t_amb_c")
    grid_g, label_g, _ = sensitivity_heatmap_plot_grid(raw_g, "g_poa_wm2")
    grid_t, label_t, _ = sensitivity_heatmap_plot_grid(raw_t, "t_amb_c")

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), constrained_layout=True)
    panels = (
        (axes[0], grid_g, label_g),
        (axes[1], grid_t, label_t),
    )
    for ax, grid, colorbar_label in panels:
        g_values = [float(value) for value in grid.columns]
        t_values = [float(value) for value in grid.index]
        mesh = ax.pcolormesh(
            _coordinate_edges(g_values),
            _coordinate_edges(t_values),
            grid.to_numpy(dtype=float),
            shading="flat",
            edgecolors="white",
            linewidth=0.55,
        )
        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label(colorbar_label)
        ax.set_xticks(g_values)
        ax.set_yticks(t_values)
        ax.set_xlabel(r"Einstrahlung in Modulebene $G_{poa}$ [W/m$^2$]")
        ax.set_ylabel(r"Umgebungstemperatur $T_{amb}$ [$^\circ$C]")
    figures_dir.mkdir(parents=True, exist_ok=True)
    stem = "fig08_heatmap_g_t_sensitivity_p_slack_combined"
    png_path = figures_dir / f"{stem}.png"
    pdf_path = figures_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


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
    ax.set_xlabel(r"Umgebungstemperatur $T_{amb}$ [$^\circ$C]")
    ax.set_ylabel(r"Slack-Wirkleistung $P_{slack}$ [MW]")
    ax.legend(title="Szenario", fontsize=8, title_fontsize=9, frameon=True)
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

    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    mesh = ax.pcolormesh(
        g_edges,
        t_edges,
        matrix,
        shading="flat",
        edgecolors="white",
        linewidth=0.55,
    )
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(r"Slack-Wirkleistung $P_{slack}$ [MW]")
    ax.set_xticks(GRID_G_POA_TICKS)
    ax.set_yticks(GRID_T_AMB_TICKS)
    ax.set_xlabel(r"Einstrahlung in Modulebene $G_{poa}$ [W/m$^2$]")
    ax.set_ylabel(r"Umgebungstemperatur $T_{amb}$ [$^\circ$C]")
    ax.text(
        0.01,
        -0.20,
        SLACK_SIGN_NOTE,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
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

    y_values = np.asarray(all_y, dtype=float)
    if y_values.size and np.all(y_values >= 0.0):
        ax.set_ylim(0.0, float(np.nanmax(y_values)) * 1.15)
    elif y_values.size:
        span = float(np.nanmax(y_values) - np.nanmin(y_values))
        pad = max(span * 0.5, float(np.nanmax(np.abs(y_values))) * 0.1, 1e-9)
        ax.set_ylim(float(np.nanmin(y_values)) - pad, float(np.nanmax(y_values)) + pad)

    ax.set_xticks(SWEEP_T_AMB_TICKS)
    ax.set_xlabel(r"Umgebungstemperatur $T_{amb}$ [$^\circ$C]")
    ax.set_ylabel(r"$\partial P_{slack} / \partial T_{amb}$ [kW/$^\circ$C]")
    ax.legend(title="Szenario", fontsize=8, title_fontsize=9, frameon=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _plot_export(fig, "fig03_sensitivity_p_slack_vs_t_amb", figures_dir)


def plot_g_sweep_p_slack(
    scenario_rows: list[dict],
    figures_dir: Path = FIGURES_DIR,
) -> list[Path]:
    """Figure 4: 1D irradiance sweep of slack active power."""

    grouped = select_fig04_rows(scenario_rows)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for scenario in SCENARIO_ORDER:
        rows = grouped[scenario]
        x = [_as_float(row, "g_poa_wm2") for row in rows]
        y = [_as_float(row, "value") for row in rows]
        ax.plot(
            x,
            y,
            marker="o",
            markersize=4.2,
            linewidth=1.6,
            label=_scenario_label(scenario),
        )

    ax.set_xticks(G_SWEEP_TICKS)
    ax.set_xlabel(r"Einstrahlung in Modulebene $G_{poa}$ [W/m$^2$]")
    ax.set_ylabel(r"Slack-Wirkleistung $P_{slack}$ [MW]")
    ax.text(
        0.99,
        -0.22,
        G_SWEEP_FIXED_WEATHER_LABEL,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )
    ax.text(
        0.01,
        -0.22,
        SLACK_SIGN_NOTE,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )
    ax.legend(title="Szenario", fontsize=8, title_fontsize=9, frameon=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
    return list(_plot_export(fig, "fig04_g_sweep_p_slack", figures_dir))


def plot_g_sweep_p_slack_sensitivity(
    sensitivity_rows: list[dict],
    figures_dir: Path = FIGURES_DIR,
) -> list[Path]:
    """Figure 5: d(P_slack)/d(G_poa) over the 1D irradiance sweep."""

    grouped = select_fig05_rows(sensitivity_rows)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    all_y: list[float] = []
    for scenario in SCENARIO_ORDER:
        rows = grouped[scenario]
        x = [_as_float(row, "g_poa_wm2") for row in rows]
        y = sensitivity_values_kw_per_wm2(rows)
        all_y.extend(y)
        ax.plot(
            x,
            y,
            marker="o",
            markersize=4.2,
            linewidth=1.6,
            label=_scenario_label(scenario),
        )

    ax.set_xticks(G_SWEEP_TICKS)
    ax.set_xlabel(r"Einstrahlung in Modulebene $G_{poa}$ [W/m$^2$]")
    ax.set_ylabel(r"$\partial P_{slack} / \partial G_{poa}$ [kW/(W/m$^2$)]")
    ax.text(
        0.99,
        -0.18,
        G_SWEEP_FIXED_WEATHER_LABEL,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )
    ax.legend(title="Szenario", fontsize=8, title_fontsize=9, frameon=True)
    ax.grid(True, alpha=0.3)
    if all_y:
        ax.set_ylim(*padded_limits(all_y, pad_fraction=0.15))
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
    return list(_plot_export(fig, "fig05_sensitivity_p_slack_vs_g_poa", figures_dir))


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
Lines compare `base`, `load_low`, and `load_high`. The raw artifact stores
sensitivities in MW/degC; the figure converts them to kW/degC by multiplying
`value` by 1000.

## Sign convention

For all slack active-power plots, negative `P_slack` values denote export to
the upstream grid.

## Figure 4

Files: `fig04_g_sweep_p_slack.png` and `fig04_g_sweep_p_slack.pdf`.
Data source: `scenario_grid.csv`. Filter:
`weather_case_type == "sweep_g_1d"` and `observable == "p_slack_mw"`.
Lines compare `base`, `load_low`, and `load_high` over `g_poa_wm2` at fixed
`t_amb_c = 25` and `wind_ms = 2`. The irradiance sweep shows the dominant
direct influence of PV irradiance on PV active power and the slack active-power
balance.

## Figure 5

Files: `fig05_sensitivity_p_slack_vs_g_poa.png` and
`fig05_sensitivity_p_slack_vs_g_poa.pdf`. Data source:
`sensitivity_table.csv`. Filter: `weather_case_type == "sweep_g_1d"`,
`observable == "p_slack_mw"`, and `input_parameter == "g_poa_wm2"`.
Lines compare the local AD sensitivity of slack active power to irradiance
over the same sweep. The raw artifact stores sensitivities in MW/(W/m^2); the
figure converts them to kW/(W/m^2) by multiplying `value` by 1000.

## Figure 6

Files: `fig06_heatmap_g_t_sensitivity_p_slack_wrt_g.png` and
`fig06_heatmap_g_t_sensitivity_p_slack_wrt_g.pdf`. Data source:
`sensitivity_table.csv`. Filter: `weather_case_type == "grid_2d"`,
`network_scenario == "base"`, `observable == "p_slack_mw"`, and
`input_parameter == "g_poa_wm2"`. Values are pivoted to a
`t_amb_c x g_poa_wm2` matrix and displayed in the same discrete 5 x 5 style as
Figure 2. The raw artifact stores sensitivities in MW/(W/m^2); the figure
converts them to kW per 100 W/m^2 by multiplying `value` by 1000 x 100.

This heatmap shows local derivatives, not global finite differences. Negative
irradiance sensitivity means higher irradiance increases PV injection and makes
`P_slack` more negative under the project's slack-power sign convention.

## Figure 7

Files: `fig07_heatmap_g_t_sensitivity_p_slack_wrt_t_amb.png` and
`fig07_heatmap_g_t_sensitivity_p_slack_wrt_t_amb.pdf`. Data source:
`sensitivity_table.csv`. Filter: `weather_case_type == "grid_2d"`,
`network_scenario == "base"`, `observable == "p_slack_mw"`, and
`input_parameter == "t_amb_c"`. Values are pivoted to a
`t_amb_c x g_poa_wm2` matrix and displayed in the same discrete 5 x 5 style as
Figure 2. The raw artifact stores sensitivities in MW/degC; the figure
converts them to kW/degC by multiplying `value` by 1000.

This heatmap also shows local derivatives. Positive temperature sensitivity
means higher ambient temperature reduces PV active power and makes `P_slack`
less negative.

## Limitations

The figures are generated from existing Experiment 3 artifacts. Plotting does
not run new power-flow solves, does not compute new AD sensitivities, does not
run new finite-difference checks, and does not change the PV model, the
power-flow core, or solver logic.
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
    outputs.extend(plot_g_sweep_p_slack(scenario_rows, target_dir))
    outputs.extend(plot_g_sweep_p_slack_sensitivity(sensitivity_rows, target_dir))
    outputs.extend(
        plot_fig06_heatmap_g_t_sensitivity_p_slack_wrt_g(
            sensitivity_rows,
            target_dir,
        )
    )
    outputs.extend(
        plot_fig07_heatmap_g_t_sensitivity_p_slack_wrt_t_amb(
            sensitivity_rows,
            target_dir,
        )
    )
    outputs.extend(
        plot_fig08_heatmap_g_t_sensitivity_p_slack_combined(
            sensitivity_rows,
            target_dir,
        )
    )
    outputs.append(write_figures_readme(target_dir, results_dir))
    return outputs


def main() -> None:
    outputs = generate_figures(RESULTS_DIR, FIGURES_DIR)
    print(f"Figures directory: {FIGURES_DIR}")
    for path in outputs:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
