"""Create publication-style figures from existing Experiment 5 artifacts.

This script reads only already exported CSV/JSON files from Experiment 5a and
Experiment 5b. It does not run power-flow solves, does not compute
sensitivities, does not run an optimization, and does not import JAX or
``pandapower``.

Run:
    python experiments/plot_exp05_figures.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXP05A_RESULTS_DIR = Path(__file__).resolve().parent / "results" / "exp05a_network_screening"
EXP05B_RESULTS_DIR = (
    Path(__file__).resolve().parent / "results" / "exp05b_optimize_pv_curtailment"
)
FIGURES_DIR = Path(__file__).resolve().parent / "results" / "exp05_figures"

SELECTED_CASE_ID = "selected_realistic_load0p4_g1200_t30"
EXPORT_LIMIT_MW = 7.0
EXPORT_TARGET_MW = 6.99
PNG_DPI = 300

FIGURE_STEMS = (
    "fig51_screening_export_overview",
    "fig53_export_before_after_reference",
    "fig54_grid_reference_export_vs_curtailment",
    "fig55_optimization_trace_export_and_curtailment",
)


@dataclass(frozen=True)
class FigureInputs:
    """Resolved artifact locations for the Exp. 5 figure pipeline."""

    exp05a_dir: Path = EXP05A_RESULTS_DIR
    exp05b_dir: Path = EXP05B_RESULTS_DIR

    @property
    def screening_results(self) -> Path:
        return self.exp05a_dir / "screening_results.csv"

    @property
    def selected_realistic_case(self) -> Path:
        return self.exp05a_dir / "selected_realistic_case.csv"

    @property
    def selected_case_baseline(self) -> Path:
        return self.exp05b_dir / "selected_case_baseline.csv"

    @property
    def final_solution(self) -> Path:
        return self.exp05b_dir / "final_solution.csv"

    @property
    def grid_reference(self) -> Path:
        return self.exp05b_dir / "grid_reference.csv"

    @property
    def optimization_trace(self) -> Path:
        return self.exp05b_dir / "optimization_trace.csv"

    @property
    def constraint_diagnostics(self) -> Path:
        return self.exp05b_dir / "constraint_diagnostics.csv"

    def as_dict(self) -> dict[str, str]:
        return {
            "screening_results": str(self.screening_results),
            "selected_realistic_case": str(self.selected_realistic_case),
            "selected_case_baseline": str(self.selected_case_baseline),
            "final_solution": str(self.final_solution),
            "grid_reference": str(self.grid_reference),
            "optimization_trace": str(self.optimization_trace),
            "constraint_diagnostics": str(self.constraint_diagnostics),
        }


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV artifact with a clear error for missing files."""

    if not path.exists():
        raise FileNotFoundError(f"Required CSV artifact not found: {path}")
    return pd.read_csv(path)


def require_columns(df: pd.DataFrame, columns: Iterable[str], artifact_name: str) -> None:
    """Validate required columns and include available columns in the error."""

    missing = sorted(set(columns) - set(df.columns))
    if missing:
        available = ", ".join(map(str, df.columns))
        raise ValueError(
            f"{artifact_name} is missing required columns {missing}. "
            f"Available columns: [{available}]"
        )


def ensure_output_dir(path: Path) -> Path:
    """Create and return the output directory."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(fig, output_dir: Path, stem: str) -> tuple[Path, Path]:
    """Save a Matplotlib figure as PNG and PDF and close it."""

    ensure_output_dir(output_dir)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=PNG_DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def read_first_row(path: Path) -> pd.Series:
    """Read a one-row artifact and return the first row."""

    df = load_csv(path)
    if df.empty:
        raise ValueError(f"{path.name} must contain at least one row.")
    return df.iloc[0]


def get_scalar(df: pd.DataFrame, column: str, artifact_name: str) -> float:
    """Extract a finite scalar from a one-row DataFrame."""

    require_columns(df, [column], artifact_name)
    if len(df) != 1:
        raise ValueError(f"{artifact_name} must contain exactly one row, got {len(df)}.")
    value = float(pd.to_numeric(df[column], errors="raise").iloc[0])
    if not np.isfinite(value):
        raise ValueError(f"{artifact_name}.{column} is not finite: {value}")
    return value


def find_grid_best_feasible(grid_df: pd.DataFrame, limit_mw: float = EXPORT_LIMIT_MW) -> pd.Series:
    """Return the largest curtailment factor with ``p_export_mw <= limit_mw``."""

    require_columns(
        grid_df,
        ["curtailment_factor", "p_export_mw"],
        "grid_reference.csv",
    )
    work = grid_df.copy()
    work["curtailment_factor"] = pd.to_numeric(work["curtailment_factor"], errors="coerce")
    work["p_export_mw"] = pd.to_numeric(work["p_export_mw"], errors="coerce")
    feasible = work[
        np.isfinite(work["curtailment_factor"])
        & np.isfinite(work["p_export_mw"])
        & (work["p_export_mw"] <= limit_mw)
    ]
    if feasible.empty:
        raise ValueError(
            f"grid_reference.csv contains no feasible row with p_export_mw <= {limit_mw}."
        )
    idx = feasible["curtailment_factor"].idxmax()
    return work.loc[idx]


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (7.2, 4.4),
            "figure.dpi": 120,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "font.size": 10,
            "savefig.facecolor": "white",
        }
    )


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce")


def _artifact_limit_and_target(final_df: pd.DataFrame) -> tuple[float, float]:
    limit = (
        get_scalar(final_df, "p_export_limit_mw", "final_solution.csv")
        if "p_export_limit_mw" in final_df.columns
        else EXPORT_LIMIT_MW
    )
    target = (
        get_scalar(final_df, "p_export_target_mw", "final_solution.csv")
        if "p_export_target_mw" in final_df.columns
        else EXPORT_TARGET_MW
    )
    return limit, target


def _screening_cases(screening_df: pd.DataFrame) -> pd.DataFrame:
    if "case_type" in screening_df.columns:
        return screening_df[screening_df["case_type"].astype(str) == "screening"].copy()
    if "case_id" in screening_df.columns:
        return screening_df[
            ~screening_df["case_id"].astype(str).str.contains("no_pv|ref_", case=False, regex=True)
        ].copy()
    return screening_df.copy()


def plot_fig51(
    screening_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    output_dir: Path,
    limit_mw: float = EXPORT_LIMIT_MW,
) -> tuple[Path, Path]:
    """Plot screening export over sorted operating points."""

    require_columns(screening_df, ["p_export_mw"], "screening_results.csv")
    require_columns(selected_df, ["p_export_mw"], "selected_realistic_case.csv")
    screening = _screening_cases(screening_df)
    if screening.empty:
        raise ValueError("screening_results.csv contains no screening cases to plot.")

    screening = screening.copy()
    selected = selected_df.copy().head(1)
    screening["plot_kind"] = "screening"
    selected["plot_kind"] = "selected"
    combined = pd.concat([screening, selected], ignore_index=True, sort=False)
    combined["p_export_mw"] = _numeric(combined, "p_export_mw")
    combined = combined[np.isfinite(combined["p_export_mw"])].copy()
    combined = combined.sort_values("p_export_mw", ascending=False).reset_index(drop=True)
    combined["plot_index"] = np.arange(1, len(combined) + 1)

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    g_values = _numeric(combined, "g_poa_wm2") if "g_poa_wm2" in combined.columns else None

    if g_values is not None and np.isfinite(g_values).any():
        screening_mask = combined["plot_kind"] == "screening"
        scatter = ax.scatter(
            combined.loc[screening_mask, "plot_index"],
            combined.loc[screening_mask, "p_export_mw"],
            c=g_values.loc[screening_mask],
            cmap="viridis",
            s=38,
            edgecolor="white",
            linewidth=0.5,
            label="Screening-Faelle",
        )
        cbar = fig.colorbar(scatter, ax=ax, pad=0.015)
        cbar.set_label(r"Einstrahlung $G_{poa}$ [W/m$^2$]")
    else:
        screening_mask = combined["plot_kind"] == "screening"
        ax.scatter(
            combined.loc[screening_mask, "plot_index"],
            combined.loc[screening_mask, "p_export_mw"],
            s=38,
            color="#4c78a8",
            edgecolor="white",
            linewidth=0.5,
            label="Screening-Faelle",
        )

    selected_mask = combined["plot_kind"] == "selected"
    selected_rows = combined[selected_mask]
    if selected_rows.empty:
        raise ValueError("selected_realistic_case.csv did not produce a selected plot point.")
    sel = selected_rows.iloc[0]
    ax.scatter(
        [sel["plot_index"]],
        [sel["p_export_mw"]],
        marker="*",
        s=190,
        color="#d95f02",
        edgecolor="black",
        linewidth=0.8,
        zorder=4,
        label="ausgewaehlter 30 degC-Fall",
    )
    ax.annotate(
        "selected 30 degC case",
        xy=(sel["plot_index"], sel["p_export_mw"]),
        xytext=(8, 12),
        textcoords="offset points",
        fontsize=8.5,
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "0.25"},
    )

    ax.axhline(
        limit_mw,
        color="#b2182b",
        linestyle="--",
        linewidth=1.4,
        label="7.0-MW-Grenze",
    )
    ax.set_title("Abb. 5.1: Screening-Export ueber Betriebspunkte")
    ax.set_xlabel("Betriebspunkte, nach Export sortiert")
    ax.set_ylabel(r"Exportleistung $p_{export}$ [MW]")
    ax.set_xlim(0.2, len(combined) + 0.8)
    ax.legend(loc="best", frameon=True)
    return save_figure(fig, output_dir, "fig51_screening_export_overview")


def _baseline_row(baseline_df: pd.DataFrame, baseline_type: str) -> pd.Series:
    require_columns(
        baseline_df,
        ["baseline_type", "curtailment_factor", "p_export_mw"],
        "selected_case_baseline.csv",
    )
    match = baseline_df[baseline_df["baseline_type"].astype(str) == baseline_type]
    if match.empty:
        raise ValueError(f"selected_case_baseline.csv has no baseline_type={baseline_type!r}.")
    return match.iloc[0]


def _bar_value(row: pd.Series, value_col: str = "p_export_mw") -> float:
    value = float(row[value_col])
    if not np.isfinite(value):
        raise ValueError(f"Expected finite {value_col}, got {value}.")
    return value


def plot_fig53(
    baseline_df: pd.DataFrame,
    final_df: pd.DataFrame,
    grid_df: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Plot full-PV, optimized, grid-reference, and zero-PV export bars."""

    require_columns(
        final_df,
        ["final_curtailment_factor", "final_p_export_mw", "grid_best_curtailment_factor"],
        "final_solution.csv",
    )
    limit_mw, target_mw = _artifact_limit_and_target(final_df)
    full = _baseline_row(baseline_df, "full_pv")
    zero = _baseline_row(baseline_df, "zero_pv")
    final = final_df.iloc[0]
    grid_best = find_grid_best_feasible(grid_df, limit_mw=limit_mw)

    labels = ["Full PV", "Optimiert", "Grid-Referenz", "Zero PV"]
    exports = [
        _bar_value(full),
        float(final["final_p_export_mw"]),
        _bar_value(grid_best),
        _bar_value(zero),
    ]
    c_values = [
        float(full["curtailment_factor"]),
        float(final["final_curtailment_factor"]),
        float(grid_best["curtailment_factor"]),
        float(zero["curtailment_factor"]),
    ]
    colors = ["#d95f02", "#1b9e77", "#7570b3", "#4c78a8"]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = np.arange(len(labels))
    bars = ax.bar(x, exports, color=colors, width=0.62, edgecolor="0.2", linewidth=0.7)
    ax.axhline(limit_mw, color="#b2182b", linestyle="--", linewidth=1.4, label="7.0-MW-Grenze")
    ax.axhline(
        target_mw,
        color="#b2182b",
        linestyle=":",
        linewidth=1.1,
        label="Optimierungsziel 6.99 MW",
    )
    for bar, export, c_value in zip(bars, exports, c_values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            export + 0.035,
            f"{export:.3f}\nc = {c_value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    ax.set_title("Abb. 5.3: Export vor und nach Curtailment-Optimierung")
    ax.set_ylabel(r"Exportleistung $p_{export}$ [MW]")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    y_min = max(0.0, min(exports) - 0.5)
    ax.set_ylim(y_min, max(max(exports), limit_mw) + 0.65)
    ax.legend(loc="upper right", frameon=True)
    return save_figure(fig, output_dir, "fig53_export_before_after_reference")


def plot_fig54(
    grid_df: pd.DataFrame,
    final_df: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Plot the grid reference curve with optimizer and grid-best points."""

    require_columns(grid_df, ["curtailment_factor", "p_export_mw"], "grid_reference.csv")
    require_columns(
        final_df,
        ["final_curtailment_factor", "final_p_export_mw"],
        "final_solution.csv",
    )
    limit_mw, target_mw = _artifact_limit_and_target(final_df)
    grid = grid_df.copy()
    grid["curtailment_factor"] = _numeric(grid, "curtailment_factor")
    grid["p_export_mw"] = _numeric(grid, "p_export_mw")
    grid = grid[np.isfinite(grid["curtailment_factor"]) & np.isfinite(grid["p_export_mw"])]
    grid = grid.sort_values("curtailment_factor")
    if grid.empty:
        raise ValueError("grid_reference.csv contains no finite grid curve rows.")

    final = final_df.iloc[0]
    opt_c = float(final["final_curtailment_factor"])
    opt_export = float(final["final_p_export_mw"])
    grid_best = find_grid_best_feasible(grid, limit_mw=limit_mw)
    grid_c = float(grid_best["curtailment_factor"])
    grid_export = float(grid_best["p_export_mw"])

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(
        grid["curtailment_factor"],
        grid["p_export_mw"],
        color="#4c78a8",
        linewidth=1.8,
        label="Grid-Referenzkurve",
    )
    ax.axhline(limit_mw, color="#b2182b", linestyle="--", linewidth=1.4, label="7.0-MW-Grenze")
    ax.axhline(
        target_mw,
        color="#b2182b",
        linestyle=":",
        linewidth=1.1,
        label="Optimierungsziel 6.99 MW",
    )
    ax.scatter([opt_c], [opt_export], color="#1b9e77", s=70, zorder=4, label="Optimizer")
    ax.scatter([grid_c], [grid_export], color="#7570b3", s=70, zorder=4, label="Grid-Referenz")
    ax.annotate(
        f"Optimizer: c = {opt_c:.3f}",
        xy=(opt_c, opt_export),
        xytext=(-90, -35),
        textcoords="offset points",
        fontsize=8.5,
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "0.25"},
    )
    ax.annotate(
        f"Grid: c = {grid_c:.3f}",
        xy=(grid_c, grid_export),
        xytext=(10, 15),
        textcoords="offset points",
        fontsize=8.5,
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "0.25"},
    )
    ax.set_title("Abb. 5.4: Grid-Referenz fuer Export ueber PV-Nutzungsfaktor")
    ax.set_xlabel("PV-Nutzungsfaktor c [-]")
    ax.set_ylabel(r"Exportleistung $p_{export}$ [MW]")
    ax.set_xlim(0.0, 1.0)
    ax.legend(loc="best", frameon=True)
    return save_figure(fig, output_dir, "fig54_grid_reference_export_vs_curtailment")


def plot_fig55(
    trace_df: pd.DataFrame,
    output_dir: Path,
    limit_mw: float = EXPORT_LIMIT_MW,
    target_mw: float = EXPORT_TARGET_MW,
) -> tuple[Path, Path]:
    """Plot optimization trace for export and curtailment factor."""

    require_columns(
        trace_df,
        ["iteration", "p_export_mw", "curtailment_factor"],
        "optimization_trace.csv",
    )
    trace = trace_df.copy()
    trace["iteration"] = _numeric(trace, "iteration")
    trace["p_export_mw"] = _numeric(trace, "p_export_mw")
    trace["curtailment_factor"] = _numeric(trace, "curtailment_factor")
    trace = trace[
        np.isfinite(trace["iteration"])
        & np.isfinite(trace["p_export_mw"])
        & np.isfinite(trace["curtailment_factor"])
    ].sort_values("iteration")
    if trace.empty:
        raise ValueError("optimization_trace.csv contains no finite trace rows.")
    final = trace.iloc[-1]

    fig, (ax_export, ax_c) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(7.2, 5.5),
        sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1.0]},
    )
    ax_export.plot(
        trace["iteration"],
        trace["p_export_mw"],
        color="#4c78a8",
        linewidth=1.8,
        label=r"$p_{export}$",
    )
    ax_export.axhline(
        limit_mw,
        color="#b2182b",
        linestyle="--",
        linewidth=1.3,
        label="7.0-MW-Grenze",
    )
    ax_export.axhline(
        target_mw,
        color="#b2182b",
        linestyle=":",
        linewidth=1.0,
        label="Optimierungsziel 6.99 MW",
    )
    ax_export.scatter(
        [final["iteration"]],
        [final["p_export_mw"]],
        color="#1b9e77",
        s=52,
        zorder=4,
    )
    ax_export.annotate(
        f"{final['p_export_mw']:.3f} MW",
        xy=(final["iteration"], final["p_export_mw"]),
        xytext=(-65, 12),
        textcoords="offset points",
        fontsize=8.5,
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "0.25"},
    )
    ax_export.set_title("Abb. 5.5: Verlauf der Curtailment-Optimierung")
    ax_export.set_ylabel(r"Exportleistung $p_{export}$ [MW]")
    ax_export.legend(loc="best", frameon=True)

    ax_c.plot(
        trace["iteration"],
        trace["curtailment_factor"],
        color="#1b9e77",
        linewidth=1.8,
        label="PV-Nutzungsfaktor c",
    )
    ax_c.scatter(
        [final["iteration"]],
        [final["curtailment_factor"]],
        color="#1b9e77",
        edgecolor="black",
        linewidth=0.5,
        s=52,
        zorder=4,
    )
    ax_c.annotate(
        f"c = {final['curtailment_factor']:.3f}",
        xy=(final["iteration"], final["curtailment_factor"]),
        xytext=(-70, -25),
        textcoords="offset points",
        fontsize=8.5,
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "0.25"},
    )
    ax_c.set_xlabel("Iteration")
    ax_c.set_ylabel("PV-Nutzungsfaktor c [-]")
    ax_c.set_ylim(0.0, 1.02)
    ax_c.legend(loc="best", frameon=True)
    fig.tight_layout()
    return save_figure(fig, output_dir, "fig55_optimization_trace_export_and_curtailment")


def write_readme(output_dir: Path) -> Path:
    """Write the figure directory README."""

    text = """# Experiment 5 Figures

This directory contains report-ready figures for Experiment 5. The figures are
created exclusively from existing CSV/JSON artifacts of Experiment 5a and
Experiment 5b. The plotting script does not run new power-flow solves, does not
compute sensitivities, does not run a grid search, and does not start a new
curtailment optimization.

## Figures

- `fig51_screening_export_overview.png/pdf`: Screening export over all
  screening operating points from `screening_results.csv`, with the selected
  realistic 30 degC case from `selected_realistic_case.csv` highlighted. The
  figure shows that high-export cases occur mainly under high-PV and low-load
  conditions; the selected 30 degC case remains export-critical without using
  the colder -10 degC stress point as the main narrative.
- `fig53_export_before_after_reference.png/pdf`: Export bars for full PV,
  optimized curtailment, the independent grid reference, and zero PV. Full PV
  exceeds the 7.0 MW demonstrator target; the optimized solution reduces export
  to about 6.99 MW and remains close to the largest feasible grid point.
- `fig54_grid_reference_export_vs_curtailment.png/pdf`: One-dimensional grid
  reference curve `p_export(c)` with optimizer and grid-reference points. The
  optimizer is slightly conservative because it tracks the 6.99 MW target below
  the 7.0 MW limit.
- `fig55_optimization_trace_export_and_curtailment.png/pdf`: Optimization
  trace showing `p_export_mw` and `curtailment_factor` over iterations. The
  optimizer moves export into the target range and stabilizes near `c = 0.714`.

## Data Sources

- `experiments/results/exp05a_network_screening/screening_results.csv`
- `experiments/results/exp05a_network_screening/selected_realistic_case.csv`
- `experiments/results/exp05b_optimize_pv_curtailment/selected_case_baseline.csv`
- `experiments/results/exp05b_optimize_pv_curtailment/final_solution.csv`
- `experiments/results/exp05b_optimize_pv_curtailment/grid_reference.csv`
- `experiments/results/exp05b_optimize_pv_curtailment/optimization_trace.csv`
- `experiments/results/exp05b_optimize_pv_curtailment/constraint_diagnostics.csv`

## Scope Notes

The 7.0 MW line is a demonstrator-internal target, not a normative grid-code
limit. PV remains a PQ injection. There is no PV-bus voltage regulation, no
Q-limit handling, no PV-PQ switching, no controller logic, and no normative
thermal equipment-limit assessment.
"""
    ensure_output_dir(output_dir)
    path = output_dir / "README.md"
    path.write_text(text, encoding="utf-8")
    return path


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def write_metadata(
    output_dir: Path,
    inputs: FigureInputs,
    figure_paths: list[Path],
    readme_path: Path,
) -> Path:
    """Write lightweight reproducibility metadata for the figures."""

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_hash": _git_hash(),
        "script": "experiments/plot_exp05_figures.py",
        "selected_case_id": SELECTED_CASE_ID,
        "export_limit_mw": EXPORT_LIMIT_MW,
        "export_target_mw": EXPORT_TARGET_MW,
        "input_artifacts": inputs.as_dict(),
        "output_files": [str(path) for path in figure_paths] + [str(readme_path)],
        "notes": [
            "Figures are generated only from existing Exp. 5 artifacts.",
            "No power-flow solves, sensitivities, grid search, or optimization are run.",
        ],
    }
    path = output_dir / "figure_metadata.json"
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def generate_figures(
    exp05a_dir: Path = EXP05A_RESULTS_DIR,
    exp05b_dir: Path = EXP05B_RESULTS_DIR,
    output_dir: Path = FIGURES_DIR,
) -> list[Path]:
    """Generate all Experiment 5 figures from existing artifacts."""

    _configure_matplotlib()
    inputs = FigureInputs(exp05a_dir=exp05a_dir, exp05b_dir=exp05b_dir)
    output_dir = ensure_output_dir(output_dir)

    screening_df = load_csv(inputs.screening_results)
    selected_df = load_csv(inputs.selected_realistic_case)
    baseline_df = load_csv(inputs.selected_case_baseline)
    final_df = load_csv(inputs.final_solution)
    grid_df = load_csv(inputs.grid_reference)
    trace_df = load_csv(inputs.optimization_trace)

    limit_mw, target_mw = _artifact_limit_and_target(final_df)
    figure_paths: list[Path] = []
    for png_path, pdf_path in [
        plot_fig51(screening_df, selected_df, output_dir, limit_mw),
        plot_fig53(baseline_df, final_df, grid_df, output_dir),
        plot_fig54(grid_df, final_df, output_dir),
        plot_fig55(trace_df, output_dir, limit_mw=limit_mw, target_mw=target_mw),
    ]:
        figure_paths.extend([png_path, pdf_path])

    readme_path = write_readme(output_dir)
    metadata_path = write_metadata(output_dir, inputs, figure_paths, readme_path)
    return figure_paths + [readme_path, metadata_path]


def main() -> None:
    print("=" * 72)
    print("Experiment 5: figure generation from existing artifacts")
    print(f"Output directory: {FIGURES_DIR}")
    print("=" * 72)
    paths = generate_figures()
    print("\nGenerated files:")
    for path in paths:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
