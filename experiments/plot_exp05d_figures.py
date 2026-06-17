"""Create figures for Experiment 5d from existing CSV/JSON artifacts.

This script reads already exported artifacts from Experiment 5a and Experiment
5d. It does not run power-flow solves, does not run an optimization, and does
not perform a grid search.

Run:
    python experiments/plot_exp05d_figures.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import plot_exp05_figures as exp05_plot


EXP05A_RESULTS_DIR = exp05_plot.EXP05A_RESULTS_DIR
EXP05D_RESULTS_DIR = (
    Path(__file__).resolve().parent
    / "results"
    / "exp05d_optimize_pv_curtailment_simple_objective"
)
EXP05C_RESULTS_DIR = (
    Path(__file__).resolve().parent / "results" / "exp05c_optimize_pv_curtailment_nn"
)
FIGURES_DIR = Path(__file__).resolve().parent / "results" / "exp05d_figures"

FIGURE_STEMS = exp05_plot.FIGURE_STEMS
SELECTED_CASE_ID = exp05_plot.SELECTED_CASE_ID
EXPORT_LIMIT_MW = 7.0
EXPORT_TARGET_MW = 7.0
EXPORT_TARGET_LABEL = "Exportziel 7.0 MW"
MODEL_LABELS = {
    "5d": "Analytisches Modell",
    "5c": "NN-Surrogat",
}


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _grid_best_objective(grid_df: pd.DataFrame) -> pd.Series:
    exp05_plot.require_columns(
        grid_df,
        ["curtailment_factor", "p_export_mw", "objective"],
        "grid_reference.csv",
    )
    work = grid_df.copy()
    work["objective"] = pd.to_numeric(work["objective"], errors="coerce")
    valid = work[np.isfinite(work["objective"])]
    if valid.empty:
        raise ValueError("grid_reference.csv contains no finite objective values.")
    return work.loc[valid["objective"].idxmin()]


def _grid_best_feasible(grid_df: pd.DataFrame, limit_mw: float) -> pd.Series | None:
    try:
        return exp05_plot.find_grid_best_feasible(grid_df, limit_mw=limit_mw)
    except ValueError:
        return None


def plot_fig53(
    baseline_df: pd.DataFrame,
    final_df: pd.DataFrame,
    grid_df: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Plot full-PV, optimized-simple, grid objective, and zero-PV export bars."""

    exp05_plot.require_columns(
        final_df,
        ["final_curtailment_factor", "final_p_export_mw"],
        "final_solution.csv",
    )
    limit_mw, _target_mw = exp05_plot._artifact_limit_and_target(final_df)
    full = exp05_plot._baseline_row(baseline_df, "full_pv")
    zero = exp05_plot._baseline_row(baseline_df, "zero_pv")
    final = final_df.iloc[0]
    grid_objective = _grid_best_objective(grid_df)
    grid_feasible = _grid_best_feasible(grid_df, limit_mw)

    labels = ["Full PV", "Optimiert\nanalytisch", "Grid-Referenz\nanalytisch", "Zero PV"]
    exports = [
        exp05_plot._bar_value(full),
        float(final["final_p_export_mw"]),
        exp05_plot._bar_value(grid_objective),
        exp05_plot._bar_value(zero),
    ]
    c_values = [
        float(full["curtailment_factor"]),
        float(final["final_curtailment_factor"]),
        float(grid_objective["curtailment_factor"]),
        float(zero["curtailment_factor"]),
    ]
    colors = ["#d95f02", "#1b9e77", "#7570b3", "#4c78a8"]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = np.arange(len(labels))
    bars = ax.bar(x, exports, color=colors, width=0.62, edgecolor="0.2", linewidth=0.7)
    ax.axhline(limit_mw, color="#b2182b", linestyle="--", linewidth=1.4, label=EXPORT_TARGET_LABEL)
    if grid_feasible is not None:
        ax.scatter(
            [2],
            [float(grid_feasible["p_export_mw"])],
            marker="D",
            color="#f1a340",
            edgecolor="0.2",
            s=42,
            zorder=4,
            label="Grid machbar",
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
    ax.set_ylabel(r"Exportleistung $p_{export}$ [MW]")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    y_min = max(0.0, min(exports) - 0.5)
    ax.set_ylim(y_min, max(max(exports), limit_mw) + 0.65)
    ax.legend(loc="upper right", frameon=True)
    return exp05_plot.save_figure(fig, output_dir, "fig53_export_before_after_reference")


def plot_fig54(
    grid_df: pd.DataFrame,
    final_df: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Plot Exp. 5d grid curve with optimizer and grid-best points."""

    exp05_plot.require_columns(grid_df, ["curtailment_factor", "p_export_mw"], "grid_reference.csv")
    exp05_plot.require_columns(
        final_df,
        ["final_curtailment_factor", "final_p_export_mw"],
        "final_solution.csv",
    )
    limit_mw, _target_mw = exp05_plot._artifact_limit_and_target(final_df)
    grid = grid_df.copy()
    grid["curtailment_factor"] = exp05_plot._numeric(grid, "curtailment_factor")
    grid["p_export_mw"] = exp05_plot._numeric(grid, "p_export_mw")
    grid = grid[np.isfinite(grid["curtailment_factor"]) & np.isfinite(grid["p_export_mw"])]
    grid = grid.sort_values("curtailment_factor")
    if grid.empty:
        raise ValueError("grid_reference.csv contains no finite grid curve rows.")

    final = final_df.iloc[0]
    opt_c = float(final["final_curtailment_factor"])
    opt_export = float(final["final_p_export_mw"])
    grid_objective = _grid_best_objective(grid)
    grid_obj_c = float(grid_objective["curtailment_factor"])
    grid_obj_export = float(grid_objective["p_export_mw"])
    grid_feasible = _grid_best_feasible(grid, limit_mw)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(
        grid["curtailment_factor"],
        grid["p_export_mw"],
        color="#4c78a8",
        linewidth=1.8,
        label="Grid-Referenzkurve",
    )
    ax.axhline(limit_mw, color="#b2182b", linestyle="--", linewidth=1.4, label=EXPORT_TARGET_LABEL)
    ax.scatter([opt_c], [opt_export], color="#1b9e77", s=70, zorder=4, label="Optimierer")
    ax.scatter(
        [grid_obj_c],
        [grid_obj_export],
        color="#7570b3",
        s=70,
        zorder=4,
        label="Grid-Ziel",
    )
    if grid_feasible is not None:
        ax.scatter(
            [float(grid_feasible["curtailment_factor"])],
            [float(grid_feasible["p_export_mw"])],
            marker="D",
            color="#f1a340",
            edgecolor="0.2",
            s=58,
            zorder=4,
            label="Grid machbar",
        )
    ax.annotate(
        f"Optimierer: c = {opt_c:.3f}",
        xy=(opt_c, opt_export),
        xytext=(-90, -35),
        textcoords="offset points",
        fontsize=8.5,
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "0.25"},
    )
    ax.annotate(
        f"Grid-Ziel: c = {grid_obj_c:.3f}",
        xy=(grid_obj_c, grid_obj_export),
        xytext=(10, 15),
        textcoords="offset points",
        fontsize=8.5,
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "0.25"},
    )
    ax.set_xlabel("PV-Nutzungsfaktor c [-]")
    ax.set_ylabel(r"Exportleistung $p_{export}$ [MW]")
    ax.set_xlim(0.0, 1.0)
    ax.legend(loc="best", frameon=True)
    return exp05_plot.save_figure(fig, output_dir, "fig54_grid_reference_export_vs_curtailment")


def plot_fig55(
    trace_df: pd.DataFrame,
    output_dir: Path,
    limit_mw: float = EXPORT_LIMIT_MW,
) -> tuple[Path, Path]:
    """Plot Exp. 5d optimization trace for export and curtailment factor."""

    exp05_plot.require_columns(
        trace_df,
        ["iteration", "p_export_mw", "curtailment_factor"],
        "optimization_trace.csv",
    )
    trace = trace_df.copy()
    trace["iteration"] = exp05_plot._numeric(trace, "iteration")
    trace["p_export_mw"] = exp05_plot._numeric(trace, "p_export_mw")
    trace["curtailment_factor"] = exp05_plot._numeric(trace, "curtailment_factor")
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
        label=EXPORT_TARGET_LABEL,
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
    return exp05_plot.save_figure(fig, output_dir, "fig55_optimization_trace_export_and_curtailment")


def _variant_inputs_available(results_dir: Path) -> bool:
    required = (
        "selected_case_baseline.csv",
        "final_solution.csv",
        "grid_reference.csv",
        "optimization_trace.csv",
    )
    return all((results_dir / name).exists() for name in required)


def _load_variant_artifacts(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        exp05_plot.load_csv(results_dir / "selected_case_baseline.csv"),
        exp05_plot.load_csv(results_dir / "final_solution.csv"),
        exp05_plot.load_csv(results_dir / "grid_reference.csv"),
        exp05_plot.load_csv(results_dir / "optimization_trace.csv"),
    )


def _fig53_values(
    baseline_df: pd.DataFrame,
    final_df: pd.DataFrame,
    grid_df: pd.DataFrame,
    variant: str,
) -> tuple[list[str], list[float], list[float], pd.Series | None, float, float]:
    full = exp05_plot._baseline_row(baseline_df, "full_pv")
    zero = exp05_plot._baseline_row(baseline_df, "zero_pv")
    final = final_df.iloc[0]
    limit_mw, target_mw = exp05_plot._artifact_limit_and_target(final_df)

    if variant == "5d":
        grid_bar = _grid_best_objective(grid_df)
        grid_marker = _grid_best_feasible(grid_df, limit_mw)
        labels = ["Full PV", "Optimiert", "Grid-Referenz", "Zero PV"]
    else:
        grid_bar = exp05_plot.find_grid_best_feasible(grid_df, limit_mw=limit_mw)
        grid_marker = None
        labels = ["Full PV", "Optimiert", "Grid-Referenz", "Zero PV"]

    exports = [
        exp05_plot._bar_value(full),
        float(final["final_p_export_mw"]),
        exp05_plot._bar_value(grid_bar),
        exp05_plot._bar_value(zero),
    ]
    c_values = [
        float(full["curtailment_factor"]),
        float(final["final_curtailment_factor"]),
        float(grid_bar["curtailment_factor"]),
        float(zero["curtailment_factor"]),
    ]
    return labels, exports, c_values, grid_marker, limit_mw, target_mw


def _plot_fig53_panel(
    ax: plt.Axes,
    baseline_df: pd.DataFrame,
    final_df: pd.DataFrame,
    grid_df: pd.DataFrame,
    variant: str,
    y_limit: tuple[float, float],
    show_grid_marker: bool = True,
) -> None:
    labels, exports, c_values, grid_marker, limit_mw, target_mw = _fig53_values(
        baseline_df,
        final_df,
        grid_df,
        variant,
    )
    colors = ["#d95f02", "#1b9e77", "#7570b3", "#4c78a8"]
    x = np.arange(len(labels))
    bars = ax.bar(x, exports, color=colors, width=0.62, edgecolor="0.2", linewidth=0.7)
    ax.axhline(limit_mw, color="#b2182b", linestyle="--", linewidth=1.3, label=EXPORT_TARGET_LABEL)
    if abs(target_mw - limit_mw) > 1e-9:
        ax.axhline(
            target_mw,
            color="#b2182b",
            linestyle=":",
            linewidth=1.0,
            label=f"Exportziel {target_mw:.2f} MW",
        )
    if show_grid_marker and grid_marker is not None:
        ax.scatter(
            [2],
            [float(grid_marker["p_export_mw"])],
            marker="D",
            color="#f1a340",
            edgecolor="0.2",
            s=42,
            zorder=4,
            label="Grid machbar",
        )
    for bar, export, c_value in zip(bars, exports, c_values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            export + 0.035,
            f"{export:.3f}\nc = {c_value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.2,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_xlabel(MODEL_LABELS[variant])
    ax.set_ylim(*y_limit)
    ax.grid(True, axis="y", alpha=0.25)


def plot_fig53_exp05d_vs_exp05c_combined(
    exp05d_dir: Path = EXP05D_RESULTS_DIR,
    exp05c_dir: Path = EXP05C_RESULTS_DIR,
    output_dir: Path = FIGURES_DIR,
) -> tuple[Path, Path]:
    """Combine Fig. 53 from Exp. 5d (left) and Exp. 5c (right)."""

    d_baseline, d_final, d_grid, _ = _load_variant_artifacts(exp05d_dir)
    c_baseline, c_final, c_grid, _ = _load_variant_artifacts(exp05c_dir)
    d_values = _fig53_values(d_baseline, d_final, d_grid, "5d")
    c_values = _fig53_values(c_baseline, c_final, c_grid, "5c")
    all_exports = d_values[1] + c_values[1]
    y_min = max(0.0, min(all_exports) - 0.5)
    y_max = max(max(all_exports), d_values[4], c_values[4]) + 0.8

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), sharey=True)
    _plot_fig53_panel(
        axes[0],
        d_baseline,
        d_final,
        d_grid,
        "5d",
        (y_min, y_max),
        show_grid_marker=False,
    )
    _plot_fig53_panel(
        axes[1],
        c_baseline,
        c_final,
        c_grid,
        "5c",
        (y_min, y_max),
        show_grid_marker=False,
    )
    axes[0].set_ylabel(r"Exportleistung $p_{export}$ [MW]")

    handles: list = []
    labels: list[str] = []
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=True)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    return exp05_plot.save_figure(fig, output_dir, "fig53_exp05d_left_exp05c_right_comparison")


def _prepare_trace(trace_df: pd.DataFrame) -> pd.DataFrame:
    exp05_plot.require_columns(
        trace_df,
        ["iteration", "p_export_mw", "curtailment_factor"],
        "optimization_trace.csv",
    )
    trace = trace_df.copy()
    trace["iteration"] = exp05_plot._numeric(trace, "iteration")
    trace["p_export_mw"] = exp05_plot._numeric(trace, "p_export_mw")
    trace["curtailment_factor"] = exp05_plot._numeric(trace, "curtailment_factor")
    trace = trace[
        np.isfinite(trace["iteration"])
        & np.isfinite(trace["p_export_mw"])
        & np.isfinite(trace["curtailment_factor"])
    ].sort_values("iteration")
    if trace.empty:
        raise ValueError("optimization_trace.csv contains no finite trace rows.")
    return trace


def _add_axis_break_marks(ax_top: plt.Axes, ax_bottom: plt.Axes) -> None:
    mark_kwargs = {
        "marker": [(-1, -0.7), (1, 0.7)],
        "markersize": 8,
        "linestyle": "none",
        "color": "0.2",
        "mec": "0.2",
        "mew": 1.0,
        "clip_on": False,
    }
    ax_top.plot([0], [0], transform=ax_top.transAxes, **mark_kwargs)
    ax_bottom.plot([0], [1], transform=ax_bottom.transAxes, **mark_kwargs)


def _plot_fig55_column(
    ax_export: plt.Axes,
    ax_c_upper: plt.Axes,
    ax_c_lower: plt.Axes,
    trace_df: pd.DataFrame,
    final_df: pd.DataFrame,
    variant: str,
) -> None:
    trace = _prepare_trace(trace_df)
    final = trace.iloc[-1]
    limit_mw, target_mw = exp05_plot._artifact_limit_and_target(final_df)
    ax_export.plot(
        trace["iteration"],
        trace["p_export_mw"],
        color="#4c78a8",
        linewidth=1.8,
        label=r"$p_{export}$",
    )
    ax_export.axhline(limit_mw, color="#b2182b", linestyle="--", linewidth=1.3, label=EXPORT_TARGET_LABEL)
    if abs(target_mw - limit_mw) > 1e-9:
        ax_export.axhline(
            target_mw,
            color="#b2182b",
            linestyle=":",
            linewidth=1.0,
            label=f"Exportziel {target_mw:.2f} MW",
        )
    ax_export.scatter([final["iteration"]], [final["p_export_mw"]], color="#1b9e77", s=48, zorder=4)
    ax_export.annotate(
        f"{final['p_export_mw']:.3f} MW",
        xy=(final["iteration"], final["p_export_mw"]),
        xytext=(-62, 12),
        textcoords="offset points",
        fontsize=8.2,
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "0.25"},
    )
    ax_export.grid(True, alpha=0.25)

    for ax_c, label in (
        (ax_c_upper, "PV-Nutzungsfaktor c"),
        (ax_c_lower, None),
    ):
        ax_c.plot(
            trace["iteration"],
            trace["curtailment_factor"],
            color="#1b9e77",
            linewidth=1.8,
            label=label,
        )
        ax_c.scatter(
            [final["iteration"]],
            [final["curtailment_factor"]],
            color="#1b9e77",
            edgecolor="black",
            linewidth=0.5,
            s=48,
            zorder=4,
        )
        ax_c.grid(True, alpha=0.25)

    ax_c_upper.annotate(
        f"c = {final['curtailment_factor']:.3f}",
        xy=(final["iteration"], final["curtailment_factor"]),
        xytext=(-66, -25),
        textcoords="offset points",
        fontsize=8.2,
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "0.25"},
    )
    ax_c_upper.set_ylim(0.6, 1.02)
    ax_c_lower.set_ylim(-0.01, 0.04)
    ax_c_upper.set_yticks(np.arange(0.6, 1.01, 0.05))
    ax_c_lower.set_yticks([0.0])
    ax_c_upper.tick_params(labelbottom=False, bottom=False)
    ax_c_lower.tick_params(top=False)
    ax_c_upper.spines.bottom.set_visible(False)
    ax_c_lower.spines.top.set_visible(False)
    ax_c_lower.grid(False)
    ax_c_lower.set_xlabel(f"Iteration ({MODEL_LABELS[variant]})")
    _add_axis_break_marks(ax_c_upper, ax_c_lower)


def plot_fig55_exp05d_vs_exp05c_combined(
    exp05d_dir: Path = EXP05D_RESULTS_DIR,
    exp05c_dir: Path = EXP05C_RESULTS_DIR,
    output_dir: Path = FIGURES_DIR,
) -> tuple[Path, Path]:
    """Combine Fig. 55 from Exp. 5d (left) and Exp. 5c (right)."""

    _, d_final, _, d_trace = _load_variant_artifacts(exp05d_dir)
    _, c_final, _, c_trace = _load_variant_artifacts(exp05c_dir)
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(12.4, 6.8),
        sharex="col",
        gridspec_kw={"height_ratios": [1.15, 0.9, 0.16], "hspace": 0.06},
    )
    _plot_fig55_column(axes[0, 0], axes[1, 0], axes[2, 0], d_trace, d_final, "5d")
    _plot_fig55_column(axes[0, 1], axes[1, 1], axes[2, 1], c_trace, c_final, "5c")
    axes[0, 0].set_ylabel(r"Exportleistung $p_{export}$ [MW]")
    axes[1, 0].set_ylabel("PV-Nutzungsfaktor c [-]")

    handles: list = []
    labels: list[str] = []
    for ax in axes.ravel():
        for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=3, frameon=True)
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.11,
        top=0.87,
        wspace=0.20,
        hspace=0.06,
    )
    return exp05_plot.save_figure(fig, output_dir, "fig55_exp05d_left_exp05c_right_comparison")


def write_readme(output_dir: Path) -> Path:
    """Write the Exp. 5d figure directory README."""

    text = """# Experiment 5d Figures

This directory contains figures for Experiment 5d. The figures are created
only from existing CSV/JSON artifacts of Experiment 5a and Experiment 5d. The
plotting script does not run power-flow solves, does not run a grid search, and
does not start a curtailment optimization.

## Figures

- `fig51_screening_export_overview.png/pdf`: Existing Exp.-5a screening export
  context with the selected 30 degC case highlighted.
- `fig53_export_before_after_reference.png/pdf`: Full PV, optimized simple
  objective, grid-best-objective, and zero-PV exports. If available, the
  largest feasible grid point is shown as a separate marker.
- `fig54_grid_reference_export_vs_curtailment.png/pdf`: Exp.-5d grid
  reference curve `p_export(c)` with optimizer, grid-best-objective, and
  grid-best-feasible points.
- `fig55_optimization_trace_export_and_curtailment.png/pdf`: Exp.-5d optimizer
  trace for export and curtailment factor.
- `fig53_exp05d_left_exp05c_right_comparison.png/pdf`: Side-by-side comparison
  of Fig. 53, with Exp. 5d on the left and Exp. 5c on the right.
- `fig55_exp05d_left_exp05c_right_comparison.png/pdf`: Side-by-side comparison
  of Fig. 55, with Exp. 5d on the left and Exp. 5c on the right.

## Data Sources

- `experiments/results/exp05a_network_screening/screening_results.csv`
- `experiments/results/exp05a_network_screening/selected_realistic_case.csv`
- `experiments/results/exp05d_optimize_pv_curtailment_simple_objective/selected_case_baseline.csv`
- `experiments/results/exp05d_optimize_pv_curtailment_simple_objective/final_solution.csv`
- `experiments/results/exp05d_optimize_pv_curtailment_simple_objective/grid_reference.csv`
- `experiments/results/exp05d_optimize_pv_curtailment_simple_objective/optimization_trace.csv`

## Scope Notes

Experiment 5d uses the analytical PV weather model and the simple objective
`((p_export_proxy - 7.0) / p_scale_mw)^2`. It intentionally has no 6.99-MW
target line, no softplus penalty, and no curtailment regularization. The 7.0 MW
line is a demonstrator-internal target, not a normative grid-code limit.
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
        "script": "experiments/plot_exp05d_figures.py",
        "experiment": "exp05d_optimize_pv_curtailment_simple_objective",
        "objective_variant": "simple_target_7mw",
        "selected_case_id": SELECTED_CASE_ID,
        "export_limit_mw": EXPORT_LIMIT_MW,
        "export_target_mw": EXPORT_TARGET_MW,
        "input_artifacts": inputs.as_dict(),
        "output_files": [str(path) for path in figure_paths] + [str(readme_path)],
        "notes": [
            "Figures are generated only from existing Exp. 5a and Exp. 5d artifacts.",
            "No power-flow solves, grid search, or optimization are run.",
            "Exp. 5d uses only the simple 7.0-MW quadratic target objective.",
        ],
    }
    path = output_dir / "figure_metadata.json"
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def generate_figures(
    exp05a_dir: Path = EXP05A_RESULTS_DIR,
    exp05d_dir: Path = EXP05D_RESULTS_DIR,
    exp05c_dir: Path = EXP05C_RESULTS_DIR,
    output_dir: Path = FIGURES_DIR,
) -> list[Path]:
    """Generate all Experiment 5d figures from existing artifacts."""

    exp05_plot._configure_matplotlib()
    inputs = exp05_plot.FigureInputs(exp05a_dir=exp05a_dir, exp05b_dir=exp05d_dir)
    output_dir = exp05_plot.ensure_output_dir(output_dir)

    screening_df = exp05_plot.load_csv(inputs.screening_results)
    selected_df = exp05_plot.load_csv(inputs.selected_realistic_case)
    baseline_df = exp05_plot.load_csv(inputs.selected_case_baseline)
    final_df = exp05_plot.load_csv(inputs.final_solution)
    grid_df = exp05_plot.load_csv(inputs.grid_reference)
    trace_df = exp05_plot.load_csv(inputs.optimization_trace)

    limit_mw, _target_mw = exp05_plot._artifact_limit_and_target(final_df)
    figure_paths: list[Path] = []
    for png_path, pdf_path in [
        exp05_plot.plot_fig51(screening_df, selected_df, output_dir, limit_mw),
        plot_fig53(baseline_df, final_df, grid_df, output_dir),
        plot_fig54(grid_df, final_df, output_dir),
        plot_fig55(trace_df, output_dir, limit_mw=limit_mw),
    ]:
        figure_paths.extend([png_path, pdf_path])

    should_combine = _variant_inputs_available(exp05c_dir) and (
        exp05d_dir == EXP05D_RESULTS_DIR or exp05c_dir != EXP05C_RESULTS_DIR
    )
    if should_combine:
        for png_path, pdf_path in [
            plot_fig53_exp05d_vs_exp05c_combined(exp05d_dir, exp05c_dir, output_dir),
            plot_fig55_exp05d_vs_exp05c_combined(exp05d_dir, exp05c_dir, output_dir),
        ]:
            figure_paths.extend([png_path, pdf_path])

    readme_path = write_readme(output_dir)
    metadata_path = write_metadata(output_dir, inputs, figure_paths, readme_path)
    return figure_paths + [readme_path, metadata_path]


def main() -> None:
    print("=" * 72)
    print("Experiment 5d: figure generation from existing artifacts")
    print(f"Output directory: {FIGURES_DIR}")
    print("=" * 72)
    paths = generate_figures()
    print("\nGenerated files:")
    for path in paths:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
