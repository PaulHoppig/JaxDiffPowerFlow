"""Create publication-style figures from existing Experiment 2b artifacts.

This script reads only the already exported CSV files in
``experiments/results/exp02_example_simple_gradients``. It does not run new
power-flow solves, does not compute new gradients, and does not import
``pandapower`` or JAX.

Run:
    python experiments/plot_exp02_gradient_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_DIR = (
    Path(__file__).resolve().parent / "results" / "exp02_example_simple_gradients"
)
FIGURES_DIR = RESULTS_DIR / "figures"

EPS_LOG = 1e-16

SCENARIO_ORDER = ("base", "load_high", "sgen_high")
INPUT_ORDER = (
    "load_scale_mv_bus_2",
    "sgen_scale_static_generator",
    "shunt_q_scale",
    "trafo_x_scale",
)
OBSERVABLE_ORDER = (
    "vm_mv_bus_2_pu",
    "p_slack_mw",
    "total_p_loss_mw",
    "p_trafo_hv_mw",
)

LABELS = {
    "base": "Base",
    "load_high": "High load",
    "sgen_high": "High static generator",
    "load_scale_mv_bus_2": "Load scale MV Bus 2",
    "sgen_scale_static_generator": "Static generator scale",
    "shunt_q_scale": "Shunt Q scale",
    "trafo_x_scale": "Transformer x scale",
    "vm_mv_bus_2_pu": "|V| MV Bus 2",
    "p_slack_mw": "Slack P",
    "total_p_loss_mw": "Total P loss",
    "p_trafo_hv_mw": "Transformer HV P",
}

GRADIENT_REQUIRED_COLUMNS = {
    "scenario",
    "input_parameter",
    "ad_grad",
    "fd_grad",
}
ERROR_SUMMARY_REQUIRED_COLUMNS = {"scenario"}
FD_STEP_REQUIRED_COLUMNS = {"fd_step", "ad_grad", "fd_grad"}


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV artifact with a clear error for missing files."""

    if not path.exists():
        raise FileNotFoundError(f"Required CSV artifact not found: {path}")
    return pd.read_csv(path)


def require_columns(df: pd.DataFrame, required: set[str], table_name: str) -> None:
    """Raise a descriptive error if required columns are missing."""

    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")


def find_column(df: pd.DataFrame, candidates: tuple[str, ...], table_name: str) -> str:
    """Return the first available column from a list of aliases."""

    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        f"{table_name} must contain one of these columns: {list(candidates)}"
    )


def safe_log10(values, eps: float = EPS_LOG) -> np.ndarray:
    """Return log10(max(abs(values), eps)) for robust log-display."""

    arr = np.asarray(values, dtype=float)
    return np.log10(np.maximum(np.abs(arr), eps))


def prettify_label(name: str) -> str:
    """Map technical artifact labels to compact plot labels."""

    return LABELS.get(str(name), str(name).replace("_", " "))


def save_figure(fig, output_dir: Path, stem: str, dpi: int = 300) -> tuple[Path, Path]:
    """Save a Matplotlib figure as PNG and PDF and close it."""

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def load_artifacts(
    results_dir: Path = RESULTS_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and validate the existing Exp. 2b CSV artifacts."""

    gradient = load_csv(results_dir / "gradient_table.csv")
    error_summary = load_csv(results_dir / "error_summary.csv")
    fd_step = load_csv(results_dir / "fd_step_study.csv")

    require_columns(gradient, GRADIENT_REQUIRED_COLUMNS, "gradient_table.csv")
    require_columns(error_summary, ERROR_SUMMARY_REQUIRED_COLUMNS, "error_summary.csv")
    require_columns(fd_step, FD_STEP_REQUIRED_COLUMNS, "fd_step_study.csv")

    find_column(gradient, ("output_observable", "observable"), "gradient_table.csv")
    find_column(gradient, ("rel_error", "relative_error"), "gradient_table.csv")
    find_column(fd_step, ("rel_error", "relative_error"), "fd_step_study.csv")
    return gradient, error_summary, fd_step


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce")


def _gradient_columns(df: pd.DataFrame) -> tuple[str, str, str, str, str]:
    obs_col = find_column(df, ("output_observable", "observable"), "gradient_table.csv")
    rel_col = find_column(df, ("rel_error", "relative_error"), "gradient_table.csv")
    abs_col = find_column(df, ("abs_error", "absolute_error"), "gradient_table.csv")
    ad_col = find_column(df, ("ad_grad", "ad_gradient"), "gradient_table.csv")
    fd_col = find_column(df, ("fd_grad", "fd_gradient"), "gradient_table.csv")
    return obs_col, rel_col, abs_col, ad_col, fd_col


def aggregate_error_heatmap(
    gradient_df: pd.DataFrame,
) -> tuple[list[str], list[str], np.ndarray]:
    """Aggregate max relative error by observable and input parameter."""

    obs_col, rel_col, _, _, _ = _gradient_columns(gradient_df)
    require_columns(
        gradient_df,
        {"input_parameter", obs_col, rel_col},
        "gradient_table.csv",
    )

    work = gradient_df.copy()
    work[rel_col] = _numeric_series(work, rel_col)
    pivot = work.pivot_table(
        index=obs_col,
        columns="input_parameter",
        values=rel_col,
        aggfunc="max",
    )

    observables = [name for name in OBSERVABLE_ORDER if name in pivot.index]
    inputs = [name for name in INPUT_ORDER if name in pivot.columns]
    if not observables or not inputs:
        raise ValueError("Could not build non-empty gradient error heatmap.")

    matrix = pivot.loc[observables, inputs].to_numpy(dtype=float)
    return observables, inputs, matrix


def grouped_relative_errors_by_observable(
    gradient_df: pd.DataFrame,
) -> tuple[list[str], list[np.ndarray]]:
    """Return relative-error arrays grouped by output observable."""

    obs_col, rel_col, _, _, _ = _gradient_columns(gradient_df)
    work = gradient_df.copy()
    work[rel_col] = _numeric_series(work, rel_col).abs()

    observables = [name for name in OBSERVABLE_ORDER if name in set(work[obs_col])]
    groups: list[np.ndarray] = []
    for observable in observables:
        vals = work.loc[work[obs_col] == observable, rel_col].dropna().to_numpy(float)
        if vals.size == 0:
            raise ValueError(f"No relative-error values for {observable!r}.")
        groups.append(vals)
    if not groups:
        raise ValueError("No observable groups available for relative-error boxplot.")
    return observables, groups


def _parity_limit(fd_values: np.ndarray, ad_values: np.ndarray) -> tuple[float, float]:
    """Return shared x/y limits for one parity panel."""

    all_vals = np.concatenate([fd_values, ad_values])
    finite = all_vals[np.isfinite(all_vals)]
    if finite.size == 0:
        return -1.0, 1.0

    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if lo == hi:
        pad = max(abs(lo) * 0.05, 1e-12)
    else:
        pad = 0.05 * (hi - lo)
    return lo - pad, hi + pad


def plot_global_parity(
    gradient_df: pd.DataFrame,
    output_dir: Path = FIGURES_DIR,
) -> tuple[Path, Path]:
    """Optional Figure 1a: global AD-vs-FD parity plot for all gradients."""

    _, _, _, ad_col, fd_col = _gradient_columns(gradient_df)
    work = gradient_df.copy()
    work[ad_col] = _numeric_series(work, ad_col)
    work[fd_col] = _numeric_series(work, fd_col)
    work = work.dropna(subset=[ad_col, fd_col, "scenario"])
    if work.empty:
        raise ValueError("No finite AD/FD gradient pairs available for parity plot.")

    all_vals = np.concatenate([work[ad_col].to_numpy(float), work[fd_col].to_numpy(float)])
    max_abs = float(np.nanmax(np.abs(all_vals)))
    limit = max_abs * 1.05 if max_abs > 0.0 else 1.0

    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    for scenario in SCENARIO_ORDER:
        subset = work[work["scenario"] == scenario]
        if subset.empty:
            continue
        ax.scatter(
            subset[fd_col],
            subset[ad_col],
            s=32,
            alpha=0.8,
            label=prettify_label(scenario),
        )
    ax.plot([-limit, limit], [-limit, limit], linewidth=1.0, color="black", alpha=0.7)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Finite-difference gradient")
    ax.set_ylabel("Implicit AD gradient")
    ax.legend(title="Scenario")
    ax.grid(True, alpha=0.3)
    ax.text(
        0.04,
        0.96,
        f"n = {len(work)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
    )
    fig.tight_layout()
    return save_figure(fig, output_dir, "fig01a_ad_vs_fd_parity_global")


def plot_parity(gradient_df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> tuple[Path, Path]:
    """Figure 1: faceted AD-vs-FD parity plot by output observable."""

    obs_col, rel_col, _, ad_col, fd_col = _gradient_columns(gradient_df)
    require_columns(
        gradient_df,
        {"scenario", "input_parameter", obs_col, rel_col, ad_col, fd_col},
        "gradient_table.csv",
    )

    work = gradient_df.copy()
    work[ad_col] = _numeric_series(work, ad_col)
    work[fd_col] = _numeric_series(work, fd_col)
    work[rel_col] = _numeric_series(work, rel_col).abs()
    work = work.dropna(subset=[ad_col, fd_col, rel_col, "scenario", "input_parameter", obs_col])
    if work.empty:
        raise ValueError("No finite AD/FD gradient pairs available for faceted parity plot.")

    observables = [name for name in OBSERVABLE_ORDER if name in set(work[obs_col])]
    if not observables:
        raise ValueError("No known output observables available for faceted parity plot.")

    input_parameters = [name for name in INPUT_ORDER if name in set(work["input_parameter"])]
    scenario_names = [name for name in SCENARIO_ORDER if name in set(work["scenario"])]
    prop_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    input_colors = {
        name: prop_colors[idx % len(prop_colors)] if prop_colors else f"C{idx}"
        for idx, name in enumerate(input_parameters)
    }
    scenario_markers = {
        scenario: marker
        for scenario, marker in zip(scenario_names, ("o", "s", "^", "D", "P"))
    }

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.4), sharex=False, sharey=False)
    flat_axes = axes.ravel()
    for ax, observable in zip(flat_axes, observables):
        subset = work[work[obs_col] == observable]
        x = subset[fd_col].to_numpy(float)
        y = subset[ad_col].to_numpy(float)
        lo, hi = _parity_limit(x, y)

        for input_parameter in input_parameters:
            for scenario in scenario_names:
                points = subset[
                    (subset["input_parameter"] == input_parameter)
                    & (subset["scenario"] == scenario)
                ]
                if points.empty:
                    continue
                ax.scatter(
                    points[fd_col],
                    points[ad_col],
                    s=22,
                    alpha=0.7,
                    marker=scenario_markers[scenario],
                    facecolor=input_colors[input_parameter],
                    edgecolor="black",
                    linewidth=0.35,
                )

        ax.plot([lo, hi], [lo, hi], linewidth=0.9, color="black", alpha=0.75)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(prettify_label(observable), fontsize=10)
        ax.grid(True, alpha=0.25)
        max_rel = float(np.nanmax(subset[rel_col].to_numpy(float)))
        median_rel = float(np.nanmedian(subset[rel_col].to_numpy(float)))
        ax.text(
            0.04,
            0.96,
            f"n = {len(subset)}\nmax rel. err. = {max_rel:.1e}\nmedian = {median_rel:.1e}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
        )

    for ax in flat_axes[len(observables) :]:
        ax.set_visible(False)

    input_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=input_colors[name],
            markeredgecolor="black",
            markeredgewidth=0.35,
            label=prettify_label(name),
        )
        for name in input_parameters
    ]
    scenario_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=scenario_markers[name],
            linestyle="",
            color="black",
            markerfacecolor="white",
            markeredgecolor="black",
            label=prettify_label(name),
        )
        for name in scenario_names
    ]
    fig.legend(
        handles=input_handles,
        title="Input parameter",
        loc="center left",
        bbox_to_anchor=(0.99, 0.62),
        frameon=False,
        fontsize=8,
        title_fontsize=9,
    )
    fig.legend(
        handles=scenario_handles,
        title="Scenario",
        loc="center left",
        bbox_to_anchor=(0.99, 0.28),
        frameon=False,
        fontsize=8,
        title_fontsize=9,
    )
    fig.supxlabel("Finite-difference gradient", y=0.04)
    fig.supylabel("Implicit AD gradient", x=0.04)
    fig.subplots_adjust(
        left=0.10,
        right=0.78,
        bottom=0.10,
        top=0.94,
        wspace=0.32,
        hspace=0.32,
    )
    return save_figure(fig, output_dir, "fig01_ad_vs_fd_parity_by_observable")


def plot_error_heatmap(
    gradient_df: pd.DataFrame,
    output_dir: Path = FIGURES_DIR,
) -> tuple[Path, Path]:
    """Figure 2: heatmap of max relative error over scenarios."""

    observables, inputs, matrix = aggregate_error_heatmap(gradient_df)
    log_matrix = safe_log10(matrix)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    image = ax.imshow(log_matrix, aspect="auto")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("log10(relative error)")

    ax.set_xticks(np.arange(len(inputs)))
    ax.set_yticks(np.arange(len(observables)))
    ax.set_xticklabels([prettify_label(name) for name in inputs], rotation=30, ha="right")
    ax.set_yticklabels([prettify_label(name) for name in observables])
    ax.set_xlabel("Input parameter")
    ax.set_ylabel("Output observable")

    for i in range(len(observables)):
        for j in range(len(inputs)):
            val = matrix[i, j]
            label = f"{val:.1e}" if np.isfinite(val) else "nan"
            ax.text(j, i, label, ha="center", va="center", fontsize=8)

    fig.tight_layout()
    return save_figure(fig, output_dir, "fig02_gradient_error_heatmap")


def plot_error_boxplot(
    gradient_df: pd.DataFrame,
    output_dir: Path = FIGURES_DIR,
) -> tuple[Path, Path]:
    """Figure 3: log-scale boxplot of relative errors by observable."""

    observables, groups = grouped_relative_errors_by_observable(gradient_df)
    display_groups = [np.maximum(group, EPS_LOG) for group in groups]

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    labels = [prettify_label(name) for name in observables]
    try:
        ax.boxplot(display_groups, tick_labels=labels)
    except TypeError:
        ax.boxplot(display_groups, labels=labels)
    for idx, vals in enumerate(display_groups, start=1):
        offsets = np.linspace(-0.08, 0.08, len(vals)) if len(vals) > 1 else np.array([0.0])
        ax.scatter(
            np.full(len(vals), idx) + offsets,
            vals,
            s=14,
            alpha=0.45,
            color="black",
        )
    ax.set_yscale("log")
    ax.set_ylabel("Relative gradient error")
    ax.set_xlabel("Output observable")
    ax.grid(True, which="both", axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return save_figure(fig, output_dir, "fig03_relative_error_boxplot")


def _curve_label(row: pd.Series) -> str:
    if "selected_gradient_id" in row and isinstance(row["selected_gradient_id"], str):
        return row["selected_gradient_id"]
    parts = []
    for column in ("scenario", "input_parameter", "output_observable", "observable"):
        if column in row and pd.notna(row[column]):
            parts.append(str(row[column]))
    return " / ".join(parts) if parts else "gradient"


def plot_fd_step_study(
    fd_step_df: pd.DataFrame,
    output_dir: Path = FIGURES_DIR,
) -> tuple[Path, Path]:
    """Figure 4: log-log FD step-size study."""

    step_col = find_column(fd_step_df, ("fd_step", "step_size"), "fd_step_study.csv")
    rel_col = find_column(fd_step_df, ("rel_error", "relative_error"), "fd_step_study.csv")
    group_col = (
        "selected_gradient_id"
        if "selected_gradient_id" in fd_step_df.columns
        else find_column(fd_step_df, ("output_observable", "observable"), "fd_step_study.csv")
    )

    work = fd_step_df.copy()
    work[step_col] = _numeric_series(work, step_col)
    work[rel_col] = _numeric_series(work, rel_col).abs()
    work = work.dropna(subset=[step_col, rel_col, group_col])
    if work.empty:
        raise ValueError("No finite FD step-study values available for plotting.")

    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    for group_name, group in work.groupby(group_col, sort=False):
        group = group.sort_values(step_col)
        y = np.maximum(group[rel_col].to_numpy(float), EPS_LOG)
        label = _curve_label(group.iloc[0]) if group_col == "selected_gradient_id" else str(group_name)
        ax.plot(group[step_col], y, marker="o", linewidth=1.5, label=label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Finite-difference step size")
    ax.set_ylabel("Relative AD-FD error")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return save_figure(fig, output_dir, "fig04_fd_step_study")


def _scenario_error_summary(error_summary_df: pd.DataFrame) -> pd.DataFrame:
    max_col = find_column(
        error_summary_df,
        ("max_rel_error", "max_relative_error", "relative_error", "rel_error"),
        "error_summary.csv",
    )
    median_col = None
    for candidate in ("median_rel_error", "median_relative_error"):
        if candidate in error_summary_df.columns:
            median_col = candidate
            break

    work = error_summary_df.copy()
    work[max_col] = _numeric_series(work, max_col).abs()
    agg = work.groupby("scenario", as_index=False)[max_col].max()
    agg = agg.rename(columns={max_col: "max_rel_error"})
    if median_col is not None:
        work[median_col] = _numeric_series(work, median_col).abs()
        median = work.groupby("scenario", as_index=False)[median_col].median()
        median = median.rename(columns={median_col: "median_rel_error"})
        agg = agg.merge(median, on="scenario", how="left")
    return agg


def plot_error_by_scenario(
    error_summary_df: pd.DataFrame,
    output_dir: Path = FIGURES_DIR,
) -> tuple[Path, Path]:
    """Figure 5: aggregate relative error by operating scenario."""

    summary = _scenario_error_summary(error_summary_df)
    scenario_order = [name for name in SCENARIO_ORDER if name in set(summary["scenario"])]
    summary = summary.set_index("scenario").loc[scenario_order].reset_index()
    x = np.arange(len(summary))
    max_vals = np.maximum(summary["max_rel_error"].to_numpy(float), EPS_LOG)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    width = 0.35 if "median_rel_error" in summary.columns else 0.55
    ax.bar(x - width / 2, max_vals, width=width, label="Max relative error")
    if "median_rel_error" in summary.columns:
        median_vals = np.maximum(summary["median_rel_error"].to_numpy(float), EPS_LOG)
        ax.bar(x + width / 2, median_vals, width=width, label="Median relative error")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([prettify_label(name) for name in summary["scenario"]])
    ax.set_ylabel("Relative gradient error")
    ax.set_xlabel("Scenario")
    ax.legend()
    ax.grid(True, which="both", axis="y", alpha=0.3)
    fig.tight_layout()
    return save_figure(fig, output_dir, "fig05_error_by_scenario")


def write_figures_readme(
    figures_dir: Path = FIGURES_DIR,
    results_dir: Path = RESULTS_DIR,
) -> Path:
    """Write a concise description of figure sources and filters."""

    figures_dir.mkdir(parents=True, exist_ok=True)
    text = f"""# Experiment 2b Figures

These figures are generated only from existing Experiment 2b artifacts in
`{results_dir.as_posix()}`. The plotting script does not run new power-flow
solves, does not recompute AD gradients, and does not evaluate finite
differences.

## Figure 1

Files: `fig01_ad_vs_fd_parity_by_observable.png` and
`fig01_ad_vs_fd_parity_by_observable.pdf`. Data source: `gradient_table.csv`.
The main parity plot facets the exported gradients by output observable. Each
panel compares implicit AD gradients against central finite-difference
gradients, colors points by input parameter, marks scenarios with different
symbols, and includes the reference line `y = x`.

Optional files: `fig01a_ad_vs_fd_parity_global.png` and
`fig01a_ad_vs_fd_parity_global.pdf`. These retain the compact global parity
view for comparison but are not the main Figure 1.

## Figure 2

Files: `fig02_gradient_error_heatmap.png` and
`fig02_gradient_error_heatmap.pdf`. Data source: `gradient_table.csv`.
The color value is `log10(max(relative error, {EPS_LOG:g}))`; annotations show
the aggregated max relative error over the three scenarios for each
input-output combination.

## Figure 3

Files: `fig03_relative_error_boxplot.png` and
`fig03_relative_error_boxplot.pdf`. Data source: `gradient_table.csv`.
Relative errors are grouped by output observable. Values at or below zero are
shown as `{EPS_LOG:g}` only for log-scale display.

## Figure 4

Files: `fig04_fd_step_study.png` and `fig04_fd_step_study.pdf`.
Data source: `fd_step_study.csv`. Curves show relative AD-FD error over the
exported finite-difference step sizes.

## Figure 5

Files: `fig05_error_by_scenario.png` and `fig05_error_by_scenario.pdf`.
Data source: `error_summary.csv`. Bars summarize max relative error by
scenario; median relative error is included when present in the artifact.
"""
    path = figures_dir / "README.md"
    path.write_text(text, encoding="utf-8")
    return path


def generate_figures(
    results_dir: Path = RESULTS_DIR,
    figures_dir: Path | None = None,
) -> list[Path]:
    """Generate all mandatory Exp. 2b figures from existing artifacts."""

    target_dir = figures_dir if figures_dir is not None else results_dir / "figures"
    gradient_df, error_summary_df, fd_step_df = load_artifacts(results_dir)

    outputs: list[Path] = []
    outputs.extend(plot_parity(gradient_df, target_dir))
    outputs.extend(plot_global_parity(gradient_df, target_dir))
    outputs.extend(plot_error_heatmap(gradient_df, target_dir))
    outputs.extend(plot_error_boxplot(gradient_df, target_dir))
    outputs.extend(plot_fd_step_study(fd_step_df, target_dir))
    outputs.extend(plot_error_by_scenario(error_summary_df, target_dir))
    outputs.append(write_figures_readme(target_dir, results_dir))
    return outputs


def main() -> None:
    outputs = generate_figures(RESULTS_DIR, FIGURES_DIR)
    print(f"Figures directory: {FIGURES_DIR}")
    for path in outputs:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
