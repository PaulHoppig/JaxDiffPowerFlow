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
    "base": "Basis",
    "load_high": "Hohe Last",
    "sgen_high": "Hohe Einspeisung",
    "load_scale_mv_bus_2": "Lastskalierung MV-Bus 2",
    "sgen_scale_static_generator": "Skalierung statischer Generator",
    "shunt_q_scale": "Shunt-Q-Skalierung",
    "trafo_x_scale": "Trafo-X-Skalierung",
    "vm_mv_bus_2_pu": "|V| MV Bus 2",
    "p_slack_mw": "Slack-P",
    "total_p_loss_mw": "Wirkleistungsverluste gesamt",
    "p_trafo_hv_mw": "Trafo-HV-P",
}

COMPARISON_INPUT_LABELS = {
    "load_scale_mv_bus_2": "Lastskalierung",
    "sgen_scale_static_generator": "sgen-Skalierung",
    "shunt_q_scale": "Shunt-Q-Skalierung",
    "trafo_x_scale": "Trafo-X-Skalierung",
}
COMPARISON_OUTPUT_LABELS = {
    "vm_mv_bus_2_pu": "|V| MV Bus 2",
    "p_slack_mw": "Slack-P",
    "total_p_loss_mw": "Wirkleistungsverluste",
    "p_trafo_hv_mw": "Trafo-HV-P",
}

GRADIENT_REQUIRED_COLUMNS = {
    "scenario",
    "input_parameter",
    "ad_grad",
    "fd_grad",
}
ERROR_SUMMARY_REQUIRED_COLUMNS = {"scenario"}
FD_STEP_REQUIRED_COLUMNS = {"fd_step", "ad_grad", "fd_grad"}
FD_VS_FD_REQUIRED_COLUMNS = {
    "selected_gradient_id",
    "scenario",
    "input_parameter",
    "output_observable",
    "fd_step",
    "fd_grad",
    "fd_plus_converged",
    "fd_minus_converged",
}
GRADIENT_MAGNITUDE_ERROR_SUMMARY_COLUMNS = (
    "input_parameter",
    "output_observable",
    "n",
    "median_abs_ad_grad",
    "min_abs_ad_grad",
    "max_abs_ad_grad",
    "median_rel_error",
    "max_rel_error",
    "log10_median_abs_ad_grad",
    "log10_max_rel_error",
)
FD_VS_FD_STEP_STABILITY_COLUMNS = (
    "selected_gradient_id",
    "scenario",
    "input_parameter",
    "output_observable",
    "fd_step_large",
    "fd_step_small",
    "fd_step_pair",
    "fd_grad_large",
    "fd_grad_small",
    "fd_abs_change",
    "fd_rel_change",
    "fd_plus_converged_large",
    "fd_minus_converged_large",
    "fd_plus_converged_small",
    "fd_minus_converged_small",
    "ad_grad_large",
    "ad_grad_small",
    "ad_vs_fd_rel_error_large",
    "ad_vs_fd_rel_error_small",
)

FD_STABILITY_LABELS = {
    "base:load_scale_mv_bus_2->vm_mv_bus_2_pu": r"$\partial |V_2| / \partial \theta_\mathrm{load}$",
    "base:sgen_scale_static_generator->p_slack_mw": r"$\partial P_\mathrm{slack} / \partial \theta_\mathrm{sgen}$",
    "base:shunt_q_scale->total_p_loss_mw": r"$\partial P_\mathrm{loss} / \partial \theta_\mathrm{shunt}$",
}


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


def build_gradient_magnitude_error_comparison_table(
    gradient_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate AD-gradient magnitudes and relative errors by input/output pair."""

    obs_col, rel_col, _, ad_col, _ = _gradient_columns(gradient_df)
    require_columns(
        gradient_df,
        {"input_parameter", obs_col, ad_col, rel_col},
        "gradient_table.csv",
    )

    work = gradient_df.copy()
    work[ad_col] = _numeric_series(work, ad_col)
    work[rel_col] = _numeric_series(work, rel_col).abs()
    work["abs_ad_grad"] = work[ad_col].abs()
    work = work.dropna(subset=["input_parameter", obs_col, "abs_ad_grad", rel_col])

    rows: list[dict] = []
    for input_parameter in INPUT_ORDER:
        for observable in OBSERVABLE_ORDER:
            subset = work[
                (work["input_parameter"] == input_parameter)
                & (work[obs_col] == observable)
            ]
            if subset.empty:
                continue

            abs_ad = subset["abs_ad_grad"].to_numpy(float)
            rel_error = subset[rel_col].to_numpy(float)
            median_abs_ad = float(np.median(abs_ad))
            max_rel_error = float(np.max(rel_error))

            rows.append(
                {
                    "input_parameter": input_parameter,
                    "output_observable": observable,
                    "n": int(len(subset)),
                    "median_abs_ad_grad": median_abs_ad,
                    "min_abs_ad_grad": float(np.min(abs_ad)),
                    "max_abs_ad_grad": float(np.max(abs_ad)),
                    "median_rel_error": float(np.median(rel_error)),
                    "max_rel_error": max_rel_error,
                    "log10_median_abs_ad_grad": float(
                        np.log10(max(median_abs_ad, EPS_LOG))
                    ),
                    "log10_max_rel_error": float(
                        np.log10(max(max_rel_error, EPS_LOG))
                    ),
                }
            )

    return pd.DataFrame(rows, columns=GRADIENT_MAGNITUDE_ERROR_SUMMARY_COLUMNS)


def write_gradient_magnitude_error_summary(
    summary_df: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """Write the Fig. 6 summary table as CSV and JSON."""

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "gradient_magnitude_vs_error_summary.csv"
    json_path = output_dir / "gradient_magnitude_vs_error_summary.json"
    summary_df.to_csv(csv_path, index=False)
    summary_df.to_json(json_path, orient="records", indent=2)
    return [csv_path, json_path]


def compute_fd_vs_fd_step_stability(
    fd_step_df: pd.DataFrame,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """Compare neighboring finite-difference gradients from an FD step study."""

    require_columns(fd_step_df, FD_VS_FD_REQUIRED_COLUMNS, "fd_step_study.csv")
    rel_col = next(
        (
            candidate
            for candidate in ("rel_error", "relative_error")
            if candidate in fd_step_df.columns
        ),
        None,
    )

    group_cols = [
        "selected_gradient_id",
        "scenario",
        "input_parameter",
        "output_observable",
    ]
    work = fd_step_df.copy()
    work["fd_step"] = _numeric_series(work, "fd_step")
    work["fd_grad"] = _numeric_series(work, "fd_grad")
    if "ad_grad" in work.columns:
        work["ad_grad"] = _numeric_series(work, "ad_grad")
    if rel_col is not None:
        work[rel_col] = _numeric_series(work, rel_col)
    work = work.dropna(subset=group_cols + ["fd_step", "fd_grad"])

    rows: list[dict] = []
    for _, group in work.groupby(group_cols, sort=False):
        group = group.sort_values("fd_step", ascending=False).reset_index(drop=True)
        for idx in range(len(group) - 1):
            large = group.iloc[idx]
            small = group.iloc[idx + 1]
            fd_grad_large = float(large["fd_grad"])
            fd_grad_small = float(small["fd_grad"])
            fd_abs_change = abs(fd_grad_small - fd_grad_large)
            denominator = max(abs(fd_grad_small), abs(fd_grad_large), eps)

            row = {
                "selected_gradient_id": large["selected_gradient_id"],
                "scenario": large["scenario"],
                "input_parameter": large["input_parameter"],
                "output_observable": large["output_observable"],
                "fd_step_large": float(large["fd_step"]),
                "fd_step_small": float(small["fd_step"]),
                "fd_step_pair": f"{large['fd_step']:g} -> {small['fd_step']:g}",
                "fd_grad_large": fd_grad_large,
                "fd_grad_small": fd_grad_small,
                "fd_abs_change": float(fd_abs_change),
                "fd_rel_change": float(fd_abs_change / denominator),
                "fd_plus_converged_large": large["fd_plus_converged"],
                "fd_minus_converged_large": large["fd_minus_converged"],
                "fd_plus_converged_small": small["fd_plus_converged"],
                "fd_minus_converged_small": small["fd_minus_converged"],
            }
            if "ad_grad" in work.columns:
                row["ad_grad_large"] = float(large["ad_grad"])
                row["ad_grad_small"] = float(small["ad_grad"])
            if rel_col is not None:
                row["ad_vs_fd_rel_error_large"] = float(large[rel_col])
                row["ad_vs_fd_rel_error_small"] = float(small[rel_col])
            rows.append(row)

    return pd.DataFrame(rows, columns=FD_VS_FD_STEP_STABILITY_COLUMNS)


def write_fd_vs_fd_step_stability(
    stability_df: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """Write the FD-vs-FD step-stability table as CSV and JSON."""

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "fd_vs_fd_step_stability.csv"
    json_path = output_dir / "fd_vs_fd_step_stability.json"
    stability_df.to_csv(csv_path, index=False)
    stability_df.to_json(json_path, orient="records", indent=2)
    return [csv_path, json_path]


def build_error_heatmap_figure(
    gradient_df: pd.DataFrame,
) -> tuple[plt.Figure, plt.Axes]:
    """Build Figure 2 without saving, so labels and layout stay testable."""

    observables, inputs, matrix = aggregate_error_heatmap(gradient_df)
    log_matrix = safe_log10(matrix)
    x_edges = np.arange(len(inputs) + 1, dtype=float) - 0.5
    y_edges = np.arange(len(observables) + 1, dtype=float) - 0.5

    fig, ax = plt.subplots(figsize=(7.4, 5.0), constrained_layout=True)
    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        log_matrix,
        shading="flat",
        edgecolors="white",
        linewidth=0.7,
    )
    cbar = fig.colorbar(mesh, ax=ax, pad=0.025)
    cbar.set_label("log10(max. relativer Fehler)")

    ax.set_xticks(np.arange(len(inputs)))
    ax.set_yticks(np.arange(len(observables)))
    ax.set_xticklabels(
        [prettify_label(name) for name in inputs],
        rotation=30,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_yticklabels([prettify_label(name) for name in observables])
    ax.set_xlabel("Eingangsparameter")
    ax.set_ylabel("Ausgangsgröße")
    ax.set_xlim(-0.5, len(inputs) - 0.5)
    ax.set_ylim(len(observables) - 0.5, -0.5)
    ax.tick_params(axis="both", which="both", length=0)

    for i in range(len(observables)):
        for j in range(len(inputs)):
            val = matrix[i, j]
            label = f"{val:.1e}" if np.isfinite(val) else "nan"
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color="black",
                bbox={
                    "boxstyle": "round,pad=0.16",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.72,
                },
            )

    return fig, ax


def build_gradient_magnitude_vs_relative_error_figure(
    summary_df: pd.DataFrame,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Build Figure 6 comparing gradient magnitude and relative error heatmaps."""

    require_columns(
        summary_df,
        set(GRADIENT_MAGNITUDE_ERROR_SUMMARY_COLUMNS),
        "gradient_magnitude_vs_error_summary",
    )
    left_matrix = _summary_matrix(summary_df, "log10_median_abs_ad_grad")
    right_matrix = _summary_matrix(summary_df, "log10_max_rel_error")
    left_values = _summary_matrix(summary_df, "median_abs_ad_grad")
    right_values = _summary_matrix(summary_df, "max_rel_error")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.0, 5.2),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    _draw_comparison_heatmap(
        fig=fig,
        ax=axes[0],
        matrix=left_matrix,
        annotation_values=left_values,
        colorbar_label="log10(Median |AD-Gradient|)",
        cmap="viridis_r",
    )
    _draw_comparison_heatmap(
        fig=fig,
        ax=axes[1],
        matrix=right_matrix,
        annotation_values=right_values,
        colorbar_label="log10(max. relativer Fehler)",
        cmap="viridis",
    )

    return fig, (axes[0], axes[1])


def plot_gradient_magnitude_vs_relative_error_heatmaps(
    summary_df: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """Figure 6: side-by-side gradient magnitude and relative-error heatmaps."""

    fig, _ = build_gradient_magnitude_vs_relative_error_figure(summary_df)
    return list(
        save_figure(
            fig,
            output_dir,
            "fig06_gradient_magnitude_vs_relative_error_heatmaps",
        )
    )


def _summary_matrix(summary_df: pd.DataFrame, value_column: str) -> np.ndarray:
    pivot = summary_df.pivot(
        index="output_observable",
        columns="input_parameter",
        values=value_column,
    )
    missing_outputs = [name for name in OBSERVABLE_ORDER if name not in pivot.index]
    missing_inputs = [name for name in INPUT_ORDER if name not in pivot.columns]
    if missing_outputs or missing_inputs:
        raise ValueError(
            "Missing input/output combinations for Fig. 6 heatmap: "
            f"outputs={missing_outputs}, inputs={missing_inputs}"
        )
    return pivot.loc[OBSERVABLE_ORDER, INPUT_ORDER].to_numpy(float)


def _draw_comparison_heatmap(
    fig: plt.Figure,
    ax: plt.Axes,
    matrix: np.ndarray,
    annotation_values: np.ndarray,
    colorbar_label: str,
    cmap: str,
) -> None:
    x_edges = np.arange(len(INPUT_ORDER) + 1, dtype=float) - 0.5
    y_edges = np.arange(len(OBSERVABLE_ORDER) + 1, dtype=float) - 0.5
    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        matrix,
        shading="flat",
        edgecolors="white",
        linewidth=0.7,
        cmap=cmap,
    )
    cbar = fig.colorbar(mesh, ax=ax, pad=0.025)
    cbar.set_label(colorbar_label)

    ax.set_xticks(np.arange(len(INPUT_ORDER)))
    ax.set_yticks(np.arange(len(OBSERVABLE_ORDER)))
    ax.set_xticklabels(
        [COMPARISON_INPUT_LABELS[name] for name in INPUT_ORDER],
        rotation=30,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_yticklabels([COMPARISON_OUTPUT_LABELS[name] for name in OBSERVABLE_ORDER])
    ax.set_xlabel("Eingangsparameter")
    ax.set_ylabel("Ausgangsgröße")
    ax.set_xlim(-0.5, len(INPUT_ORDER) - 0.5)
    ax.set_ylim(len(OBSERVABLE_ORDER) - 0.5, -0.5)
    ax.tick_params(axis="both", which="both", length=0)

    for i in range(len(OBSERVABLE_ORDER)):
        for j in range(len(INPUT_ORDER)):
            value = annotation_values[i, j]
            label = f"{value:.1e}" if np.isfinite(value) else "nan"
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color="black",
                bbox={
                    "boxstyle": "round,pad=0.16",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.72,
                },
            )


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

    fig, _ = build_error_heatmap_figure(gradient_df)
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
        gid = row["selected_gradient_id"]
        if gid in FD_STABILITY_LABELS:
            return FD_STABILITY_LABELS[gid]
        return gid
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


def _fd_stability_label(group_id: str, row: pd.Series) -> str:
    if group_id in FD_STABILITY_LABELS:
        return FD_STABILITY_LABELS[group_id]
    input_label = COMPARISON_INPUT_LABELS.get(
        str(row["input_parameter"]),
        prettify_label(str(row["input_parameter"])),
    )
    output_label = COMPARISON_OUTPUT_LABELS.get(
        str(row["output_observable"]),
        prettify_label(str(row["output_observable"])),
    )
    return f"{input_label} -> {output_label}"


def build_fd_vs_fd_step_stability_figure(
    stability_df: pd.DataFrame,
) -> tuple[plt.Figure, plt.Axes]:
    """Build Figure 7 showing FD-gradient changes between neighboring steps."""

    required = {
        "selected_gradient_id",
        "input_parameter",
        "output_observable",
        "fd_step_large",
        "fd_step_small",
        "fd_rel_change",
    }
    require_columns(stability_df, required, "fd_vs_fd_step_stability")

    work = stability_df.copy()
    work["fd_step_large"] = _numeric_series(work, "fd_step_large")
    work["fd_step_small"] = _numeric_series(work, "fd_step_small")
    work["fd_rel_change"] = _numeric_series(work, "fd_rel_change").abs()
    work = work.dropna(subset=["selected_gradient_id", "fd_step_large", "fd_rel_change"])
    if work.empty:
        raise ValueError("No finite FD-vs-FD stability rows available for plotting.")

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for group_id, group in work.groupby("selected_gradient_id", sort=False):
        group = group.sort_values("fd_step_large")
        y = np.maximum(group["fd_rel_change"].to_numpy(float), EPS_LOG)
        label = _fd_stability_label(str(group_id), group.iloc[0])
        ax.plot(
            group["fd_step_large"],
            y,
            marker="o",
            linewidth=1.5,
            label=label,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("FD-vs-FD stability over finite-difference step size")
    ax.set_xlabel("Larger FD step h")
    ax.set_ylabel("Relative change between FD(h) and FD(next smaller h)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_fd_vs_fd_step_stability(
    stability_df: pd.DataFrame,
    output_dir: Path = FIGURES_DIR,
) -> tuple[Path, Path]:
    """Figure 7: FD-vs-FD step-stability diagnostic."""

    fig, _ = build_fd_vs_fd_step_stability_figure(stability_df)
    return save_figure(fig, output_dir, "fig07_fd_vs_fd_step_stability")


def write_fd_vs_fd_step_stability_detailed(
    stability_df: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """Write the detailed FD-vs-FD step-stability table as CSV and JSON."""

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "fd_vs_fd_step_stability_detailed.csv"
    json_path = output_dir / "fd_vs_fd_step_stability_detailed.json"
    stability_df.to_csv(csv_path, index=False)
    stability_df.to_json(json_path, orient="records", indent=2)
    return [csv_path, json_path]


def plot_fd_step_study_detailed(
    fd_step_detailed_df: pd.DataFrame,
    output_dir: Path = FIGURES_DIR,
) -> tuple[Path, Path]:
    """Figure 8: detailed log-log AD-vs-FD step-size study (h = 1e0 ... 1e-10)."""

    step_col = find_column(
        fd_step_detailed_df, ("fd_step", "step_size"), "fd_step_study_detailed.csv"
    )
    rel_col = find_column(
        fd_step_detailed_df, ("rel_error", "relative_error"), "fd_step_study_detailed.csv"
    )
    group_col = (
        "selected_gradient_id"
        if "selected_gradient_id" in fd_step_detailed_df.columns
        else find_column(
            fd_step_detailed_df, ("output_observable", "observable"), "fd_step_study_detailed.csv"
        )
    )

    work = fd_step_detailed_df.copy()
    work[step_col] = _numeric_series(work, step_col)
    work[rel_col] = _numeric_series(work, rel_col).abs()
    work = work.dropna(subset=[step_col, rel_col, group_col])
    if work.empty:
        raise ValueError("No finite detailed FD step-study values available for Fig. 8.")

    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    for group_name, group in work.groupby(group_col, sort=False):
        group = group.sort_values(step_col)
        y = np.maximum(group[rel_col].to_numpy(float), EPS_LOG)
        label = _curve_label(group.iloc[0]) if group_col == "selected_gradient_id" else str(group_name)
        ax.plot(group[step_col], y, marker="o", linewidth=1.5, label=label)

    # O(h²) reference: c = 3e-6, about half of sgen_scale (c_sgen≈5.85e-6), so the
    # line runs slightly below the orange P_slack-sensitivity curve on the right side.
    # Cuts off cleanly when y falls below 1e-14 (no flat floor at EPS_LOG).
    _h_o2_full = np.logspace(0, -10, 400)
    _anc_h, _anc_y = 1e-1, 3e-8   # → c = 3e-6
    _y_o2_full = _anc_y * (_h_o2_full / _anc_h) ** 2
    _mask_o2 = _y_o2_full >= 1e-14
    ax.plot(_h_o2_full[_mask_o2], _y_o2_full[_mask_o2], linestyle="--", color="dimgray",
            linewidth=1.0, alpha=0.65, zorder=0)
    # Label: left edge at x=6e-2, y=1e-9
    ax.text(6e-2, 1e-9,
            "Referenzsteigung\n$O(h^2)$",
            color="dimgray", fontsize=6, ha="left", va="top")

    # O(ε/h) reference: C=1.5e-15, anchored so the line sits just below the local
    # minimum of the blue voltage-sensitivity curve at h≈5e-8 (y_line≈3e-8 vs data≈4.4e-8).
    # Crosses O(h²) near h≈8e-4 (y≈1.9e-12, above the 1e-14 cutoff → crossing visible).
    _C_round = 1.5e-15
    _h_round_full = np.logspace(0, -10, 400)
    _y_round_full = _C_round / _h_round_full
    _mask_round = _y_round_full >= 1e-14
    ax.plot(_h_round_full[_mask_round], _y_round_full[_mask_round], linestyle="-.", color="dimgray",
            linewidth=1.0, alpha=0.65, zorder=0)
    # Label: x=1e-8, y=1e-9
    ax.text(1e-8, 1e-9,
            "Referenzsteigung\n$O(\\varepsilon/h)$",
            color="dimgray", fontsize=6, ha="left", va="top")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-15)
    ax.set_xlabel("Schrittweite $h$")
    ax.set_ylabel(r"Relativer Fehler $\epsilon_\mathrm{rel}$ (AD vs. FD)")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return save_figure(fig, output_dir, "fig08_fd_step_study_detailed")


def build_fd_vs_fd_step_stability_detailed_figure(
    stability_df: pd.DataFrame,
) -> tuple[plt.Figure, plt.Axes]:
    """Build Figure 9: detailed FD-vs-FD stability over step size."""

    required = {
        "selected_gradient_id",
        "input_parameter",
        "output_observable",
        "fd_step_large",
        "fd_step_small",
        "fd_rel_change",
    }
    require_columns(stability_df, required, "fd_vs_fd_step_stability_detailed")

    work = stability_df.copy()
    work["fd_step_large"] = _numeric_series(work, "fd_step_large")
    work["fd_step_small"] = _numeric_series(work, "fd_step_small")
    work["fd_rel_change"] = _numeric_series(work, "fd_rel_change").abs()
    work = work.dropna(subset=["selected_gradient_id", "fd_step_large", "fd_rel_change"])
    if work.empty:
        raise ValueError("No finite detailed FD-vs-FD stability rows available for Fig. 9.")

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for group_id, group in work.groupby("selected_gradient_id", sort=False):
        group = group.sort_values("fd_step_large")
        y = np.maximum(group["fd_rel_change"].to_numpy(float), EPS_LOG)
        label = _fd_stability_label(str(group_id), group.iloc[0])
        ax.plot(group["fd_step_large"], y, marker="o", linewidth=1.5, label=label)

    # O(h²) reference line: anchored on load_scale gradient (h=1, fd_rel_change≈3.4e-4)
    _h_o2 = np.logspace(0, -4, 200)
    _anc_h, _anc_y = 1e0, 1.0e-3
    _y_o2 = _anc_y * (_h_o2 / _anc_h) ** 2
    ax.plot(_h_o2, _y_o2, linestyle="--", color="dimgray",
            linewidth=1.0, alpha=0.65, zorder=0)
    ax.text(3e-1, _anc_y * (3e-1 / _anc_h) ** 2 * 3.5,
            r"$O(h^2)$", color="dimgray", fontsize=8, ha="center", va="bottom")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("FD-vs-FD stability over step size")
    ax.set_xlabel("Larger FD step h")
    ax.set_ylabel("Relative change between FD(h) and FD(next smaller h)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_fd_vs_fd_step_stability_detailed(
    stability_df: pd.DataFrame,
    output_dir: Path = FIGURES_DIR,
) -> tuple[Path, Path]:
    """Figure 9: detailed FD-vs-FD step-stability diagnostic."""

    fig, _ = build_fd_vs_fd_step_stability_detailed_figure(stability_df)
    return save_figure(fig, output_dir, "fig09_fd_vs_fd_step_stability_detailed")


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
solves, does not recompute AD gradients, and does not run new
finite-difference evaluations.

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
input-output combination. The cells are rendered as a discrete tiled heatmap
with visible white boundaries for readability, using only the existing
exported gradient artifact. No new gradients or power-flow solves are run.

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

## Figure 6

Files: `fig06_gradient_magnitude_vs_relative_error_heatmaps.png` and
`fig06_gradient_magnitude_vs_relative_error_heatmaps.pdf`. Data source:
`gradient_table.csv`. The left heatmap shows
`log10(median |AD gradient|)` over the three scenarios for each
input-output combination. The right heatmap shows
`log10(max relative error)` over the same combinations and scenario set.
The companion files `gradient_magnitude_vs_error_summary.csv` and
`gradient_magnitude_vs_error_summary.json` contain the aggregated values.

The purpose of Figure 6 is to compare gradient orders of magnitude with
relative AD-vs-FD errors. Small gradients can amplify relative
finite-difference deviations because the observable changes measured by
finite differences become very small. This is a descriptive interpretation of
existing artifacts only. No new power-flow solves, AD gradients, or finite
differences are run.

## Figure 7

Files: `fig07_fd_vs_fd_step_stability.png` and
`fig07_fd_vs_fd_step_stability.pdf`. Data source: `fd_step_study.csv`.
The companion files `fd_vs_fd_step_stability.csv` and
`fd_vs_fd_step_stability.json` contain one row per neighboring
finite-difference step-size pair and selected gradient.

Figure 7 compares FD gradients against FD gradients from the next smaller
step size, for example `FD(h)` against `FD(h/10)`. This is a diagnostic of
the finite-difference reference itself. It does not replace the AD-vs-FD
main comparison; instead, it checks whether the exported FD gradients form a
stable plateau over a useful step-size range. Small `fd_rel_change` values
indicate such a stable FD plateau. Strongly increasing `fd_rel_change` at
small `h` indicates that cancellation, rounding, or solver-tolerance noise can
dominate the finite-difference signal. The shunt-Q case is expected to be more
sensitive at small step sizes because the corresponding gradient is very small.

This Figure 7 analysis is a pure re-analysis of the existing
`fd_step_study.csv` artifact. No new power-flow solves, AD gradients, or
finite-difference runs are performed.

## Figure 8

Files: `fig08_fd_step_study_detailed.png` and `fig08_fd_step_study_detailed.pdf`.
Data source: `fd_step_study_detailed.csv` (generated by the experiment script).
Extends Figure 4 with a finer logarithmic step-size grid from h = 1e0 to
h = 1e-10 (11 steps, 3 representative gradients, 33 rows). Each curve shows
the relative AD-vs-FD error versus FD step size. This figure is only generated
when `fd_step_study_detailed.csv` is present in the results directory.

## Figure 9

Files: `fig09_fd_vs_fd_step_stability_detailed.png` and
`fig09_fd_vs_fd_step_stability_detailed.pdf`.
Data source: `fd_step_study_detailed.csv` (via the FD-vs-FD diagnostic).
The companion files `fd_vs_fd_step_stability_detailed.csv` and
`fd_vs_fd_step_stability_detailed.json` contain one row per neighbouring
step-size pair (3 gradients x 10 pairs = 30 rows).

Figure 9 extends Figure 7 to the full detailed step-size range. It compares
`FD(h)` against `FD(h/10)` for h in {{1e0, ..., 1e-1}}. Strongly increasing
`fd_rel_change` at small h is a sign of finite-difference instability due to
floating-point cancellation. The shunt-Q case is most sensitive. This analysis
does not replace the AD-vs-FD main comparison in Figure 8.
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
    comparison_summary = build_gradient_magnitude_error_comparison_table(gradient_df)
    fd_stability = compute_fd_vs_fd_step_stability(fd_step_df)

    outputs: list[Path] = []
    outputs.extend(plot_parity(gradient_df, target_dir))
    outputs.extend(plot_global_parity(gradient_df, target_dir))
    outputs.extend(plot_error_heatmap(gradient_df, target_dir))
    outputs.extend(plot_error_boxplot(gradient_df, target_dir))
    outputs.extend(plot_fd_step_study(fd_step_df, target_dir))
    outputs.extend(plot_error_by_scenario(error_summary_df, target_dir))
    outputs.extend(write_gradient_magnitude_error_summary(comparison_summary, target_dir))
    outputs.extend(
        plot_gradient_magnitude_vs_relative_error_heatmaps(
            comparison_summary,
            target_dir,
        )
    )
    outputs.extend(write_fd_vs_fd_step_stability(fd_stability, target_dir))
    outputs.extend(plot_fd_vs_fd_step_stability(fd_stability, target_dir))

    detailed_path = results_dir / "fd_step_study_detailed.csv"
    if detailed_path.exists():
        fd_step_detailed_df = load_csv(detailed_path)
        fd_stability_detailed = compute_fd_vs_fd_step_stability(fd_step_detailed_df)
        outputs.extend(write_fd_vs_fd_step_stability_detailed(fd_stability_detailed, target_dir))
        outputs.extend(plot_fd_step_study_detailed(fd_step_detailed_df, target_dir))
        outputs.extend(plot_fd_vs_fd_step_stability_detailed(fd_stability_detailed, target_dir))

    outputs.append(write_figures_readme(target_dir, results_dir))
    return outputs


def main() -> None:
    outputs = generate_figures(RESULTS_DIR, FIGURES_DIR)
    print(f"Figures directory: {FIGURES_DIR}")
    for path in outputs:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
