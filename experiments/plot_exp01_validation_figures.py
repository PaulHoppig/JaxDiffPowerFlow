"""Create final Experiment 1 validation figures from existing artifacts.

The pipeline reads the already exported Experiment 1 CSV files only. It does
not run power-flow solves, does not call pandapower, and does not touch the
numerical core.

Run:
    python experiments/plot_exp01_validation_figures.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter


RESULTS_DIR = (
    Path(__file__).resolve().parent / "results" / "exp01_example_simple_validation"
)
FIGURES_DIR = RESULTS_DIR / "figures"

SCENARIO_ORDER = (
    "base",
    "load_low",
    "load_high",
    "sgen_low",
    "sgen_high",
    "combined_high_load_low_sgen",
    "combined_low_load_high_sgen",
)

SCENARIO_LABELS = {
    "base": "Base",
    "load_low": "Load low",
    "load_high": "Load high",
    "sgen_low": "sgen low",
    "sgen_high": "sgen high",
    "combined_high_load_low_sgen": "High load\nlow sgen",
    "combined_low_load_high_sgen": "Low load\nhigh sgen",
}


@dataclass(frozen=True)
class ErrorMetric:
    key: str
    quantity: str
    metric: str
    unit: str
    panel_title: str
    heatmap_label: str


ERROR_METRICS: tuple[ErrorMetric, ...] = (
    ErrorMetric(
        "max_vm_pu_abs_diff",
        "Voltage magnitude",
        "max |Delta |V||",
        "p.u.",
        "Voltage magnitude error [p.u.]",
        "|Delta |V|| [p.u.]",
    ),
    ErrorMetric(
        "max_va_degree_abs_diff",
        "Voltage angle",
        "max |Delta theta|",
        "deg",
        "Voltage angle error [deg]",
        "|Delta theta| [deg]",
    ),
    ErrorMetric(
        "p_slack_mw_abs_diff",
        "Slack active power",
        "|Delta P_slack|",
        "MW",
        "Slack active-power error [MW]",
        "|Delta P_slack| [MW]",
    ),
    ErrorMetric(
        "q_slack_mvar_abs_diff",
        "Slack reactive power",
        "|Delta Q_slack|",
        "MVAr",
        "Slack reactive-power error [MVAr]",
        "|Delta Q_slack| [MVAr]",
    ),
    ErrorMetric(
        "total_p_loss_mw_abs_diff",
        "Total active losses",
        "|Delta P_loss|",
        "MW",
        "Total active-loss error [MW]",
        "|Delta P_loss| [MW]",
    ),
    ErrorMetric(
        "total_q_loss_mvar_abs_diff",
        "Total reactive losses",
        "|Delta Q_loss|",
        "MVAr",
        "Total reactive-loss error [MVAr]",
        "|Delta Q_loss| [MVAr]",
    ),
    ErrorMetric(
        "trafo_pl_mw_abs_diff",
        "Transformer active losses",
        "|Delta P_trafo|",
        "MW",
        "Transformer active-loss error [MW]",
        "|Delta P_trafo| [MW]",
    ),
    ErrorMetric(
        "trafo_ql_mvar_abs_diff",
        "Transformer reactive losses",
        "|Delta Q_trafo|",
        "MVAr",
        "Transformer reactive-loss error [MVAr]",
        "|Delta Q_trafo| [MVAr]",
    ),
)

OLD_FINAL_FIGURE_FILES = (
    "fig01_scope_matched_error_by_scenario.png",
    "fig01_scope_matched_error_by_scenario.pdf",
    "fig02_scope_matched_error_boxplots.png",
    "fig02_scope_matched_error_boxplots.pdf",
    "scope_matched_error_long_table.csv",
    "scope_matched_error_long_table.json",
    "scope_matched_error_stability_summary.csv",
    "scope_matched_error_stability_summary.json",
)

EXPECTED_OUTPUTS = (
    "fig01_final_max_errors_table.png",
    "fig01_final_max_errors_table.pdf",
    "final_max_errors_table.csv",
    "final_max_errors_table.md",
    "fig02_scope_matched_error_dotplot.png",
    "fig02_scope_matched_error_dotplot.pdf",
    "fig03_scope_matched_error_heatmap_log10.png",
    "fig03_scope_matched_error_heatmap_log10.pdf",
    "model_alignment_error_reduction.csv",
    "model_alignment_error_reduction.json",
    "fig04_model_alignment_error_reduction_power.png",
    "fig04_model_alignment_error_reduction_power.pdf",
    "fig05_model_alignment_diagnostic_reduction.png",
    "fig05_model_alignment_diagnostic_reduction.pdf",
    "README.md",
)

MODEL_STEP_LABELS = {
    "initial_scope_matched_before_trafo_fix": "Initial",
    "after_trafo_magnetization_fix": "After trafo fix",
    "final_after_open_line_policy": "Final",
}

POWER_REDUCTION_METRICS = (
    "p_slack_mw_abs_diff",
    "total_p_loss_mw_abs_diff",
    "trafo_pl_mw_abs_diff",
)

DIAGNOSTIC_REDUCTION_METRICS = (
    "max_vm_pu_abs_diff",
    "max_va_degree_abs_diff",
    "ybus_max_abs_complex_diff",
    "diffpf_residual_at_pandapower_solution",
)

ALIGNMENT_METRIC_LABELS = {
    "max_vm_pu_abs_diff": ("Voltage magnitude", "max |Delta |V||", "p.u."),
    "max_va_degree_abs_diff": ("Voltage angle", "max |Delta theta|", "deg"),
    "p_slack_mw_abs_diff": ("Slack active power", "|Delta P_slack|", "MW"),
    "total_p_loss_mw_abs_diff": ("Total active losses", "|Delta P_loss|", "MW"),
    "trafo_pl_mw_abs_diff": ("Transformer active losses", "|Delta P_trafo|", "MW"),
    "ybus_max_abs_complex_diff": ("Y-bus parity", "max |Delta Ybus|", "-"),
    "diffpf_residual_at_pandapower_solution": (
        "Cross residual",
        "||r_diffpf(V_pp)||",
        "-",
    ),
}

MODEL_ALIGNMENT_SOURCE_NOTES = {
    "initial_scope_matched_before_trafo_fix": (
        "documented representative offset from transformer magnetization "
        "ablation/diagnosis; archived exact maxima not found in current artifacts"
    ),
    "after_trafo_magnetization_fix": (
        "documented maximum after transformer magnetization stamp correction "
        "before open-line policy"
    ),
    "final_after_open_line_policy": (
        "documented final maximum after scope-matched open-line policy"
    ),
}

MODEL_ALIGNMENT_VALUES = {
    "initial_scope_matched_before_trafo_fix": {
        "p_slack_mw_abs_diff": 1.4364137e-02,
        "total_p_loss_mw_abs_diff": 1.4364137e-02,
        "trafo_pl_mw_abs_diff": 1.4374564e-02,
    },
    "after_trafo_magnetization_fix": {
        "max_vm_pu_abs_diff": 2.350559167396682e-05,
        "max_va_degree_abs_diff": 2.527126e-04,
        "p_slack_mw_abs_diff": 5.776101701826519e-06,
        "total_p_loss_mw_abs_diff": 5.7759869437762346e-06,
        "trafo_pl_mw_abs_diff": 4.70052885237493e-06,
        "ybus_max_abs_complex_diff": 4.178332435860909e-03,
        "diffpf_residual_at_pandapower_solution": 4.290310889485325e-03,
    },
    "final_after_open_line_policy": {
        "max_vm_pu_abs_diff": 4.773959e-14,
        "max_va_degree_abs_diff": 2.188472e-12,
        "p_slack_mw_abs_diff": 1.218972e-10,
        "total_p_loss_mw_abs_diff": 1.106781e-12,
        "trafo_pl_mw_abs_diff": 3.108624e-14,
        "ybus_max_abs_complex_diff": 0.0,
        "diffpf_residual_at_pandapower_solution": 2.650863e-10,
    },
}


def load_scope_matched_errors(results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    """Load scope-matched errors in base units from existing artifacts."""

    validation = pd.read_csv(results_dir / "validation_summary.csv")
    _require_columns(
        validation,
        {
            "scenario",
            "reference_mode",
            "strict_validation",
            "max_vm_pu_abs_diff",
            "max_va_degree_abs_diff",
            "p_slack_mw_abs_diff",
            "q_slack_mvar_abs_diff",
            "total_p_loss_mw_abs_diff",
            "total_q_loss_mvar_abs_diff",
        },
        "validation_summary.csv",
    )
    scope = validation[validation["reference_mode"] == "scope_matched"].copy()
    strict = scope["strict_validation"]
    if strict.dtype == bool:
        scope = scope[strict].copy()
    else:
        scope = scope[strict.astype(str).str.lower().isin({"true", "1", "yes"})].copy()

    trafo = pd.read_csv(results_dir / "trafo_flows.csv")
    _require_columns(
        trafo,
        {
            "scenario",
            "reference_mode",
            "pl_mw_abs_diff",
            "q_hv_mvar_diffpf",
            "q_lv_mvar_diffpf",
            "q_hv_mvar_pp",
            "q_lv_mvar_pp",
        },
        "trafo_flows.csv",
    )
    trafo_scope = trafo[trafo["reference_mode"] == "scope_matched"].copy()
    trafo_scope["trafo_ql_mvar_abs_diff"] = (
        (trafo_scope["q_hv_mvar_diffpf"] + trafo_scope["q_lv_mvar_diffpf"])
        - (trafo_scope["q_hv_mvar_pp"] + trafo_scope["q_lv_mvar_pp"])
    ).abs()
    trafo_metrics = (
        trafo_scope.groupby("scenario", as_index=False)
        .agg(
            trafo_pl_mw_abs_diff=("pl_mw_abs_diff", "max"),
            trafo_ql_mvar_abs_diff=("trafo_ql_mvar_abs_diff", "max"),
        )
    )

    errors = scope.merge(trafo_metrics, on="scenario", how="left")
    errors = _order_scenarios(errors)
    required_metric_keys = {metric.key for metric in ERROR_METRICS}
    _require_columns(errors, required_metric_keys, "scope_matched_errors")
    return errors


def build_final_max_errors_table(errors: pd.DataFrame) -> pd.DataFrame:
    """Build final maximum-error table over all scope-matched scenarios."""

    _require_columns(errors, {"scenario", *(metric.key for metric in ERROR_METRICS)}, "errors")
    rows = []
    for metric in ERROR_METRICS:
        values = pd.to_numeric(errors[metric.key], errors="coerce")
        idx = int(values.idxmax())
        rows.append(
            {
                "quantity": metric.quantity,
                "metric": metric.metric,
                "unit": metric.unit,
                "max_abs_error": float(values.loc[idx]),
                "worst_case_scenario": str(errors.loc[idx, "scenario"]),
            }
        )
    return pd.DataFrame(rows)


def export_final_max_errors_table(
    table: pd.DataFrame,
    output_dir: Path = FIGURES_DIR,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    formatted = table.copy()
    formatted["max_abs_error"] = formatted["max_abs_error"].map(format_e)

    csv_path = output_dir / "final_max_errors_table.csv"
    md_path = output_dir / "final_max_errors_table.md"
    formatted.to_csv(csv_path, index=False)
    md_path.write_text(dataframe_to_markdown(formatted), encoding="utf-8")

    png_path, pdf_path = plot_final_max_errors_table(table, output_dir)
    return [png_path, pdf_path, csv_path, md_path]


def plot_final_max_errors_table(
    table: pd.DataFrame,
    output_dir: Path = FIGURES_DIR,
) -> tuple[Path, Path]:
    """Render the maximum-error table as PNG/PDF."""

    output_dir.mkdir(parents=True, exist_ok=True)
    display = table.copy()
    display["max_abs_error"] = display["max_abs_error"].map(format_e)
    display.columns = [
        "Quantity",
        "Metric",
        "Unit",
        "Max abs. error",
        "Worst-case scenario",
    ]

    fig, ax = plt.subplots(figsize=(12.0, 4.8))
    ax.axis("off")
    table_artist = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=[0.22, 0.20, 0.08, 0.17, 0.25],
    )
    table_artist.auto_set_font_size(False)
    table_artist.set_fontsize(9.0)
    table_artist.scale(1.0, 1.45)
    for (row, _col), cell in table_artist.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#2f3b52")
        else:
            cell.set_facecolor("#f6f8fb" if row % 2 == 0 else "white")
        cell.set_edgecolor("#d0d5dd")

    fig.suptitle(
        "Experiment 1: final scope-matched maximum validation errors",
        fontsize=13,
        fontweight="bold",
        y=0.95,
    )
    return _save_figure(fig, output_dir, "fig01_final_max_errors_table")


def plot_scope_matched_error_dotplot(
    errors: pd.DataFrame,
    output_dir: Path = FIGURES_DIR,
) -> tuple[Path, Path]:
    """Create a faceted dotplot of final errors by scenario."""

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 2, figsize=(12.0, 12.0), sharex=True)
    x = np.arange(len(SCENARIO_ORDER))
    labels = [SCENARIO_LABELS[scenario] for scenario in SCENARIO_ORDER]

    for ax, metric in zip(axes.ravel(), ERROR_METRICS):
        values = pd.to_numeric(errors[metric.key], errors="coerce").to_numpy(float)
        positive = np.where(values > 0.0, values, np.nan)
        ax.scatter(x, positive, s=42, color="#2f6f9f", edgecolor="white", linewidth=0.7)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(scientific_tick))
        ax.set_title(metric.panel_title, fontsize=10.5)
        ax.set_ylabel("Absolute error")
        ax.grid(True, axis="y", which="both", alpha=0.28)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")

    fig.suptitle(
        "Experiment 1: final scope-matched errors by scenario",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))
    return _save_figure(fig, output_dir, "fig02_scope_matched_error_dotplot")


def compute_log10_errors(values: pd.DataFrame, eps: float = 1e-300) -> pd.DataFrame:
    """Return log10(abs(error)) with a floor for exact zeros.

    The floor avoids ``-inf`` in the heatmap while preserving all non-zero
    values. Tile annotations still show the original unmodified values.
    """

    numeric = values.apply(pd.to_numeric, errors="coerce").astype(float)
    clipped = numeric.abs().clip(lower=eps)
    return np.log10(clipped)


def plot_scope_matched_error_heatmap(
    errors: pd.DataFrame,
    output_dir: Path = FIGURES_DIR,
) -> tuple[Path, Path]:
    """Create a scenarios x metrics heatmap of log10 absolute errors."""

    output_dir.mkdir(parents=True, exist_ok=True)
    metric_keys = [metric.key for metric in ERROR_METRICS]
    heatmap_values = errors[metric_keys].copy()
    log_values = compute_log10_errors(heatmap_values)
    labels_x = [metric.heatmap_label for metric in ERROR_METRICS]
    labels_y = [SCENARIO_LABELS[scenario].replace("\n", " ") for scenario in errors["scenario"]]

    fig, ax = plt.subplots(figsize=(13.0, 6.2))
    im = ax.imshow(log_values.to_numpy(float), aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(labels_x)))
    ax.set_xticklabels(labels_x, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(labels_y)))
    ax.set_yticklabels(labels_y)
    ax.set_xlabel("Error metric")
    ax.set_ylabel("Scenario")
    ax.set_title(
        "Experiment 1: log10 absolute validation errors",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )

    raw_values = heatmap_values.to_numpy(float)
    for row in range(raw_values.shape[0]):
        for col in range(raw_values.shape[1]):
            ax.text(
                col,
                row,
                format_e(raw_values[row, col]),
                ha="center",
                va="center",
                color="white",
                fontsize=7.5,
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log10(|Delta|)")
    fig.tight_layout()
    return _save_figure(fig, output_dir, "fig03_scope_matched_error_heatmap_log10")


def build_model_alignment_error_reduction_table() -> pd.DataFrame:
    """Build documented model-alignment error-reduction data.

    Step 0 uses documented representative offsets because the main Exp.-1
    artifacts have been regenerated for the final model state.
    """

    rows = []
    for model_step, metric_values in MODEL_ALIGNMENT_VALUES.items():
        for metric, abs_error in metric_values.items():
            quantity, _metric_label, unit = ALIGNMENT_METRIC_LABELS[metric]
            rows.append(
                {
                    "model_step": model_step,
                    "model_step_label": MODEL_STEP_LABELS[model_step],
                    "metric": metric,
                    "quantity": quantity,
                    "unit": unit,
                    "abs_error": float(abs_error),
                    "source_note": MODEL_ALIGNMENT_SOURCE_NOTES[model_step],
                }
            )
    return pd.DataFrame(rows)


def export_model_alignment_error_reduction(
    table: pd.DataFrame,
    output_dir: Path = FIGURES_DIR,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    formatted = table.copy()
    formatted["abs_error"] = formatted["abs_error"].map(format_e)

    csv_path = output_dir / "model_alignment_error_reduction.csv"
    json_path = output_dir / "model_alignment_error_reduction.json"
    formatted.to_csv(csv_path, index=False)
    formatted.to_json(json_path, orient="records", indent=2)
    return [csv_path, json_path]


def plot_model_alignment_power_reduction(
    table: pd.DataFrame,
    output_dir: Path = FIGURES_DIR,
) -> tuple[Path, Path]:
    """Create the grouped log bar chart for active-power error reduction."""

    steps = (
        "initial_scope_matched_before_trafo_fix",
        "after_trafo_magnetization_fix",
        "final_after_open_line_policy",
    )
    metric_labels = {
        "p_slack_mw_abs_diff": "|Delta P_slack|",
        "total_p_loss_mw_abs_diff": "|Delta P_loss|",
        "trafo_pl_mw_abs_diff": "|Delta P_trafo|",
    }
    return _plot_grouped_log_bars(
        table=table,
        metrics=POWER_REDUCTION_METRICS,
        metric_labels=metric_labels,
        steps=steps,
        title="Experiment 1: error reduction by model alignment",
        subtitle="scope-matched pandapower reference vs. diffpf",
        ylabel="absolute error [MW]",
        stem="fig04_model_alignment_error_reduction_power",
        output_dir=output_dir,
    )


def plot_model_alignment_diagnostic_reduction(
    table: pd.DataFrame,
    output_dir: Path = FIGURES_DIR,
) -> tuple[Path, Path]:
    """Create the grouped log bar chart for residual topology diagnostics."""

    steps = (
        "after_trafo_magnetization_fix",
        "final_after_open_line_policy",
    )
    metric_labels = {
        "max_vm_pu_abs_diff": "max |Delta |V||\n[p.u.]",
        "max_va_degree_abs_diff": "max |Delta theta|\n[deg]",
        "ybus_max_abs_complex_diff": "max |Delta Ybus|\n[-]",
        "diffpf_residual_at_pandapower_solution": "cross residual\n[-]",
    }
    return _plot_grouped_log_bars(
        table=table,
        metrics=DIAGNOSTIC_REDUCTION_METRICS,
        metric_labels=metric_labels,
        steps=steps,
        title="Experiment 1: residual topology error after final model alignment",
        subtitle=None,
        ylabel="absolute error",
        stem="fig05_model_alignment_diagnostic_reduction",
        output_dir=output_dir,
    )


def _plot_grouped_log_bars(
    table: pd.DataFrame,
    metrics: tuple[str, ...],
    metric_labels: dict[str, str],
    steps: tuple[str, ...],
    title: str,
    subtitle: str | None,
    ylabel: str,
    stem: str,
    output_dir: Path,
    eps: float = 1e-300,
) -> tuple[Path, Path]:
    _require_columns(
        table,
        {"model_step", "model_step_label", "metric", "abs_error"},
        "model_alignment_error_reduction",
    )

    lookup = table.set_index(["model_step", "metric"])["abs_error"].astype(float)
    x = np.arange(len(metrics))
    width = min(0.23, 0.72 / max(len(steps), 1))
    colors = ("#2f6f9f", "#6a8f3a", "#b45f3c")

    fig, ax = plt.subplots(figsize=(10.8, 5.7))
    for step_idx, step in enumerate(steps):
        offset = (step_idx - (len(steps) - 1) / 2.0) * width
        values = np.array(
            [float(lookup.loc[(step, metric)]) for metric in metrics],
            dtype=float,
        )
        plot_values = np.where(values > 0.0, values, eps)
        bars = ax.bar(
            x + offset,
            plot_values,
            width=width,
            label=MODEL_STEP_LABELS[step],
            color=colors[step_idx % len(colors)],
            edgecolor="white",
            linewidth=0.7,
        )
        for bar, raw_value in zip(bars, values):
            ax.annotate(
                format_e(raw_value),
                xy=(bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=8,
            )

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(scientific_tick))
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([metric_labels[metric] for metric in metrics])
    ax.grid(True, axis="y", which="both", alpha=0.25)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title(title, fontsize=13.5, fontweight="bold", pad=18)
    if subtitle:
        ax.text(
            0.5,
            1.01,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=10,
            color="#4b5563",
        )
    fig.tight_layout()
    return _save_figure(fig, output_dir, stem)


def write_figures_readme(output_dir: Path = FIGURES_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    text = """# Experiment 1 Final Validation Figures

These final figures are generated from the already updated Experiment 1
artifacts. The data source is `validation_summary.csv`, filtered to
`reference_mode == "scope_matched"`. Transformer-loss metrics are read from the
existing `trafo_flows.csv` artifact and joined by scenario.

## Figures

`fig01_final_max_errors_table.png/pdf` presents the maximum absolute error over
all seven scope-matched scenarios for each validation quantity. The same table
is exported as `final_max_errors_table.csv` and `final_max_errors_table.md`.

`fig02_scope_matched_error_dotplot.png/pdf` shows scenario-wise absolute errors
as points on logarithmic y-axes. Scenarios are categorical and are therefore not
connected by lines.

`fig03_scope_matched_error_heatmap_log10.png/pdf` shows scenarios by error
metrics. Tile color encodes `log10(|Delta|)`, while tile annotations show the
original absolute error in scientific notation. Exact zero values, if present,
are floored at `1e-300` only for the logarithmic color calculation.

`fig04_model_alignment_error_reduction_power.png/pdf` is a grouped logarithmic
bar chart showing the reduction of Slack active-power error, total active-loss
error, and transformer active-loss error across three model states: the initial
scope-matched comparison before the transformer correction, the state after the
transformer magnetization fix, and the final state after the open-line policy.

`fig05_model_alignment_diagnostic_reduction.png/pdf` is a diagnostic grouped
logarithmic bar chart showing how the voltage error, angle error, Y-bus
difference, and cross-residual changed from the post-transformer-fix state to
the final open-line-policy state.

`model_alignment_error_reduction.csv/json` contains the numeric values used for
`fig04` and `fig05`.

## Units And Formatting

All values are shown in base units: p.u. for voltage magnitude, deg for voltage
angle, MW for active power, and MVAr for reactive power. No kW, kVAr, mdeg, or
m.p.u. conversion is used. Numeric labels and tables use scientific e-notation.
The initial model-alignment step uses documented representative offset values
from the ablation diagnosis because the main artifacts have since been updated
to the final model state.

## Interpretation

Boxplots were intentionally removed because there are only seven deterministic
scenario points. After the scope-matched topology alignment, the final errors
are in the range of numerical roundoff.

The model-alignment reduction plots show that the original deviations were
model-structural rather than solver-driven: the transformer magnetization fix
removed the dominant active-power offset, and the open-line policy removed the
remaining topology mismatch.
"""
    path = output_dir / "README.md"
    path.write_text(text, encoding="utf-8")
    return path


def cleanup_old_plot_outputs(output_dir: Path = FIGURES_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in OLD_FINAL_FIGURE_FILES:
        path = output_dir / name
        if path.exists():
            path.unlink()


def generate_figures(
    results_dir: Path = RESULTS_DIR,
    figures_dir: Path = FIGURES_DIR,
) -> list[Path]:
    """Generate all final Exp.-1 figure artifacts from existing CSV files."""

    cleanup_old_plot_outputs(figures_dir)
    errors = load_scope_matched_errors(results_dir)
    max_table = build_final_max_errors_table(errors)

    outputs: list[Path] = []
    outputs.extend(export_final_max_errors_table(max_table, figures_dir))
    outputs.extend(plot_scope_matched_error_dotplot(errors, figures_dir))
    outputs.extend(plot_scope_matched_error_heatmap(errors, figures_dir))
    alignment_table = build_model_alignment_error_reduction_table()
    outputs.extend(export_model_alignment_error_reduction(alignment_table, figures_dir))
    outputs.extend(plot_model_alignment_power_reduction(alignment_table, figures_dir))
    outputs.extend(plot_model_alignment_diagnostic_reduction(alignment_table, figures_dir))
    outputs.append(write_figures_readme(figures_dir))
    return outputs


def main() -> None:
    outputs = generate_figures(RESULTS_DIR, FIGURES_DIR)
    print(f"Figures directory: {FIGURES_DIR}")
    for path in outputs:
        print(f"  {path.name}")


def format_e(value: float) -> str:
    return f"{float(value):.3e}"


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render a simple GitHub-flavored Markdown table without tabulate."""

    columns = list(df.columns)
    rows = []
    rows.append("| " + " | ".join(columns) + " |")
    rows.append("| " + " | ".join("---" for _ in columns) + " |")
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return "\n".join(rows) + "\n"


def scientific_tick(value: float, _position: int | None = None) -> str:
    if value <= 0 or not np.isfinite(value):
        return ""
    exponent = int(np.floor(np.log10(value)))
    return f"1e{exponent}"


def _require_columns(df: pd.DataFrame, required: set[str], table_name: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")


def _order_scenarios(df: pd.DataFrame) -> pd.DataFrame:
    missing = [name for name in SCENARIO_ORDER if name not in set(df["scenario"])]
    if missing:
        raise ValueError(f"Missing required Experiment 1 scenarios: {missing}")
    ordered = df[df["scenario"].isin(SCENARIO_ORDER)].copy()
    ordered["scenario"] = pd.Categorical(
        ordered["scenario"],
        categories=SCENARIO_ORDER,
        ordered=True,
    )
    ordered = ordered.sort_values("scenario").reset_index(drop=True)
    ordered["scenario"] = ordered["scenario"].astype(str)
    return ordered


def _save_figure(
    fig,
    output_dir: Path,
    stem: str,
    dpi: int = 300,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


if __name__ == "__main__":
    main()
