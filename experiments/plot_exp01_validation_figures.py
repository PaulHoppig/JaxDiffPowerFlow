"""Create Experiment 1 validation-stability figures from existing artifacts.

This script reads only the already exported CSV files in
``experiments/results/exp01_example_simple_validation``. It does not run new
power-flow solves, does not call pandapower, and does not modify the numerical
JAX core.

Run:
    python experiments/plot_exp01_validation_figures.py
"""

from __future__ import annotations

from pathlib import Path
from textwrap import fill

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

ERROR_METRICS = {
    "max_vm_pu_abs_diff": {
        "label": "Max voltage magnitude error",
        "unit": "m.p.u.",
        "scale": 1e3,
    },
    "max_va_degree_abs_diff": {
        "label": "Max voltage angle error",
        "unit": "mdeg",
        "scale": 1e3,
    },
    "p_slack_mw_abs_diff": {
        "label": "Slack active-power error",
        "unit": "kW",
        "scale": 1e3,
    },
    "q_slack_mvar_abs_diff": {
        "label": "Slack reactive-power error",
        "unit": "kVAr",
        "scale": 1e3,
    },
    "total_p_loss_mw_abs_diff": {
        "label": "Total active-loss error",
        "unit": "kW",
        "scale": 1e3,
    },
    "total_q_loss_mvar_abs_diff": {
        "label": "Total reactive-loss error",
        "unit": "kVAr",
        "scale": 1e3,
    },
}

SUMMARY_REQUIRED_COLUMNS = {"scenario", "reference_mode", *ERROR_METRICS.keys()}
LONG_TABLE_COLUMNS = (
    "scenario",
    "scenario_order",
    "metric_key",
    "metric_label",
    "raw_value",
    "display_value",
    "display_unit",
    "reference_mode",
)


def load_validation_summary(result_dir: Path) -> pd.DataFrame:
    """Load the existing Experiment 1 validation summary CSV artifact."""

    path = result_dir / "validation_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Required CSV artifact not found: {path}")
    df = pd.read_csv(path)
    _require_columns(df, SUMMARY_REQUIRED_COLUMNS, "validation_summary.csv")
    return df


def filter_scope_matched(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only strict scope-matched validation rows."""

    _require_columns(df, {"reference_mode"}, "validation_summary.csv")
    filtered = df.loc[df["reference_mode"] == "scope_matched"].copy()

    if "strict_validation" in filtered.columns:
        strict = filtered["strict_validation"]
        if strict.dtype == bool:
            strict_mask = strict
        else:
            strict_mask = strict.astype(str).str.lower().isin({"true", "1", "yes"})
        filtered = filtered.loc[strict_mask].copy()

    return filtered.reset_index(drop=True)


def build_error_long_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build a tidy error table with one row per scenario and metric."""

    _require_columns(df, SUMMARY_REQUIRED_COLUMNS, "validation_summary.csv")
    selected = _order_scope_matched_rows(df)

    rows: list[dict] = []
    for scenario_order, row in enumerate(selected.itertuples(index=False), start=1):
        row_data = row._asdict()
        for metric_key, meta in ERROR_METRICS.items():
            raw_value = float(row_data[metric_key])
            rows.append(
                {
                    "scenario": str(row_data["scenario"]),
                    "scenario_order": scenario_order,
                    "metric_key": metric_key,
                    "metric_label": str(meta["label"]),
                    "raw_value": raw_value,
                    "display_value": raw_value * float(meta["scale"]),
                    "display_unit": str(meta["unit"]),
                    "reference_mode": str(row_data["reference_mode"]),
                }
            )

    return pd.DataFrame(rows, columns=LONG_TABLE_COLUMNS)


def build_error_stability_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize stability of each error metric over scenarios."""

    _require_columns(
        long_df,
        {"metric_key", "raw_value", "display_value"},
        "scope_matched_error_long_table",
    )

    rows: list[dict] = []
    for metric_key, group in long_df.groupby("metric_key", sort=False):
        raw = pd.to_numeric(group["raw_value"], errors="coerce").to_numpy(float)
        display = pd.to_numeric(group["display_value"], errors="coerce").to_numpy(float)
        raw = raw[np.isfinite(raw)]
        display = display[np.isfinite(display)]
        first = group.iloc[0]

        mean_raw = _mean(raw)
        std_raw = _std(raw)
        mean_display = _mean(display)
        std_display = _std(display)

        rows.append(
            {
                "metric_key": metric_key,
                "metric_label": first.get("metric_label", metric_key),
                "display_unit": first.get("display_unit", ""),
                "n": int(raw.size),
                "mean_raw": mean_raw,
                "std_raw": std_raw,
                "min_raw": _min(raw),
                "max_raw": _max(raw),
                "range_raw": _range(raw),
                "coefficient_of_variation_raw": _coefficient_of_variation(
                    mean_raw, std_raw
                ),
                "mean_display": mean_display,
                "std_display": std_display,
                "min_display": _min(display),
                "max_display": _max(display),
                "range_display": _range(display),
                "coefficient_of_variation_display": _coefficient_of_variation(
                    mean_display, std_display
                ),
            }
        )

    return pd.DataFrame(rows)


def plot_error_by_scenario(
    long_df: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """Figure 1: scenario-wise error values with per-metric means."""

    _require_columns(long_df, set(LONG_TABLE_COLUMNS), "scope_matched_error_long_table")
    summary = build_error_stability_summary(long_df)

    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.2), sharex=False)
    flat_axes = axes.ravel()
    x = np.arange(len(SCENARIO_ORDER))
    labels = [SCENARIO_LABELS[name] for name in SCENARIO_ORDER]

    for ax, metric_key in zip(flat_axes, ERROR_METRICS):
        subset = _metric_rows(long_df, metric_key)
        values = subset["display_value"].to_numpy(float)
        unit = str(subset["display_unit"].iloc[0])
        metric_label = str(subset["metric_label"].iloc[0])
        stat = summary.loc[summary["metric_key"] == metric_key].iloc[0]

        ax.plot(x, values, marker="o", markersize=4.5, linewidth=1.2)
        ax.axhline(
            float(stat["mean_display"]),
            color="black",
            linestyle="--",
            linewidth=0.9,
            alpha=0.65,
        )
        ax.set_title(fill(metric_label, width=30), fontsize=10)
        ax.set_ylabel(unit)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        cv = _format_cv(float(stat["coefficient_of_variation_display"]))
        ax.text(
            0.03,
            0.95,
            f"mean = {float(stat['mean_display']):.3g} {unit}\ncv = {cv}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
        )

    fig.suptitle(
        "Experiment 1: scope-matched validation errors across scenarios",
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    return list(_save_figure(fig, output_dir, "fig01_scope_matched_error_by_scenario"))


def plot_error_boxplots(
    long_df: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """Figure 2: per-metric boxplots with individual scenario points."""

    _require_columns(long_df, set(LONG_TABLE_COLUMNS), "scope_matched_error_long_table")
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.8), sharex=False)
    flat_axes = axes.ravel()

    for ax, metric_key in zip(flat_axes, ERROR_METRICS):
        subset = _metric_rows(long_df, metric_key)
        values = subset["display_value"].to_numpy(float)
        unit = str(subset["display_unit"].iloc[0])
        metric_label = str(subset["metric_label"].iloc[0])

        try:
            ax.boxplot([values], tick_labels=[""])
        except TypeError:
            ax.boxplot([values], labels=[""])
        jitter = np.linspace(-0.055, 0.055, len(values))
        ax.scatter(
            np.ones(len(values)) + jitter,
            values,
            s=28,
            color="black",
            alpha=0.7,
            zorder=3,
        )
        ax.set_title(fill(metric_label, width=28), fontsize=10)
        ax.set_ylabel(unit)
        ax.set_xlim(0.75, 1.25)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "Experiment 1: distribution of scope-matched validation errors",
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    return list(_save_figure(fig, output_dir, "fig02_scope_matched_error_boxplots"))


def write_summary_tables(summary_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Write the error stability summary table as CSV and JSON."""

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "scope_matched_error_stability_summary.csv"
    json_path = output_dir / "scope_matched_error_stability_summary.json"
    summary_df.to_csv(csv_path, index=False)
    summary_df.to_json(json_path, orient="records", indent=2)
    return [csv_path, json_path]


def write_long_table(long_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Write the tidy long error table as CSV and JSON."""

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "scope_matched_error_long_table.csv"
    json_path = output_dir / "scope_matched_error_long_table.json"
    long_df.to_csv(csv_path, index=False)
    long_df.to_json(json_path, orient="records", indent=2)
    return [csv_path, json_path]


def write_figures_readme(output_dir: Path) -> Path:
    """Write a concise description of figure sources, filters, and limits."""

    output_dir.mkdir(parents=True, exist_ok=True)
    metric_lines = "\n".join(f"- `{name}`" for name in ERROR_METRICS)
    text = f"""# Experiment 1 Validation Figures

These figures are generated only from the existing
`validation_summary.csv` artifact in
`{RESULTS_DIR.as_posix()}`. The plotting script does not run new power-flow
solves, does not call pandapower, and does not modify the numerical core.

## Filter

Rows are filtered with `reference_mode == "scope_matched"`. If the
`strict_validation` column is present, only rows with `strict_validation == True`
are used.

## Error Metrics

{metric_lines}

## Unit Conversion

`max_vm_pu_abs_diff` is multiplied by 1000 and shown in m.p.u.
`max_va_degree_abs_diff` is multiplied by 1000 and shown in mdeg. MW and MVAr
error metrics are multiplied by 1000 and shown in kW or kVAr.

## Figure 1

Files: `fig01_scope_matched_error_by_scenario.png` and
`fig01_scope_matched_error_by_scenario.pdf`. This figure shows whether the
error metrics stay stable over the seven operating scenarios. After the
2026-05-19 transformer magnetization-stamp correction, active-power errors are
expected to be near the few-watt numerical-noise range rather than the former
systematic 14-kW offset.

## Figure 2

Files: `fig02_scope_matched_error_boxplots.png` and
`fig02_scope_matched_error_boxplots.pdf`. This figure is descriptive. It shows
median, spread, and the individual scenario values for each metric, but it is
not statistical proof of full pandapower parity.

## Limitations

The evidence is descriptive only. No new solves, no new validation logic, no
optimization, no gradient computation, and no numerical-core changes are part
of this plotting pipeline.
"""
    path = output_dir / "README.md"
    path.write_text(text, encoding="utf-8")
    return path


def generate_figures(
    results_dir: Path = RESULTS_DIR,
    figures_dir: Path | None = None,
) -> list[Path]:
    """Generate all Experiment 1 plot artifacts from existing CSV data."""

    target_dir = figures_dir if figures_dir is not None else results_dir / "figures"
    summary = load_validation_summary(results_dir)
    filtered = filter_scope_matched(summary)
    long_df = build_error_long_table(filtered)
    stability = build_error_stability_summary(long_df)

    outputs: list[Path] = []
    outputs.extend(write_long_table(long_df, target_dir))
    outputs.extend(write_summary_tables(stability, target_dir))
    outputs.extend(plot_error_by_scenario(long_df, target_dir))
    outputs.extend(plot_error_boxplots(long_df, target_dir))
    outputs.append(write_figures_readme(target_dir))
    return outputs


def main() -> None:
    outputs = generate_figures(RESULTS_DIR, FIGURES_DIR)
    print(f"Figures directory: {FIGURES_DIR}")
    for path in outputs:
        print(f"  {path.name}")


def _require_columns(df: pd.DataFrame, required: set[str], table_name: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")


def _order_scope_matched_rows(df: pd.DataFrame) -> pd.DataFrame:
    missing = [name for name in SCENARIO_ORDER if name not in set(df["scenario"])]
    if missing:
        raise ValueError(f"Missing required Experiment 1 scenarios: {missing}")

    selected = df.loc[df["scenario"].isin(SCENARIO_ORDER)].copy()
    selected["scenario"] = pd.Categorical(
        selected["scenario"],
        categories=SCENARIO_ORDER,
        ordered=True,
    )
    selected = selected.sort_values("scenario")
    if selected["scenario"].duplicated().any():
        duplicates = selected.loc[selected["scenario"].duplicated(), "scenario"].tolist()
        raise ValueError(f"Duplicate scope-matched scenario rows: {duplicates}")
    return selected.reset_index(drop=True)


def _metric_rows(long_df: pd.DataFrame, metric_key: str) -> pd.DataFrame:
    subset = long_df.loc[long_df["metric_key"] == metric_key].copy()
    subset = subset.sort_values("scenario_order")
    if subset.empty:
        raise ValueError(f"No rows available for metric {metric_key!r}.")
    return subset


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


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else float("nan")


def _std(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1)) if values.size > 1 else float("nan")


def _min(values: np.ndarray) -> float:
    return float(np.min(values)) if values.size else float("nan")


def _max(values: np.ndarray) -> float:
    return float(np.max(values)) if values.size else float("nan")


def _range(values: np.ndarray) -> float:
    return float(np.max(values) - np.min(values)) if values.size else float("nan")


def _coefficient_of_variation(mean: float, std: float) -> float:
    if not np.isfinite(mean) or not np.isfinite(std) or np.isclose(mean, 0.0):
        return float("nan")
    return float(std / abs(mean))


def _format_cv(value: float) -> str:
    return "nan" if not np.isfinite(value) else f"{value:.2e}"


if __name__ == "__main__":
    main()
