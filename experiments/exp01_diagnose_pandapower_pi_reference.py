"""Experiment 1 diagnostic: explicit pandapower pi transformer reference.

This script reruns the Exp.-1 ``scope_matched`` comparison while explicitly
forcing the pandapower reference solve to use ``trafo_model="pi"``. It writes
diagnostic artifacts to a separate results directory and does not overwrite the
main Experiment 1 artifacts.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import matplotlib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import experiments.exp01_validate_example_simple as exp01

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


RESULTS_DIR = (
    Path(__file__).resolve().parent
    / "results"
    / "exp01_pandapower_pi_reference_diagnostic"
)
MAIN_EXP01_RESULTS_DIR = (
    Path(__file__).resolve().parent / "results" / "exp01_example_simple_validation"
)
REFERENCE_MODE = "scope_matched"


@dataclass(frozen=True)
class PiReferenceSummaryRow:
    scenario: str
    reference_mode: str
    pandapower_trafo_model: str
    diffpf_converged: bool
    pandapower_converged: bool
    diffpf_iterations: int
    diffpf_residual_norm: float
    max_vm_pu_abs_diff: float
    max_va_degree_abs_diff: float
    p_slack_mw_abs_diff: float
    q_slack_mvar_abs_diff: float
    total_p_loss_mw_abs_diff: float
    total_q_loss_mvar_abs_diff: float
    trafo_pl_mw_abs_diff: float
    trafo_ql_mvar_abs_diff: float


@dataclass(frozen=True)
class PiReferenceComparisonRow:
    metric: str
    previous_exp01_max: float
    pi_reference_max: float
    absolute_change: float
    reduction_factor: float
    unit: str
    interpretation: str


def pi_runpp_kwargs() -> dict:
    """Return Exp.-1 pandapower options with explicit trafo_model='pi'."""

    kwargs = dict(exp01.PP_RUNPP_KWARGS)
    kwargs["trafo_model"] = "pi"
    return kwargs


@contextmanager
def force_exp01_pi_reference() -> Iterator[None]:
    """Temporarily force the imported Exp.-1 module to use trafo_model='pi'."""

    old_kwargs = dict(exp01.PP_RUNPP_KWARGS)
    exp01.PP_RUNPP_KWARGS = pi_runpp_kwargs()
    try:
        yield
    finally:
        exp01.PP_RUNPP_KWARGS = old_kwargs


def run_pi_reference_diagnostic() -> pd.DataFrame:
    """Run all seven Exp.-1 scenarios in scope_matched mode."""

    rows: list[PiReferenceSummaryRow] = []
    with force_exp01_pi_reference():
        for scenario_name, load_factor, sgen_factor in exp01.SCENARIOS:
            print(f"  Running {scenario_name} / explicit pandapower pi ...", end=" ")
            result = exp01.run_scenario(
                scenario_name,
                load_factor,
                sgen_factor,
                REFERENCE_MODE,
            )
            if len(result.trafo_rows) != 1:
                raise RuntimeError(
                    f"Expected one active transformer, got {len(result.trafo_rows)}"
                )
            trafo = result.trafo_rows[0]
            trafo_ql_diffpf = trafo.q_hv_mvar_diffpf + trafo.q_lv_mvar_diffpf
            trafo_ql_pp = trafo.q_hv_mvar_pp + trafo.q_lv_mvar_pp
            row = PiReferenceSummaryRow(
                scenario=scenario_name,
                reference_mode=REFERENCE_MODE,
                pandapower_trafo_model="pi",
                diffpf_converged=result.summary.diffpf_converged,
                pandapower_converged=result.summary.pandapower_converged,
                diffpf_iterations=result.summary.diffpf_iterations,
                diffpf_residual_norm=result.summary.diffpf_residual_norm,
                max_vm_pu_abs_diff=result.summary.max_vm_pu_abs_diff,
                max_va_degree_abs_diff=result.summary.max_va_degree_abs_diff,
                p_slack_mw_abs_diff=result.summary.p_slack_mw_abs_diff,
                q_slack_mvar_abs_diff=result.summary.q_slack_mvar_abs_diff,
                total_p_loss_mw_abs_diff=result.summary.total_p_loss_mw_abs_diff,
                total_q_loss_mvar_abs_diff=result.summary.total_q_loss_mvar_abs_diff,
                trafo_pl_mw_abs_diff=trafo.pl_mw_abs_diff,
                trafo_ql_mvar_abs_diff=abs(trafo_ql_diffpf - trafo_ql_pp),
            )
            rows.append(row)
            print(
                "OK "
                f"max_dV={row.max_vm_pu_abs_diff:.3e} "
                f"dP_slack={row.p_slack_mw_abs_diff * 1e3:.6f} kW"
            )

    return pd.DataFrame([asdict(row) for row in rows])


def _safe_reduction_factor(previous: float, current: float) -> float:
    if not math.isfinite(previous) or not math.isfinite(current):
        return float("nan")
    if abs(current) < 1e-15:
        return float("inf")
    return previous / current


def _interpret_metric(previous: float, current: float) -> str:
    if not math.isfinite(previous) or not math.isfinite(current):
        return "not_comparable"
    if current < previous * 0.5:
        return "clearly_reduced"
    if current < previous * 0.9:
        return "slightly_reduced"
    if current <= previous * 1.1:
        return "similar"
    return "increased"


def _max_from_existing_exp01(metric: str, main_results_dir: Path) -> float:
    if metric in {"trafo_pl_mw_abs_diff", "trafo_ql_mvar_abs_diff"}:
        path = main_results_dir / "trafo_flows.csv"
        df = pd.read_csv(path)
        scope = df[df["reference_mode"] == REFERENCE_MODE].copy()
        if metric == "trafo_pl_mw_abs_diff":
            return float(scope["pl_mw_abs_diff"].max())
        q_diffpf = scope["q_hv_mvar_diffpf"] + scope["q_lv_mvar_diffpf"]
        q_pp = scope["q_hv_mvar_pp"] + scope["q_lv_mvar_pp"]
        return float((q_diffpf - q_pp).abs().max())

    path = main_results_dir / "validation_summary.csv"
    df = pd.read_csv(path)
    scope = df[df["reference_mode"] == REFERENCE_MODE]
    return float(scope[metric].max())


def build_comparison_summary(
    pi_summary: pd.DataFrame,
    main_results_dir: Path = MAIN_EXP01_RESULTS_DIR,
) -> pd.DataFrame:
    """Compare explicit-pi diagnostic maxima to current Exp.-1 artifacts."""

    metric_units = {
        "max_vm_pu_abs_diff": "p.u.",
        "max_va_degree_abs_diff": "deg",
        "p_slack_mw_abs_diff": "MW",
        "q_slack_mvar_abs_diff": "MVAr",
        "total_p_loss_mw_abs_diff": "MW",
        "total_q_loss_mvar_abs_diff": "MVAr",
        "trafo_pl_mw_abs_diff": "MW",
        "trafo_ql_mvar_abs_diff": "MVAr",
    }

    rows: list[PiReferenceComparisonRow] = []
    for metric, unit in metric_units.items():
        previous = _max_from_existing_exp01(metric, main_results_dir)
        current = float(pi_summary[metric].max())
        rows.append(
            PiReferenceComparisonRow(
                metric=metric,
                previous_exp01_max=previous,
                pi_reference_max=current,
                absolute_change=current - previous,
                reduction_factor=_safe_reduction_factor(previous, current),
                unit=unit,
                interpretation=_interpret_metric(previous, current),
            )
        )
    return pd.DataFrame([asdict(row) for row in rows])


def _write_dataframe(df: pd.DataFrame, stem: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / f"{stem}.csv", index=False)
    df.to_json(output_dir / f"{stem}.json", orient="records", indent=2)


def _git_hash() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def write_metadata(output_dir: Path) -> Path:
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_hash": _git_hash(),
        "python_version": sys.version,
        "experiment": "exp01_pandapower_pi_reference_diagnostic",
        "reference_mode": REFERENCE_MODE,
        "scenarios": [name for name, _, _ in exp01.SCENARIOS],
        "pandapower_runpp_options": pi_runpp_kwargs(),
        "main_exp01_results_dir": str(MAIN_EXP01_RESULTS_DIR),
        "core_changed_by_diagnostic": False,
        "main_exp01_artifacts_overwritten": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "metadata.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _display_value(metric: str, value: float) -> tuple[float, str]:
    if metric == "max_vm_pu_abs_diff":
        return value, "p.u."
    if metric == "max_va_degree_abs_diff":
        return value, "deg"
    if metric.endswith("_mw_abs_diff"):
        return value * 1e3, "kW"
    if metric.endswith("_mvar_abs_diff"):
        return value * 1e3, "kVAr"
    return value, ""


def write_readme(
    pi_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    key_metrics = [
        "max_vm_pu_abs_diff",
        "max_va_degree_abs_diff",
        "p_slack_mw_abs_diff",
        "total_p_loss_mw_abs_diff",
        "trafo_pl_mw_abs_diff",
    ]

    current_lines = []
    comparison_lines = []
    for metric in key_metrics:
        current_raw = float(pi_summary[metric].max())
        current_value, current_unit = _display_value(metric, current_raw)
        current_lines.append(f"| `{metric}` | {current_value:.8g} {current_unit} |")

        row = comparison[comparison["metric"] == metric].iloc[0]
        previous_value, previous_unit = _display_value(
            metric, float(row["previous_exp01_max"])
        )
        pi_value, pi_unit = _display_value(metric, float(row["pi_reference_max"]))
        comparison_lines.append(
            "| `{metric}` | {prev:.8g} {prev_unit} | {cur:.8g} {cur_unit} | "
            "{factor:.6g} | {interp} |".format(
                metric=metric,
                prev=previous_value,
                prev_unit=previous_unit,
                cur=pi_value,
                cur_unit=pi_unit,
                factor=float(row["reduction_factor"]),
                interp=str(row["interpretation"]),
            )
        )

    vm_row = comparison[comparison["metric"] == "max_vm_pu_abs_diff"].iloc[0]
    vm_interpretation = str(vm_row["interpretation"])
    if vm_interpretation in {"clearly_reduced", "slightly_reduced"}:
        diagnostic_interpretation = (
            "The voltage error is smaller with the explicit pandapower pi "
            "reference. This suggests that the remaining voltage deviation was "
            "at least partly caused by a T-vs-pi reference-model mismatch."
        )
    else:
        diagnostic_interpretation = (
            "The voltage error is similar to the current Exp.-1 artifact. In "
            "this repository state, the main Exp.-1 reference already uses "
            "`trafo_model=\"pi\"`, so the remaining voltage deviation is more "
            "likely due to other small conventions such as tap/shift side, "
            "per-unit base mapping, result definitions, or other minor model "
            "differences."
        )

    already_pi_note = (
        "The imported Exp.-1 module currently has "
        f"`PP_RUNPP_KWARGS['trafo_model'] = {exp01.PP_RUNPP_KWARGS.get('trafo_model')!r}`."
    )

    text = f"""# Experiment 1 Diagnostic: Explicit pandapower Pi Reference

## Goal

This diagnostic reruns the seven Exp.-1 `scope_matched` scenarios while
explicitly forcing the pandapower reference solver to use
`trafo_model=\"pi\"`.

`diffpf` itself is unchanged. The numerical core, solver, residuals, Y-bus and
observables are not modified by this diagnostic. Existing Exp.-1 main artifacts
under `experiments/results/exp01_example_simple_validation/` are read for
comparison only and are not overwritten.

{already_pi_note}

## New Pi-Reference Error Maxima

| Metric | Explicit pi-reference max |
|--------|--------------------------:|
{chr(10).join(current_lines)}

## Comparison To Current Exp.-1 Artifacts

| Metric | Current Exp.-1 max | Explicit pi-reference max | Reduction factor | Interpretation |
|--------|-------------------:|--------------------------:|-----------------:|----------------|
{chr(10).join(comparison_lines)}

## Interpretation

{diagnostic_interpretation}

## Artifacts

- `pi_reference_validation_summary.csv/json`: one row per scope-matched
  scenario with the explicit pandapower pi reference.
- `pi_reference_comparison_summary.csv/json`: max-error comparison against the
  current Exp.-1 main artifacts.
- `metadata.json`: run metadata and pandapower options.
- `figures/fig01_default_vs_pi_reference_errors.png/pdf`: optional bar plot for
  the main comparison metrics.
"""
    path = output_dir / "README.md"
    path.write_text(text, encoding="utf-8")
    return path


def write_bar_plot(comparison: pd.DataFrame, output_dir: Path) -> list[Path]:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        "max_vm_pu_abs_diff",
        "max_va_degree_abs_diff",
        "p_slack_mw_abs_diff",
        "total_p_loss_mw_abs_diff",
        "trafo_pl_mw_abs_diff",
    ]
    labels = ["max |dV| [p.u.]", "max |dtheta| [deg]", "P slack [kW]", "P loss [kW]", "Trafo PL [kW]"]
    display_previous = []
    display_current = []
    for metric in metrics:
        row = comparison[comparison["metric"] == metric].iloc[0]
        prev, _ = _display_value(metric, float(row["previous_exp01_max"]))
        cur, _ = _display_value(metric, float(row["pi_reference_max"]))
        display_previous.append(prev)
        display_current.append(cur)

    x = np.arange(len(metrics))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.bar(x - width / 2, display_previous, width, label="current Exp. 1 artifacts")
    ax.bar(x + width / 2, display_current, width, label="explicit pandapower pi")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Displayed error value")
    ax.set_title("Exp. 1 diagnostic: current reference vs explicit pandapower pi")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    png = figures_dir / "fig01_default_vs_pi_reference_errors.png"
    pdf = figures_dir / "fig01_default_vs_pi_reference_errors.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return [png, pdf]


def export_artifacts(
    pi_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    output_dir: Path = RESULTS_DIR,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_dataframe(pi_summary, "pi_reference_validation_summary", output_dir)
    _write_dataframe(comparison, "pi_reference_comparison_summary", output_dir)
    outputs = [
        output_dir / "pi_reference_validation_summary.csv",
        output_dir / "pi_reference_validation_summary.json",
        output_dir / "pi_reference_comparison_summary.csv",
        output_dir / "pi_reference_comparison_summary.json",
        write_metadata(output_dir),
        write_readme(pi_summary, comparison, output_dir),
    ]
    outputs.extend(write_bar_plot(comparison, output_dir))
    return outputs


def main() -> None:
    print("Experiment 1 diagnostic: explicit pandapower trafo_model='pi'")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"pandapower runpp kwargs: {pi_runpp_kwargs()}")
    pi_summary = run_pi_reference_diagnostic()
    comparison = build_comparison_summary(pi_summary)
    outputs = export_artifacts(pi_summary, comparison, RESULTS_DIR)
    print()
    print("Exported artifacts:")
    for path in outputs:
        print(f"  {path.relative_to(RESULTS_DIR)}")

    vm = comparison[comparison["metric"] == "max_vm_pu_abs_diff"].iloc[0]
    print()
    print(
        "max_vm_pu_abs_diff: "
        f"current={float(vm['previous_exp01_max']):.8e}, "
        f"explicit_pi={float(vm['pi_reference_max']):.8e}, "
        f"interpretation={vm['interpretation']}"
    )


if __name__ == "__main__":
    main()

