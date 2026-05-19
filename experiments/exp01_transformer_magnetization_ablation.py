"""Experiment 1 diagnostic: transformer magnetization ablation.

This script checks whether the systematic active-power offset in the
``scope_matched`` validation of pandapower ``example_simple()`` is mainly
explained by the transformer magnetizing branch / iron losses.

It intentionally does not change the numerical core. Both pandapower and
diffpf receive the same pandapower network variant before the diffpf
``NetworkSpec`` is built.
"""

from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import experiments.exp01_validate_example_simple as exp01

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


RESULTS_DIR = (
    Path(__file__).resolve().parent
    / "results"
    / "exp01_transformer_magnetization_ablation"
)

REFERENCE_MODE = "scope_matched"
TRAFO_NAME = "110kV/20kV transformer"
BASELINE_VARIANT = "baseline"
ABLATION_VARIANT = "no_trafo_magnetization"
VARIANTS = (BASELINE_VARIANT, ABLATION_VARIANT)
PFE_REFERENCE_MW = 0.014
SUPPORT_ABS_THRESHOLD_MW = 1e-3
SUPPORT_REDUCTION_FACTOR = 10.0
CORRECTED_BASELINE_ABS_THRESHOLD_MW = 1e-3


@dataclass(frozen=True)
class AblationResultRow:
    variant: str
    scenario: str
    reference_mode: str
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
    trafo_pl_mw_diffpf: float
    trafo_pl_mw_pandapower: float
    trafo_ql_mvar_diffpf: float
    trafo_ql_mvar_pandapower: float
    total_p_loss_mw_diffpf: float
    total_p_loss_mw_pandapower: float
    total_q_loss_mvar_diffpf: float
    total_q_loss_mvar_pandapower: float
    p_slack_mw_diffpf: float
    p_slack_mw_pandapower: float
    q_slack_mvar_diffpf: float
    q_slack_mvar_pandapower: float


def find_trafo_index(net: pp.pandapowerNet, trafo_name: str = TRAFO_NAME) -> int:
    """Return the unique transformer index with ``trafo_name``.

    Raises
    ------
    ValueError
        If no transformer or multiple transformers with this name exist.
    """
    matches = net.trafo.index[net.trafo["name"].astype(str) == trafo_name].tolist()
    if not matches:
        available = sorted(str(name) for name in net.trafo.get("name", []))
        raise ValueError(
            f"Transformer '{trafo_name}' not found. Available transformer names: {available}"
        )
    if len(matches) > 1:
        raise ValueError(f"Transformer name '{trafo_name}' is not unique: {matches}")
    return int(matches[0])


def apply_trafo_magnetization_ablation(
    net: pp.pandapowerNet,
    trafo_name: str = TRAFO_NAME,
) -> pp.pandapowerNet:
    """Return a copy with transformer iron loss and no-load current removed."""
    ablated = copy.deepcopy(net)
    idx = find_trafo_index(ablated, trafo_name)
    ablated.trafo.at[idx, "pfe_kw"] = 0.0
    ablated.trafo.at[idx, "i0_percent"] = 0.0
    return ablated


def make_variant_scope_matched_net(
    load_factor: float,
    sgen_factor: float,
    variant: str,
) -> pp.pandapowerNet:
    """Build one scenario in Exp.-1 ``scope_matched`` form for a variant."""
    net = exp01.make_scenario_net(load_factor, sgen_factor)
    if variant == BASELINE_VARIANT:
        variant_net = net
    elif variant == ABLATION_VARIANT:
        variant_net = apply_trafo_magnetization_ablation(net)
    else:
        raise ValueError(f"Unknown ablation variant: {variant}")
    return exp01.convert_gen_to_sgen(variant_net)


def run_variant_scenario(
    variant: str,
    scenario_name: str,
    load_factor: float,
    sgen_factor: float,
) -> AblationResultRow:
    """Run one variant and one scenario through the Exp.-1 scope-matched path."""
    net_scope = make_variant_scope_matched_net(load_factor, sgen_factor, variant)
    net_pp_run = copy.deepcopy(net_scope)
    net_diffpf_input = copy.deepcopy(net_scope)

    bus_to_repr, disabled_lines, disabled_trafos = exp01._build_switch_info(net_scope)
    pp_sol = exp01.solve_pandapower(net_pp_run)
    diffpf_sol = exp01.solve_diffpf(net_diffpf_input)

    spec = exp01.from_pandapower(net_diffpf_input)
    topology, params = exp01.compile_network(spec)
    active_line_idx = exp01._active_pp_line_indices(
        net_diffpf_input, bus_to_repr, disabled_lines
    )
    active_trafo_idx = exp01._active_pp_trafo_indices(
        net_diffpf_input, bus_to_repr, disabled_trafos
    )

    bus_rows = exp01._extract_bus_rows(
        scenario_name,
        REFERENCE_MODE,
        net_diffpf_input,
        spec,
        diffpf_sol,
        pp_sol,
        gen_bus_repr_ids=set(),
    )
    slack_row = exp01._extract_slack_row(scenario_name, REFERENCE_MODE, topology, diffpf_sol, pp_sol)
    line_rows = exp01._extract_line_rows(
        scenario_name,
        REFERENCE_MODE,
        net_diffpf_input,
        spec,
        params,
        topology,
        active_line_idx,
        diffpf_sol,
        pp_sol,
    )
    trafo_rows = exp01._extract_trafo_rows(
        scenario_name,
        REFERENCE_MODE,
        net_diffpf_input,
        spec,
        params,
        active_trafo_idx,
        diffpf_sol,
        pp_sol,
    )
    loss_row = exp01._extract_loss_row(
        scenario_name,
        REFERENCE_MODE,
        line_rows,
        trafo_rows,
        params,
        diffpf_sol,
        pp_sol,
    )

    if len(trafo_rows) != 1:
        raise RuntimeError(f"Expected exactly one active transformer, got {len(trafo_rows)}")
    trafo_row = trafo_rows[0]

    vm_diffs = [row.vm_pu_abs_diff for row in bus_rows if math.isfinite(row.vm_pu_abs_diff)]
    va_diffs = [row.va_degree_abs_diff for row in bus_rows if math.isfinite(row.va_degree_abs_diff)]
    max_vm = max(vm_diffs) if vm_diffs else float("nan")
    max_va = max(va_diffs) if va_diffs else float("nan")

    trafo_ql_diffpf = trafo_row.q_hv_mvar_diffpf + trafo_row.q_lv_mvar_diffpf
    trafo_ql_pp = trafo_row.q_hv_mvar_pp + trafo_row.q_lv_mvar_pp

    return AblationResultRow(
        variant=variant,
        scenario=scenario_name,
        reference_mode=REFERENCE_MODE,
        diffpf_converged=diffpf_sol.converged,
        pandapower_converged=pp_sol.converged,
        diffpf_iterations=diffpf_sol.iterations,
        diffpf_residual_norm=diffpf_sol.residual_norm,
        max_vm_pu_abs_diff=max_vm,
        max_va_degree_abs_diff=max_va,
        p_slack_mw_abs_diff=slack_row.p_slack_mw_abs_diff,
        q_slack_mvar_abs_diff=slack_row.q_slack_mvar_abs_diff,
        total_p_loss_mw_abs_diff=loss_row.total_p_loss_mw_abs_diff,
        total_q_loss_mvar_abs_diff=loss_row.total_q_loss_mvar_abs_diff,
        trafo_pl_mw_abs_diff=trafo_row.pl_mw_abs_diff,
        trafo_ql_mvar_abs_diff=abs(trafo_ql_diffpf - trafo_ql_pp),
        trafo_pl_mw_diffpf=trafo_row.pl_mw_diffpf,
        trafo_pl_mw_pandapower=trafo_row.pl_mw_pp,
        trafo_ql_mvar_diffpf=trafo_ql_diffpf,
        trafo_ql_mvar_pandapower=trafo_ql_pp,
        total_p_loss_mw_diffpf=loss_row.total_p_loss_mw_diffpf,
        total_p_loss_mw_pandapower=loss_row.total_p_loss_mw_pp,
        total_q_loss_mvar_diffpf=loss_row.total_q_loss_mvar_diffpf,
        total_q_loss_mvar_pandapower=loss_row.total_q_loss_mvar_pp,
        p_slack_mw_diffpf=slack_row.p_slack_mw_diffpf,
        p_slack_mw_pandapower=slack_row.p_slack_mw_pp,
        q_slack_mvar_diffpf=slack_row.q_slack_mvar_diffpf,
        q_slack_mvar_pandapower=slack_row.q_slack_mvar_pp,
    )


def build_all_results() -> pd.DataFrame:
    """Run all seven Exp.-1 scenarios for both diagnostic variants."""
    rows: list[AblationResultRow] = []
    for variant in VARIANTS:
        for scenario_name, load_factor, sgen_factor in exp01.SCENARIOS:
            print(f"  Running {variant} / {scenario_name} ...", end=" ", flush=True)
            row = run_variant_scenario(variant, scenario_name, load_factor, sgen_factor)
            rows.append(row)
            status = "OK" if row.diffpf_converged and row.pandapower_converged else "CHECK"
            print(
                f"{status} "
                f"dP_loss={row.total_p_loss_mw_abs_diff * 1e3:.3f} kW "
                f"dP_trafo={row.trafo_pl_mw_abs_diff * 1e3:.3f} kW"
            )
    return pd.DataFrame([asdict(row) for row in rows])


def build_ablation_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """Build one compact summary row per diagnostic variant."""
    rows: list[dict[str, Any]] = []
    for variant, group in results_df.groupby("variant", sort=False):
        rows.append(
            {
                "variant": variant,
                "n_scenarios": int(len(group)),
                "all_diffpf_converged": bool(group["diffpf_converged"].all()),
                "all_pandapower_converged": bool(group["pandapower_converged"].all()),
                "mean_p_slack_mw_abs_diff": float(group["p_slack_mw_abs_diff"].mean()),
                "max_p_slack_mw_abs_diff": float(group["p_slack_mw_abs_diff"].max()),
                "mean_total_p_loss_mw_abs_diff": float(
                    group["total_p_loss_mw_abs_diff"].mean()
                ),
                "max_total_p_loss_mw_abs_diff": float(
                    group["total_p_loss_mw_abs_diff"].max()
                ),
                "mean_trafo_pl_mw_abs_diff": float(group["trafo_pl_mw_abs_diff"].mean()),
                "max_trafo_pl_mw_abs_diff": float(group["trafo_pl_mw_abs_diff"].max()),
                "mean_q_slack_mvar_abs_diff": float(group["q_slack_mvar_abs_diff"].mean()),
                "mean_total_q_loss_mvar_abs_diff": float(
                    group["total_q_loss_mvar_abs_diff"].mean()
                ),
                "mean_trafo_ql_mvar_abs_diff": float(
                    group["trafo_ql_mvar_abs_diff"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _summary_value(summary_df: pd.DataFrame, variant: str, column: str) -> float:
    match = summary_df.loc[summary_df["variant"] == variant, column]
    if match.empty:
        raise ValueError(f"Variant '{variant}' missing in summary table")
    return float(match.iloc[0])


def _reduction_factor(baseline: float, ablated: float) -> float:
    if not math.isfinite(baseline) or not math.isfinite(ablated):
        return float("nan")
    if abs(ablated) < 1e-15:
        return float("inf")
    return baseline / ablated


def build_hypothesis_check(summary_df: pd.DataFrame) -> dict[str, Any]:
    """Evaluate whether the active-power offset collapses after ablation."""
    metrics = {
        "p_slack": "mean_p_slack_mw_abs_diff",
        "total_p_loss": "mean_total_p_loss_mw_abs_diff",
        "trafo_pl": "mean_trafo_pl_mw_abs_diff",
    }

    payload: dict[str, Any] = {
        "hypothesis": (
            "Historical diagnostic: the formerly observed approximately 14 kW "
            "active-power offset in the Exp.-1 scope_matched comparison was "
            "mainly caused by transformer magnetization / iron-loss handling "
            "in pandapower example_simple()."
        ),
        "ablation": {"pfe_kw": 0.0, "i0_percent": 0.0},
        "reference_pfe_mw": PFE_REFERENCE_MW,
        "current_core_note": (
            "The main diffpf transformer pi stamp now splits the total "
            "magnetizing admittance over both terminals. Therefore the current "
            "baseline is expected to be small already; ablation no longer needs "
            "to produce a dramatic collapse."
        ),
        "support_rule": (
            "Legacy diagnostic support is true if each active-power mean offset "
            f"drops by at least a factor of {SUPPORT_REDUCTION_FACTOR:g} or "
            f"below {SUPPORT_ABS_THRESHOLD_MW:g} MW after ablation."
        ),
    }

    metric_support: dict[str, bool] = {}
    for label, column in metrics.items():
        baseline = _summary_value(summary_df, BASELINE_VARIANT, column)
        ablated = _summary_value(summary_df, ABLATION_VARIANT, column)
        reduction = baseline - ablated
        factor = _reduction_factor(baseline, ablated)
        supported = bool(
            (math.isfinite(factor) and factor >= SUPPORT_REDUCTION_FACTOR)
            or ablated < SUPPORT_ABS_THRESHOLD_MW
        )
        metric_support[label] = supported
        payload[f"baseline_mean_{label}_mw_abs_diff"] = baseline
        payload[f"ablated_mean_{label}_mw_abs_diff"] = ablated
        payload[f"reduction_mean_{label}_mw_abs_diff"] = reduction
        payload[f"reduction_factor_{label}"] = factor
        payload[f"baseline_mean_{label}_minus_0p014_mw"] = baseline - PFE_REFERENCE_MW
        payload[f"{label}_supports_hypothesis"] = supported

    payload["supports_pfe_offset_hypothesis"] = bool(all(metric_support.values()))
    payload["current_baseline_below_corrected_threshold"] = bool(
        all(
            _summary_value(summary_df, BASELINE_VARIANT, column)
            < CORRECTED_BASELINE_ABS_THRESHOLD_MW
            for column in metrics.values()
        )
    )
    payload["corrected_baseline_threshold_mw"] = CORRECTED_BASELINE_ABS_THRESHOLD_MW
    return payload


def _to_native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_native(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_native(v) for v in value]
    if isinstance(value, tuple):
        return [_to_native(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _write_dataframe_artifacts(df: pd.DataFrame, stem: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / f"{stem}.csv", index=False)
    df.to_json(output_dir / f"{stem}.json", orient="records", indent=2)


def _git_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def write_metadata(output_dir: Path, hypothesis_check: dict[str, Any]) -> None:
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_hash": _git_hash(),
        "python_version": sys.version,
        "experiment": "exp01_transformer_magnetization_ablation",
        "reference_mode": REFERENCE_MODE,
        "scenarios": [name for name, _, _ in exp01.SCENARIOS],
        "variants": list(VARIANTS),
        "transformer_name": TRAFO_NAME,
        "baseline_transformer_parameters": {
            "pfe_kw": 14.0,
            "i0_percent": 0.07,
        },
        "ablation_transformer_parameters": {
            "pfe_kw": 0.0,
            "i0_percent": 0.0,
        },
        "pandapower_runpp_options": exp01.PP_RUNPP_KWARGS,
        "diffpf_solver_options": {
            "max_iters": exp01.NEWTON_OPTIONS.max_iters,
            "tolerance": exp01.NEWTON_OPTIONS.tolerance,
            "damping": exp01.NEWTON_OPTIONS.damping,
        },
        "core_uses_corrected_pi_magnetization_stamp": True,
        "core_change_note": (
            "The diagnostic itself does not alter the numerical core during the "
            "run. The repository core may already include the corrected pi "
            "magnetizing-admittance stamp."
        ),
        "hypothesis_check": hypothesis_check,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(
        json.dumps(_to_native(meta), indent=2),
        encoding="utf-8",
    )


def write_readme(
    output_dir: Path,
    summary_df: pd.DataFrame,
    hypothesis_check: dict[str, Any],
) -> None:
    baseline_loss_kw = (
        _summary_value(summary_df, BASELINE_VARIANT, "mean_total_p_loss_mw_abs_diff") * 1e3
    )
    ablated_loss_kw = (
        _summary_value(summary_df, ABLATION_VARIANT, "mean_total_p_loss_mw_abs_diff") * 1e3
    )
    baseline_slack_kw = (
        _summary_value(summary_df, BASELINE_VARIANT, "mean_p_slack_mw_abs_diff") * 1e3
    )
    ablated_slack_kw = (
        _summary_value(summary_df, ABLATION_VARIANT, "mean_p_slack_mw_abs_diff") * 1e3
    )
    baseline_trafo_kw = (
        _summary_value(summary_df, BASELINE_VARIANT, "mean_trafo_pl_mw_abs_diff") * 1e3
    )
    ablated_trafo_kw = (
        _summary_value(summary_df, ABLATION_VARIANT, "mean_trafo_pl_mw_abs_diff") * 1e3
    )

    if hypothesis_check.get("current_baseline_below_corrected_threshold", False):
        interpretation = (
            "Die aktuelle Baseline liegt bereits unter "
            f"{CORRECTED_BASELINE_ABS_THRESHOLD_MW * 1e3:.3f} kW. Der fruehere "
            "14-kW-Befund ist damit im korrigierten Hauptmodell nicht mehr als "
            "aktueller Offset vorhanden."
        )
    elif hypothesis_check["supports_pfe_offset_hypothesis"]:
        interpretation = "Die historische Ablationshypothese wird weiterhin unterstuetzt."
    else:
        interpretation = "Die historische Ablationshypothese wird in diesem Lauf nicht unterstuetzt."

    text = f"""# Experiment 1 Diagnostic: Transformer Magnetization Ablation

## Ziel

Dieser Diagnose-Lauf prueft den frueheren systematischen Wirkleistungsoffset im
`scope_matched`-Vergleich von Experiment 1. Nach der Korrektur der
Trafo-Magnetisierungsstempelung im Hauptmodell dient der Lauf als Regression:
Er trennt den historischen Diagnosebefund von der aktuellen korrigierten
Pi-Stempelung.

## Hypothese

Der Trafo `{TRAFO_NAME}` besitzt im Basismodell `pfe_kw = 14.0` und
`i0_percent = 0.07`. Der frueher beobachtete Offset von etwa 14 kW in
Slackleistung, Gesamtwirkverlust und Trafowirkverlust lag in derselben
Groessenordnung.

## Varianten

`baseline` verwendet das unveraenderte `example_simple()`-Netz, aber weiterhin
das Exp.-1-`scope_matched`-Handling mit `gen -> sgen(P, Q=0)`.

`no_trafo_magnetization` verwendet dasselbe Netz, setzt jedoch vor der
diffpf-Konvertierung und vor dem pandapower-Referenzlauf beim Trafo
`pfe_kw = 0.0` und `i0_percent = 0.0`.

Es wird ausschliesslich `reference_mode = scope_matched` betrachtet. Der
aktuelle numerische Kern verwendet weiterhin ein Pi-Ersatzschaltbild, verteilt
die gesamte Magnetisierungsadmittanz aber korrekt auf beide Klemmen. Residuen,
Newton-Solver und implizite Differentiation werden durch diesen Diagnose-Lauf
nicht veraendert.

## Zentrale Ergebnisse

| Groesse | baseline mean [kW] | ablated mean [kW] | Reduktionsfaktor |
|---------|--------------------:|------------------:|-----------------:|
| p_slack | {baseline_slack_kw:.6f} | {ablated_slack_kw:.6f} | {hypothesis_check["reduction_factor_p_slack"]:.3g} |
| total_p_loss | {baseline_loss_kw:.6f} | {ablated_loss_kw:.6f} | {hypothesis_check["reduction_factor_total_p_loss"]:.3g} |
| trafo_pl | {baseline_trafo_kw:.6f} | {ablated_trafo_kw:.6f} | {hypothesis_check["reduction_factor_trafo_pl"]:.3g} |

## Interpretation

{interpretation}

Das historische Ablationskriterium ist erfuellt, wenn die mittleren
Wirkleistungsoffsets im ablierten Fall mindestens um Faktor
{SUPPORT_REDUCTION_FACTOR:g} sinken oder unter
{SUPPORT_ABS_THRESHOLD_MW * 1e3:.3f} kW fallen. Im korrigierten Hauptmodell ist
jedoch vor allem relevant, ob bereits die Baseline klein bleibt.

## Artefakte

| Datei | Inhalt |
|-------|--------|
| `ablation_results.csv/json` | Eine Zeile pro Variante und Szenario |
| `ablation_summary.csv/json` | Kompakte Variante-zu-Variante-Zusammenfassung |
| `hypothesis_check.json` | Datenbasierte Hypothesenbewertung |
| `metadata.json` | Reproduzierbarkeits- und Solver-Metadaten |
| `figures/fig01_p_offset_baseline_vs_ablation.png/pdf` | Optionaler Balkenvergleich der Wirkleistungsoffsets |

## Einschraenkung

Der Test isoliert nur die Wirkung von `pfe_kw` und `i0_percent`. Er ist kein
Nachweis vollstaendiger pandapower-Kompatibilitaet fuer alle Trafofaelle.
"""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def write_offset_bar_figure(summary_df: pd.DataFrame, output_dir: Path) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    metric_labels = ["p_slack", "total_p_loss", "trafo_pl"]
    summary_columns = [
        "mean_p_slack_mw_abs_diff",
        "mean_total_p_loss_mw_abs_diff",
        "mean_trafo_pl_mw_abs_diff",
    ]
    x = np.arange(len(metric_labels))
    width = 0.36
    baseline = [
        _summary_value(summary_df, BASELINE_VARIANT, column) * 1e3
        for column in summary_columns
    ]
    ablated = [
        _summary_value(summary_df, ABLATION_VARIANT, column) * 1e3
        for column in summary_columns
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars_base = ax.bar(x - width / 2, baseline, width, label="baseline", color="#4C78A8")
    bars_abl = ax.bar(
        x + width / 2,
        ablated,
        width,
        label="no_trafo_magnetization",
        color="#F58518",
    )
    ax.set_title("Experiment 1 diagnostic: transformer magnetization ablation")
    ax.set_ylabel("Absolute Abweichung [kW]")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.axhline(14.0, color="0.35", linestyle="--", linewidth=1.0, label="pfe = 14 kW")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)

    for bars in (bars_base, bars_abl):
        for bar in bars:
            height = float(bar.get_height())
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(figures_dir / "fig01_p_offset_baseline_vs_ablation.png", dpi=300)
    fig.savefig(figures_dir / "fig01_p_offset_baseline_vs_ablation.pdf")
    plt.close(fig)


def export_artifacts(
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    hypothesis_check: dict[str, Any],
    output_dir: Path = RESULTS_DIR,
) -> None:
    """Write all CSV/JSON/README artifacts for the ablation experiment."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_dataframe_artifacts(results_df, "ablation_results", output_dir)
    _write_dataframe_artifacts(summary_df, "ablation_summary", output_dir)
    (output_dir / "hypothesis_check.json").write_text(
        json.dumps(_to_native(hypothesis_check), indent=2),
        encoding="utf-8",
    )
    write_metadata(output_dir, hypothesis_check)
    write_readme(output_dir, summary_df, hypothesis_check)
    write_offset_bar_figure(summary_df, output_dir)


def main() -> None:
    print("Running Experiment 1 transformer magnetization ablation")
    results_df = build_all_results()
    summary_df = build_ablation_summary(results_df)
    hypothesis_check = build_hypothesis_check(summary_df)
    export_artifacts(results_df, summary_df, hypothesis_check, RESULTS_DIR)

    status = "SUPPORTED" if hypothesis_check["supports_pfe_offset_hypothesis"] else "NOT SUPPORTED"
    print(f"\nHypothesis check: {status}")
    print(
        "  mean total_p_loss offset: "
        f"{hypothesis_check['baseline_mean_total_p_loss_mw_abs_diff'] * 1e3:.6f} kW -> "
        f"{hypothesis_check['ablated_mean_total_p_loss_mw_abs_diff'] * 1e3:.6f} kW"
    )
    print(
        "  mean p_slack offset: "
        f"{hypothesis_check['baseline_mean_p_slack_mw_abs_diff'] * 1e3:.6f} kW -> "
        f"{hypothesis_check['ablated_mean_p_slack_mw_abs_diff'] * 1e3:.6f} kW"
    )
    print(
        "  mean trafo_pl offset: "
        f"{hypothesis_check['baseline_mean_trafo_pl_mw_abs_diff'] * 1e3:.6f} kW -> "
        f"{hypothesis_check['ablated_mean_trafo_pl_mw_abs_diff'] * 1e3:.6f} kW"
    )
    print(f"Artifacts written to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
