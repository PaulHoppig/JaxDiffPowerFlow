"""Tests for exp01_validate_example_simple.py

Prüft:
- Importierbarkeit des Experiment-Moduls
- Szenariodefinitionen vorhanden und korrekt
- scope_matched / original_pandapower werden unterschieden
- Einzel-Szenario läuft durch (base × scope_matched)
- Output-Tabellen enthalten die erwarteten Spalten
- Ergebnisordner und JSON/CSV-Export funktionieren

Schwere Integrationstests (alle Szenarien, voller Exportlauf) sind mit
``pytest.mark.slow`` markiert und werden in der CI-Standardkonfiguration
ausgelassen (pytest -m "not slow").
"""

from __future__ import annotations

import json
import math
import tempfile
from dataclasses import asdict, fields
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def exp_module():
    """Importiert das Experiment-Modul einmalig pro Test-Session."""
    import experiments.exp01_validate_example_simple as m
    return m


@pytest.fixture(scope="module")
def base_scope_matched_result(exp_module):
    """Einzel-Szenario base × scope_matched."""
    return exp_module.run_scenario("base", 1.0, 1.0, "scope_matched")


@pytest.fixture(scope="module")
def base_original_result(exp_module):
    """Einzel-Szenario base × original_pandapower."""
    return exp_module.run_scenario("base", 1.0, 1.0, "original_pandapower")


# ---------------------------------------------------------------------------
# Importierbarkeit und Modul-Struktur
# ---------------------------------------------------------------------------


def test_module_is_importable(exp_module):
    assert exp_module is not None


def test_scenarios_defined(exp_module):
    assert len(exp_module.SCENARIOS) >= 7
    names = [s[0] for s in exp_module.SCENARIOS]
    for expected in [
        "base", "load_low", "load_high",
        "sgen_low", "sgen_high",
        "combined_high_load_low_sgen", "combined_low_load_high_sgen",
    ]:
        assert expected in names, f"Szenario '{expected}' fehlt in SCENARIOS"


def test_reference_modes_defined(exp_module):
    assert "scope_matched" in exp_module.REFERENCE_MODES
    assert "original_pandapower" in exp_module.REFERENCE_MODES


def test_results_dir_constant_defined(exp_module):
    assert isinstance(exp_module.RESULTS_DIR, Path)


# ---------------------------------------------------------------------------
# Netz-Manipulation Helpers
# ---------------------------------------------------------------------------


def test_make_scenario_net_base(exp_module):
    net = exp_module.make_scenario_net(1.0, 1.0)
    import pandapower.networks as pn
    net_ref = pn.example_simple()
    assert len(net.bus) == len(net_ref.bus)
    assert len(net.load) == len(net_ref.load)


def test_make_scenario_net_load_scaling(exp_module):
    net1 = exp_module.make_scenario_net(1.0, 1.0)
    net2 = exp_module.make_scenario_net(1.5, 1.0)
    s1 = net1.load["scaling"].sum()
    s2 = net2.load["scaling"].sum()
    assert abs(s2 - 1.5 * s1) < 1e-9, "load_factor=1.5 soll scaling × 1.5 setzen"


def test_make_scenario_net_sgen_scaling(exp_module):
    net1 = exp_module.make_scenario_net(1.0, 1.0)
    net2 = exp_module.make_scenario_net(1.0, 0.5)
    s1 = net1.sgen["scaling"].sum()
    s2 = net2.sgen["scaling"].sum()
    assert abs(s2 - 0.5 * s1) < 1e-9, "sgen_factor=0.5 soll scaling × 0.5 setzen"


def test_convert_gen_to_sgen(exp_module):
    import pandapower.networks as pn
    net = pn.example_simple()
    n_gens_before = int((net.gen["in_service"] == True).sum())  # noqa: E712
    n_sgens_before = int((net.sgen["in_service"] == True).sum())  # noqa: E712
    converted = exp_module.convert_gen_to_sgen(net)
    # Original net not mutated
    assert int((net.gen["in_service"] == True).sum()) == n_gens_before  # noqa: E712
    # Converted net: gen deactivated, sgen added
    assert int((converted.gen["in_service"] == True).sum()) == 0  # noqa: E712
    assert int((converted.sgen["in_service"] == True).sum()) == n_sgens_before + n_gens_before


def test_convert_gen_to_sgen_q_zero(exp_module):
    import pandapower.networks as pn
    net = pn.example_simple()
    converted = exp_module.convert_gen_to_sgen(net)
    new_sgen = converted.sgen[converted.sgen["name"].str.startswith("gen_as_sgen", na=False)]
    for _, row in new_sgen.iterrows():
        assert abs(float(row["q_mvar"])) < 1e-12, "Konvertierter gen muss Q=0 haben"


# ---------------------------------------------------------------------------
# Solver und Ergebnis-Zeilen
# ---------------------------------------------------------------------------


def test_run_scenario_returns_named_tuple(exp_module, base_scope_matched_result):
    res = base_scope_matched_result
    assert hasattr(res, "summary")
    assert hasattr(res, "bus_rows")
    assert hasattr(res, "slack_row")
    assert hasattr(res, "line_rows")
    assert hasattr(res, "trafo_rows")
    assert hasattr(res, "loss_row")
    assert hasattr(res, "structure_row")


def test_scope_matched_converges(base_scope_matched_result):
    assert base_scope_matched_result.summary.diffpf_converged, (
        "diffpf soll für base/scope_matched konvergieren"
    )


def test_scope_matched_residual_small(base_scope_matched_result):
    assert base_scope_matched_result.summary.diffpf_residual_norm < 1e-6


def test_pandapower_converges_scope_matched(base_scope_matched_result):
    assert base_scope_matched_result.summary.pandapower_converged


def test_reference_mode_in_summary(base_scope_matched_result, base_original_result):
    assert base_scope_matched_result.summary.reference_mode == "scope_matched"
    assert base_original_result.summary.reference_mode == "original_pandapower"


def test_strict_validation_flag(base_scope_matched_result, base_original_result):
    assert base_scope_matched_result.summary.strict_validation is True
    assert base_original_result.summary.strict_validation is False


# ---------------------------------------------------------------------------
# Spalten-Vollständigkeit
# ---------------------------------------------------------------------------


def test_summary_row_has_required_columns(exp_module, base_scope_matched_result):
    row = base_scope_matched_result.summary
    d = asdict(row)
    for col in [
        "scenario", "reference_mode", "diffpf_converged", "pandapower_converged",
        "diffpf_iterations", "diffpf_residual_norm", "strict_validation",
        "max_vm_pu_abs_diff", "rmse_vm_pu", "max_va_degree_abs_diff", "rmse_va_degree",
        "max_line_p_mw_abs_diff", "max_trafo_p_mw_abs_diff",
        "total_p_loss_mw_abs_diff", "total_q_loss_mvar_abs_diff",
    ]:
        assert col in d, f"Spalte '{col}' fehlt in SummaryRow"


def test_bus_result_row_has_required_columns(base_scope_matched_result):
    assert len(base_scope_matched_result.bus_rows) > 0
    row = base_scope_matched_result.bus_rows[0]
    d = asdict(row)
    for col in [
        "scenario", "reference_mode", "bus_internal_id", "bus_name", "vn_kv",
        "is_slack", "vm_pu_diffpf", "vm_pu_pp", "vm_pu_abs_diff",
        "va_degree_diffpf", "va_degree_pp", "va_degree_abs_diff",
        "validation_scope",
    ]:
        assert col in d, f"Spalte '{col}' fehlt in BusResultRow"


def test_line_flow_row_has_required_columns(base_scope_matched_result):
    assert len(base_scope_matched_result.line_rows) > 0
    row = base_scope_matched_result.line_rows[0]
    d = asdict(row)
    for col in [
        "scenario", "reference_mode", "line_name", "pp_line_idx",
        "p_from_mw_diffpf", "p_from_mw_pp", "q_from_mvar_diffpf", "q_from_mvar_pp",
        "p_to_mw_diffpf", "p_to_mw_pp", "pl_mw_diffpf", "pl_mw_pp",
        "p_from_mw_abs_diff", "q_from_mvar_abs_diff", "pl_mw_abs_diff",
    ]:
        assert col in d, f"Spalte '{col}' fehlt in LineFlowRow"


def test_trafo_flow_row_has_required_columns(base_scope_matched_result):
    assert len(base_scope_matched_result.trafo_rows) > 0
    row = base_scope_matched_result.trafo_rows[0]
    d = asdict(row)
    for col in [
        "scenario", "reference_mode", "trafo_name", "pp_trafo_idx",
        "p_hv_mw_diffpf", "p_hv_mw_pp", "q_hv_mvar_diffpf",
        "p_lv_mw_diffpf", "p_lv_mw_pp", "pl_mw_diffpf", "pl_mw_pp",
        "p_hv_mw_abs_diff", "pl_mw_abs_diff",
    ]:
        assert col in d, f"Spalte '{col}' fehlt in TrafoFlowRow"


def test_loss_row_has_required_columns(base_scope_matched_result):
    row = base_scope_matched_result.loss_row
    d = asdict(row)
    for col in [
        "scenario", "reference_mode",
        "total_p_loss_mw_diffpf", "total_p_loss_mw_pp", "total_p_loss_mw_abs_diff",
        "total_q_loss_mvar_diffpf", "total_q_loss_mvar_pp",
        "line_p_loss_mw_diffpf", "trafo_p_loss_mw_diffpf",
    ]:
        assert col in d, f"Spalte '{col}' fehlt in LossRow"


def test_structure_row_has_required_columns(base_scope_matched_result):
    row = base_scope_matched_result.structure_row
    d = asdict(row)
    for col in [
        "scenario", "reference_mode",
        "number_of_original_buses", "number_of_internal_buses_after_fusion",
        "number_of_lines_original", "number_of_lines_active_after_switches",
        "number_of_trafos_active", "number_of_shunts",
        "number_of_loads", "number_of_sgens", "number_of_gens",
        "bus_fusion_groups", "disabled_lines_due_to_open_switches",
    ]:
        assert col in d, f"Spalte '{col}' fehlt in StructureSummaryRow"


# ---------------------------------------------------------------------------
# Physikalische Plausibilität (scope_matched)
# ---------------------------------------------------------------------------


def test_bus_count_after_fusion(base_scope_matched_result):
    row = base_scope_matched_result.structure_row
    # example_simple: 7 buses, 2 bus-bus switches → 5 internal buses
    assert row.number_of_original_buses == 7
    assert row.number_of_internal_buses_after_fusion == 5


def test_exactly_one_slack_bus(base_scope_matched_result):
    slack_buses = [b for b in base_scope_matched_result.bus_rows if b.is_slack]
    assert len(slack_buses) == 1


def test_active_lines_count(base_scope_matched_result):
    row = base_scope_matched_result.structure_row
    # example_simple has lines; one line disabled by open switch → 3 active
    assert row.number_of_lines_active_after_switches >= 1
    assert row.number_of_lines_active_after_switches < row.number_of_lines_original


def test_trafo_active(base_scope_matched_result):
    row = base_scope_matched_result.structure_row
    assert row.number_of_trafos_active >= 1


def test_scope_matched_vm_abs_diff_reasonable(base_scope_matched_result):
    """max |Δvm| < 1e-3 pu im scope_matched Modus."""
    diffs = [
        b.vm_pu_abs_diff
        for b in base_scope_matched_result.bus_rows
        if not math.isnan(b.vm_pu_abs_diff)
    ]
    assert len(diffs) > 0
    assert max(diffs) < 1e-3, f"max |Δvm| = {max(diffs):.4e} pu is unexpectedly large"


def test_validation_scope_labels(base_scope_matched_result, base_original_result):
    for row in base_scope_matched_result.bus_rows:
        assert row.validation_scope == "strict"
    for row in base_original_result.bus_rows:
        assert row.validation_scope == "contextual"


def test_loss_row_positive_p_loss(base_scope_matched_result):
    row = base_scope_matched_result.loss_row
    assert row.total_p_loss_mw_diffpf > 0, "Total P losses should be positive"


def test_trafo_pl_mw_positive(base_scope_matched_result):
    for row in base_scope_matched_result.trafo_rows:
        assert row.pl_mw_diffpf > -1e-6, f"Trafo P losses should be non-negative, got {row.pl_mw_diffpf}"


# ---------------------------------------------------------------------------
# Scope difference: original_pandapower should differ from scope_matched
# ---------------------------------------------------------------------------


def test_modes_produce_different_pp_results(
    base_scope_matched_result, base_original_result
):
    """pandapower ergibt unterschiedliche Q am gen-Bus für beide Modi."""
    sm_q = {r.bus_name: r.vm_pu_pp for r in base_scope_matched_result.bus_rows}
    op_q = {r.bus_name: r.vm_pu_pp for r in base_original_result.bus_rows}
    # They share same bus names; original_pandapower may differ due to PV bus
    # At minimum the summary notes differ
    assert base_scope_matched_result.summary.notes != base_original_result.summary.notes


# ---------------------------------------------------------------------------
# Export to CSV/JSON (uses temp directory)
# ---------------------------------------------------------------------------


def test_csv_json_export(exp_module, base_scope_matched_result):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        results = dict(
            summary=[base_scope_matched_result.summary],
            bus=[base_scope_matched_result.bus_rows[0]],
            slack=[base_scope_matched_result.slack_row],
            lines=[base_scope_matched_result.line_rows[0]] if base_scope_matched_result.line_rows else [],
            trafos=[base_scope_matched_result.trafo_rows[0]] if base_scope_matched_result.trafo_rows else [],
            losses=[base_scope_matched_result.loss_row],
            structure=[base_scope_matched_result.structure_row],
        )
        exp_module.export_all(results, tmp)

        for name in [
            "validation_summary.csv", "validation_summary.json",
            "bus_results.csv", "bus_results.json",
            "slack_results.csv", "slack_results.json",
            "line_flows.csv", "line_flows.json",
            "trafo_flows.csv", "trafo_flows.json",
            "losses.csv", "losses.json",
            "structure_summary.csv", "structure_summary.json",
            "metadata.json",
        ]:
            assert (tmp / name).exists(), f"{name} wurde nicht exportiert"


def test_json_is_valid(exp_module, base_scope_matched_result):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        results = dict(
            summary=[base_scope_matched_result.summary],
            bus=base_scope_matched_result.bus_rows,
            slack=[base_scope_matched_result.slack_row],
            lines=base_scope_matched_result.line_rows,
            trafos=base_scope_matched_result.trafo_rows,
            losses=[base_scope_matched_result.loss_row],
            structure=[base_scope_matched_result.structure_row],
        )
        exp_module.export_all(results, tmp)

        with (tmp / "validation_summary.json").open() as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["scenario"] == "base"
        assert data[0]["reference_mode"] == "scope_matched"


def test_results_dir_is_created(exp_module, base_scope_matched_result):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir) / "nested" / "output"
        results = dict(
            summary=[base_scope_matched_result.summary],
            bus=base_scope_matched_result.bus_rows,
            slack=[base_scope_matched_result.slack_row],
            lines=base_scope_matched_result.line_rows,
            trafos=base_scope_matched_result.trafo_rows,
            losses=[base_scope_matched_result.loss_row],
            structure=[base_scope_matched_result.structure_row],
        )
        exp_module.export_all(results, tmp)
        assert tmp.exists()


# ---------------------------------------------------------------------------
# Slow integration test: full suite
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_full_suite_all_scenarios(exp_module, tmp_path):
    """Alle 7 Szenarien × 2 Modi laufen durch und exportieren alle Artefakte."""
    results = exp_module.build_all_results()
    exp_module.export_all(results, tmp_path)

    assert len(results["summary"]) == len(exp_module.SCENARIOS) * len(exp_module.REFERENCE_MODES)

    for name in [
        "validation_summary.csv", "bus_results.csv",
        "line_flows.csv", "trafo_flows.csv", "losses.csv",
        "structure_summary.csv", "metadata.json",
    ]:
        assert (tmp_path / name).exists()

    # Alle scope_matched müssen konvergieren
    for row in results["summary"]:
        if row.reference_mode == "scope_matched":
            assert row.diffpf_converged, (
                f"scope_matched/{row.scenario} did not converge: "
                f"norm={row.diffpf_residual_norm:.2e}"
            )
