"""Lightweight schema tests for Experiment 5a network screening artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def exp_module():
    import experiments.exp05a_network_screening as module

    return module


def test_module_is_importable(exp_module):
    assert exp_module is not None


def test_scenario_grid_size_matches_specification(exp_module):
    assert exp_module.LOAD_MULTIPLIERS_MV_BUS_2 == (0.40, 0.70, 1.00, 1.30)
    assert exp_module.G_POA_LEVELS_WM2 == (200.0, 600.0, 1200.0)
    assert exp_module.T_AMB_LEVELS_C == (-10.0, 5.0, 25.0, 45.0)
    assert exp_module.screening_case_count() == 48
    assert exp_module.no_pv_reference_count() == 4


def test_fixed_curtailment_and_pv_constants(exp_module):
    assert exp_module.CURTAILMENT_FACTOR == 1.0
    assert exp_module.PV_SIZE_FACTOR == 1.0
    assert exp_module.EXP5A_KAPPA == -0.25
    assert exp_module.WIND_MS == 2.0


def test_required_artifacts_are_declared(exp_module):
    required = set(exp_module.REQUIRED_ARTIFACTS)
    for name in [
        "screening_results.csv",
        "top_critical_cases.csv",
        "sensitivity_top20.csv",
        "branch_flows.csv",
        "run_summary.csv",
        "metadata.json",
        "README.md",
    ]:
        assert name in required


def test_screening_columns_contain_required_fields(exp_module):
    cols = exp_module.SCREENING_COLUMNS
    for name in [
        "case_id",
        "case_type",
        "load_multiplier_mv_bus_2",
        "g_poa_wm2",
        "t_amb_c",
        "wind_ms",
        "curtailment_factor",
        "pv_size_factor",
        "p_pv_mw",
        "q_pv_mvar",
        "p_export_mw",
        "vm_mv_bus_2_pu",
        "max_vm_pu",
        "s_trafo_hv_mva",
        "max_line_s_mva",
        "delta_p_slack_vs_no_pv_mw",
        "loss_delta_pct_vs_no_pv",
        "numeric_critical",
        "export_critical",
        "voltage_critical_demo",
        "criticality_score",
        "selected_for_sensitivity",
        "top20_rank",
    ]:
        assert name in cols


def test_sensitivity_columns_are_top20_curtailment_focused(exp_module):
    cols = exp_module.SENSITIVITY_COLUMNS
    for name in [
        "case_id",
        "top20_rank",
        "input_parameter",
        "observable",
        "value",
        "ad_converged",
    ]:
        assert name in cols
    observable_names = [name for name, _ in exp_module.SENSITIVITY_OBSERVABLES]
    assert "criticality_score" in observable_names
    assert "p_export_mw" in observable_names
    assert "vm_mv_bus_2_pu" in observable_names


def test_branch_flow_schema_does_not_claim_loading_percent(exp_module):
    cols = exp_module.BRANCH_FLOW_COLUMNS
    assert "max_end_s_mva" in cols
    assert "is_mv_branch" in cols
    assert "active" in cols
    assert all("loading_percent" not in col for col in cols)


def test_score_components_follow_requested_formula(exp_module):
    export_score, voltage_score, loss_score, trafo_score, total = exp_module._score_components(
        p_export_mw=7.25,
        max_vm_pu=1.012,
        loss_delta_pct_vs_no_pv=0.20,
        delta_s_trafo_hv_vs_no_pv_mva=1.0,
    )

    assert export_score == pytest.approx(1.0)
    assert voltage_score == pytest.approx(1.0)
    assert loss_score == pytest.approx(2.0)
    assert trafo_score == pytest.approx(2.0)
    assert total == pytest.approx(9.0)


def test_select_top_critical_cases_uses_screening_only(exp_module):
    rows = [
        {
            "case_id": "ref",
            "case_type": "no_pv_reference",
            "criticality_score": 999.0,
            "p_export_mw": 999.0,
            "max_vm_pu": 999.0,
            "p_pv_mw": 0.0,
        },
        {
            "case_id": "low",
            "case_type": "screening",
            "criticality_score": 1.0,
            "p_export_mw": 1.0,
            "max_vm_pu": 1.0,
            "p_pv_mw": 1.0,
        },
        {
            "case_id": "high",
            "case_type": "screening",
            "criticality_score": 2.0,
            "p_export_mw": 1.0,
            "max_vm_pu": 1.0,
            "p_pv_mw": 1.0,
        },
    ]

    selected = exp_module.select_top_critical_cases(rows, n_top=2)
    assert [row["case_id"] for row in selected] == ["high", "low"]


def _stub_screening_row(exp_module):
    return exp_module.ScreeningRow(
        case_id="screen_load1_g1200_t45",
        case_type="screening",
        selected_for_sensitivity=True,
        top20_rank=1,
        load_multiplier_mv_bus_2=1.0,
        g_poa_wm2=1200.0,
        t_amb_c=45.0,
        wind_ms=2.0,
        curtailment_factor=1.0,
        pv_size_factor=1.0,
        kappa=-0.25,
        t_cell_c=60.0,
        p_pv_mw=2.1,
        q_pv_mvar=-0.525,
        q_over_p=-0.25,
        converged=True,
        iterations=6,
        residual_norm=1e-11,
        p_slack_mw=-7.1,
        q_slack_mvar=1.0,
        p_export_mw=7.1,
        vm_mv_bus_2_pu=1.01,
        max_vm_pu=1.012,
        max_vm_bus="MV Bus 2",
        total_p_loss_mw=0.1,
        total_q_loss_mvar=0.2,
        p_trafo_hv_mw=7.0,
        q_trafo_hv_mvar=1.0,
        s_trafo_hv_mva=7.1,
        trafo_loading_proxy=0.284,
        max_line_s_mva=2.0,
        max_line_id=1,
        max_line_name="line",
        disabled_lines_due_to_open_switches="2",
        disabled_trafos_due_to_open_switches="",
        delta_p_slack_vs_no_pv_mw=-2.0,
        delta_p_export_vs_no_pv_mw=2.0,
        delta_vm_mv_bus_2_vs_no_pv_pu=0.002,
        delta_total_p_loss_vs_no_pv_mw=0.01,
        loss_delta_pct_vs_no_pv=0.12,
        delta_s_trafo_hv_vs_no_pv_mva=0.6,
        numeric_critical=False,
        export_warning=True,
        export_critical=True,
        export_severe=False,
        voltage_warning=True,
        voltage_critical_demo=True,
        pv_voltage_impact=True,
        loss_delta_warning=True,
        trafo_delta_warning=True,
        export_score=0.4,
        voltage_score=1.0,
        loss_score=1.2,
        trafo_score=1.2,
        criticality_score=5.6,
        notes="",
    )


def _stub_top_row(exp_module):
    return exp_module.TopCriticalCaseRow(
        rank=1,
        case_id="screen_load1_g1200_t45",
        load_multiplier_mv_bus_2=1.0,
        g_poa_wm2=1200.0,
        t_amb_c=45.0,
        wind_ms=2.0,
        p_pv_mw=2.1,
        p_export_mw=7.1,
        max_vm_pu=1.012,
        vm_mv_bus_2_pu=1.01,
        total_p_loss_mw=0.1,
        s_trafo_hv_mva=7.1,
        delta_vm_mv_bus_2_vs_no_pv_pu=0.002,
        delta_total_p_loss_vs_no_pv_mw=0.01,
        loss_delta_pct_vs_no_pv=0.12,
        delta_s_trafo_hv_vs_no_pv_mva=0.6,
        numeric_critical=False,
        export_critical=True,
        voltage_critical_demo=True,
        criticality_score=5.6,
    )


def _stub_sensitivity_row(exp_module):
    return exp_module.SensitivityRow(
        case_id="screen_load1_g1200_t45",
        top20_rank=1,
        input_parameter="curtailment_factor",
        input_unit="dimensionless",
        observable="p_export_mw",
        observable_unit="MW",
        value=2.0,
        ad_converged=True,
        load_multiplier_mv_bus_2=1.0,
        g_poa_wm2=1200.0,
        t_amb_c=45.0,
        wind_ms=2.0,
        curtailment_factor=1.0,
        pv_size_factor=1.0,
        kappa=-0.25,
    )


def _stub_branch_row(exp_module):
    return exp_module.BranchFlowRow(
        case_id="screen_load1_g1200_t45",
        case_type="screening",
        load_multiplier_mv_bus_2=1.0,
        pp_line_idx=1,
        compiled_line_idx=0,
        line_name="line",
        from_bus_name="MV Bus 0",
        to_bus_name="MV Bus 1",
        is_mv_branch=True,
        active=True,
        p_from_mw=1.0,
        q_from_mvar=0.1,
        s_from_mva=1.005,
        p_to_mw=-0.99,
        q_to_mvar=-0.09,
        s_to_mva=0.994,
        max_end_s_mva=1.005,
        notes="Flow magnitude only.",
    )


def _stub_summary_row(exp_module):
    return exp_module.RunSummaryRow(
        metric="n_screening_cases",
        value=48.0,
        unit="count",
        notes="PV screening cases only.",
    )


def test_export_all_writes_mandatory_artifacts(exp_module, tmp_path: Path):
    exp_module.export_all(
        [_stub_screening_row(exp_module)],
        [_stub_top_row(exp_module)],
        [_stub_sensitivity_row(exp_module)],
        [_stub_branch_row(exp_module)],
        [_stub_summary_row(exp_module)],
        tmp_path,
    )

    for name in exp_module.REQUIRED_ARTIFACTS:
        assert (tmp_path / name).exists(), f"Missing artifact: {name}"


def test_exported_csv_schemas_are_stable(exp_module, tmp_path: Path):
    exp_module.export_all(
        [_stub_screening_row(exp_module)],
        [_stub_top_row(exp_module)],
        [_stub_sensitivity_row(exp_module)],
        [_stub_branch_row(exp_module)],
        [_stub_summary_row(exp_module)],
        tmp_path,
    )

    with (tmp_path / "screening_results.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == exp_module.SCREENING_COLUMNS
        assert len(list(reader)) == 1

    with (tmp_path / "sensitivity_top20.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == exp_module.SENSITIVITY_COLUMNS
        rows = list(reader)
        assert rows[0]["input_parameter"] == "curtailment_factor"


def test_metadata_documents_no_optimization_and_top20_scope(exp_module, tmp_path: Path):
    exp_module.write_metadata(tmp_path)
    with (tmp_path / "metadata.json").open(encoding="utf-8") as handle:
        meta = json.load(handle)

    assert meta["experiment"] == "exp05a_network_screening"
    assert meta["scenario_design"]["n_screening_cases"] == 48
    assert meta["scenario_design"]["n_no_pv_reference_cases"] == 4
    assert meta["scenario_design"]["n_top_cases_for_sensitivity"] == 20
    assert meta["sensitivity_scope"]["input_parameter"] == "curtailment_factor"
    assert "no optimization" in meta["purpose"].lower()


def test_readme_mentions_non_normative_indicators(exp_module, tmp_path: Path):
    exp_module.write_readme(tmp_path)
    text = (tmp_path / "README.md").read_text(encoding="utf-8")

    assert "does not run an optimization" in text
    assert "demonstrator-internal stress indicators" in text
    assert "no line" in text.lower()
