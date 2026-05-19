from __future__ import annotations

import math

import pandas as pd

import experiments.exp01_diagnose_pandapower_pi_reference as diagnostic
import experiments.exp01_validate_example_simple as exp01


def test_pi_runpp_kwargs_forces_pi_transformer_model() -> None:
    kwargs = diagnostic.pi_runpp_kwargs()

    assert kwargs["trafo_model"] == "pi"


def test_force_exp01_pi_reference_restores_original_kwargs() -> None:
    original_kwargs = dict(exp01.PP_RUNPP_KWARGS)
    try:
        exp01.PP_RUNPP_KWARGS = dict(original_kwargs, trafo_model="t")

        with diagnostic.force_exp01_pi_reference():
            assert exp01.PP_RUNPP_KWARGS["trafo_model"] == "pi"

        assert exp01.PP_RUNPP_KWARGS == dict(original_kwargs, trafo_model="t")
    finally:
        exp01.PP_RUNPP_KWARGS = original_kwargs


def test_build_comparison_summary_uses_scope_matched_maxima(tmp_path) -> None:
    main_results_dir = tmp_path / "main_exp01"
    main_results_dir.mkdir()
    pd.DataFrame(
        [
            {
                "scenario": "base",
                "reference_mode": "scope_matched",
                "max_vm_pu_abs_diff": 4.0,
                "max_va_degree_abs_diff": 8.0,
                "p_slack_mw_abs_diff": 2.0,
                "q_slack_mvar_abs_diff": 3.0,
                "total_p_loss_mw_abs_diff": 5.0,
                "total_q_loss_mvar_abs_diff": 6.0,
            },
            {
                "scenario": "base",
                "reference_mode": "all_elements",
                "max_vm_pu_abs_diff": 99.0,
                "max_va_degree_abs_diff": 99.0,
                "p_slack_mw_abs_diff": 99.0,
                "q_slack_mvar_abs_diff": 99.0,
                "total_p_loss_mw_abs_diff": 99.0,
                "total_q_loss_mvar_abs_diff": 99.0,
            },
        ]
    ).to_csv(main_results_dir / "validation_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario": "base",
                "reference_mode": "scope_matched",
                "pl_mw_abs_diff": 7.0,
                "q_hv_mvar_diffpf": 10.0,
                "q_lv_mvar_diffpf": 2.0,
                "q_hv_mvar_pp": 8.0,
                "q_lv_mvar_pp": 1.0,
            },
            {
                "scenario": "base",
                "reference_mode": "all_elements",
                "pl_mw_abs_diff": 99.0,
                "q_hv_mvar_diffpf": 99.0,
                "q_lv_mvar_diffpf": 99.0,
                "q_hv_mvar_pp": 0.0,
                "q_lv_mvar_pp": 0.0,
            },
        ]
    ).to_csv(main_results_dir / "trafo_flows.csv", index=False)
    pi_summary = pd.DataFrame(
        [
            {
                "max_vm_pu_abs_diff": 2.0,
                "max_va_degree_abs_diff": 4.0,
                "p_slack_mw_abs_diff": 1.0,
                "q_slack_mvar_abs_diff": 1.5,
                "total_p_loss_mw_abs_diff": 2.5,
                "total_q_loss_mvar_abs_diff": 3.0,
                "trafo_pl_mw_abs_diff": 3.5,
                "trafo_ql_mvar_abs_diff": 1.0,
            }
        ]
    )

    comparison = diagnostic.build_comparison_summary(pi_summary, main_results_dir)

    by_metric = comparison.set_index("metric")
    assert by_metric.loc["max_vm_pu_abs_diff", "previous_exp01_max"] == 4.0
    assert by_metric.loc["max_vm_pu_abs_diff", "pi_reference_max"] == 2.0
    assert by_metric.loc["max_vm_pu_abs_diff", "interpretation"] == "slightly_reduced"
    assert math.isclose(
        by_metric.loc["max_vm_pu_abs_diff", "reduction_factor"],
        2.0,
    )
    assert by_metric.loc["trafo_pl_mw_abs_diff", "previous_exp01_max"] == 7.0
    assert by_metric.loc["trafo_ql_mvar_abs_diff", "previous_exp01_max"] == 3.0
