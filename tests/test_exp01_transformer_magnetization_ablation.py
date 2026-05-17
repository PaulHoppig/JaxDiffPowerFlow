"""Tests for the Exp.-1 transformer magnetization ablation diagnostic."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pandapower.networks as pn
import pytest


@pytest.fixture(scope="module")
def exp_module():
    import experiments.exp01_transformer_magnetization_ablation as m

    return m


def test_module_is_importable(exp_module):
    assert exp_module is not None


def test_find_trafo_index_finds_example_simple_trafo(exp_module):
    net = pn.example_simple()
    idx = exp_module.find_trafo_index(net)

    assert str(net.trafo.at[idx, "name"]) == exp_module.TRAFO_NAME


def test_apply_ablation_sets_pfe_and_i0_to_zero(exp_module):
    net = pn.example_simple()
    ablated = exp_module.apply_trafo_magnetization_ablation(net)
    idx = exp_module.find_trafo_index(ablated)

    assert float(ablated.trafo.at[idx, "pfe_kw"]) == 0.0
    assert float(ablated.trafo.at[idx, "i0_percent"]) == 0.0


def test_apply_ablation_preserves_other_trafo_parameters(exp_module):
    net = pn.example_simple()
    idx = exp_module.find_trafo_index(net)
    keep_columns = ["vk_percent", "vkr_percent", "shift_degree", "tap_pos"]
    before = {col: net.trafo.at[idx, col] for col in keep_columns}

    ablated = exp_module.apply_trafo_magnetization_ablation(net)
    idx_after = exp_module.find_trafo_index(ablated)

    for col, expected in before.items():
        got = ablated.trafo.at[idx_after, col]
        assert got == expected


def test_apply_ablation_does_not_mutate_original(exp_module):
    net = pn.example_simple()
    idx = exp_module.find_trafo_index(net)

    _ = exp_module.apply_trafo_magnetization_ablation(net)

    assert float(net.trafo.at[idx, "pfe_kw"]) == 14.0
    assert float(net.trafo.at[idx, "i0_percent"]) == 0.07


def test_build_ablation_summary_on_synthetic_dataframe(exp_module):
    df = pd.DataFrame(
        [
            _synthetic_result("baseline", "s0", 0.010, 0.011, 0.012),
            _synthetic_result("baseline", "s1", 0.020, 0.021, 0.022),
            _synthetic_result("no_trafo_magnetization", "s0", 0.001, 0.002, 0.003),
            _synthetic_result("no_trafo_magnetization", "s1", 0.003, 0.004, 0.005),
        ]
    )

    summary = exp_module.build_ablation_summary(df)
    baseline = summary[summary["variant"] == "baseline"].iloc[0]
    ablated = summary[summary["variant"] == "no_trafo_magnetization"].iloc[0]

    assert int(baseline["n_scenarios"]) == 2
    assert bool(baseline["all_diffpf_converged"])
    assert math.isclose(float(baseline["mean_p_slack_mw_abs_diff"]), 0.015)
    assert math.isclose(float(baseline["max_total_p_loss_mw_abs_diff"]), 0.021)
    assert math.isclose(float(ablated["mean_trafo_pl_mw_abs_diff"]), 0.004)


def test_hypothesis_check_true_when_offsets_collapse(exp_module):
    summary = pd.DataFrame(
        [
            _synthetic_summary("baseline", 0.014, 0.014, 0.014),
            _synthetic_summary("no_trafo_magnetization", 0.0005, 0.0005, 0.0005),
        ]
    )

    check = exp_module.build_hypothesis_check(summary)

    assert check["supports_pfe_offset_hypothesis"] is True
    assert check["reduction_factor_total_p_loss"] > 10.0


def test_hypothesis_check_false_when_offsets_do_not_collapse(exp_module):
    summary = pd.DataFrame(
        [
            _synthetic_summary("baseline", 0.014, 0.014, 0.014),
            _synthetic_summary("no_trafo_magnetization", 0.012, 0.012, 0.012),
        ]
    )

    check = exp_module.build_hypothesis_check(summary)

    assert check["supports_pfe_offset_hypothesis"] is False


def test_export_artifacts_writes_expected_files(exp_module, tmp_path: Path):
    results = pd.DataFrame([_synthetic_result("baseline", "s0", 0.014, 0.014, 0.014)])
    summary = pd.DataFrame(
        [
            _synthetic_summary("baseline", 0.014, 0.014, 0.014),
            _synthetic_summary("no_trafo_magnetization", 0.0004, 0.0004, 0.0004),
        ]
    )
    hypothesis = exp_module.build_hypothesis_check(summary)

    exp_module.export_artifacts(results, summary, hypothesis, tmp_path)

    for name in [
        "ablation_results.csv",
        "ablation_results.json",
        "ablation_summary.csv",
        "ablation_summary.json",
        "hypothesis_check.json",
        "metadata.json",
        "README.md",
        "figures/fig01_p_offset_baseline_vs_ablation.png",
        "figures/fig01_p_offset_baseline_vs_ablation.pdf",
    ]:
        assert (tmp_path / name).exists(), f"{name} was not written"

    payload = json.loads((tmp_path / "hypothesis_check.json").read_text(encoding="utf-8"))
    assert payload["supports_pfe_offset_hypothesis"] is True


def _synthetic_result(
    variant: str,
    scenario: str,
    p_slack_diff: float,
    total_p_loss_diff: float,
    trafo_pl_diff: float,
) -> dict:
    return {
        "variant": variant,
        "scenario": scenario,
        "reference_mode": "scope_matched",
        "diffpf_converged": True,
        "pandapower_converged": True,
        "diffpf_iterations": 4,
        "diffpf_residual_norm": 1e-12,
        "max_vm_pu_abs_diff": 1e-6,
        "max_va_degree_abs_diff": 1e-5,
        "p_slack_mw_abs_diff": p_slack_diff,
        "q_slack_mvar_abs_diff": 0.01,
        "total_p_loss_mw_abs_diff": total_p_loss_diff,
        "total_q_loss_mvar_abs_diff": 0.02,
        "trafo_pl_mw_abs_diff": trafo_pl_diff,
        "trafo_ql_mvar_abs_diff": 0.03,
        "trafo_pl_mw_diffpf": 0.04,
        "trafo_pl_mw_pandapower": 0.03,
        "trafo_ql_mvar_diffpf": 0.2,
        "trafo_ql_mvar_pandapower": 0.17,
        "total_p_loss_mw_diffpf": 0.05,
        "total_p_loss_mw_pandapower": 0.04,
        "total_q_loss_mvar_diffpf": 0.3,
        "total_q_loss_mvar_pandapower": 0.28,
        "p_slack_mw_diffpf": -3.0,
        "p_slack_mw_pandapower": -3.014,
        "q_slack_mvar_diffpf": -1.0,
        "q_slack_mvar_pandapower": -1.01,
    }


def _synthetic_summary(
    variant: str,
    mean_p_slack: float,
    mean_total_p_loss: float,
    mean_trafo_pl: float,
) -> dict:
    return {
        "variant": variant,
        "n_scenarios": 7,
        "all_diffpf_converged": True,
        "all_pandapower_converged": True,
        "mean_p_slack_mw_abs_diff": mean_p_slack,
        "max_p_slack_mw_abs_diff": mean_p_slack,
        "mean_total_p_loss_mw_abs_diff": mean_total_p_loss,
        "max_total_p_loss_mw_abs_diff": mean_total_p_loss,
        "mean_trafo_pl_mw_abs_diff": mean_trafo_pl,
        "max_trafo_pl_mw_abs_diff": mean_trafo_pl,
        "mean_q_slack_mvar_abs_diff": 0.01,
        "mean_total_q_loss_mvar_abs_diff": 0.02,
        "mean_trafo_ql_mvar_abs_diff": 0.03,
    }
