"""Lightweight tests for Experiment 4 training-history plotting."""

from __future__ import annotations

import inspect
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def plot_module():
    import experiments.plot_exp04_training_figures as module

    return module


@pytest.fixture()
def sample_history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "step": [0, 50, 100],
            "train_mse": [1.0, 0.1, 0.01],
            "val_mse": [1.2, 0.11, 0.02],
            "train_mae_mw": [2.0, 1.0, 0.5],
            "val_mae_mw": [2.1, 1.1, 0.6],
            "learning_rate": [0.05, 0.02, 0.0005],
        }
    )


def test_plot_module_is_importable(plot_module):
    assert plot_module is not None
    assert callable(plot_module.load_training_history)
    assert callable(plot_module.detect_step_column)
    assert callable(plot_module.detect_loss_columns)
    assert callable(plot_module.detect_learning_rate_column)


def test_load_training_history_reads_existing_csv(plot_module):
    history = plot_module.load_training_history()

    assert not history.empty
    assert "train_mse" in history.columns
    assert "val_mse" in history.columns


def test_detect_step_column(plot_module, sample_history):
    assert plot_module.detect_step_column(sample_history) == "step"

    epoch_history = sample_history.rename(columns={"step": "epoch"})
    assert plot_module.detect_step_column(epoch_history) == "epoch"


def test_detect_loss_columns(plot_module, sample_history):
    assert plot_module.detect_loss_columns(sample_history) == ("train_mse", "val_mse")


def test_detect_mae_columns(plot_module, sample_history):
    assert plot_module.detect_mae_columns(sample_history) == (
        "train_mae_mw",
        "val_mae_mw",
    )


def test_detect_learning_rate_column(plot_module, sample_history):
    assert plot_module.detect_learning_rate_column(sample_history) == "learning_rate"


def test_validate_training_history_rejects_missing_loss_columns(
    plot_module,
    sample_history,
):
    invalid = sample_history.drop(columns=["train_mse", "val_mse"])

    with pytest.raises(ValueError, match="loss columns"):
        plot_module.validate_training_history(invalid)


def test_should_use_log_y_requires_positive_values(plot_module):
    assert plot_module.should_use_log_y(pd.Series([1.0, 0.1]), pd.Series([0.2]))
    assert not plot_module.should_use_log_y(pd.Series([1.0, 0.0]), pd.Series([0.2]))


def test_loss_curve_uses_log_scale_for_positive_values(plot_module, sample_history):
    train_col, val_col = plot_module.detect_loss_columns(sample_history)
    fig, ax, use_log_y, final_train, final_val = plot_module._plot_metric_curve(
        sample_history,
        train_column=train_col,
        val_column=val_col,
        title="test",
        y_label="loss",
    )

    assert use_log_y
    assert ax.get_yscale() == "log"
    assert final_train == pytest.approx(0.01)
    assert final_val == pytest.approx(0.02)
    plt.close(fig)


def test_training_loss_plot_writes_png_and_pdf(
    plot_module,
    sample_history,
    tmp_path: Path,
):
    png_path, pdf_path, use_log_y, final_train, final_val = (
        plot_module.plot_training_loss_curve(sample_history, tmp_path)
    )

    assert png_path.exists()
    assert pdf_path.exists()
    assert png_path.name == "fig01_training_loss_curve.png"
    assert pdf_path.name == "fig01_training_loss_curve.pdf"
    assert use_log_y
    assert np.isfinite([final_train, final_val]).all()


def test_training_mae_plot_writes_png_and_pdf(
    plot_module,
    sample_history,
    tmp_path: Path,
):
    result = plot_module.plot_training_mae_curve(sample_history, tmp_path)

    assert result is not None
    png_path, pdf_path, use_log_y, final_train, final_val = result
    assert png_path.exists()
    assert pdf_path.exists()
    assert png_path.name == "fig02_training_mae_curve.png"
    assert pdf_path.name == "fig02_training_mae_curve.pdf"
    assert use_log_y
    assert np.isfinite([final_train, final_val]).all()


def test_learning_rate_plot_writes_png_and_pdf(
    plot_module,
    sample_history,
    tmp_path: Path,
):
    result = plot_module.plot_learning_rate_schedule(sample_history, tmp_path)

    assert result is not None
    png_path, pdf_path, use_log_y, first_lr, final_lr = result
    assert png_path.exists()
    assert pdf_path.exists()
    assert png_path.name == "fig03_learning_rate_schedule.png"
    assert pdf_path.name == "fig03_learning_rate_schedule.pdf"
    assert use_log_y
    assert first_lr == pytest.approx(0.05)
    assert final_lr == pytest.approx(0.0005)


def test_generate_figures_from_existing_artifacts(plot_module, tmp_path: Path):
    outputs = plot_module.generate_figures(
        results_dir=plot_module.RESULTS_DIR,
        figures_dir=tmp_path,
    )
    output_names = {path.name for path in outputs}

    assert "fig01_training_loss_curve.png" in output_names
    assert "fig01_training_loss_curve.pdf" in output_names
    assert "fig02_training_mae_curve.png" in output_names
    assert "fig02_training_mae_curve.pdf" in output_names
    assert "fig03_learning_rate_schedule.png" in output_names
    assert "fig03_learning_rate_schedule.pdf" in output_names
    assert "README.md" in output_names
    assert all(path.exists() for path in outputs)


def test_figures_readme_documents_scope(plot_module, sample_history, tmp_path: Path):
    loss_png, loss_pdf, loss_log_y, final_train_loss, final_val_loss = (
        plot_module.plot_training_loss_curve(sample_history, tmp_path)
    )
    mae_png, mae_pdf, mae_log_y, final_train_mae, final_val_mae = (
        plot_module.plot_training_mae_curve(sample_history, tmp_path)
    )
    assert loss_png.exists()
    assert loss_pdf.exists()
    assert mae_png.exists()
    assert mae_pdf.exists()

    readme_path = plot_module.write_figures_readme(
        tmp_path,
        loss_summary=(loss_log_y, final_train_loss, final_val_loss),
        mae_summary=(mae_log_y, final_train_mae, final_val_mae),
        lr_summary=(True, 0.05, 0.0005),
    )

    text = readme_path.read_text(encoding="utf-8")
    assert "training_history.csv" in text
    assert "Final train MSE" in text
    assert "Final validation MSE" in text
    assert "Learning-rate schedule" in text
    assert "does not start new power-flow solves" in text
    assert "does not retrain the NN surrogate" in text


def test_plot_script_does_not_depend_on_training_or_power_flow(plot_module):
    source = inspect.getsource(plot_module)

    assert "pandapower" not in source
    assert "jax" not in source
    assert "train_surrogate(" not in source
    assert "run_experiment(" not in source
    assert "solve_power_flow" not in source
