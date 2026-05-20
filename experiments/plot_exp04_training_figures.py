"""Create training-history figures from existing Experiment 4 artifacts.

This script reads only
``experiments/results/exp04_modular_upstream_nn_surrogate/training_history.csv``.
It does not train the surrogate, does not run power-flow solves, and does not
modify the numerical JAX core.

Run:
    python experiments/plot_exp04_training_figures.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_DIR = (
    Path(__file__).resolve().parent / "results" / "exp04_modular_upstream_nn_surrogate"
)
FIGURES_DIR = RESULTS_DIR / "figures"
TRAINING_HISTORY_FILENAME = "training_history.csv"

STEP_COLUMN_CANDIDATES = ("step", "iteration", "epoch")
LOSS_COLUMN_CANDIDATES = (
    ("train_mse", "val_mse"),
    ("train_loss", "val_loss"),
    ("training_loss", "validation_loss"),
)
MAE_COLUMN_CANDIDATES = (
    ("train_mae_mw", "val_mae_mw"),
    ("train_mae", "val_mae"),
    ("training_mae", "validation_mae"),
)
LEARNING_RATE_COLUMN_CANDIDATES = ("learning_rate", "lr")


def load_training_history(results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    """Load and validate the Experiment 4 training-history CSV artifact."""

    path = results_dir / TRAINING_HISTORY_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Required training-history artifact not found: {path}")

    history = pd.read_csv(path)
    validate_training_history(history)
    return history


def validate_training_history(history: pd.DataFrame) -> None:
    """Validate that a training-history table has a usable plotting schema."""

    if history.empty:
        raise ValueError("Training-history table is empty.")
    detect_step_column(history)
    detect_loss_columns(history)


def _find_existing_pair(
    history: pd.DataFrame,
    candidates: Sequence[tuple[str, str]],
    metric_name: str,
) -> tuple[str, str]:
    for train_column, val_column in candidates:
        if train_column in history.columns and val_column in history.columns:
            return train_column, val_column
    candidate_text = ", ".join(f"({train}, {val})" for train, val in candidates)
    raise ValueError(
        f"Could not find train/validation {metric_name} columns. "
        f"Expected one of: {candidate_text}."
    )


def detect_step_column(history: pd.DataFrame) -> str:
    """Return the column used as x-axis for the training curve."""

    for column in STEP_COLUMN_CANDIDATES:
        if column in history.columns:
            return column
    raise ValueError(
        "Could not find a training step column. "
        f"Expected one of: {', '.join(STEP_COLUMN_CANDIDATES)}."
    )


def detect_loss_columns(history: pd.DataFrame) -> tuple[str, str]:
    """Return the train/validation loss columns."""

    return _find_existing_pair(history, LOSS_COLUMN_CANDIDATES, "loss")


def detect_mae_columns(history: pd.DataFrame) -> tuple[str, str] | None:
    """Return train/validation MAE columns if present."""

    try:
        return _find_existing_pair(history, MAE_COLUMN_CANDIDATES, "MAE")
    except ValueError:
        return None


def detect_learning_rate_column(history: pd.DataFrame) -> str | None:
    """Return the learning-rate column if present."""

    for column in LEARNING_RATE_COLUMN_CANDIDATES:
        if column in history.columns:
            return column
    return None


def _numeric_series(history: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(history[column], errors="coerce")
    if values.isna().all():
        raise ValueError(f"Column {column!r} does not contain numeric values.")
    return values


def _prepared_metric_frame(
    history: pd.DataFrame,
    train_column: str,
    val_column: str,
) -> pd.DataFrame:
    step_column = detect_step_column(history)
    frame = pd.DataFrame(
        {
            "step": _numeric_series(history, step_column),
            "train": _numeric_series(history, train_column),
            "validation": _numeric_series(history, val_column),
        }
    ).dropna()
    if frame.empty:
        raise ValueError(
            f"No numeric rows remain for {step_column}, {train_column}, {val_column}."
        )
    return frame.sort_values("step")


def should_use_log_y(*series: pd.Series) -> bool:
    """Return True if all plotted y-values are finite and strictly positive."""

    values = np.concatenate([pd.to_numeric(item, errors="coerce").to_numpy() for item in series])
    finite = values[np.isfinite(values)]
    return bool(finite.size and np.all(finite > 0.0))


def _format_metric(value: float) -> str:
    return f"{value:.4g}"


def _plot_metric_curve(
    history: pd.DataFrame,
    train_column: str,
    val_column: str,
    title: str,
    y_label: str,
) -> tuple[plt.Figure, plt.Axes, bool, float, float]:
    frame = _prepared_metric_frame(history, train_column, val_column)
    final = frame.iloc[-1]
    final_train = float(final["train"])
    final_val = float(final["validation"])
    use_log_y = should_use_log_y(frame["train"], frame["validation"])

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(
        frame["step"],
        frame["train"],
        marker="o",
        markersize=3.5,
        linewidth=1.8,
        label=f"Train (final={_format_metric(final_train)})",
    )
    ax.plot(
        frame["step"],
        frame["validation"],
        marker="s",
        markersize=3.5,
        linewidth=1.8,
        label=f"Validation (final={_format_metric(final_val)})",
    )

    ax.set_title(title, pad=12)
    ax.set_xlabel("Training step")
    ax.set_ylabel(y_label)
    if use_log_y:
        ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.7)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    return fig, ax, use_log_y, final_train, final_val


def plot_training_loss_curve(
    history: pd.DataFrame,
    figures_dir: Path = FIGURES_DIR,
) -> tuple[Path, Path, bool, float, float]:
    """Plot train and validation loss from the training-history artifact."""

    train_column, val_column = detect_loss_columns(history)
    fig, _ax, use_log_y, final_train, final_val = _plot_metric_curve(
        history,
        train_column=train_column,
        val_column=val_column,
        title="Experiment 4: NN Surrogate Training Loss",
        y_label="MSE loss",
    )
    png_path, pdf_path = _save_figure(fig, "fig01_training_loss_curve", figures_dir)
    return png_path, pdf_path, use_log_y, final_train, final_val


def plot_training_mae_curve(
    history: pd.DataFrame,
    figures_dir: Path = FIGURES_DIR,
) -> tuple[Path, Path, bool, float, float] | None:
    """Plot train and validation MAE when MAE columns are available."""

    mae_columns = detect_mae_columns(history)
    if mae_columns is None:
        return None

    train_column, val_column = mae_columns
    fig, _ax, use_log_y, final_train, final_val = _plot_metric_curve(
        history,
        train_column=train_column,
        val_column=val_column,
        title="Experiment 4: NN Surrogate Training MAE",
        y_label="MAE [MW]",
    )
    png_path, pdf_path = _save_figure(fig, "fig02_training_mae_curve", figures_dir)
    return png_path, pdf_path, use_log_y, final_train, final_val


def plot_learning_rate_schedule(
    history: pd.DataFrame,
    figures_dir: Path = FIGURES_DIR,
) -> tuple[Path, Path, bool, float, float] | None:
    """Plot the exported learning-rate schedule when available."""

    lr_column = detect_learning_rate_column(history)
    if lr_column is None:
        return None
    step_column = detect_step_column(history)
    frame = pd.DataFrame(
        {
            "step": _numeric_series(history, step_column),
            "learning_rate": _numeric_series(history, lr_column),
        }
    ).dropna()
    if frame.empty:
        raise ValueError(f"No numeric rows remain for {step_column} and {lr_column}.")
    frame = frame.sort_values("step")
    use_log_y = should_use_log_y(frame["learning_rate"])

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(
        frame["step"],
        frame["learning_rate"],
        marker="o",
        markersize=3.5,
        linewidth=1.8,
        label="Learning rate",
    )
    ax.set_title("Experiment 4: Learning-Rate Schedule", pad=12)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Learning rate")
    if use_log_y:
        ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.7)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    png_path, pdf_path = _save_figure(fig, "fig03_learning_rate_schedule", figures_dir)
    return (
        png_path,
        pdf_path,
        use_log_y,
        float(frame["learning_rate"].iloc[0]),
        float(frame["learning_rate"].iloc[-1]),
    )


def _save_figure(fig: plt.Figure, stem: str, figures_dir: Path) -> tuple[Path, Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    png_path = figures_dir / f"{stem}.png"
    pdf_path = figures_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def _log_scale_text(use_log_y: bool) -> str:
    return "yes" if use_log_y else "no"


def write_figures_readme(
    figures_dir: Path,
    loss_summary: tuple[bool, float, float],
    mae_summary: tuple[bool, float, float] | None = None,
    lr_summary: tuple[bool, float, float] | None = None,
) -> Path:
    """Write a compact README for the Experiment 4 training-history figures."""

    figures_dir.mkdir(parents=True, exist_ok=True)
    loss_log_y, final_train_loss, final_val_loss = loss_summary

    lines = [
        "# Experiment 4 training figures",
        "",
        "This folder contains figures generated only from the existing "
        "`training_history.csv` artifact of Experiment 4.",
        "",
        "## Fig. 1 - NN surrogate training loss",
        "",
        "`fig01_training_loss_curve.png` and `fig01_training_loss_curve.pdf` "
        "plot the training and validation MSE loss over the recorded training "
        "steps. The y-axis uses logarithmic scaling: "
        f"{_log_scale_text(loss_log_y)}.",
        "",
        f"Final train MSE: `{final_train_loss:.8g}`.",
        "",
        f"Final validation MSE: `{final_val_loss:.8g}`.",
    ]

    if mae_summary is not None:
        mae_log_y, final_train_mae, final_val_mae = mae_summary
        lines.extend(
            [
                "",
                "## Fig. 2 - NN surrogate training MAE",
                "",
                "`fig02_training_mae_curve.png` and "
                "`fig02_training_mae_curve.pdf` plot the training and "
                "validation MAE in MW because the corresponding MAE columns are "
                "available in `training_history.csv`. The y-axis uses "
                f"logarithmic scaling: {_log_scale_text(mae_log_y)}.",
                "",
                f"Final train MAE: `{final_train_mae:.8g}` MW.",
                "",
                f"Final validation MAE: `{final_val_mae:.8g}` MW.",
            ]
        )

    if lr_summary is not None:
        lr_log_y, first_lr, final_lr = lr_summary
        lines.extend(
            [
                "",
                "## Fig. 3 - Learning-rate schedule",
                "",
                "`fig03_learning_rate_schedule.png` and "
                "`fig03_learning_rate_schedule.pdf` plot the learning rate "
                "recorded in `training_history.csv`. The y-axis uses "
                f"logarithmic scaling: {_log_scale_text(lr_log_y)}.",
                "",
                f"Initial exported learning rate: `{first_lr:.8g}`.",
                "",
                f"Final exported learning rate: `{final_lr:.8g}`.",
            ]
        )

    lines.extend(
        [
            "",
            "## Scope",
            "",
            "These figures are a pure re-visualization of existing Experiment 4 "
            "artifacts. The plotting pipeline does not start new power-flow "
            "solves, does not retrain the NN surrogate, and does not compute "
            "new AD or finite-difference sensitivities.",
            "",
        ]
    )

    readme_path = figures_dir / "README.md"
    readme_path.write_text("\n".join(lines), encoding="utf-8")
    return readme_path


def generate_figures(
    results_dir: Path = RESULTS_DIR,
    figures_dir: Path | None = None,
) -> list[Path]:
    """Generate all Experiment 4 training-history figures."""

    if figures_dir is None:
        figures_dir = results_dir / "figures"

    history = load_training_history(results_dir)
    outputs: list[Path] = []

    loss_png, loss_pdf, loss_log_y, final_train_loss, final_val_loss = (
        plot_training_loss_curve(history, figures_dir)
    )
    outputs.extend([loss_png, loss_pdf])

    mae_result = plot_training_mae_curve(history, figures_dir)
    mae_summary = None
    if mae_result is not None:
        mae_png, mae_pdf, mae_log_y, final_train_mae, final_val_mae = mae_result
        outputs.extend([mae_png, mae_pdf])
        mae_summary = (mae_log_y, final_train_mae, final_val_mae)

    lr_result = plot_learning_rate_schedule(history, figures_dir)
    lr_summary = None
    if lr_result is not None:
        lr_png, lr_pdf, lr_log_y, first_lr, final_lr = lr_result
        outputs.extend([lr_png, lr_pdf])
        lr_summary = (lr_log_y, first_lr, final_lr)

    readme_path = write_figures_readme(
        figures_dir,
        loss_summary=(loss_log_y, final_train_loss, final_val_loss),
        mae_summary=mae_summary,
        lr_summary=lr_summary,
    )
    outputs.append(readme_path)
    return outputs


def main() -> None:
    outputs = generate_figures()
    print("Generated Experiment 4 training figures:")
    for path in outputs:
        print(f"- {path}")


if __name__ == "__main__":
    main()
