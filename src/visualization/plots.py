from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _plot_confusion_matrix(confusion, output_dir: Path, fmt: str, dpi: int) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(np.array(confusion)).plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix")
    file_path = output_dir / f"confusion_matrix.{fmt}"
    fig.savefig(file_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return file_path


def _plot_roc(fpr, tpr, output_dir: Path, fmt: str, dpi: int) -> Path | None:
    if fpr is None or tpr is None:
        return None
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label="ROC Curve")
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    file_path = output_dir / f"roc_curve.{fmt}"
    fig.savefig(file_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return file_path


def _plot_feature_importance(model, output_dir: Path, fmt: str, dpi: int) -> Path | None:
    if not hasattr(model, "feature_importances_"):
        return None
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=importances[indices], y=[f"f{i}" for i in indices], ax=ax, palette="viridis")
    ax.set_title("Feature Importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    file_path = output_dir / f"feature_importance.{fmt}"
    fig.savefig(file_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return file_path


def _plot_learning_curve(curve_data: Dict, output_dir: Path, fmt: str, dpi: int) -> Path:
    train_sizes = curve_data["train_sizes"]
    train_scores = curve_data["train_scores"]
    validation_scores = curve_data["validation_scores"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(train_sizes, train_scores.mean(axis=1), label="Training score", marker="o")
    ax.fill_between(
        train_sizes,
        train_scores.mean(axis=1) - train_scores.std(axis=1),
        train_scores.mean(axis=1) + train_scores.std(axis=1),
        alpha=0.2,
    )
    ax.plot(train_sizes, validation_scores.mean(axis=1), label="Validation score", marker="s")
    ax.fill_between(
        train_sizes,
        validation_scores.mean(axis=1) - validation_scores.std(axis=1),
        validation_scores.mean(axis=1) + validation_scores.std(axis=1),
        alpha=0.2,
    )
    ax.set_xlabel("Training examples")
    ax.set_ylabel("F1 score")
    ax.set_title("Learning Curve")
    ax.legend()
    file_path = output_dir / f"learning_curve.{fmt}"
    fig.savefig(file_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return file_path


def generate_plots(model, eval_artifacts: Dict, curve_data: Dict, paths: Dict, viz_config: Dict, logger) -> Dict[str, Path]:
    """Generate confusion matrix, ROC curve, feature importance, and learning curve plots."""
    outputs_dir = Path(paths["outputs_dir"])
    figures_dir = outputs_dir / "figures"
    _ensure_dir(figures_dir)

    fmt = viz_config.get("figure_format", "png")
    dpi = viz_config.get("dpi", 120)

    artifacts = {}
    artifacts["confusion_matrix"] = _plot_confusion_matrix(eval_artifacts["confusion"], figures_dir, fmt, dpi)
    roc_path = _plot_roc(eval_artifacts["fpr"], eval_artifacts["tpr"], figures_dir, fmt, dpi)
    if roc_path:
        artifacts["roc_curve"] = roc_path

    fi_path = _plot_feature_importance(model, figures_dir, fmt, dpi)
    if fi_path:
        artifacts["feature_importance"] = fi_path

    artifacts["learning_curve"] = _plot_learning_curve(curve_data, figures_dir, fmt, dpi)

    logger.info("Saved visualization artifacts: %s", artifacts)
    return artifacts
