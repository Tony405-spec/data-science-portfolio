"""
Visualization node for distributed workflow.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import logging

import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def plot_confusion_matrix(confusion: List[List[int]], output_dir: Path, labels: Optional[List[str]] = None) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / "confusion_matrix.png"

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    if labels:
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels, rotation=0)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    return fig_path


def plot_roc_curve(fpr: List[float], tpr: List[float], roc_auc: float, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / "roc_curve.png"

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})", color="darkorange")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    return fig_path


def plot_feature_importance(importances: List[float], feature_names: List[str], output_dir: Path, top_k: int = 15) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / "feature_importance.png"

    # Order features by importance
    pairs = list(zip(feature_names, importances))
    top_pairs = sorted(pairs, key=lambda pair: pair[1], reverse=True)[:top_k]
    names, scores = zip(*top_pairs)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=list(scores), y=list(names), ax=ax, palette="viridis")
    ax.set_title("Feature Importance (Top Features)")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    return fig_path


def plot_learning_curves(train_sizes: List[float], train_scores: List[float], validation_scores: List[float], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / "learning_curves.png"

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, train_scores, marker="o", label="Training score")
    ax.plot(train_sizes, validation_scores, marker="s", label="Validation score")
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score (F1 weighted)")
    ax.set_title("Training vs Validation Curves")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    return fig_path
