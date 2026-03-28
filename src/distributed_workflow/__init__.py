"""
Distributed workflow package containing modular pipeline nodes.
"""

from .ingestion_node import ingest_data, summarize_dataframe
from .preprocessing_node import preprocess_dataset, split_dataset
from .training_node import TrainingArtifacts, tune_and_train_classifier
from .evaluation_node import EvaluationArtifacts, evaluate_classifier
from .visualization_node import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_learning_curves,
    plot_roc_curve,
)

__all__ = [
    "EvaluationArtifacts",
    "TrainingArtifacts",
    "evaluate_classifier",
    "ingest_data",
    "plot_confusion_matrix",
    "plot_feature_importance",
    "plot_learning_curves",
    "plot_roc_curve",
    "preprocess_dataset",
    "split_dataset",
    "summarize_dataframe",
    "tune_and_train_classifier",
]
