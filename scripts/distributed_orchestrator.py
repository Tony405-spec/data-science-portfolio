#!/usr/bin/env python3
"""
Distributed orchestrator that wires ingestion, preprocessing, training, evaluation, and visualization nodes.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
import sys

import joblib
from dask.distributed import Client, LocalCluster

sys.path.append(str(Path(__file__).parent.parent))

from distributed_workflow import (
    evaluate_classifier,
    ingest_data,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_learning_curves,
    plot_roc_curve,
    preprocess_dataset,
    split_dataset,
    summarize_dataframe,
    tune_and_train_classifier,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _persist_metrics(metrics: dict, roc_auc: float | None, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(metrics)
    if roc_auc is not None:
        payload["roc_auc"] = roc_auc
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return metrics_path


def _persist_model(model, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_path = output_dir / f"random_forest_{timestamp}.joblib"
    joblib.dump(model, model_path)
    return model_path


def _persist_learning_curve(data: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    curve_path = output_dir / "learning_curve.json"
    with curve_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    return curve_path


def orchestrate_pipeline(
    input_path: Path, target_column: str, results_dir: Path, n_workers: int = 3, test_size: float = 0.2
) -> dict:
    """
    Execute the distributed workflow using a LocalCluster to simulate multiple nodes.
    """
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=1,
        processes=True,
        dashboard_address=None,
        memory_limit="1GB",
    )
    logger.info("Started Dask cluster with %d workers", n_workers)

    with Client(cluster) as client:
        # Ingestion
        ingest_future = client.submit(ingest_data, input_path)
        summary_future = client.submit(summarize_dataframe, ingest_future)

        # Preprocessing and split
        preprocess_future = client.submit(preprocess_dataset, ingest_future, target_column)
        split_future = client.submit(split_dataset, preprocess_future, test_size=test_size)

        # Training with hyperparameter tuning
        training_future = client.submit(
            lambda split: tune_and_train_classifier(split["X_train"], split["y_train"], split["feature_names"]),
            split_future,
        )

        # Evaluation
        evaluation_future = client.submit(
            lambda split, training: evaluate_classifier(training.model, split["X_test"], split["y_test"]),
            split_future,
            training_future,
        )

        # Materialize outputs
        data_summary = summary_future.result()
        splits = split_future.result()
        training_artifacts = training_future.result()
        evaluation_artifacts = evaluation_future.result()

    figures_dir = results_dir / "figures"
    metrics_dir = results_dir / "metrics"
    models_dir = Path("models") / "distributed"

    confusion_path = plot_confusion_matrix(evaluation_artifacts.confusion_matrix, figures_dir)

    roc_path = None
    if evaluation_artifacts.roc_curve:
        roc_data = evaluation_artifacts.roc_curve
        roc_path = plot_roc_curve(roc_data["fpr"], roc_data["tpr"], roc_data["roc_auc"], figures_dir)

    feature_path = plot_feature_importance(
        list(training_artifacts.model.feature_importances_), training_artifacts.feature_names, figures_dir
    )

    learning_path = plot_learning_curves(
        training_artifacts.learning_curve_data["train_sizes"],
        training_artifacts.learning_curve_data["train_scores"],
        training_artifacts.learning_curve_data["validation_scores"],
        figures_dir,
    )

    metrics_path = _persist_metrics(
        metrics=evaluation_artifacts.metrics,
        roc_auc=evaluation_artifacts.roc_curve["roc_auc"] if evaluation_artifacts.roc_curve else None,
        output_dir=metrics_dir,
    )
    model_path = _persist_model(training_artifacts.model, models_dir)
    learning_curve_path = _persist_learning_curve(training_artifacts.learning_curve_data, metrics_dir)

    logger.info("Artifacts written to %s", results_dir)

    return {
        "data_summary": data_summary,
        "confusion_matrix_fig": confusion_path,
        "roc_curve_fig": roc_path,
        "feature_importance_fig": feature_path,
        "learning_curve_fig": learning_path,
        "metrics_path": metrics_path,
        "learning_curve_path": learning_curve_path,
        "model_path": model_path,
        "cluster_workers": n_workers,
        "test_size": test_size,
    }


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Run distributed ML workflow.")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=repo_root / "data" / "raw" / "sample_data.csv",
        help="Path to the raw CSV dataset.",
    )
    parser.add_argument("--target-column", type=str, default="churned", help="Name of the target column.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=repo_root / "reports" / "distributed",
        help="Directory to store metrics and figures.",
    )
    parser.add_argument("--workers", type=int, default=3, help="Number of workers to simulate.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proportion of test data.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = orchestrate_pipeline(
        input_path=args.input_path,
        target_column=args.target_column,
        results_dir=args.results_dir,
        n_workers=args.workers,
        test_size=args.test_size,
    )
    logger.info("Pipeline complete. Key artifacts: %s", outputs)


if __name__ == "__main__":
    main()
