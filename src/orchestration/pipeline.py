from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from dask.distributed import Client, LocalCluster

from src.ingestion import ingest_data
from src.preprocessing import preprocess_data, split_data
from src.training import generate_learning_curve, train_model
from src.evaluation import evaluate_model
from src.visualization import generate_plots
from src.utils import load_config, setup_logging


def _create_client(dist_config: dict) -> Client:
    cluster = LocalCluster(
        n_workers=dist_config.get("n_workers", 2),
        threads_per_worker=dist_config.get("threads_per_worker", 1),
        memory_limit=dist_config.get("memory_limit", "1GB"),
    )
    return Client(cluster)


def run_pipeline(config_path: str = "configs/config.yaml", logging_config: str = "configs/logging.yaml") -> dict:
    """Run the full multi-node ML workflow."""
    setup_logging(logging_config)
    logger = logging.getLogger("ml_pipeline")
    config = load_config(config_path)
    paths = config["paths"]

    Path(paths["outputs_dir"]).mkdir(parents=True, exist_ok=True)

    with _create_client(config["distributed"]) as client:
        logger.info("Dask dashboard: %s", client.dashboard_link)

        raw_path = ingest_data(config["ingestion"], paths, logger)
        processed_path = preprocess_data(raw_path, config["preprocessing"], paths, logger)

        X_train, X_test, y_train, y_test = split_data(processed_path, config["preprocessing"])
        model, cv_results = train_model(X_train, y_train, config["training"], client, paths, logger)

        metrics, eval_artifacts = evaluate_model(model, X_test, y_test, paths, logger)

        # Re-load full dataset for learning curve to avoid data leakage from test split
        full_df = __import__("pandas").read_parquet(processed_path)
        curve_data = generate_learning_curve(
            model, full_df.drop(columns=["target"]).to_numpy(), full_df["target"].to_numpy(), logger
        )

        viz_artifacts = generate_plots(model, eval_artifacts, curve_data, paths, config["visualization"], logger)

    summary = {
        "metrics": metrics,
        "cv_results": cv_results,
        "visualizations": {k: str(v) for k, v in viz_artifacts.items()},
    }
    logger.info("Pipeline finished successfully: %s", summary)
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the distributed ML pipeline.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to pipeline configuration YAML.")
    parser.add_argument("--logging", dest="logging_config", default="configs/logging.yaml", help="Path to logging YAML.")
    args = parser.parse_args()

    run_pipeline(config_path=args.config, logging_config=args.logging_config)
