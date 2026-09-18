from __future__ import annotations

import logging
from pathlib import Path
from sklearn.datasets import make_classification
import pandas as pd


def ingest_data(ingestion_config: dict, paths: dict, logger: logging.Logger) -> Path:
    """Ingest or generate data for the pipeline."""
    logger.info("Starting data ingestion...")
    
    # Generate synthetic data for testing
    dataset_name = ingestion_config.get("dataset_name", "synthetic_data")
    n_samples = ingestion_config.get("n_samples", 1000)
    n_features = ingestion_config.get("n_features", 20)
    n_informative = ingestion_config.get("n_informative", 10)
    n_redundant = ingestion_config.get("n_redundant", 5)
    n_classes = ingestion_config.get("n_classes", 2)
    class_sep = ingestion_config.get("class_sep", 1.0)
    random_state = ingestion_config.get("random_state", 42)
    
    logger.info(f"Generating {dataset_name} with {n_samples} samples, {n_features} features")
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=n_classes,
        class_sep=class_sep,
        random_state=random_state,
    )
    
    # Create DataFrame
    feature_cols = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_cols)
    df['target'] = y
    
    # Save raw data
    raw_path = Path(paths["raw_dir"]) / "raw_data.parquet"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(raw_path)
    
    logger.info(f"Saved raw data to {raw_path}")
    return raw_path
