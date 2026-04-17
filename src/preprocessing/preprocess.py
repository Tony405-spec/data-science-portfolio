from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import dask.dataframe as dd
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def _load_raw(raw_path: Path) -> dd.DataFrame:
    return dd.read_parquet(raw_path)


def preprocess_data(raw_path: Path, config: Dict, paths: Dict, logger) -> Path:
    """Clean, impute, and scale features; save processed dataset."""
    processed_dir = Path(paths["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    ddf = _load_raw(raw_path)
    logger.info("Loaded raw dataset with shape %s", ddf.shape)

    df = ddf.compute()
    
    # Handle case where 'target' column might not exist (e.g., in test data generation)
    if "target" in df.columns:
        features = df.drop(columns=["target"])
        target = df["target"]
    else:
        # If no target column, assume all columns are features
        features = df
        target = None

    imputer = SimpleImputer(strategy=config.get("impute_strategy", "median"))
    imputed = imputer.fit_transform(features)

    if config.get("scale", True):
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(imputed)
        feature_names = list(features.columns)
        processed_df = pd.DataFrame(scaled_features, columns=feature_names)
    else:
        processed_df = pd.DataFrame(imputed, columns=features.columns)

    if target is not None:
        processed_df["target"] = target.to_numpy()

    processed_path = processed_dir / "processed.parquet"
    processed_df.to_parquet(processed_path, index=False)
    logger.info("Saved processed dataset to %s", processed_path)
    return processed_path


def split_data(processed_path: Path, config: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split processed data into train and test sets."""
    df = pd.read_parquet(processed_path)
    
    # Check if target column exists
    if "target" not in df.columns:
        raise ValueError("Processed data must contain a 'target' column for splitting")
    
    X = df.drop(columns=["target"]).to_numpy()
    y = df["target"].to_numpy()

    # Use stratify only if it makes sense (for classification with balanced classes)
    stratify_param = y if len(np.unique(y)) > 1 else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config.get("test_size", 0.2), 
        random_state=config.get("random_state", 42), 
        stratify=stratify_param
    )
    return X_train, X_test, y_train, y_test
