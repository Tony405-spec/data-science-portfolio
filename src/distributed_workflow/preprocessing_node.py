"""
Preprocessing node for distributed workflow.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple
import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing.tools import clean_data, create_features, handle_missing_values, scale_features

logger = logging.getLogger(__name__)


def preprocess_dataset(df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
    """
    Clean, impute, encode, and scale features while separating the target.
    """
    if target_column not in df:
        raise ValueError(f"Target column '{target_column}' not found in dataframe")

    df_clean = clean_data(handle_missing_values(df))
    y = df_clean[target_column]
    X = df_clean.drop(columns=[target_column])
    X_encoded = pd.get_dummies(X, drop_first=True)
    X_engineered = create_features(X_encoded)
    X_scaled, scaler = scale_features(X_engineered)

    logger.info("Preprocessed dataset with %d features", X_scaled.shape[1])
    return {"features": X_scaled, "target": y, "feature_names": list(X_scaled.columns), "scaler": scaler}


def split_dataset(
    processed: Dict[str, Any], test_size: float = 0.2, random_state: int = 42
) -> Dict[str, Any]:
    """
    Perform a train/test split for modeling.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        processed["features"], processed["target"], test_size=test_size, random_state=random_state, stratify=None
    )

    logger.info("Completed train/test split: %d train rows, %d test rows", len(X_train), len(X_test))
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": processed["feature_names"],
        "scaler": processed["scaler"],
    }
