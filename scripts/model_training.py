#!/usr/bin/env python3
"""
Machine Learning Model Training Script
Trains and evaluates models on processed data
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path
import logging
import json
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_processed_data(file_path):
    """Load processed dataset"""
    df = pd.read_csv(file_path)
    logger.info(f"Loaded {file_path}: {df.shape}")
    return df


def prepare_data_for_modeling(df):
    """Prepare data for machine learning"""
    # Identify target column (look for common names)
    target_cols = ["churned", "target", "purchase_amount"]
    target = None

    for col in target_cols:
        if col in df.columns:
            target = col
            break

    if target is None:
        logger.warning("No target column found, skipping modeling")
        return None, None, None, None, None

    # Separate features and target
    X = df.drop(columns=[target])
    y = df[target]

    # Handle categorical variables
    categorical_cols = X.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Handle missing values
    X = X.fillna(X.mean())

    return X, y, target, categorical_cols, le


def train_classification_model(X_train, X_test, y_train, y_test):
    """Train a classification model"""
    logger.info("Training classification model")

    # Convert target to categorical if needed
    if y_train.dtype not in ["int64", "int32"]:
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
    }

    # Feature importance
    feature_importance = dict(zip(X_train.columns, model.feature_importances_))

    return model, metrics, feature_importance


def train_regression_model(X_train, X_test, y_train, y_test):
    """Train a regression model"""
    logger.info("Training regression model")

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    metrics = {
        "mse": mean_squared_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2_score": r2_score(y_test, y_pred),
    }

    # Feature importance
    feature_importance = dict(zip(X_train.columns, model.feature_importances_))

    return model, metrics, feature_importance


def save_model(model, metrics, feature_importance, dataset_name, model_type):
    """Save model and metrics"""
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = models_dir / f"{dataset_name}_{model_type}_{timestamp}.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")

    # Save metrics
    metrics_path = models_dir / f"{dataset_name}_{model_type}_{timestamp}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "dataset": dataset_name,
                "model_type": model_type,
                "metrics": metrics,
                "feature_importance": feature_importance,
                "timestamp": timestamp,
            },
            f,
            indent=4,
        )

    return model_path


def main():
    """Main execution function"""
    logger.info("Starting model training pipeline")

    processed_dir = Path("data/processed")
    processed_files = list(processed_dir.glob("*_processed.csv"))

    if not processed_files:
        logger.warning("No processed datasets found")
        return

    for file_path in processed_files:
        try:
            # Load data
            df = load_processed_data(file_path)
            dataset_name = file_path.stem.replace("_processed", "")

            # Prepare data
            X, y, target, categorical_cols, label_encoder = prepare_data_for_modeling(df)

            if X is None:
                continue

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Determine problem type and train
            if y.dtype in ["int64", "int32"] and y.nunique() <= 20:
                # Classification problem
                model, metrics, feature_importance = train_classification_model(X_train, X_test, y_train, y_test)
                model_type = "classification"
            else:
                # Regression problem
                model, metrics, feature_importance = train_regression_model(X_train, X_test, y_train, y_test)
                model_type = "regression"

            # Save model and metrics
            save_model(model, metrics, feature_importance, dataset_name, model_type)

            logger.info(f"Successfully trained {model_type} model for {dataset_name}")
            logger.info(f"Metrics: {metrics}")

        except Exception as e:
            logger.error(f"Failed to train model for {file_path}: {str(e)}")
            raise


if __name__ == "__main__":
    main()
