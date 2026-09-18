from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
from joblib import parallel_backend
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, learning_curve


def _build_model(config: Dict) -> Tuple[RandomForestClassifier, Dict]:
    estimator = RandomForestClassifier(
        random_state=config.get("random_state", 42),
        n_jobs=config.get("n_jobs", -1),
    )
    param_grid = config.get(
        "hyperparameters",
        {"n_estimators": [150, 250], "max_depth": [5, 10, None], "min_samples_split": [2, 4]},
    )
    return estimator, param_grid


def train_model(X_train: np.ndarray, y_train: np.ndarray, config: Dict, client, paths: Dict, logger):
    """Train a model with distributed hyperparameter tuning using the Dask backend."""
    models_dir = Path(paths["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    estimator, param_grid = _build_model(config)
    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=3,
        scoring="f1",
        n_jobs=-1,
        refit=True,
        verbose=0,
    )

    logger.info("Starting distributed grid search with parameters: %s", param_grid)
    with parallel_backend("dask", client=client):
        search.fit(X_train, y_train)

    best_model = search.best_estimator_
    model_path = models_dir / "best_model.joblib"
    joblib.dump(best_model, model_path)
    logger.info("Best model saved to %s with score %.4f", model_path, search.best_score_)

    return best_model, search.cv_results_


def generate_learning_curve(model, X: np.ndarray, y: np.ndarray, logger):
    """Compute learning curve data points."""
    logger.info("Computing learning curves")
    train_sizes, train_scores, validation_scores = learning_curve(
        model, X, y, cv=3, scoring="f1", n_jobs=-1, train_sizes=np.linspace(0.2, 1.0, 5)
    )
    return {
        "train_sizes": train_sizes,
        "train_scores": train_scores,
        "validation_scores": validation_scores,
    }
