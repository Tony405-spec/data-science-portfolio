"""
Training node for distributed workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
import logging

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, learning_curve

logger = logging.getLogger(__name__)


@dataclass
class TrainingArtifacts:
    model: RandomForestClassifier
    cv_results: Dict[str, Any]
    learning_curve_data: Dict[str, List[float]]
    feature_names: List[str]


def _hyperparameter_space() -> Dict[str, List[Any]]:
    return {
        "n_estimators": [100, 150, 200],
        "max_depth": [None, 5, 8, 12],
        "min_samples_split": [2, 4, 8],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
    }


def tune_and_train_classifier(
    X_train, y_train, feature_names: List[str], random_state: int = 42
) -> TrainingArtifacts:
    """
    Train a classification model with hyperparameter tuning and capture learning curves.
    """
    model = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=_hyperparameter_space(),
        n_iter=8,
        cv=3,
        n_jobs=-1,
        random_state=random_state,
        scoring="f1_weighted",
        return_train_score=True,
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    logger.info("Best RandomForest params: %s", search.best_params_)

    train_sizes, train_scores, validation_scores = learning_curve(
        best_model,
        X_train,
        y_train,
        cv=3,
        n_jobs=-1,
        train_sizes=np.linspace(0.5, 1.0, 5),
        scoring="f1_weighted",
    )

    learning_curve_data = {
        "train_sizes": train_sizes.tolist(),
        "train_scores": train_scores.mean(axis=1).tolist(),
        "validation_scores": validation_scores.mean(axis=1).tolist(),
    }

    return TrainingArtifacts(
        model=best_model,
        cv_results=search.cv_results_,
        learning_curve_data=learning_curve_data,
        feature_names=feature_names,
    )
