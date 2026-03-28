"""
Evaluation node for distributed workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import logging

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationArtifacts:
    metrics: Dict[str, float]
    confusion_matrix: List[List[int]]
    roc_curve: Optional[Dict[str, List[float]]]
    y_true: List[Any]
    y_pred: List[Any]
    y_score: Optional[List[float]]


def evaluate_classifier(model, X_test, y_test) -> EvaluationArtifacts:
    """
    Compute evaluation metrics, confusion matrix, and ROC curve where applicable.
    """
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
    }

    cm = confusion_matrix(y_test, y_pred).tolist()

    roc_payload: Optional[Dict[str, List[float]]] = None
    y_score: Optional[List[float]] = None
    try:
        if hasattr(model, "predict_proba"):
            score = model.predict_proba(X_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test, score, drop_intermediate=False)
            roc_payload = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": thresholds.tolist(),
                "roc_auc": float(roc_auc_score(y_test, score)),
            }
            y_score = score.tolist()
    except ValueError as exc:
        logger.warning("ROC computation skipped: %s", exc)

    logger.info("Evaluation metrics: %s", metrics)
    return EvaluationArtifacts(
        metrics=metrics, confusion_matrix=cm, roc_curve=roc_payload, y_true=y_test.tolist(), y_pred=y_pred.tolist(), y_score=y_score
    )
