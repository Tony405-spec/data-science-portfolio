from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn import metrics


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, paths: Dict, logger) -> Tuple[Dict, Dict]:
    """Evaluate the trained model and persist metrics."""
    outputs_dir = Path(paths["outputs_dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    results = {
        "accuracy": metrics.accuracy_score(y_test, y_pred),
        "precision": metrics.precision_score(y_test, y_pred, zero_division=0),
        "recall": metrics.recall_score(y_test, y_pred, zero_division=0),
        "f1": metrics.f1_score(y_test, y_pred, zero_division=0),
    }

    if y_proba is not None and len(np.unique(y_test)) == 2:
        results["roc_auc"] = metrics.roc_auc_score(y_test, y_proba)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_proba)
    else:
        fpr, tpr, thresholds = None, None, None

    confusion = metrics.confusion_matrix(y_test, y_pred).tolist()

    metrics_path = outputs_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump({"metrics": results, "confusion_matrix": confusion}, handle, indent=2)

    logger.info("Evaluation metrics saved to %s", metrics_path)
    return results, {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "confusion": confusion,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }
