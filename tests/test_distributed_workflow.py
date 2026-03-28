from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent / "src"))

from distributed_workflow import (
    evaluate_classifier,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_learning_curves,
    plot_roc_curve,
    preprocess_dataset,
    split_dataset,
    tune_and_train_classifier,
)


def _sample_dataframe() -> pd.DataFrame:
    rows = []
    for i in range(20):
        rows.append(
            {
                "age": 25 + i,
                "income": 40000 + (i * 1500),
                "credit_score": 600 + i,
                "segment": "premium" if i % 2 == 0 else "standard",
                "churned": 0 if i % 3 else 1,
            }
        )
    return pd.DataFrame(rows)


def test_preprocess_and_split_produces_expected_shapes():
    df = _sample_dataframe()
    processed = preprocess_dataset(df, target_column="churned")
    splits = split_dataset(processed, test_size=0.25, random_state=0)

    assert processed["features"].shape[0] == df.shape[0]
    assert len(processed["feature_names"]) == processed["features"].shape[1]
    assert splits["X_train"].shape[0] + splits["X_test"].shape[0] == df.shape[0]
    assert set(["X_train", "X_test", "y_train", "y_test"]).issubset(splits.keys())


def test_training_and_evaluation_flow(tmp_path):
    df = _sample_dataframe()
    processed = preprocess_dataset(df, target_column="churned")
    splits = split_dataset(processed, test_size=0.25, random_state=1)

    training = tune_and_train_classifier(splits["X_train"], splits["y_train"], splits["feature_names"], random_state=0)
    evaluation = evaluate_classifier(training.model, splits["X_test"], splits["y_test"])

    # Metrics exist
    for key in ["accuracy", "precision", "recall", "f1_score"]:
        assert key in evaluation.metrics

    # Visualizations generate files
    figures_dir = tmp_path / "figures"
    confusion_path = plot_confusion_matrix(evaluation.confusion_matrix, figures_dir)
    feature_path = plot_feature_importance(
        list(training.model.feature_importances_), training.feature_names, figures_dir
    )
    learning_path = plot_learning_curves(
        training.learning_curve_data["train_sizes"],
        training.learning_curve_data["train_scores"],
        training.learning_curve_data["validation_scores"],
        figures_dir,
    )

    assert confusion_path.exists()
    assert feature_path.exists()
    assert learning_path.exists()

    if evaluation.roc_curve:
        roc_path = plot_roc_curve(
            evaluation.roc_curve["fpr"],
            evaluation.roc_curve["tpr"],
            evaluation.roc_curve["roc_auc"],
            figures_dir,
        )
        assert roc_path.exists()
