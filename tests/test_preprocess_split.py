from pathlib import Path

import pandas as pd

from src.preprocessing.preprocess import _stratify_target, split_data


def test_stratify_target_skips_single_sample_class():
    y = pd.Series([0, 0, 0, 1]).to_numpy()

    assert _stratify_target(y) is None


def test_stratify_target_keeps_balanced_classes():
    y = pd.Series([0, 0, 1, 1]).to_numpy()

    assert _stratify_target(y) is y


def test_split_data_handles_rare_class_without_stratify(tmp_path: Path):
    processed_path = tmp_path / "processed.parquet"
    df = pd.DataFrame(
        {
            "feature_0": [1.0, 2.0, 3.0, 4.0],
            "feature_1": [0.1, 0.2, 0.3, 0.4],
            "target": [0, 0, 0, 1],
        }
    )
    df.to_parquet(processed_path, index=False)

    X_train, X_test, y_train, y_test = split_data(
        processed_path,
        {"test_size": 0.25, "random_state": 7},
    )

    assert len(X_train) == 3
    assert len(X_test) == 1
    assert len(y_train) == 3
    assert len(y_test) == 1
