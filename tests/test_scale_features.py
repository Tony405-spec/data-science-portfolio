import pandas as pd

from src.preprocessing import scale_features


def test_scale_features_without_numeric_columns_returns_tuple():
    df = pd.DataFrame({"category": ["A", "B", "C"]})

    df_scaled, scaler = scale_features(df)

    assert df_scaled.equals(df)
    assert scaler is None
