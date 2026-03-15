"""
Data preprocessing utilities
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

logger = logging.getLogger(__name__)


def handle_missing_values(df, strategy="median"):
    """
    Handle missing values in dataframe

    Parameters:
    df: pandas DataFrame
    strategy: 'median', 'mean', 'mode', or 'drop'
    """
    df_clean = df.copy()
    for col in df_clean.columns:
        if df_clean[col].isnull().any():
            if df_clean[col].dtype in ["int64", "float64"]:
                if strategy == "median":
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                elif strategy == "mean":
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
                elif strategy == "mode":
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
            else:
                if not df_clean[col].mode().empty:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
                else:
                    df_clean[col] = df_clean[col].fillna("Unknown")
    return df_clean


def clean_data(df):
    """
    Basic data cleaning operations
    """
    df_clean = df.copy()
    string_cols = df_clean.select_dtypes(include=["object"]).columns
    for col in string_cols:
        df_clean[col] = df_clean[col].str.strip()
    date_cols = [col for col in df_clean.columns if "date" in col.lower() or "time" in col.lower()]
    for col in date_cols:
        try:
            df_clean[col] = pd.to_datetime(df_clean[col])
        except (ValueError, TypeError):
            pass
    return df_clean


def scale_features(df, columns=None, scaler_type="standard"):
    """
    Scale numeric features

    Parameters:
    df: pandas DataFrame
    columns: list of columns to scale (if None, scale all numeric)
    scaler_type: 'standard' or 'minmax'
    """
    if columns is None:
        columns = df.select_dtypes(include=["int64", "float64"]).columns
    if len(columns) == 0:
        return df
    if scaler_type == "standard":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[columns] = scaler.fit_transform(df[columns])
    return df_scaled, scaler


def create_features(df):
    """
    Create new features from existing ones
    """
    df_new = df.copy()
    numeric_cols = df_new.select_dtypes(include=["int64", "float64"]).columns
    if len(numeric_cols) >= 2:
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                col1, col2 = numeric_cols[i], numeric_cols[j]
                df_new[f"{col1}_div_{col2}"] = df_new[col1] / (df_new[col2] + 1e-10)
                df_new[f"{col1}_mult_{col2}"] = df_new[col1] * df_new[col2]
    return df_new


def text_length_feature(rows, key):
    """
    Computes the length of the specified key in each row.
    Returns a list of dictionaries containing the length under the new key.
    """
    return [{f"{key}_length": len(row[key])} for row in rows]
