from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional


def handle_missing_values(df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    """Handle missing values in the dataframe."""
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    
    if strategy == "median":
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
    elif strategy == "mean":
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
    elif strategy == "drop":
        df_clean = df_clean.dropna()
    
    return df_clean


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic data cleaning operations."""
    df_clean = df.copy()
    # Remove duplicate rows
    df_clean = df_clean.drop_duplicates()
    return df_clean


def scale_features(df: pd.DataFrame, feature_cols: Optional[list] = None) -> pd.DataFrame:
    """Scale numerical features."""
    from sklearn.preprocessing import StandardScaler
    
    df_scaled = df.copy()
    if feature_cols is None:
        feature_cols = df_scaled.select_dtypes(include=[np.number]).columns
    
    scaler = StandardScaler()
    df_scaled[feature_cols] = scaler.fit_transform(df_scaled[feature_cols])
    
    return df_scaled


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional features."""
    df_feat = df.copy()
    numeric_cols = df_feat.select_dtypes(include=[np.number]).columns
    
    # Create interaction features between numeric columns
    if len(numeric_cols) >= 2:
        for i in range(min(3, len(numeric_cols))):
            for j in range(i+1, min(3, len(numeric_cols))):
                col_name = f"{numeric_cols[i]}_{numeric_cols[j]}_interaction"
                df_feat[col_name] = df_feat[numeric_cols[i]] * df_feat[numeric_cols[j]]
    
    return df_feat


def text_length_feature(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Extract text length as a feature."""
    df_feat = df.copy()
    if text_col in df_feat.columns:
        df_feat["text_length"] = df_feat[text_col].astype(str).str.len()
    
    return df_feat
