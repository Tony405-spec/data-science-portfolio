"""
Data ingestion node for distributed workflow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def ingest_data(input_path: str | Path) -> pd.DataFrame:
    """
    Load a raw dataset into a pandas DataFrame.
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {path}")

    df = pd.read_csv(path)
    logger.info("Ingested dataset %s with %d rows and %d columns", path, len(df), len(df.columns))
    return df


def summarize_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """
    Build a lightweight summary for downstream logging/metadata.
    """
    return {
        "rows": len(df),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "null_counts": df.isnull().sum().to_dict(),
    }
