from __future__ import annotations

from pathlib import Path
from typing import Dict

import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask import delayed
from sklearn.datasets import make_classification


def _generate_partition(config: Dict, seed: int) -> pd.DataFrame:
    """Generate a synthetic classification partition."""
    X, y = make_classification(
        n_samples=int(config["n_samples"] / config["chunks"]),
        n_features=config["n_features"],
        n_informative=config["n_informative"],
        n_redundant=config["n_redundant"],
        n_classes=config["n_classes"],
        class_sep=config.get("class_sep", 1.0),
        random_state=seed,
    )
    feature_cols = [f"feature_{i}" for i in range(config["n_features"])]
    df = pd.DataFrame(X, columns=feature_cols)
    df["target"] = y
    return df


def ingest_data(config: Dict, paths: Dict, logger) -> Path:
    """Create a synthetic dataset and persist to the raw data directory."""
    raw_dir = Path(paths["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = config["dataset_name"]
    logger.info("Generating synthetic dataset '%s' with %s partitions", dataset_name, config["chunks"])

    first_df = _generate_partition(config, seed=config["random_state"])
    delayed_parts = [
        delayed(_generate_partition)(config, seed=config["random_state"] + i + 1) for i in range(config["chunks"] - 1)
    ]

    ddf = dd.concat([dd.from_pandas(first_df, npartitions=1)] + [dd.from_delayed(d, meta=first_df) for d in delayed_parts])
    ddf = ddf.persist()

    raw_path = raw_dir / f"{dataset_name}.parquet"
    dd.to_parquet(ddf, raw_path, engine="pyarrow", overwrite=True)
    logger.info("Saved raw dataset to %s", raw_path)
    return raw_path
