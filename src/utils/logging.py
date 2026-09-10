from __future__ import annotations

import logging
import logging.config
from pathlib import Path

import yaml


def setup_logging(config_path: str | Path) -> None:
    """Configure logging using a YAML config file."""
    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)

    with config_file.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    log_dir = Path(config.get("handlers", {}).get("file", {}).get("filename", "logs/pipeline.log")).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.config.dictConfig(config)
