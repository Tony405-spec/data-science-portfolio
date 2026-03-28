import json
import logging
import sys
from pathlib import Path

import yaml

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from src.orchestration import run_pipeline


def _write_config(tmp_path: Path) -> Path:
    config = {
        "distributed": {"n_workers": 2, "threads_per_worker": 1, "memory_limit": "512MB"},
        "paths": {
            "raw_dir": str(tmp_path / "data" / "raw"),
            "processed_dir": str(tmp_path / "data" / "processed"),
            "models_dir": str(tmp_path / "models"),
            "outputs_dir": str(tmp_path / "outputs"),
        },
        "ingestion": {
            "dataset_name": "test_dataset",
            "n_samples": 200,
            "n_features": 8,
            "n_informative": 4,
            "n_redundant": 2,
            "n_classes": 2,
            "class_sep": 1.0,
            "chunks": 2,
            "random_state": 7,
        },
        "preprocessing": {"impute_strategy": "median", "scale": True, "test_size": 0.25, "random_state": 7},
        "training": {
            "model_type": "random_forest",
            "hyperparameters": {"n_estimators": [50, 100], "max_depth": [3, None], "min_samples_split": [2]},
            "n_jobs": -1,
        },
        "evaluation": {"metrics": ["accuracy", "precision", "recall", "f1", "roc_auc"]},
        "visualization": {"figure_format": "png", "dpi": 80},
    }
    config_path = tmp_path / "config.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle)
    return config_path


def _write_logging(tmp_path: Path) -> Path:
    logging_config = {
        "version": 1,
        "formatters": {"standard": {"format": "%(levelname)s:%(name)s:%(message)s"}},
        "handlers": {"console": {"class": "logging.StreamHandler", "formatter": "standard", "level": "INFO"}},
        "loggers": {"ml_pipeline": {"handlers": ["console"], "level": "INFO", "propagate": False}},
    }
    log_path = tmp_path / "logging.yaml"
    with log_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(logging_config, handle)
    return log_path


def test_run_pipeline_end_to_end(tmp_path, monkeypatch):
    # Ensure clean logging for test
    logging.getLogger("ml_pipeline").handlers.clear()

    config_path = _write_config(tmp_path)
    logging_path = _write_logging(tmp_path)

    result = run_pipeline(config_path=str(config_path), logging_config=str(logging_path))

    assert "metrics" in result
    assert result["metrics"]["accuracy"] > 0

    outputs_dir = Path(yaml.safe_load(config_path.read_text())["paths"]["outputs_dir"])
    metrics_file = outputs_dir / "metrics.json"
    assert metrics_file.exists()
    metrics_data = json.loads(metrics_file.read_text())
    assert "metrics" in metrics_data
    figures_dir = outputs_dir / "figures"
    assert figures_dir.exists()
    assert any(figures_dir.glob("*.png"))
