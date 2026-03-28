# Multi-Node Machine Learning Workflow

This repository delivers a production-ready, portfolio-quality machine learning pipeline that runs across multiple nodes (simulated with Dask). It implements ingestion, preprocessing, training with distributed hyperparameter search, evaluation, and rich visualizations in modular, fine-grained steps.

## High-level architecture

```
[Ingestion] -> [Preprocessing] -> [Training] -> [Evaluation] -> [Visualization]
       |              |                |             |                 |
   Raw data       Processed       Tuned model     Metrics         Reports/plots
       |              |                |             |                 |
       +------[Orchestration & Config-driven control via Dask]---------+
```

- **Ingestion node**: Generates or loads raw data, partitions work across Dask workers, and stores Parquet shards.
- **Preprocessing node**: Cleans, imputes, scales, and materializes processed features.
- **Training node**: Runs distributed `GridSearchCV` (via the Dask joblib backend) to tune a Random Forest.
- **Evaluation node**: Computes accuracy, precision, recall, F1, ROC-AUC, and saves metrics to disk.
- **Visualization node**: Produces confusion matrix, ROC curve, feature importance, and learning/validation curves.
- **Orchestration**: `src/orchestration/pipeline.py` wires everything together under a configurable Dask `LocalCluster` (multi-node simulation).

## Repository structure

```
data-science-portfolio/
├── configs/
│   ├── config.yaml            # Pipeline configuration
│   └── logging.yaml           # Logging configuration
├── data/
│   ├── raw/                   # Raw ingested data (Parquet)
│   └── processed/             # Cleaned & transformed data
├── logs/                      # Pipeline logs
├── models/                    # Trained model artifacts
├── notebooks/                 # Exploratory & report notebooks
├── outputs/                   # Metrics and visualization outputs
├── reports/                   # Static report assets
├── src/
│   ├── ingestion/             # Data ingestion node
│   ├── preprocessing/         # Cleaning, imputing, scaling
│   ├── training/              # Distributed training + tuning
│   ├── evaluation/            # Metrics and diagnostics
│   ├── visualization/         # Plotting utilities
│   ├── orchestration/         # Pipeline controller
│   └── utils/                 # Config + logging helpers
├── tests/                     # Unit and pipeline tests
├── main.py                    # Entry point
├── requirements.txt           # Dependencies
└── README.md
```

## Pipeline steps (fine-grained)

1. **Ingestion** (`src/ingestion/data_ingestion.py`)
   - Generate synthetic classification data via scikit-learn.
   - Partition generation across Dask workers (multi-node simulation).
   - Persist raw Parquet shards to `data/raw/`.
2. **Preprocessing** (`src/preprocessing/preprocess.py`)
   - Read raw Parquet with Dask.
   - Impute missing values (configurable strategy).
   - Scale features (standard scaling on/off via config).
   - Save processed Parquet to `data/processed/`.
3. **Training** (`src/training/trainer.py`)
   - Use Dask-backed `GridSearchCV` for distributed hyperparameter tuning of Random Forests.
   - Persist best model to `models/best_model.joblib`.
   - Capture CV results for diagnostics.
4. **Evaluation** (`src/evaluation/evaluation.py`)
   - Compute accuracy, precision, recall, F1, ROC-AUC.
   - Save `outputs/metrics.json` with metrics and confusion matrix.
5. **Visualization** (`src/visualization/plots.py`)
   - Confusion matrix, ROC curve, feature importance, learning/validation curves.
   - Save plots to `outputs/figures/`.
6. **Orchestration** (`src/orchestration/pipeline.py`)
   - Spins up Dask `LocalCluster` (configurable workers/threads) to simulate a multi-node environment.
   - Executes each node in sequence with clear logging and persisted artifacts.

## Configuration

- **Pipeline config**: `configs/config.yaml` controls paths, Dask cluster sizing, ingestion size, preprocessing options, training grid, and visualization settings.
- **Logging**: `configs/logging.yaml` sets console/file handlers; logs are written to `logs/pipeline.log`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Running the pipeline

```bash
python main.py                 # uses configs/config.yaml by default
# or
python -m src.orchestration.pipeline --config configs/config.yaml
```

Artifacts created:
- Raw data: `data/raw/*.parquet`
- Processed data: `data/processed/processed.parquet`
- Model: `models/best_model.joblib`
- Metrics: `outputs/metrics.json`
- Figures: `outputs/figures/{confusion_matrix,roc_curve,feature_importance,learning_curve}.png`

## Testing

```bash
pytest -q
```

The test suite includes an end-to-end pipeline run on a small synthetic dataset to validate ingestion, preprocessing, training, evaluation, and visualization steps.

## Production scaling

- **Cluster sizing**: Increase `distributed.n_workers` / `threads_per_worker` in `configs/config.yaml` and point to a remote Dask scheduler for real multi-node clusters.
- **Data sources**: Swap the synthetic ingestion step with connectors to object storage, warehouses, or streaming systems while keeping preprocessing/training nodes unchanged.
- **Resource isolation**: Containerize with Docker, deploy on Kubernetes, and bind Dask workers to dedicated node pools for elasticity.
- **Observability**: Extend logging to structured logs/metrics (OpenTelemetry, Prometheus) and wire alerts for SLA breaches.
- **Model governance**: Add model registries (e.g., MLflow) and drift detection while reusing the modular node structure.
