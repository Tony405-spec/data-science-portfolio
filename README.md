# data-science-portfolio

Simple practice workflow to simulate MLOps habits.

## Project structure

- `data/`: Raw, processed, and external datasets (tracked with placeholders only).
- `notebooks/`: Exploratory, modeling, and reporting notebooks.
- `scripts/`: CLI helpers for preprocessing, feature engineering, modeling, and visualization.
- `src/`: Reusable Python modules for data workflows.
- `tests/`: Pytest coverage for the reusable modules.
- `models/`: Saved model artifacts.
- `reports/`: Generated figures and HTML outputs.

## Getting started

1. Create a virtual environment.
2. Install dependencies:
   ```bash
   make install
   ```
3. Run tests:
   ```bash
   make test
   ```
