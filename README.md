# Data Science Portfolio

[![Data Science Portfolio CI](https://github.com/YOUR_USERNAME/data-science-portfolio/actions/workflows/data-science-ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/data-science-portfolio/actions/workflows/data-science-ci.yml)

A professional data science portfolio demonstrating automated CI/CD pipelines, reproducible analysis, and production-ready code practices.

## 🚀 Features

- **Automated CI/CD Pipeline**: GitHub Actions workflow that automatically tests, executes, and validates all code
- **Reproducible Analysis**: Jupyter notebooks that are automatically executed and validated
- **Professional Code Quality**: Linting, formatting, and unit tests
- **Automated Visualizations**: Scripts that generate plots from processed data
- **ML Model Training**: Automated model training and evaluation
- **Artifact Generation**: HTML reports and visualizations saved as artifacts

## 📁 Repository Structure
├── .github/workflows/ # CI/CD pipeline definitions
├── data/
│ ├── raw/ # Original, immutable data
│ └── processed/ # Cleaned and transformed data
├── notebooks/
│ ├── exploratory/ # EDA notebooks
│ └── reports/ # Final analysis notebooks
├── scripts/ # Reusable Python scripts
├── src/ # Source code modules
├── tests/ # Unit tests
├── models/ # Trained model artifacts
└── reports/ # Generated reports and figures
├── figures/
└── html/

## 🛠️ Technology Stack

- **Python 3.10+**
- **Data Analysis**: pandas, numpy
- **Machine Learning**: scikit-learn
- **Visualization**: matplotlib, seaborn
- **Distributed Execution**: Dask (LocalCluster simulation)
- **Notebooks**: Jupyter
- **Testing**: pytest
- **Code Quality**: black, flake8
- **CI/CD**: GitHub Actions

## 🧭 Distributed Multi-Node Workflow

The repository now includes a modular, Dask-powered workflow that simulates multiple nodes:

- **Ingestion Node**: Reads raw CSV data and reports dataset metadata.
- **Preprocessing Node**: Cleans data, imputes missing values, encodes categoricals, engineers features, and scales inputs.
- **Training Node**: Runs hyperparameter tuning for a RandomForest classifier and captures learning curves.
- **Evaluation Node**: Computes accuracy, precision, recall, F1, confusion matrix, and ROC/AUC.
- **Visualization Node**: Renders confusion matrix, ROC curve, feature importance, and training vs validation curves.
- **Orchestrator**: Coordinates all nodes on a Dask `LocalCluster` to simulate multi-worker execution.

Run the end-to-end workflow from the repository root with absolute paths:

```bash
python scripts/distributed_orchestrator.py \
  --input-path /home/runner/work/data-science-portfolio/data-science-portfolio/data/raw/sample_data.csv \
  --results-dir /home/runner/work/data-science-portfolio/data-science-portfolio/reports/distributed \
  --target-column churned \
  --workers 3
```

## 📊 Example Workflow

When you push code to this repository, GitHub Actions automatically:

1. ✅ Sets up Python environment
2. 📦 Installs all dependencies
3. 🔍 Runs code quality checks (flake8, black)
4. 🧪 Executes unit tests (pytest)
5. 📓 Runs all Jupyter notebooks
6. 🔄 Executes data preprocessing scripts
7. 📈 Generates visualizations
8. 🤖 Trains ML models
9. 📑 Creates HTML reports
10. 📤 Uploads all artifacts

## 🚦 Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/Tony405-spec/data-science-portfolio.git
   cd data-science-portfolio
