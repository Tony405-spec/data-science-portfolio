# Data Science Portfolio

[![Data Science Portfolio CI](https://github.com/skynet-datagrid-labs/data-science-portfolio/actions/workflows/data-science-ci.yml/badge.svg)](https://github.com/skynet-datagrid-labs/data-science-portfolio/actions/workflows/data-science-ci.yml)

A professional data science portfolio demonstrating automated CI/CD pipelines, reproducible analysis, and production-ready code practices.

## 🚀 Features

- **Automated CI/CD Pipeline**: GitHub Actions workflow that automatically tests, executes, and validates all code
- **Reproducible Analysis**: Jupyter notebooks that are automatically executed and validated
- **Professional Code Quality**: Linting, formatting, and unit tests
- **Automated Visualizations**: Scripts that generate plots from processed data
- **ML Model Training**: Automated model training and evaluation
- **Artifact Generation**: HTML reports and visualizations saved as artifacts

## 📁 Repository Structure
```text
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
```

## 🛠️ Technology Stack

- **Python 3.10+**
- **Data Analysis**: pandas, numpy
- **Machine Learning**: scikit-learn
- **Visualization**: matplotlib, seaborn
- **Notebooks**: Jupyter
- **Testing**: pytest
- **Code Quality**: black, flake8
- **CI/CD**: GitHub Actions

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
   git clone https://github.com/skynet-datagrid-labs/data-science-portfolio.git
   cd data-science-portfolio
   ```

2. **Create an environment and install dependencies**
   ```bash
   python -m venv .venv
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

3. **Run the test suite**
   ```bash
   python -m pytest
   ```

4. **Run the core scripts**
   ```bash
   python scripts/data_preprocessing.py
   python scripts/feature_engineering.py
   python scripts/model_training.py
   python scripts/visualization_generator.py
   ```

5. **Work with notebooks**
   ```bash
   jupyter notebook
   ```

   Start with `notebooks/exploratory/01_initial_eda.ipynb`, then review `notebooks/reports/02_final_analysis.ipynb`.

## Development Checks

Use these commands before opening a pull request:

```bash
python -m pytest
python -m black --check .
python -m flake8 .
```
