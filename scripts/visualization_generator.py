#!/usr/bin/env python3
"""
Visualization Generator Script
Creates automated plots from processed data
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import sys
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))


def setup_plotting_style():
    """Configure plotting style for consistent visuals"""
    plt.style.use("seaborn-v0_8-darkgrid")
    sns.set_palette("husl")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["font.size"] = 12


def load_processed_data(file_path):
    """Load processed dataset"""
    df = pd.read_csv(file_path)
    logger.info(f"Loaded {file_path}: {df.shape}")
    return df


def create_distribution_plots(df, output_dir, dataset_name):
    """Create distribution plots for numeric columns"""
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    if len(numeric_cols) == 0:
        logger.warning(f"No numeric columns found in {dataset_name}")
        return

    # Limit to first 8 columns to avoid too many plots
    for col in numeric_cols[:8]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        axes[0].hist(df[col].dropna(), bins=30, edgecolor="black", alpha=0.7)
        axes[0].set_title(f"Distribution of {col}")
        axes[0].set_xlabel(col)
        axes[0].set_ylabel("Frequency")

        # Box plot
        axes[1].boxplot(df[col].dropna())
        axes[1].set_title(f"Box Plot of {col}")
        axes[1].set_ylabel(col)

        plt.tight_layout()

        # Save plot
        plot_path = output_dir / f"{dataset_name}_{col}_distribution.png"
        plt.savefig(plot_path, dpi=100, bbox_inches="tight")
        plt.close()

        logger.info(f"Created distribution plot for {col}")


def create_correlation_heatmap(df, output_dir, dataset_name):
    """Create correlation heatmap for numeric features"""
    numeric_df = df.select_dtypes(include=["int64", "float64"])

    if len(numeric_df.columns) < 2:
        logger.warning(f"Not enough numeric columns for correlation in {dataset_name}")
        return

    plt.figure(figsize=(12, 8))
    correlation_matrix = numeric_df.corr()

    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap="coolwarm", center=0, fmt=".2f", square=True, linewidths=1)

    plt.title(f"Feature Correlation Heatmap - {dataset_name}")
    plt.tight_layout()

    # Save plot
    plot_path = output_dir / f"{dataset_name}_correlation_heatmap.png"
    plt.savefig(plot_path, dpi=100, bbox_inches="tight")
    plt.close()

    logger.info(f"Created correlation heatmap for {dataset_name}")


def create_pairplot(df, output_dir, dataset_name):
    """Create pairplot for small datasets"""
    numeric_df = df.select_dtypes(include=["int64", "float64"])

    if len(numeric_df.columns) <= 5 and len(numeric_df.columns) >= 2:
        if len(df) <= 500:  # Limit to 500 rows for performance
            g = sns.pairplot(numeric_df, diag_kind="kde")
            g.fig.suptitle(f"Pairplot - {dataset_name}", y=1.02)

            # Save plot
            plot_path = output_dir / f"{dataset_name}_pairplot.png"
            plt.savefig(plot_path, dpi=100, bbox_inches="tight")
            plt.close()

            logger.info(f"Created pairplot for {dataset_name}")


def create_target_analysis(df, output_dir, dataset_name):
    """Create analysis plots if target column exists"""
    # Look for common target column names
    target_cols = ["churned", "target", "purchase_amount", "label"]

    for target in target_cols:
        if target in df.columns:
            numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

            # Create box plots by target
            if df[target].nunique() <= 10:  # Categorical target
                for col in numeric_cols[:5]:
                    if col != target:
                        plt.figure(figsize=(10, 6))
                        df.boxplot(column=col, by=target)
                        plt.title(f"{col} by {target}")
                        plt.suptitle("")
                        plt.tight_layout()

                        plot_path = output_dir / f"{dataset_name}_{col}_by_{target}.png"
                        plt.savefig(plot_path, dpi=100, bbox_inches="tight")
                        plt.close()

            break


def main():
    """Main execution function"""
    logger.info("Starting visualization generation")

    # Setup
    setup_plotting_style()
    figures_dir = Path("reports/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Process each dataset
    processed_dir = Path("data/processed")
    processed_files = list(processed_dir.glob("*_processed.csv"))

    if not processed_files:
        logger.warning("No processed datasets found")
        return

    for file_path in processed_files:
        try:
            # Load data
            df = load_processed_data(file_path)
            dataset_name = file_path.stem.replace("_processed", "")

            # Create dataset-specific subdirectory
            dataset_fig_dir = figures_dir / dataset_name
            dataset_fig_dir.mkdir(exist_ok=True)

            logger.info(f"Generating visualizations for {dataset_name}")

            # Generate all plots
            create_distribution_plots(df, dataset_fig_dir, dataset_name)
            create_correlation_heatmap(df, dataset_fig_dir, dataset_name)
            create_pairplot(df, dataset_fig_dir, dataset_name)
            create_target_analysis(df, dataset_fig_dir, dataset_name)

            logger.info(f"Completed visualizations for {dataset_name}")

        except Exception as e:
            logger.error(f"Failed to generate visualizations for {file_path}: {str(e)}")
            raise


if __name__ == "__main__":
    main()
