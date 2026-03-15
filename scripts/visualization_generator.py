"""
Generate automated visualizations from processed data.
These plots will be saved as artifacts.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def setup_plotting_style():
    """Configure matplotlib and seaborn styles."""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")

def generate_distribution_plots(df, output_dir):
    """Generate distribution plots for numeric columns."""
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    for col in numeric_cols[:5]:  # Limit to first 5 columns
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Histogram
        axes[0].hist(df[col].dropna(), bins=30, edgecolor='black')
        axes[0].set_title(f'Distribution of {col}')
        axes[0].set_xlabel(col)
        axes[0].set_ylabel('Frequency')
        
        # Box plot
        axes[1].boxplot(df[col].dropna())
        axes[1].set_title(f'Box Plot of {col}')
        axes[1].set_ylabel(col)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'distribution_{col}.png', dpi=100, bbox_inches='tight')
        plt.close()

def generate_correlation_heatmap(df, output_dir):
    """Generate correlation heatmap for numeric features."""
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    if len(numeric_df.columns) > 1:
        plt.figure(figsize=(10, 8))
        correlation_matrix = numeric_df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(output_dir / 'correlation_heatmap.png', dpi=100, bbox_inches='tight')
        plt.close()

def main():
    """Main execution function."""
    # Setup directories
    processed_data_dir = Path('data/processed')
    figures_dir = Path('reports/figures')
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    setup_plotting_style()
    
    # Generate visualizations for each processed dataset
    for processed_file in processed_data_dir.glob('*.csv'):
        print(f"Generating visualizations for {processed_file.name}")
        
        # Load data
        df = pd.read_csv(processed_file)
        
        # Create dataset-specific subdirectory
        dataset_name = processed_file.stem.replace('processed_', '')
        dataset_fig_dir = figures_dir / dataset_name
        dataset_fig_dir.mkdir(exist_ok=True)
        
        # Generate plots
        generate_distribution_plots(df, dataset_fig_dir)
        generate_correlation_heatmap(df, dataset_fig_dir)
        
        print(f"Generated visualizations in {dataset_fig_dir}")

if __name__ == "__main__":
    main()
