#!/usr/bin/env python3
"""
Data Preprocessing Script
Loads raw data, performs cleaning, and saves processed data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.preprocessing import clean_data, handle_missing_values

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories if they don't exist"""
    dirs = [
        Path('data/processed'),
        Path('reports/figures'),
        Path('models')
    ]
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory exists: {dir_path}")

def load_raw_data(file_path):
    """Load raw data from CSV file"""
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded {file_path}: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading {file_path}: {str(e)}")
        raise

def preprocess_dataset(df, dataset_name):
    """Apply preprocessing steps to dataset"""
    logger.info(f"Preprocessing {dataset_name}")
    
    # Make a copy to avoid modifying original
    df_processed = df.copy()
    
    # Handle missing values
    df_processed = handle_missing_values(df_processed)
    
    # Remove duplicates
    initial_rows = len(df_processed)
    df_processed = df_processed.drop_duplicates()
    if initial_rows > len(df_processed):
        logger.info(f"Removed {initial_rows - len(df_processed)} duplicate rows")
    
    # Basic data cleaning
    df_processed = clean_data(df_processed)
    
    return df_processed

def save_processed_data(df, output_path):
    """Save processed data to CSV"""
    df.to_csv(output_path, index=False)
    logger.info(f"Saved processed data to {output_path}")

def generate_data_summary(df, dataset_name):
    """Generate and save data summary statistics"""
    summary = {
        'dataset': dataset_name,
        'rows': len(df),
        'columns': list(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.astype(str).to_dict(),
        'numeric_summary': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {}
    }
    
    # Save summary as text file
    summary_path = Path('reports') / f'{dataset_name}_summary.txt'
    with open(summary_path, 'w') as f:
        f.write(f"Dataset Summary: {dataset_name}\n")
        f.write("="*50 + "\n")
        f.write(f"Rows: {summary['rows']}\n")
        f.write(f"Columns: {len(summary['columns'])}\n")
        f.write("\nColumn Names:\n")
        for col in summary['columns']:
            f.write(f"  - {col}\n")
    
    logger.info(f"Generated summary for {dataset_name}")

def main():
    """Main execution function"""
    logger.info("Starting data preprocessing pipeline")
    
    # Setup directories
    setup_directories()
    
    # Define paths
    raw_data_dir = Path('data/raw')
    processed_data_dir = Path('data/processed')
    
    # Process each CSV file in raw data directory
    csv_files = list(raw_data_dir.glob('*.csv'))
    
    if not csv_files:
        logger.warning("No CSV files found in data/raw/")
        return
    
    for raw_file in csv_files:
        try:
            # Load data
            df = load_raw_data(raw_file)
            
            # Preprocess
            dataset_name = raw_file.stem
            df_processed = preprocess_dataset(df, dataset_name)
            
            # Save processed data
            output_file = processed_data_dir / f"{dataset_name}_processed.csv"
            save_processed_data(df_processed, output_file)
            
            # Generate summary
            generate_data_summary(df_processed, dataset_name)
            
            logger.info(f"Successfully processed {raw_file.name}")
            
        except Exception as e:
            logger.error(f"Failed to process {raw_file.name}: {str(e)}")
            raise

if __name__ == "__main__":
    main()
