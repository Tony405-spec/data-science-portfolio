"""
Data preprocessing script that automatically cleans and prepares datasets.
This script will be executed by the CI pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_raw_data(filepath):
    """Load raw data from CSV file."""
    logger.info(f"Loading data from {filepath}")
    return pd.read_csv(filepath)

def clean_data(df):
    """Perform data cleaning operations."""
    logger.info("Cleaning data...")
    
    # Remove duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    logger.info(f"Removed {initial_rows - len(df)} duplicates")
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].fillna('Unknown')
    
    return df

def save_processed_data(df, output_path):
    """Save processed data to CSV."""
    logger.info(f"Saving processed data to {output_path}")
    df.to_csv(output_path, index=False)

def main():
    """Main execution function."""
    # Define paths
    raw_data_dir = Path('data/raw')
    processed_data_dir = Path('data/processed')
    
    # Create processed directory if it doesn't exist
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each raw dataset
    for raw_file in raw_data_dir.glob('*.csv'):
        logger.info(f"Processing {raw_file.name}")
        
        # Load and clean data
        df = load_raw_data(raw_file)
        df_clean = clean_data(df)
        
        # Save processed data
        output_file = processed_data_dir / f"processed_{raw_file.name}"
        save_processed_data(df_clean, output_file)
        
        logger.info(f"Successfully processed {raw_file.name}")

if __name__ == "__main__":
    main()
