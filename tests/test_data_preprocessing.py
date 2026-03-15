"""
Unit tests for data preprocessing
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.preprocessing import handle_missing_values, clean_data, scale_features

class TestDataPreprocessing:
    
    def setup_method(self):
        """Setup test data before each test"""
        self.test_df = pd.DataFrame({
            'numeric_col': [1, 2, np.nan, 4, 5],
            'categorical_col': ['A', 'B', None, 'A', 'C'],
            'target': [0, 1, 0, 1, 0]
        })
    
    def test_handle_missing_values_median(self):
        """Test handling missing values with median strategy"""
        df_clean = handle_missing_values(self.test_df, strategy='median')
        
        # Check that no missing values remain
        assert df_clean.isnull().sum().sum() == 0
        
        # Check that numeric column was filled with median (3.0)
        assert df_clean.loc[2, 'numeric_col'] == 3.0
    
    def test_handle_missing_values_mean(self):
        """Test handling missing values with mean strategy"""
        df_clean = handle_missing_values(self.test_df, strategy='mean')
        
        # Calculate mean excluding NaN
        mean_val = self.test_df['numeric_col'].mean()
        assert df_clean.loc[2, 'numeric_col'] == mean_val
    
    def test_handle_missing_values_categorical(self):
        """Test handling missing values in categorical columns"""
        df_clean = handle_missing_values(self.test_df)
        
        # Check categorical column was filled
        assert pd.notna(df_clean.loc[2, 'categorical_col'])
        # Should be filled with mode ('A')
        assert df_clean.loc[2, 'categorical_col'] == 'A'
    
    def test_clean_data_whitespace(self):
        """Test cleaning removes whitespace"""
        df_with_whitespace = pd.DataFrame({
            'string_col': ['  hello  ', 'world  ', '  test']
        })
        
        df_clean = clean_data(df_with_whitespace)
        
        assert df_clean['string_col'].iloc[0] == 'hello'
        assert df_clean['string_col'].iloc[1] == 'world'
        assert df_clean['string_col'].iloc[2] == 'test'
    
    def test_scale_features_standard(self):
        """Test standard scaling"""
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': [10, 20, 30, 40, 50]
        })
        
        df_scaled, scaler = scale_features(df, scaler_type='standard')
        
        # Check that scaled data has mean approx 0 and std approx 1
        assert abs(df_scaled['col1'].mean()) < 1e-10
        assert abs(df_scaled['col1'].std() - 1) < 1e-10
    
    def test_scale_features_minmax(self):
        """Test min-max scaling"""
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5]
        })
        
        df_scaled, scaler = scale_features(df, scaler_type='minmax')
        
        # Check that scaled data is between 0 and 1
        assert df_scaled['col1'].min() == 0
        assert df_scaled['col1'].max() == 1

if __name__ == '__main__':
    pytest.main([__file__])
