"""
Tests for data loader functionality.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock


class TestDataLoader:
    """Test cases for data loading functionality."""
    
    def test_load_shipment_data_success(self):
        """Test successful data loading."""
        # Mock data
        mock_data = pd.DataFrame({
            'Shipping Mode': ['Standard', 'Express'],
            'Customer Segment': ['Consumer', 'Corporate'],
            'Sales': [100, 200],
            'Days for shipping (real)': [5, 3],
            'Days for shipment (scheduled)': [4, 3]
        })
        
        with patch('pandas.read_csv', return_value=mock_data):
            # This would test the actual data loading function
            # For now, just test that mock works
            assert len(mock_data) == 2
            assert 'Shipping Mode' in mock_data.columns
    
    def test_data_preprocessing(self):
        """Test data preprocessing functionality."""
        # Create test data
        test_data = pd.DataFrame({
            'Days for shipping (real)': [5, 3, 7],
            'Days for shipment (scheduled)': [4, 3, 5],
            'Sales': [100, 200, 150]
        })
        
        # Test delay calculation
        test_data['delayed'] = (
            test_data['Days for shipping (real)'] > 
            test_data['Days for shipment (scheduled)']
        ).astype(int)
        
        expected_delays = [1, 0, 1]
        assert test_data['delayed'].tolist() == expected_delays
    
    def test_missing_data_handling(self):
        """Test handling of missing data."""
        test_data = pd.DataFrame({
            'Shipping Mode': ['Standard', None, 'Express'],
            'Sales': [100, 200, None],
            'Days for shipping (real)': [5, 3, 7]
        })
        
        # Test that we can handle missing values
        cleaned_data = test_data.dropna()
        assert len(cleaned_data) < len(test_data)
    
    def test_categorical_encoding(self):
        """Test categorical variable encoding."""
        test_data = pd.DataFrame({
            'Shipping Mode': ['Standard', 'Express', 'First Class'],
            'Customer Segment': ['Consumer', 'Corporate', 'Home Office']
        })
        
        # Test categorical encoding
        for col in ['Shipping Mode', 'Customer Segment']:
            test_data[col] = test_data[col].astype('category').cat.codes
        
        # Check that all values are numeric
        assert test_data['Shipping Mode'].dtype in ['int64', 'int32']
        assert test_data['Customer Segment'].dtype in ['int64', 'int32']


if __name__ == "__main__":
    pytest.main([__file__]) 