import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
import numpy as np
from src.utils import compute_spillover_index

@pytest.fixture
def sample_data():
    """Load test data from rv5_sqrt_24.csv"""
    return pd.read_csv('data/rv5_sqrt_24.csv', index_col=0) * 100

def test_compute_spillover_index_basic(sample_data):
    """Test basic functionality of compute_spillover_index"""
    # Test with small subset of data
    test_data = sample_data.iloc[:1000]  # Use first 100 rows
    
    # Test parameters
    horizon = 5
    lag = 22
    scarcity_prop = 0.0
    
    # Compute spillover index
    spillover_matrix = compute_spillover_index(
        test_data, 
        horizon=horizon, 
        lag=lag, 
        scarcity_prop=scarcity_prop,
        standardized=True
    )
    
    # Basic checks
    assert isinstance(spillover_matrix, np.ndarray)
    assert spillover_matrix.shape[0] == spillover_matrix.shape[1]
    assert spillover_matrix.shape[0] == len(test_data.columns)
    
    # Check if matrix elements are percentages (0-100)
    assert np.all(spillover_matrix >= 0)
    assert np.all(spillover_matrix <= 100)
    
    # Check diagonal elements are zero (as per the code)
    assert np.allclose(np.diag(spillover_matrix), 0)

def test_compute_spillover_index_scarcity(sample_data):
    """Test spillover index with different scarcity levels"""
    test_data = sample_data.iloc[:100]
    horizon = 5
    lag = 22
    
    # Compare results with different scarcity levels
    matrix_no_scarcity = compute_spillover_index(
        test_data, 
        horizon=horizon, 
        lag=lag, 
        scarcity_prop=0.0,
        standardized=True
    )
    
    matrix_with_scarcity = compute_spillover_index(
        test_data, 
        horizon=horizon, 
        lag=lag, 
        scarcity_prop=0.5,  # 50% scarcity
        standardized=True
    )
    
    # Matrix with scarcity should have more zeros
    assert np.sum(matrix_with_scarcity == 0) > np.sum(matrix_no_scarcity == 0)

def test_compute_spillover_index_standardization(sample_data):
    """Test spillover index with and without standardization"""
    test_data = sample_data.iloc[:100]
    horizon = 5
    lag = 22
    scarcity_prop = 0.0
    
    # Compare standardized vs non-standardized results
    matrix_standardized = compute_spillover_index(
        test_data, 
        horizon=horizon, 
        lag=lag, 
        scarcity_prop=scarcity_prop,
        standardized=True
    )
    
    matrix_non_standardized = compute_spillover_index(
        test_data, 
        horizon=horizon, 
        lag=lag, 
        scarcity_prop=scarcity_prop,
        standardized=False
    )
    
    # Results should be different
    assert not np.allclose(matrix_standardized, matrix_non_standardized)

def test_compute_spillover_index_parameters(sample_data):
    """Test spillover index with different parameters"""
    test_data = sample_data.iloc[:100]
    
    # Test with different horizons
    matrix_h5 = compute_spillover_index(
        test_data, 
        horizon=5, 
        lag=22, 
        scarcity_prop=0.0,
        standardized=True
    )
    
    matrix_h10 = compute_spillover_index(
        test_data, 
        horizon=10, 
        lag=22, 
        scarcity_prop=0.0,
        standardized=True
    )
    
    # Results should be different for different horizons
    assert not np.allclose(matrix_h5, matrix_h10)
    
    # Test with different lags
    matrix_lag22 = compute_spillover_index(
        test_data, 
        horizon=5, 
        lag=22, 
        scarcity_prop=0.0,
        standardized=True
    )
    
    matrix_lag10 = compute_spillover_index(
        test_data, 
        horizon=5, 
        lag=10, 
        scarcity_prop=0.0,
        standardized=True
    )
    
    # Results should be different for different lags
    assert not np.allclose(matrix_lag22, matrix_lag10)

def test_compute_spillover_index_invalid_inputs():
    """Test spillover index with invalid inputs"""
    # Create dummy data
    dummy_data = pd.DataFrame(
        np.random.randn(100, 3),
        columns=['A', 'B', 'C']
    )
    
    # Test with invalid horizon
    with pytest.raises(ValueError):
        compute_spillover_index(
            dummy_data, 
            horizon=0,  # Invalid horizon
            lag=22, 
            scarcity_prop=0.0,
            standardized=True
        )
    
    # Test with invalid lag
    with pytest.raises(ValueError):
        compute_spillover_index(
            dummy_data, 
            horizon=5,
            lag=0,  # Invalid lag
            scarcity_prop=0.0,
            standardized=True
        )
    
    # Test with invalid scarcity_prop
    with pytest.raises(ValueError):
        compute_spillover_index(
            dummy_data, 
            horizon=5,
            lag=22,
            scarcity_prop=1.5,  # Invalid scarcity proportion
            standardized=True
        )