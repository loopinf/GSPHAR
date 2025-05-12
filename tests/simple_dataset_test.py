#!/usr/bin/env python
"""
Simple test script to verify the basic functionality of the dataset implementations.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import numpy as np
import pandas as pd
from src.data.dataset import LegacyGSPHAR_Dataset, GSPHAR_Dataset

def test_legacy_dataset():
    """Test the LegacyGSPHAR_Dataset implementation."""
    print("Testing LegacyGSPHAR_Dataset...")

    # Create a simple dictionary for testing
    data_dict = {
        'date1': {
            'y': pd.Series([1.0, 2.0, 3.0], index=['market_0', 'market_1', 'market_2']),
            'x_lag1': pd.Series([0.1, 0.2, 0.3], index=['market_0', 'market_1', 'market_2']),
            'x_lag5': pd.DataFrame(np.random.rand(3, 5),
                                  index=['market_0', 'market_1', 'market_2'],
                                  columns=['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']),
            'x_lag22': pd.DataFrame(np.random.rand(3, 22),
                                   index=['market_0', 'market_1', 'market_2'],
                                   columns=[f'lag_{i}' for i in range(1, 23)])
        },
        'date2': {
            'y': pd.Series([4.0, 5.0, 6.0], index=['market_0', 'market_1', 'market_2']),
            'x_lag1': pd.Series([0.4, 0.5, 0.6], index=['market_0', 'market_1', 'market_2']),
            'x_lag5': pd.DataFrame(np.random.rand(3, 5),
                                  index=['market_0', 'market_1', 'market_2'],
                                  columns=['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']),
            'x_lag22': pd.DataFrame(np.random.rand(3, 22),
                                   index=['market_0', 'market_1', 'market_2'],
                                   columns=[f'lag_{i}' for i in range(1, 23)])
        }
    }

    # Create dataset
    dataset = LegacyGSPHAR_Dataset(data_dict)

    # Test length
    assert len(dataset) == 2, f"Expected length 2, got {len(dataset)}"
    print(f"Dataset length: {len(dataset)}")

    # Test getitem
    for i in range(len(dataset)):
        x_lag1, x_lag5, x_lag22, y = dataset[i]
        print(f"Sample {i}:")
        print(f"  x_lag1 shape: {x_lag1.shape}")
        print(f"  x_lag5 shape: {x_lag5.shape}")
        print(f"  x_lag22 shape: {x_lag22.shape}")
        print(f"  y shape: {y.shape}")

        # Check shapes
        assert x_lag1.shape == (3,), f"Expected x_lag1 shape (3,), got {x_lag1.shape}"
        assert x_lag5.shape == (3, 5), f"Expected x_lag5 shape (3, 5), got {x_lag5.shape}"
        assert x_lag22.shape == (3, 22), f"Expected x_lag22 shape (3, 22), got {x_lag22.shape}"
        assert y.shape == (3,), f"Expected y shape (3,), got {y.shape}"

    print("LegacyGSPHAR_Dataset tests passed!")
    return True

def test_improved_dataset():
    """Test the improved GSPHAR_Dataset implementation."""
    print("\nTesting GSPHAR_Dataset...")

    # Create a simple DataFrame for testing
    np.random.seed(42)
    n_markets = 3
    n_samples = 50  # Increased to ensure we have data after dropping NaNs

    # Create base data with dates as index
    dates = pd.date_range(start='2020-01-01', periods=n_samples)
    data = pd.DataFrame(
        np.random.rand(n_samples, n_markets),
        index=dates,
        columns=[f'market_{i}' for i in range(n_markets)]
    )

    # Add lagged columns
    market_indices = data.columns.tolist()

    # Add lag1 columns
    for market in market_indices:
        data[f"{market}_1"] = data[market].shift(1)

    # Add lag5 columns (1-5)
    for market in market_indices:
        for lag in range(1, 6):
            data[f"{market}_{lag}"] = data[market].shift(lag)

    # Add lag22 columns (1-22)
    for market in market_indices:
        for lag in range(1, 23):
            if f"{market}_{lag}" not in data.columns:  # Skip if already added
                data[f"{market}_{lag}"] = data[market].shift(lag)

    # Drop NaN values
    data = data.dropna()
    print(f"Created test data with {len(data)} rows")

    # Define column groups
    y_columns = market_indices  # Use market indices as y_columns

    # Make sure we have exactly 3 lag1 columns (one for each market)
    columns_lag1 = []
    for market in market_indices:
        columns_lag1.append(f"{market}_1")

    # Make sure we have exactly 15 lag5 columns (5 for each market)
    columns_lag5 = []
    for market in market_indices:
        for lag in range(1, 6):
            columns_lag5.append(f"{market}_{lag}")

    # Make sure we have exactly 66 lag22 columns (22 for each market)
    columns_lag22 = []
    for market in market_indices:
        for lag in range(1, 23):
            columns_lag22.append(f"{market}_{lag}")

    print(f"y_columns: {len(y_columns)}")
    print(f"columns_lag1: {len(columns_lag1)}")
    print(f"columns_lag5: {len(columns_lag5)}")
    print(f"columns_lag22: {len(columns_lag22)}")

    # Create dataset
    dataset = GSPHAR_Dataset(
        data,
        y_columns,
        columns_lag1,
        columns_lag5,
        columns_lag22,
        market_indices
    )

    # Test length
    assert len(dataset) == len(data), f"Expected length {len(data)}, got {len(dataset)}"
    print(f"Dataset length: {len(dataset)}")

    # Test getitem
    for i in range(min(5, len(dataset))):  # Test first 5 samples
        x_lag1, x_lag5, x_lag22, y = dataset[i]
        print(f"Sample {i}:")
        print(f"  x_lag1 shape: {x_lag1.shape}")
        print(f"  x_lag5 shape: {x_lag5.shape}")
        print(f"  x_lag22 shape: {x_lag22.shape}")
        print(f"  y shape: {y.shape}")

        # Check shapes
        assert x_lag1.shape == (3,), f"Expected x_lag1 shape (3,), got {x_lag1.shape}"
        assert x_lag5.shape == (3, 5), f"Expected x_lag5 shape (3, 5), got {x_lag5.shape}"
        assert x_lag22.shape == (3, 22), f"Expected x_lag22 shape (3, 22), got {x_lag22.shape}"
        assert y.shape == (3,), f"Expected y shape (3,), got {y.shape}"

    # Test DataLoader compatibility
    print("\nTesting DataLoader compatibility...")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

    for batch_idx, (x_lag1, x_lag5, x_lag22, y) in enumerate(dataloader):
        if batch_idx >= 2:  # Only check first 2 batches
            break

        print(f"Batch {batch_idx}:")
        print(f"  x_lag1 shape: {x_lag1.shape}")
        print(f"  x_lag5 shape: {x_lag5.shape}")
        print(f"  x_lag22 shape: {x_lag22.shape}")
        print(f"  y shape: {y.shape}")

        # Check batch shapes
        assert x_lag1.shape[1:] == (3,), f"Expected x_lag1 shape (batch_size, 3), got {x_lag1.shape}"
        assert x_lag5.shape[1:] == (3, 5), f"Expected x_lag5 shape (batch_size, 3, 5), got {x_lag5.shape}"
        assert x_lag22.shape[1:] == (3, 22), f"Expected x_lag22 shape (batch_size, 3, 22), got {x_lag22.shape}"
        assert y.shape[1:] == (3,), f"Expected y shape (batch_size, 3), got {y.shape}"

    print("DataLoader tests passed!")
    print("GSPHAR_Dataset tests passed!")
    return True

if __name__ == "__main__":
    # Test both implementations
    legacy_passed = test_legacy_dataset()
    improved_passed = test_improved_dataset()

    if legacy_passed and improved_passed:
        print("\nAll tests passed! Both dataset implementations are working correctly.")
    else:
        print("\nSome tests failed. Please check the error messages above.")
