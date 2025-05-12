#!/usr/bin/env python
"""
Test script to verify the refactored create_lagged_features function works correctly.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.data_utils import create_lagged_features

def main():
    # Create a sample dataframe
    dates = pd.date_range(start='2020-01-01', periods=100)
    data = {
        'Market1': np.random.randn(100),
        'Market2': np.random.randn(100),
        'Market3': np.random.randn(100)
    }
    df = pd.DataFrame(data, index=dates)

    # Define parameters
    market_indices = ['Market1', 'Market2', 'Market3']
    h = 5
    look_back_window = 10

    # Create lagged features
    print("Creating lagged features...")
    df_lagged = create_lagged_features(df, market_indices, h, look_back_window)

    # Print information about the result
    print(f"\nOriginal dataframe shape: {df.shape}")
    print(f"Lagged dataframe shape: {df_lagged.shape}")

    # Expected number of columns: original columns + (market_indices * look_back_window)
    expected_columns = len(df.columns) + (len(market_indices) * look_back_window)
    print(f"Expected number of columns: {expected_columns}")
    print(f"Actual number of columns: {len(df_lagged.columns)}")

    # Check if all expected columns are present
    expected_column_names = []
    for market in market_indices:
        expected_column_names.append(market)  # Original column
        for lag in range(1, look_back_window + 1):
            expected_column_names.append(f"{market}_{lag}")  # Lagged column

    missing_columns = [col for col in expected_column_names if col not in df_lagged.columns]
    if missing_columns:
        print(f"\nWARNING: Missing expected columns: {missing_columns}")
    else:
        print("\nAll expected columns are present.")

    # Print the first few rows of the dataframe
    print("\nFirst few rows of the lagged dataframe:")
    print(df_lagged.head())

if __name__ == "__main__":
    main()
