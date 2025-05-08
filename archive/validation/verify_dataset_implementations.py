#!/usr/bin/env python
"""
Simple script to verify that the dataset implementations work correctly.
This script creates small test datasets and verifies that both implementations
produce the same results.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import numpy as np
import pandas as pd
from src.data.dataset import LegacyGSPHAR_Dataset, GSPHAR_Dataset
from torch.utils.data import DataLoader

def create_test_data():
    """Create small test data for verification."""
    # Create sample data with enough rows to handle the shifts
    test_size = 40  # Increased size to ensure we have data after dropping NaNs
    n_markets = 3

    # Set random seed for reproducibility
    np.random.seed(42)

    # Create sample data
    test_data = pd.DataFrame(
        np.random.randn(test_size, n_markets),
        columns=[f'market_{i}' for i in range(n_markets)]
    )

    # Add lagged columns
    look_back_window = 22
    h = 5
    market_indices_list = test_data.columns.tolist()

    for market_index in market_indices_list:
        for lag in range(look_back_window):
            test_data[market_index + f'_{lag+1}'] = test_data[market_index].shift(lag+h)

    # Drop rows with NaN values
    test_data = test_data.dropna()

    print(f"Created test data with {len(test_data)} rows after dropping NaNs")

    return test_data, market_indices_list

def create_legacy_dict(data, market_indices_list):
    """Create a dictionary for the legacy dataset implementation."""
    # Define column groups
    y_columns = [col for col in data.columns if '_' not in col]
    columns_lag1 = [x for x in data.columns if x[-2:] == '_1']
    columns_lag5 = [x for x in data.columns if (x[-2]=='_') and (float(x[-1]) in range(1,6))]
    columns_lag22 = [x for x in data.columns if '_' in x]

    row_index_order = market_indices_list
    column_index_order_5 = [f'lag_{i}' for i in range(1,6)]
    column_index_order_22 = [f'lag_{i}' for i in range(1,23)]

    data_dict = {}
    for date_idx, date in enumerate(data.index):
        y = data.loc[date, y_columns]

        x_lag1 = data.loc[date, columns_lag1]
        new_index = [ind[:-2] for ind in x_lag1.index.tolist()]
        x_lag1.index = new_index

        # Handle lag5 data
        x_lag5 = data.loc[date, columns_lag5]

        # Create a DataFrame with unique identifiers to avoid duplicate indices
        data_lag5 = []
        for i, index in enumerate(x_lag5.index):
            market = index.split('_')[0]
            lag = f'lag_{index.split("_")[1]}'
            data_lag5.append({
                'Market': market,
                'Lag': lag,
                'Value': x_lag5.values[i],
                'UniqueID': f"{market}_{lag}_{i}"  # Add a unique identifier
            })

        df_lag5 = pd.DataFrame(data_lag5)

        # Create a pivot table manually to avoid duplicate index issues
        pivot_df5 = pd.DataFrame(index=row_index_order, columns=column_index_order_5)
        for row in data_lag5:
            market = row['Market']
            lag = row['Lag']
            if market in pivot_df5.index and lag in pivot_df5.columns:
                pivot_df5.loc[market, lag] = row['Value']

        # Fill NaN values with 0
        df_lag5 = pivot_df5.fillna(0)

        # Handle lag22 data similarly
        x_lag22 = data.loc[date, columns_lag22]

        # Create a DataFrame with unique identifiers
        data_lag22 = []
        for i, index in enumerate(x_lag22.index):
            market = index.split('_')[0]
            lag = f'lag_{index.split("_")[1]}'
            data_lag22.append({
                'Market': market,
                'Lag': lag,
                'Value': x_lag22.values[i],
                'UniqueID': f"{market}_{lag}_{i}"
            })

        df_lag22 = pd.DataFrame(data_lag22)

        # Create a pivot table manually
        pivot_df22 = pd.DataFrame(index=row_index_order, columns=column_index_order_22)
        for row in data_lag22:
            market = row['Market']
            lag = row['Lag']
            if market in pivot_df22.index and lag in pivot_df22.columns:
                pivot_df22.loc[market, lag] = row['Value']

        # Fill NaN values with 0
        df_lag22 = pivot_df22.fillna(0)

        # Ensure all required columns are present
        x_lag1 = x_lag1.reindex(row_index_order)

        dfs_dict = {
            'y': y,
            'x_lag1': x_lag1,
            'x_lag5': df_lag5,
            'x_lag22': df_lag22
        }
        data_dict[date] = dfs_dict

    return data_dict, y_columns, columns_lag1, columns_lag5, columns_lag22

def verify_dataset_implementations():
    """Verify that both dataset implementations produce the same results."""
    print("Creating test data...")
    test_data, market_indices_list = create_test_data()

    print("Creating legacy dataset...")
    legacy_dict, y_columns, columns_lag1, columns_lag5, columns_lag22 = create_legacy_dict(test_data, market_indices_list)
    legacy_dataset = LegacyGSPHAR_Dataset(legacy_dict)

    print("Creating improved dataset...")
    improved_dataset = GSPHAR_Dataset(
        test_data,
        y_columns,
        columns_lag1,
        columns_lag5,
        columns_lag22,
        market_indices_list
    )

    print("Verifying dataset lengths...")
    assert len(legacy_dataset) == len(improved_dataset), "Dataset lengths don't match"
    print(f"Both datasets have {len(legacy_dataset)} samples.")

    print("Verifying dataset items...")
    for i in range(len(legacy_dataset)):
        legacy_x1, legacy_x5, legacy_x22, legacy_y = legacy_dataset[i]
        improved_x1, improved_x5, improved_x22, improved_y = improved_dataset[i]

        print(f"Sample {i}:")
        print(f"  Legacy shapes: x1={legacy_x1.shape}, x5={legacy_x5.shape}, x22={legacy_x22.shape}, y={legacy_y.shape}")
        print(f"  Improved shapes: x1={improved_x1.shape}, x5={improved_x5.shape}, x22={improved_x22.shape}, y={improved_y.shape}")

        assert torch.allclose(legacy_x1, improved_x1, rtol=1e-5), f"x_lag1 mismatch at sample {i}"
        assert torch.allclose(legacy_x5, improved_x5, rtol=1e-5), f"x_lag5 mismatch at sample {i}"
        assert torch.allclose(legacy_x22, improved_x22, rtol=1e-5), f"x_lag22 mismatch at sample {i}"
        assert torch.allclose(legacy_y, improved_y, rtol=1e-5), f"y mismatch at sample {i}"

    print("\nAll tests passed! Legacy and improved implementations produce identical results.")

    print("\nTesting DataLoader compatibility...")
    legacy_loader = DataLoader(legacy_dataset, batch_size=2, shuffle=False)
    improved_loader = DataLoader(improved_dataset, batch_size=2, shuffle=False)

    for batch_idx, ((legacy_x1, legacy_x5, legacy_x22, legacy_y),
                   (improved_x1, improved_x5, improved_x22, improved_y)) in enumerate(zip(legacy_loader, improved_loader)):
        print(f"Batch {batch_idx}:")
        print(f"  Legacy shapes: x1={legacy_x1.shape}, x5={legacy_x5.shape}, x22={legacy_x22.shape}, y={legacy_y.shape}")
        print(f"  Improved shapes: x1={improved_x1.shape}, x5={improved_x5.shape}, x22={improved_x22.shape}, y={improved_y.shape}")

        assert torch.allclose(legacy_x1, improved_x1, rtol=1e-5), f"x_lag1 mismatch in batch {batch_idx}"
        assert torch.allclose(legacy_x5, improved_x5, rtol=1e-5), f"x_lag5 mismatch in batch {batch_idx}"
        assert torch.allclose(legacy_x22, improved_x22, rtol=1e-5), f"x_lag22 mismatch in batch {batch_idx}"
        assert torch.allclose(legacy_y, improved_y, rtol=1e-5), f"y mismatch in batch {batch_idx}"

    print("\nDataLoader tests passed! Both implementations work correctly with DataLoader.")
    print("\nVerification complete. The dataset implementations are working correctly.")

if __name__ == "__main__":
    verify_dataset_implementations()
