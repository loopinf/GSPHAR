#!/usr/bin/env python
"""
Script to load and prepare absolute percentage change data for GSPHAR model.
This script loads the daily_pct_change_crypto.csv file, takes the absolute value,
and prepares it for use with the GSPHAR model.
"""

import pandas as pd
import numpy as np
import argparse
import os
import sys

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from local modules
from config import settings
from src.data import load_data, split_data, create_lagged_features, prepare_data_dict, create_dataloaders


def load_abs_pct_change_data(filepath):
    """
    Load percentage change data from a CSV file and take the absolute value.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data with absolute values.
    """
    try:
        # Load the data
        data = pd.read_csv(filepath)
        
        # Convert the 'index' column to datetime and set as index
        data['index'] = pd.to_datetime(data['index'])
        data.set_index('index', inplace=True)
        
        # Drop the first row which contains NaN values
        data = data.dropna(how='all')
        
        # Fill any remaining NaN values with 0
        data = data.fillna(0)
        
        # Take the absolute value of the percentage changes
        abs_data = data.abs()
        
        print(f"Loaded data from {filepath} with shape {data.shape}")
        print(f"Date range: {data.index[0]} to {data.index[-1]}")
        print(f"Number of cryptocurrencies: {data.shape[1]}")
        print(f"Converted to absolute values")
        
        return abs_data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def prepare_data_for_gsphar(data, train_ratio=0.7, h=5, look_back_window=22, batch_size=32):
    """
    Prepare data for GSPHAR model.

    Args:
        data (pd.DataFrame): Input data.
        train_ratio (float): Ratio for train/test split.
        h (int): Prediction horizon.
        look_back_window (int): Number of lagged observations to use.
        batch_size (int): Batch size for dataloaders.

    Returns:
        tuple: (dataloader_train, dataloader_test, train_dataset, test_dataset, market_indices_list, DY_adj)
    """
    from src.utils.graph_utils import compute_spillover_index
    
    # Split data
    train_dataset_raw, test_dataset_raw = split_data(data, train_ratio)
    
    # Get market indices
    market_indices_list = train_dataset_raw.columns.tolist()
    
    # Compute spillover index
    try:
        DY_adj = compute_spillover_index(train_dataset_raw, h, look_back_window, 0.0, standardized=True)
    except Exception as e:
        print(f"Error computing spillover index: {e}")
        print("Using a simple adjacency matrix instead.")
        # Create a simple adjacency matrix as fallback
        n = len(market_indices_list)
        DY_adj = np.ones((n, n))  # Full connectivity
    
    # Create lagged features
    train_dataset = create_lagged_features(train_dataset_raw, market_indices_list, h, look_back_window)
    test_dataset = create_lagged_features(test_dataset_raw, market_indices_list, h, look_back_window)
    
    # Prepare data dictionaries
    train_dict = prepare_data_dict(train_dataset, market_indices_list, look_back_window)
    test_dict = prepare_data_dict(test_dataset, market_indices_list, look_back_window)
    
    # Create dataloaders
    dataloader_train, dataloader_test = create_dataloaders(train_dict, test_dict, batch_size)
    
    return dataloader_train, dataloader_test, train_dataset, test_dataset, market_indices_list, DY_adj


def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(description='Load and prepare absolute percentage change data for GSPHAR model.')
    parser.add_argument('--data-file', type=str, default='data/daily_pct_change_crypto.csv',
                        help='Path to the data file.')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='Ratio for train/test split.')
    parser.add_argument('--horizon', type=int, default=1,
                        help='Prediction horizon.')
    parser.add_argument('--look-back', type=int, default=22,
                        help='Look-back window size.')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for dataloaders.')
    parser.add_argument('--test', action='store_true',
                        help='Run in test mode to verify data loading.')
    parser.add_argument('--save', action='store_true',
                        help='Save the absolute percentage change data to a file.')
    parser.add_argument('--output-file', type=str, default='data/abs_daily_pct_change_crypto.csv',
                        help='Path to save the absolute percentage change data.')
    
    args = parser.parse_args()
    
    # Load data
    data = load_abs_pct_change_data(args.data_file)
    
    if data is None:
        print("Failed to load data. Exiting.")
        return
    
    # Save the absolute percentage change data if requested
    if args.save:
        output_file = args.output_file
        data.to_csv(output_file)
        print(f"Saved absolute percentage change data to {output_file}")
    
    # If in test mode, prepare data and print some statistics
    if args.test:
        dataloader_train, dataloader_test, train_dataset, test_dataset, market_indices_list, DY_adj = prepare_data_for_gsphar(
            data, args.train_ratio, args.horizon, args.look_back, args.batch_size
        )
        
        print("\nData preparation successful!")
        print(f"Train dataset shape: {train_dataset.shape}")
        print(f"Test dataset shape: {test_dataset.shape}")
        print(f"Number of market indices: {len(market_indices_list)}")
        print(f"Adjacency matrix shape: {DY_adj.shape}")
        print(f"Training dataloader batches: {len(dataloader_train)}")
        print(f"Testing dataloader batches: {len(dataloader_test)}")
        
        # Print a sample batch
        print("\nSample batch from training dataloader:")
        for x_lag1, x_lag5, x_lag22, y in dataloader_train:
            print(f"x_lag1 shape: {x_lag1.shape}")
            print(f"x_lag5 shape: {x_lag5.shape}")
            print(f"x_lag22 shape: {x_lag22.shape}")
            print(f"y shape: {y.shape}")
            break
    else:
        print("Data loaded successfully. Use --test to verify data preparation.")


if __name__ == '__main__':
    main()
