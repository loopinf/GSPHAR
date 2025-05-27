#!/usr/bin/env python
"""
Script to calculate 1-hour percentage change from 5-minute cryptocurrency data.
"""

import pandas as pd
import numpy as np
import os
import argparse
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(file_path):
    """
    Load the 5-minute cryptocurrency data.

    Args:
        file_path (str): Path to the 5-minute data file.

    Returns:
        pd.DataFrame: DataFrame containing the 5-minute data.
    """
    logger.info(f"Loading data from {file_path}")
    
    if file_path.endswith('.parquet'):
        df = pd.read_parquet(file_path)
    elif file_path.endswith('.pickle'):
        df = pd.read_pickle(file_path)
    else:
        raise ValueError("Unsupported file format. Please use .parquet or .pickle")
    
    # Ensure the index is a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")
    
    logger.info(f"Loaded data with shape {df.shape}")
    return df


def calculate_1h_pct_change(df, method='resample'):
    """
    Calculate 1-hour percentage change from 5-minute data.

    Args:
        df (pd.DataFrame): DataFrame containing 5-minute data.
        method (str): Method to calculate percentage change.
            Options: 'resample', 'rolling'

    Returns:
        pd.DataFrame: DataFrame containing 1-hour percentage change.
    """
    logger.info(f"Calculating 1-hour percentage change using method: {method}")
    
    if method == 'resample':
        # Resample to 1-hour intervals and calculate percentage change
        df_1h = df.resample('1h').last()
        pct_change_1h = df_1h.pct_change() * 100
    
    elif method == 'rolling':
        # Calculate rolling 1-hour percentage change
        # For 5-minute data, 1 hour = 12 periods
        pct_change_1h = (df / df.shift(12) - 1) * 100
        # Resample to 1-hour intervals to match RV data
        pct_change_1h = pct_change_1h.resample('1h').last()
    
    else:
        raise ValueError("Unsupported method. Use 'resample' or 'rolling'")
    
    logger.info(f"Calculated 1-hour percentage change with shape {pct_change_1h.shape}")
    return pct_change_1h


def save_results(pct_change_1h, output_file, data_source='crypto'):
    """
    Save the 1-hour percentage change to a file.

    Args:
        pct_change_1h (pd.DataFrame): DataFrame containing 1-hour percentage change.
        output_file (str): Path to save the results.
        data_source (str): Source of the data (e.g., 'crypto').

    Returns:
        str: Path to the saved file.
    """
    # Get number of symbols
    num_symbols = pct_change_1h.shape[1]
    
    # Get date range
    start_date = pct_change_1h.index.min().strftime('%Y%m%d')
    end_date = pct_change_1h.index.max().strftime('%Y%m%d')
    
    # Create a more descriptive filename
    dir_path = os.path.dirname(output_file)
    file_ext = os.path.splitext(output_file)[1]
    
    # If output_file already has a descriptive name, use it as is
    if f"{data_source}_pct_change_1h_{num_symbols}" in os.path.basename(output_file):
        descriptive_path = output_file
    else:
        base_name = f"{data_source}_pct_change_1h_{num_symbols}_{start_date}_{end_date}{file_ext}"
        descriptive_path = os.path.join(dir_path, base_name)
    
    logger.info(f"Saving results to {descriptive_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(descriptive_path), exist_ok=True)
    
    # Save the results
    if descriptive_path.endswith('.csv'):
        pct_change_1h.to_csv(descriptive_path)
    elif descriptive_path.endswith('.parquet'):
        pct_change_1h.to_parquet(descriptive_path)
    elif descriptive_path.endswith('.pickle'):
        pct_change_1h.to_pickle(descriptive_path)
    else:
        raise ValueError("Unsupported output format. Please use .csv, .parquet, or .pickle")
    
    logger.info(f"Results saved to {descriptive_path}")
    return descriptive_path


def main():
    """
    Main function to calculate 1-hour percentage change from 5-minute data.
    """
    parser = argparse.ArgumentParser(description='Calculate 1-hour percentage change from 5-minute cryptocurrency data.')
    parser.add_argument('--input-file', type=str, default='data/df_cl_5m.parquet',
                        help='Path to the 5-minute data file.')
    parser.add_argument('--output-file', type=str, default='data/crypto_pct_change_1h.csv',
                        help='Path to save the 1-hour percentage change.')
    parser.add_argument('--method', type=str, default='resample',
                        choices=['resample', 'rolling'],
                        help='Method to calculate percentage change.')
    parser.add_argument('--data-source', type=str, default='crypto',
                        help='Source of the data (e.g., crypto, stocks).')
    
    args = parser.parse_args()
    
    # Load the data
    df = load_data(args.input_file)
    
    # Calculate 1-hour percentage change
    pct_change_1h = calculate_1h_pct_change(df, method=args.method)
    
    # Save the results
    output_path = save_results(
        pct_change_1h, 
        args.output_file, 
        data_source=args.data_source
    )
    
    logger.info(f"1-hour percentage change calculation completed successfully.")
    logger.info(f"Results saved to {output_path}.")


if __name__ == '__main__':
    main()
