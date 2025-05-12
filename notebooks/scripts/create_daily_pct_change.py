#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Create daily percentage change data from cryptocurrency OHLCV data.
This script takes the raw cryptocurrency data and calculates daily percentage changes.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import argparse

def load_raw_crypto_data(file_path):
    """
    Load raw cryptocurrency OHLCV data.
    
    Args:
        file_path: Path to the parquet file containing OHLCV data
        
    Returns:
        DataFrame with OHLCV data
    """
    try:
        # Load data
        df = pd.read_parquet(file_path)
        print(f"Loaded data with shape: {df.shape}")
        
        # Check if 'Open Time' column exists
        if 'Open Time' not in df.columns:
            raise ValueError("'Open Time' column not found in the data")
        
        # Set 'Open Time' as index
        df.set_index('Open Time', inplace=True)
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def resample_to_daily(df):
    """
    Resample 5-minute data to daily data.
    
    Args:
        df: DataFrame with OHLCV data at 5-minute intervals
        
    Returns:
        DataFrame with daily OHLCV data
    """
    # Group by date and get the last value of each day
    daily_df = df.groupby(df.index.date).last()
    
    # Convert index to datetime
    daily_df.index = pd.to_datetime(daily_df.index)
    
    return daily_df

def calculate_pct_change(df):
    """
    Calculate percentage change for each cryptocurrency.
    
    Args:
        df: DataFrame with daily OHLCV data
        
    Returns:
        DataFrame with daily percentage changes
    """
    # Extract close prices
    close_cols = [col for col in df.columns if 'close' in col.lower()]
    close_df = df[close_cols]
    
    # Calculate percentage change
    pct_change_df = close_df.pct_change() * 100  # Multiply by 100 to get percentage
    
    # Replace column names to remove 'close' suffix
    new_cols = {col: col.replace('_close', '') for col in pct_change_df.columns}
    pct_change_df.rename(columns=new_cols, inplace=True)
    
    # Drop first row (NaN values)
    pct_change_df = pct_change_df.dropna(how='all')
    
    return pct_change_df

def main():
    parser = argparse.ArgumentParser(description='Create daily percentage change data from cryptocurrency OHLCV data')
    parser.add_argument('--input', type=str, default='../data/df_cl_5m.parquet', help='Input parquet file path')
    parser.add_argument('--output', type=str, default='../data/daily_pct_change_crypto.csv', help='Output CSV file path')
    args = parser.parse_args()
    
    # Load raw data
    df = load_raw_crypto_data(args.input)
    if df is None:
        return
    
    # Resample to daily data
    daily_df = resample_to_daily(df)
    print(f"Resampled to daily data with shape: {daily_df.shape}")
    
    # Calculate percentage change
    pct_change_df = calculate_pct_change(daily_df)
    print(f"Calculated percentage change with shape: {pct_change_df.shape}")
    
    # Save to CSV
    pct_change_df.to_csv(args.output)
    print(f"Saved daily percentage change data to {args.output}")

if __name__ == "__main__":
    main()
