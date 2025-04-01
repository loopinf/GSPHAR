import pandas as pd
import numpy as np
from pathlib import Path
import pickle

def load_pickle_data(file_path):
    """Load 5-minute close price data from pickle file"""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def calculate_realized_volatility(df, resample_freq='1D'):
    """
    Calculate realized volatility from 5-minute close prices
    
    Args:
        df: DataFrame with datetime index and columns as different symbols
        resample_freq: frequency to resample the realized volatility to
    
    Returns:
        DataFrame of realized volatility
    """
    # Calculate log returns
    log_returns = np.log(df / df.shift(1)).dropna()
    
    # Square the returns
    squared_returns = log_returns ** 2
    
    # Resample to daily frequency and sum to get realized variance
    rv = squared_returns.resample(resample_freq).sum()
    
    # Take square root to get realized volatility
    rv_sqrt = np.sqrt(rv)
    
    return rv_sqrt

def process_data(pickle_file_path, output_file_path='data/rv5_sqrt_24.csv'):
    """Process 5-minute data and save as CSV"""
    # Create output directory if it doesn't exist
    Path(output_file_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading data from {pickle_file_path}")
    data = load_pickle_data(pickle_file_path)
    
    # Check if data is a DataFrame
    if not isinstance(data, pd.DataFrame):
        print(f"Converting data to DataFrame")
        data = pd.DataFrame(data)
    
    # Calculate realized volatility
    print(f"Calculating realized volatility")
    rv = calculate_realized_volatility(data)
    
    # Save to CSV
    print(f"Saving to {output_file_path}")
    rv.to_csv(output_file_path)
    
    print(f"Processing complete. Output saved to {output_file_path}")
    return rv

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process 5-minute data to realized volatility')
    parser.add_argument('pickle_file', type=str, help='Path to pickle file containing 5-minute close prices')
    parser.add_argument('--output', type=str, default='data/rv5_sqrt_24.csv', help='Output file path')
    
    args = parser.parse_args()
    
    process_data(args.pickle_file, args.output)
