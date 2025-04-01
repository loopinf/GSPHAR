import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import gzip

def load_parquet_data(file_path):
    """Load 5-minute close price data from parquet file"""
    file_path = Path(file_path)
    # Check if file has .gz extension
    if file_path.suffix == '.gz':
        with gzip.open(file_path, 'rb') as f:
            data = pd.read_parquet(f)
    else:
        data = pd.read_parquet(file_path)
    return data

def load_pickle_data(file_path):
    """Load 5-minute close price data from pickle file"""
    file_path = Path(file_path)
    # Check if file has .gz extension
    if file_path.suffix == '.gz':
        with gzip.open(file_path, 'rb') as f:
            data = pickle.load(f)
    else:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    return data
    # with open(file_path, 'rb') as f:
    #     data = pickle.load(f)
    # return data

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
    
    # Resample to input frequency and sum to get realized variance
    rv = squared_returns.resample(resample_freq).sum()
    
    # Take square root to get realized volatility
    rv_sqrt = np.sqrt(rv)
    
    return rv_sqrt

def process_data(pickle_file_path, resample_freq = '1h', output_folder='data'):
    """Process 5-minute data and save as CSV"""
    # Create output directory if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading data from {pickle_file_path}")
    # data = load_pickle_data(pickle_file_path)
    data = load_parquet_data(pickle_file_path)
    
    # Check if data is a DataFrame
    if not isinstance(data, pd.DataFrame):
        print(f"Converting data to DataFrame")
        data = pd.DataFrame(data)
    
    # Calculate realized volatility
    print(f"Calculating realized volatility")
    rv = calculate_realized_volatility(data, resample_freq=resample_freq)
    
    output_file_path = Path(output_folder) / f'sample_{resample_freq}_rv5_sqrt_38.csv'
    # Save to CSV
    print(f"Saving to {output_file_path}")
    rv.to_csv(output_file_path)
    
    print(f"Processing complete. Output saved to {output_file_path}")
    return rv

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process 5-minute data to realized volatility')
    parser.add_argument('pickle_file', type=str, help='Path to pickle file containing 5-minute close prices')
    parser.add_argument('--resample_freq', type=str, default='1h', help='Resample frequency for realized volatility')
    parser.add_argument('--output', type=str, default='data/rv5_sqrt_24.csv', help='Output file path')
    
    args = parser.parse_args()
    
    process_data(args.pickle_file, args.resample_freq, args.output)
