import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import gzip
from volatility_utils import calculate_volatility

def load_parquet_data(file_path):
    """Load 5-minute close price data from parquet file"""
    file_path = Path(file_path)
    if file_path.suffix == '.gz':
        with gzip.open(file_path, 'rb') as f:
            data = pd.read_parquet(f)
    else:
        data = pd.read_parquet(file_path)
    return data

def load_pickle_data(file_path):
    """Load 5-minute close price data from pickle file"""
    file_path = Path(file_path)
    if file_path.suffix == '.gz':
        with gzip.open(file_path, 'rb') as f:
            data = pickle.load(f)
    else:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    return data

def process_data(file_path, resample_freq='1H', vol_method='realized', output_folder='data'):
    """Process 5-minute data and save volatility as CSV"""
    # Create output directory if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading data from {file_path}")
    data = load_parquet_data(file_path)
    
    # Check if data is a DataFrame
    if not isinstance(data, pd.DataFrame):
        print(f"Converting data to DataFrame")
        data = pd.DataFrame(data)
    
    # Calculate volatility
    print(f"Calculating {vol_method} volatility")
    volatility = calculate_volatility(data, method=vol_method, freq=resample_freq)
    
    # Save to CSV
    output_file_path = Path(output_folder) / f'volatility_{vol_method}_{resample_freq}.csv'
    print(f"Saving to {output_file_path}")
    volatility.to_csv(output_file_path)
    
    print(f"Processing complete. Output saved to {output_file_path}")
    return volatility

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process 5-minute data to volatility')
    parser.add_argument('input_file', type=str, help='Path to input file containing 5-minute prices')
    parser.add_argument('--resample_freq', type=str, default='1H', help='Resample frequency')
    parser.add_argument('--vol_method', type=str, default='realized', 
                       choices=['realized', 'pct_change'], help='Volatility calculation method')
    parser.add_argument('--output', type=str, default='data', help='Output folder path')
    
    args = parser.parse_args()
    
    process_data(args.input_file, args.resample_freq, args.vol_method, args.output)
