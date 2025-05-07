#!/usr/bin/env python
"""
Efficient script to create rv5_sqrt_38_crypto.csv from 5-minute cryptocurrency data.

This script calculates realized volatility from 5-minute data:
1. Calculate returns from the 5-minute price data
2. Square the returns
3. Sum these squared returns over a day to get the daily realized variance
4. Take the square root to get the realized volatility

This version processes the data in chunks to be more memory-efficient.
"""

import pandas as pd
import numpy as np
import os
import pyarrow.parquet as pq
from datetime import datetime

def process_chunk(chunk):
    """
    Process a chunk of data to calculate returns.
    
    Args:
        chunk (pd.DataFrame): Chunk of 5-minute price data
        
    Returns:
        pd.DataFrame: Squared returns for the chunk
    """
    # Calculate returns (log differences)
    returns = np.log(chunk).diff().dropna()
    
    # Square the returns
    squared_returns = returns ** 2
    
    return squared_returns

def main():
    # Input and output file paths
    input_file = 'data/df_cl_5m.parquet'
    output_file = 'data/rv5_sqrt_38_crypto.csv'
    
    print(f"Reading metadata from {input_file}...")
    metadata = pq.read_metadata(input_file)
    print(f"Number of rows: {metadata.num_rows}")
    print(f"Number of columns: {metadata.num_columns}")
    
    # Read the first few rows to get column names and index
    print("Reading sample to get column structure...")
    sample = pd.read_parquet(input_file, engine='pyarrow').head(1000)
    print(f"Sample shape: {sample.shape}")
    print(f"Columns: {sample.columns.tolist()}")
    
    # Initialize a DataFrame to store daily squared returns
    daily_squared_returns = pd.DataFrame()
    
    # Process the data in chunks
    chunk_size = 50000  # Adjust based on available memory
    num_chunks = (metadata.num_rows + chunk_size - 1) // chunk_size
    
    print(f"Processing data in {num_chunks} chunks of size {chunk_size}...")
    
    for i in range(num_chunks):
        start_row = i * chunk_size
        end_row = min((i + 1) * chunk_size, metadata.num_rows)
        
        print(f"Processing chunk {i+1}/{num_chunks} (rows {start_row} to {end_row})...")
        
        # Read a chunk of data
        chunk = pd.read_parquet(
            input_file,
            engine='pyarrow',
            filters=[('__index_level_0__', '>=', start_row), ('__index_level_0__', '<', end_row)]
        )
        
        # Process the chunk
        squared_returns_chunk = process_chunk(chunk)
        
        # Resample to daily and sum
        daily_chunk = squared_returns_chunk.resample('D').sum()
        
        # Append to the result
        if daily_squared_returns.empty:
            daily_squared_returns = daily_chunk
        else:
            daily_squared_returns = daily_squared_returns.add(daily_chunk, fill_value=0)
    
    # Take the square root to get realized volatility
    rv5_sqrt = daily_squared_returns.apply(np.sqrt)
    
    # Save to CSV
    print(f"Saving to {output_file}...")
    rv5_sqrt.to_csv(output_file)
    
    print(f"Done! Created {output_file} with shape {rv5_sqrt.shape}")
    
    # Print sample of the output
    print("\nSample of the output:")
    print(rv5_sqrt.head())

if __name__ == "__main__":
    main()
