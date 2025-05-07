#!/usr/bin/env python
"""
Simple script to check the structure of the cryptocurrency data in df_cl_5m.parquet.
"""

import pandas as pd
import sys

def check_parquet_file(file_path):
    """
    Check the structure of a parquet file and print basic information.
    
    Args:
        file_path (str): Path to the parquet file
    """
    try:
        # Read the first few rows to understand the structure
        print(f"Reading file: {file_path}")
        df = pd.read_parquet(file_path)
        
        # Print basic information
        print(f"\nBasic Information:")
        print(f"Shape: {df.shape}")
        print(f"Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB")
        
        # Print index information
        print(f"\nIndex Information:")
        print(f"Index type: {type(df.index)}")
        print(f"Index name: {df.index.name}")
        print(f"Time range: {df.index.min()} to {df.index.max()}")
        
        # Print column information
        print(f"\nColumn Information:")
        print(f"Number of columns: {len(df.columns)}")
        print(f"Column names: {df.columns.tolist()}")
        
        # Print data types
        print(f"\nData Types:")
        print(df.dtypes)
        
        # Print first few rows
        print(f"\nFirst 5 rows:")
        print(df.head())
        
        # Check for missing values
        missing_values = df.isna().sum()
        print(f"\nMissing Values:")
        print(missing_values[missing_values > 0])
        
        # Print basic statistics
        print(f"\nBasic Statistics for First Column:")
        first_col = df.columns[0]
        print(df[first_col].describe())
        
    except Exception as e:
        print(f"Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    file_path = "./data/df_cl_5m.parquet"
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    
    check_parquet_file(file_path)
