#!/usr/bin/env python
"""
Script to trim cryptocurrency data to start from the date when most data exists.

This script analyzes a CSV file containing cryptocurrency percentage change data
and creates a new CSV file that starts from the date when a specified threshold
of data availability is reached.
"""

import pandas as pd
import numpy as np
import argparse
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_data_availability(df):
    """
    Analyze data availability in the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame to analyze
        
    Returns:
        pd.Series: Series with data availability percentage for each date
    """
    # Calculate the percentage of non-NaN values for each row
    data_availability = df.notna().sum(axis=1) / df.shape[1]
    
    return data_availability


def plot_data_availability(data_availability, output_dir):
    """
    Plot data availability over time.
    
    Args:
        data_availability (pd.Series): Series with data availability percentage for each date
        output_dir (str): Directory to save the plot
    """
    plt.figure(figsize=(12, 6))
    plt.plot(data_availability.index, data_availability * 100, color='blue')
    plt.axhline(y=97.5, color='red', linestyle='--', label='97.5% Threshold (37/38 columns)')
    plt.title('Data Availability Over Time')
    plt.xlabel('Date')
    plt.ylabel('Data Availability (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, 'data_availability.png'), dpi=300)
    
    # Create a more detailed plot for the first year
    if len(data_availability) > 365 * 24:  # If we have more than a year of hourly data
        plt.figure(figsize=(12, 6))
        first_year = data_availability.iloc[:365*24]
        plt.plot(first_year.index, first_year * 100, color='blue')
        plt.axhline(y=97.5, color='red', linestyle='--', label='97.5% Threshold (37/38 columns)')
        plt.title('Data Availability - First Year')
        plt.xlabel('Date')
        plt.ylabel('Data Availability (%)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(output_dir, 'data_availability_first_year.png'), dpi=300)


def find_start_date(data_availability, threshold):
    """
    Find the date when data availability exceeds the threshold.
    
    Args:
        data_availability (pd.Series): Series with data availability percentage for each date
        threshold (float): Threshold for data availability (0.0-1.0)
        
    Returns:
        pd.Timestamp: Date when threshold is exceeded
    """
    # Find dates where data availability exceeds the threshold
    dates_with_sufficient_data = data_availability[data_availability >= threshold].index
    
    if len(dates_with_sufficient_data) > 0:
        return dates_with_sufficient_data[0]
    else:
        return None


def trim_data(df, start_date):
    """
    Trim the DataFrame to start from the specified date.
    
    Args:
        df (pd.DataFrame): DataFrame to trim
        start_date (pd.Timestamp): Date to start from
        
    Returns:
        pd.DataFrame: Trimmed DataFrame
    """
    return df[df.index >= start_date]


def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(description='Trim cryptocurrency data to start from the date when most data exists.')
    parser.add_argument('--input-file', type=str, required=True,
                        help='Path to the input CSV file.')
    parser.add_argument('--output-file', type=str, required=True,
                        help='Path to the output CSV file.')
    parser.add_argument('--threshold', type=float, default=0.975,
                        help='Threshold for data availability (0.0-1.0). Default: 0.975 (37/38 columns)')
    parser.add_argument('--output-dir', type=str, default='plots/data_analysis',
                        help='Directory to save the plots.')
    
    args = parser.parse_args()
    
    # Load the data
    print(f"Loading data from {args.input_file}")
    df = pd.read_csv(args.input_file, index_col=0, parse_dates=True)
    
    # Print initial data shape
    print(f"Initial data shape: {df.shape}")
    
    # Analyze data availability
    print("Analyzing data availability...")
    data_availability = analyze_data_availability(df)
    print(f"Data availability range: {data_availability.min():.2%} to {data_availability.max():.2%}")
    
    # Plot data availability
    print(f"Plotting data availability to {args.output_dir}")
    plot_data_availability(data_availability, args.output_dir)
    
    # Find the start date
    print(f"Finding start date with threshold {args.threshold:.2%}")
    start_date = find_start_date(data_availability, args.threshold)
    
    if start_date is None:
        print(f"Error: No dates found with at least {args.threshold:.2%} data availability")
        return
    
    print(f"Start date: {start_date}")
    
    # Trim the data
    print("Trimming data...")
    trimmed_df = trim_data(df, start_date)
    print(f"Trimmed data shape: {trimmed_df.shape}")
    
    # Save the trimmed data
    print(f"Saving trimmed data to {args.output_file}")
    trimmed_df.to_csv(args.output_file)
    
    print("Done!")


if __name__ == '__main__':
    main()
