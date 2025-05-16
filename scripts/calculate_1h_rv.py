#!/usr/bin/env python
"""
Script to calculate 1-hour realized volatility from 5-minute cryptocurrency data.
"""

import pandas as pd
import numpy as np
import os
import argparse
from pathlib import Path
import logging

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


def inspect_and_clean_data(df, missing_threshold=0.3, handling_method='drop',
                          fill_method='ffill', interpolate_method='linear'):
    """
    Inspect the data for missing values and clean it according to the specified method.

    Args:
        df (pd.DataFrame): DataFrame to inspect and clean.
        missing_threshold (float): Threshold for dropping dates with too many missing values.
            If a date has more than this proportion of missing values, it will be dropped.
        handling_method (str): Method to handle missing values.
            Options: 'drop', 'fill', 'interpolate', 'none'
        fill_method (str): Method to fill missing values if handling_method is 'fill'.
            Options: 'ffill', 'bfill', 'zero'
        interpolate_method (str): Method to interpolate missing values if handling_method is 'interpolate'.
            Options: 'linear', 'time', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'

    Returns:
        pd.DataFrame: Cleaned DataFrame.
        dict: Dictionary with information about missing values.
    """
    logger.info("Inspecting data for missing values")

    # Calculate missing values
    missing_values = df.isna().sum()
    missing_percentage = (missing_values / len(df)) * 100

    # Calculate missing values by date
    missing_by_date = df.isna().groupby(df.index.date).mean()

    # Identify dates with too many missing values
    dates_to_drop = missing_by_date[missing_by_date.mean(axis=1) > missing_threshold].index

    # Create a report
    missing_info = {
        'total_rows': len(df),
        'missing_values': missing_values.to_dict(),
        'missing_percentage': missing_percentage.to_dict(),
        'dates_with_high_missing': [str(date) for date in dates_to_drop],
        'num_dates_to_drop': len(dates_to_drop)
    }

    logger.info(f"Found {len(dates_to_drop)} dates with more than {missing_threshold*100}% missing values")

    # Make a copy of the DataFrame to avoid modifying the original
    df_cleaned = df.copy()

    # Handle missing values
    if handling_method == 'drop':
        # Drop dates with too many missing values
        if len(dates_to_drop) > 0:
            # Convert index.date to a list for comparison
            dates_to_drop_set = set(dates_to_drop)
            mask = [date not in dates_to_drop_set for date in df_cleaned.index.date]
            df_cleaned = df_cleaned.loc[mask]
            logger.info(f"Dropped {len(dates_to_drop)} dates with high missing values")

        # Drop any remaining rows with any missing values
        rows_before = len(df_cleaned)
        df_cleaned = df_cleaned.dropna()
        rows_dropped = rows_before - len(df_cleaned)
        logger.info(f"Dropped {rows_dropped} additional rows with any missing values")

    elif handling_method == 'fill':
        # Fill missing values
        if fill_method == 'ffill':
            df_cleaned = df_cleaned.ffill()
            # Fill any remaining NaNs at the beginning with backfill
            df_cleaned = df_cleaned.bfill()
        elif fill_method == 'bfill':
            df_cleaned = df_cleaned.bfill()
            # Fill any remaining NaNs at the end with forward fill
            df_cleaned = df_cleaned.ffill()
        elif fill_method == 'zero':
            df_cleaned = df_cleaned.fillna(0)
        logger.info(f"Filled missing values using method: {fill_method}")

    elif handling_method == 'interpolate':
        # Interpolate missing values
        df_cleaned = df_cleaned.interpolate(method=interpolate_method, axis=0)
        # Fill any remaining NaNs at the edges
        df_cleaned = df_cleaned.ffill().bfill()
        logger.info(f"Interpolated missing values using method: {interpolate_method}")

    elif handling_method == 'none':
        logger.info("No cleaning performed, keeping missing values as is")

    else:
        raise ValueError("Unsupported handling method. Use 'drop', 'fill', 'interpolate', or 'none'")

    # Calculate how many missing values remain
    remaining_missing = df_cleaned.isna().sum()
    missing_info['remaining_missing'] = remaining_missing.to_dict()
    missing_info['remaining_missing_percentage'] = ((remaining_missing / len(df_cleaned)) * 100).to_dict()

    logger.info(f"Data cleaning completed. Original shape: {df.shape}, Cleaned shape: {df_cleaned.shape}")
    return df_cleaned, missing_info


def calculate_returns(df):
    """
    Calculate log returns from price data.

    Args:
        df (pd.DataFrame): DataFrame containing price data.

    Returns:
        pd.DataFrame: DataFrame containing log returns.
    """
    logger.info("Calculating log returns")

    # Calculate log returns
    returns = np.log(df) - np.log(df.shift(1))

    # Drop the first row which will have NaN values
    returns = returns.dropna(how='all')

    logger.info(f"Calculated returns with shape {returns.shape}")
    return returns


def calculate_1h_rv(returns, method='sum_of_squares', annualize=False):
    """
    Calculate 1-hour realized volatility from 5-minute returns.

    Args:
        returns (pd.DataFrame): DataFrame containing 5-minute returns.
        method (str): Method to calculate realized volatility.
            Options: 'sum_of_squares', 'standard_deviation'
        annualize (bool): Whether to annualize the volatility.

    Returns:
        pd.DataFrame: DataFrame containing 1-hour realized volatility.
    """
    logger.info(f"Calculating 1-hour realized volatility using method: {method}")

    # Resample to 1-hour intervals
    if method == 'sum_of_squares':
        # Sum of squared returns
        rv_1h = returns.resample('1h').apply(lambda x: np.sqrt(np.sum(x**2)))
    elif method == 'standard_deviation':
        # Standard deviation of returns
        rv_1h = returns.resample('1h').std() * np.sqrt(12)  # 12 5-minute intervals in 1 hour
    else:
        raise ValueError("Unsupported method. Use 'sum_of_squares' or 'standard_deviation'")

    # Annualize if requested (multiply by sqrt of number of hours in a year)
    if annualize:
        rv_1h = rv_1h * np.sqrt(24 * 365)  # 24 hours in a day, 365 days in a year

    logger.info(f"Calculated 1-hour RV with shape {rv_1h.shape}")
    return rv_1h


def save_results(rv_1h, output_path, include_sqrt=True, data_source='crypto'):
    """
    Save the 1-hour realized volatility to a file.

    Args:
        rv_1h (pd.DataFrame): DataFrame containing 1-hour realized volatility.
        output_path (str): Path to save the results.
        include_sqrt (bool): Whether to also save the square root of RV.
        data_source (str): Source of the data (e.g., 'crypto').

    Returns:
        None
    """
    # Get number of symbols
    num_symbols = rv_1h.shape[1]

    # Get date range
    start_date = rv_1h.index.min().strftime('%Y%m%d')
    end_date = rv_1h.index.max().strftime('%Y%m%d')

    # Create a more descriptive filename
    dir_path = os.path.dirname(output_path)
    file_ext = os.path.splitext(output_path)[1]

    # If output_path already has a descriptive name, use it as is
    if f"{data_source}_rv1h_{num_symbols}" in os.path.basename(output_path):
        descriptive_path = output_path
    else:
        base_name = f"{data_source}_rv1h_{num_symbols}_{start_date}_{end_date}{file_ext}"
        descriptive_path = os.path.join(dir_path, base_name)

    logger.info(f"Saving results to {descriptive_path}")

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(descriptive_path), exist_ok=True)

    # Save the results
    if descriptive_path.endswith('.csv'):
        rv_1h.to_csv(descriptive_path)
    elif descriptive_path.endswith('.parquet'):
        rv_1h.to_parquet(descriptive_path)
    elif descriptive_path.endswith('.pickle'):
        rv_1h.to_pickle(descriptive_path)
    else:
        raise ValueError("Unsupported output format. Please use .csv, .parquet, or .pickle")

    # Save square root version if requested
    if include_sqrt:
        base_path = os.path.splitext(descriptive_path)[0]
        sqrt_path = f"{base_path}_sqrt{os.path.splitext(descriptive_path)[1]}"

        # Calculate square root of RV
        rv_1h_sqrt = np.sqrt(rv_1h)

        # Save the square root results
        if sqrt_path.endswith('.csv'):
            rv_1h_sqrt.to_csv(sqrt_path)
        elif sqrt_path.endswith('.parquet'):
            rv_1h_sqrt.to_parquet(sqrt_path)
        elif sqrt_path.endswith('.pickle'):
            rv_1h_sqrt.to_pickle(sqrt_path)

        logger.info(f"Saved square root of RV to {sqrt_path}")

    logger.info("Results saved successfully")

    return descriptive_path


def main():
    """
    Main function to calculate 1-hour realized volatility from 5-minute data.
    """
    parser = argparse.ArgumentParser(description='Calculate 1-hour realized volatility from 5-minute cryptocurrency data.')
    parser.add_argument('--input-file', type=str, default='data/df_cl_5m.parquet',
                        help='Path to the 5-minute data file.')
    parser.add_argument('--output-file', type=str, default='data/crypto_rv1h.csv',
                        help='Path to save the 1-hour realized volatility.')
    parser.add_argument('--method', type=str, default='sum_of_squares',
                        choices=['sum_of_squares', 'standard_deviation'],
                        help='Method to calculate realized volatility.')
    parser.add_argument('--annualize', action='store_true',
                        help='Whether to annualize the volatility.')
    parser.add_argument('--include-sqrt', action='store_true', default=True,
                        help='Whether to also save the square root of RV.')
    parser.add_argument('--data-source', type=str, default='crypto',
                        help='Source of the data (e.g., crypto, stocks).')
    parser.add_argument('--missing-threshold', type=float, default=0.3,
                        help='Threshold for dropping dates with too many missing values (0.0-1.0).')
    parser.add_argument('--handling-method', type=str, default='drop',
                        choices=['drop', 'fill', 'interpolate', 'none'],
                        help='Method to handle missing values.')
    parser.add_argument('--fill-method', type=str, default='ffill',
                        choices=['ffill', 'bfill', 'zero'],
                        help='Method to fill missing values if handling-method is "fill".')
    parser.add_argument('--interpolate-method', type=str, default='linear',
                        choices=['linear', 'time', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'],
                        help='Method to interpolate missing values if handling-method is "interpolate".')
    parser.add_argument('--save-missing-report', action='store_true',
                        help='Whether to save a report of missing values.')

    args = parser.parse_args()

    # Load the data
    df = load_data(args.input_file)

    # Inspect and clean the data
    df_cleaned, missing_info = inspect_and_clean_data(
        df,
        missing_threshold=args.missing_threshold,
        handling_method=args.handling_method,
        fill_method=args.fill_method,
        interpolate_method=args.interpolate_method
    )

    # Save missing values report if requested
    if args.save_missing_report:
        import json
        report_path = os.path.splitext(args.output_file)[0] + '_missing_report.json'
        with open(report_path, 'w') as f:
            json.dump(missing_info, f, indent=2)
        logger.info(f"Missing values report saved to {report_path}")

    # Calculate returns
    returns = calculate_returns(df_cleaned)

    # Calculate 1-hour realized volatility
    rv_1h = calculate_1h_rv(returns, method=args.method, annualize=args.annualize)

    # Save the results
    output_path = save_results(
        rv_1h,
        args.output_file,
        include_sqrt=args.include_sqrt,
        data_source=args.data_source
    )

    logger.info(f"1-hour realized volatility calculation completed successfully.")
    logger.info(f"Results saved to {output_path} and its sqrt version.")


if __name__ == '__main__':
    main()
