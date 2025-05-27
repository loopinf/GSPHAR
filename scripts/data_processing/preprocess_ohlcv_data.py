#!/usr/bin/env python
"""
Preprocess OHLCV data for trading strategy training.

This script:
1. Loads OHLCV data from combined files
2. Aligns time indices across all assets
3. Handles missing data appropriately
4. Creates clean datasets for model training
5. Prepares data for both long (low prices) and short (high prices) strategies
"""

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_ohlcv_combined_data(data_dir="data/ohlcv_1h"):
    """
    Load combined OHLCV data from CSV files.
    
    Args:
        data_dir (str): Directory containing OHLCV files
        
    Returns:
        dict: Dictionary with keys ['open', 'high', 'low', 'close', 'volume']
              and values as DataFrames with assets as columns
    """
    logger.info(f"Loading OHLCV data from {data_dir}")
    
    ohlcv_data = {}
    components = ['open', 'high', 'low', 'close', 'volume']
    
    for component in components:
        file_path = os.path.join(data_dir, f"crypto_{component}_1h_38.csv")
        
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            ohlcv_data[component] = df
            logger.info(f"  {component.upper()}: {df.shape}")
        else:
            logger.warning(f"  {component.upper()}: File not found - {file_path}")
    
    return ohlcv_data


def analyze_data_quality(ohlcv_data):
    """
    Analyze data quality and missing values.
    
    Args:
        ohlcv_data (dict): OHLCV data dictionary
        
    Returns:
        dict: Analysis results
    """
    logger.info("Analyzing data quality...")
    
    analysis = {}
    
    for component, df in ohlcv_data.items():
        component_analysis = {
            'shape': df.shape,
            'date_range': (df.index.min(), df.index.max()),
            'missing_values': df.isnull().sum().sum(),
            'missing_by_asset': df.isnull().sum().to_dict(),
            'complete_cases': df.dropna().shape[0],
            'assets_with_missing': (df.isnull().sum() > 0).sum()
        }
        analysis[component] = component_analysis
        
        logger.info(f"  {component.upper()}:")
        logger.info(f"    Shape: {component_analysis['shape']}")
        logger.info(f"    Date range: {component_analysis['date_range'][0]} to {component_analysis['date_range'][1]}")
        logger.info(f"    Missing values: {component_analysis['missing_values']}")
        logger.info(f"    Complete cases: {component_analysis['complete_cases']}")
        logger.info(f"    Assets with missing data: {component_analysis['assets_with_missing']}")
    
    return analysis


def align_time_indices(ohlcv_data):
    """
    Align time indices across all OHLCV components.
    
    Args:
        ohlcv_data (dict): OHLCV data dictionary
        
    Returns:
        tuple: (aligned_data, common_index)
    """
    logger.info("Aligning time indices...")
    
    # Find common time index across all components
    common_index = None
    for component, df in ohlcv_data.items():
        if common_index is None:
            common_index = df.index
        else:
            common_index = common_index.intersection(df.index)
    
    logger.info(f"  Common time periods: {len(common_index)}")
    logger.info(f"  Date range: {common_index.min()} to {common_index.max()}")
    
    # Align all components to common index
    aligned_data = {}
    for component, df in ohlcv_data.items():
        aligned_df = df.loc[common_index].copy()
        aligned_data[component] = aligned_df
        logger.info(f"  {component.upper()}: {df.shape} -> {aligned_df.shape}")
    
    return aligned_data, common_index


def handle_missing_data(aligned_data, strategy='drop_assets'):
    """
    Handle missing data in aligned OHLCV data.
    
    Args:
        aligned_data (dict): Aligned OHLCV data
        strategy (str): Strategy for handling missing data
                       'drop_assets' - drop assets with any missing data
                       'drop_periods' - drop time periods with any missing data
                       'forward_fill' - forward fill missing values
                       'interpolate' - interpolate missing values
    
    Returns:
        dict: Cleaned OHLCV data
    """
    logger.info(f"Handling missing data with strategy: {strategy}")
    
    if strategy == 'drop_assets':
        # Find assets with complete data across all components
        complete_assets = None
        
        for component, df in aligned_data.items():
            assets_complete = df.columns[df.isnull().sum() == 0]
            if complete_assets is None:
                complete_assets = assets_complete
            else:
                complete_assets = complete_assets.intersection(assets_complete)
        
        logger.info(f"  Assets with complete data: {len(complete_assets)}")
        logger.info(f"  Complete assets: {list(complete_assets)}")
        
        # Keep only complete assets
        cleaned_data = {}
        for component, df in aligned_data.items():
            cleaned_df = df[complete_assets].copy()
            cleaned_data[component] = cleaned_df
            logger.info(f"  {component.upper()}: {df.shape} -> {cleaned_df.shape}")
    
    elif strategy == 'drop_periods':
        # Find periods with complete data across all components and assets
        complete_periods = None
        
        for component, df in aligned_data.items():
            periods_complete = df.index[df.isnull().sum(axis=1) == 0]
            if complete_periods is None:
                complete_periods = periods_complete
            else:
                complete_periods = complete_periods.intersection(periods_complete)
        
        logger.info(f"  Periods with complete data: {len(complete_periods)}")
        
        # Keep only complete periods
        cleaned_data = {}
        for component, df in aligned_data.items():
            cleaned_df = df.loc[complete_periods].copy()
            cleaned_data[component] = cleaned_df
            logger.info(f"  {component.upper()}: {df.shape} -> {cleaned_df.shape}")
    
    elif strategy == 'forward_fill':
        # Forward fill missing values
        cleaned_data = {}
        for component, df in aligned_data.items():
            cleaned_df = df.fillna(method='ffill').copy()
            remaining_na = cleaned_df.isnull().sum().sum()
            cleaned_data[component] = cleaned_df
            logger.info(f"  {component.upper()}: {df.isnull().sum().sum()} -> {remaining_na} missing values")
    
    elif strategy == 'interpolate':
        # Interpolate missing values
        cleaned_data = {}
        for component, df in aligned_data.items():
            cleaned_df = df.interpolate(method='time').copy()
            remaining_na = cleaned_df.isnull().sum().sum()
            cleaned_data[component] = cleaned_df
            logger.info(f"  {component.upper()}: {df.isnull().sum().sum()} -> {remaining_na} missing values")
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return cleaned_data


def validate_ohlcv_consistency(cleaned_data):
    """
    Validate OHLCV data consistency (High >= Low, etc.).
    
    Args:
        cleaned_data (dict): Cleaned OHLCV data
        
    Returns:
        dict: Validation results
    """
    logger.info("Validating OHLCV consistency...")
    
    high_df = cleaned_data['high']
    low_df = cleaned_data['low']
    open_df = cleaned_data['open']
    close_df = cleaned_data['close']
    
    validation_results = {}
    
    # Check High >= Low
    high_low_violations = (high_df < low_df).sum().sum()
    validation_results['high_low_violations'] = high_low_violations
    
    # Check High >= Open, Close
    high_open_violations = (high_df < open_df).sum().sum()
    high_close_violations = (high_df < close_df).sum().sum()
    validation_results['high_open_violations'] = high_open_violations
    validation_results['high_close_violations'] = high_close_violations
    
    # Check Low <= Open, Close
    low_open_violations = (low_df > open_df).sum().sum()
    low_close_violations = (low_df > close_df).sum().sum()
    validation_results['low_open_violations'] = low_open_violations
    validation_results['low_close_violations'] = low_close_violations
    
    logger.info("  Validation results:")
    for violation_type, count in validation_results.items():
        logger.info(f"    {violation_type}: {count}")
    
    total_violations = sum(validation_results.values())
    if total_violations == 0:
        logger.info("  ✅ All OHLCV data is consistent!")
    else:
        logger.warning(f"  ⚠️  Found {total_violations} OHLCV consistency violations")
    
    return validation_results


def create_combined_ohlcv_tensor(cleaned_data):
    """
    Create a combined OHLCV tensor for model training.
    
    Args:
        cleaned_data (dict): Cleaned OHLCV data
        
    Returns:
        tuple: (ohlcv_tensor, asset_names, time_index)
               ohlcv_tensor shape: [time_periods, assets, 5] (O,H,L,C,V)
    """
    logger.info("Creating combined OHLCV tensor...")
    
    # Get dimensions
    time_periods = len(cleaned_data['close'])
    assets = list(cleaned_data['close'].columns)
    n_assets = len(assets)
    
    # Create tensor: [time_periods, assets, 5]
    ohlcv_tensor = np.zeros((time_periods, n_assets, 5))
    
    # Fill tensor
    component_order = ['open', 'high', 'low', 'close', 'volume']
    for i, component in enumerate(component_order):
        ohlcv_tensor[:, :, i] = cleaned_data[component].values
    
    time_index = cleaned_data['close'].index
    
    logger.info(f"  OHLCV tensor shape: {ohlcv_tensor.shape}")
    logger.info(f"  Assets: {n_assets}")
    logger.info(f"  Time periods: {time_periods}")
    logger.info(f"  Components: {component_order}")
    
    return ohlcv_tensor, assets, time_index


def save_preprocessed_data(cleaned_data, ohlcv_tensor, assets, time_index, output_dir="data/preprocessed"):
    """
    Save preprocessed data for model training.
    
    Args:
        cleaned_data (dict): Cleaned OHLCV data
        ohlcv_tensor (np.ndarray): Combined OHLCV tensor
        assets (list): Asset names
        time_index (pd.DatetimeIndex): Time index
        output_dir (str): Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving preprocessed data to {output_dir}")
    
    # Save individual components
    for component, df in cleaned_data.items():
        output_file = os.path.join(output_dir, f"crypto_{component}_1h_clean.csv")
        df.to_csv(output_file)
        logger.info(f"  Saved {component}: {output_file}")
    
    # Save combined tensor as numpy
    tensor_file = os.path.join(output_dir, "ohlcv_tensor.npy")
    np.save(tensor_file, ohlcv_tensor)
    logger.info(f"  Saved tensor: {tensor_file}")
    
    # Save metadata
    metadata = {
        'assets': assets,
        'time_range': (str(time_index.min()), str(time_index.max())),
        'shape': ohlcv_tensor.shape,
        'components': ['open', 'high', 'low', 'close', 'volume']
    }
    
    metadata_file = os.path.join(output_dir, "metadata.txt")
    with open(metadata_file, 'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    logger.info(f"  Saved metadata: {metadata_file}")
    
    # Save time index
    time_index_file = os.path.join(output_dir, "time_index.csv")
    pd.DataFrame({'datetime': time_index}).to_csv(time_index_file, index=False)
    logger.info(f"  Saved time index: {time_index_file}")


def main():
    """
    Main preprocessing function.
    """
    logger.info("Starting OHLCV data preprocessing...")
    
    # Step 1: Load data
    ohlcv_data = load_ohlcv_combined_data()
    
    if not ohlcv_data:
        logger.error("No OHLCV data loaded. Check file paths.")
        return
    
    # Step 2: Analyze data quality
    analysis = analyze_data_quality(ohlcv_data)
    
    # Step 3: Align time indices
    aligned_data, common_index = align_time_indices(ohlcv_data)
    
    # Step 4: Handle missing data
    # You can change strategy here: 'drop_assets', 'drop_periods', 'forward_fill', 'interpolate'
    cleaned_data = handle_missing_data(aligned_data, strategy='drop_assets')
    
    # Step 5: Validate consistency
    validation_results = validate_ohlcv_consistency(cleaned_data)
    
    # Step 6: Create combined tensor
    ohlcv_tensor, assets, time_index = create_combined_ohlcv_tensor(cleaned_data)
    
    # Step 7: Save preprocessed data
    save_preprocessed_data(cleaned_data, ohlcv_tensor, assets, time_index)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("PREPROCESSING SUMMARY")
    logger.info("="*60)
    logger.info(f"✅ Final dataset shape: {ohlcv_tensor.shape}")
    logger.info(f"✅ Assets: {len(assets)}")
    logger.info(f"✅ Time periods: {len(time_index)}")
    logger.info(f"✅ Date range: {time_index.min()} to {time_index.max()}")
    logger.info(f"✅ Components: [open, high, low, close, volume]")
    logger.info(f"✅ Data saved to: data/preprocessed/")
    
    if sum(validation_results.values()) == 0:
        logger.info("✅ All OHLCV data is consistent")
    else:
        logger.warning(f"⚠️  {sum(validation_results.values())} OHLCV consistency violations found")


if __name__ == "__main__":
    main()
