import pandas as pd
import numpy as np
from pathlib import Path

def calculate_realized_volatility(df, resample_freq='1H'):
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

def calculate_pct_change_volatility(df, freq='1H'):
    """
    Calculate volatility using simple percentage changes
    
    Args:
        df: DataFrame with price data (can be 1H or 5min)
        freq: frequency for resampling (default '1H')
    
    Returns:
        DataFrame with percentage change volatility
    """
    # If data is not already at target frequency, resample it
    if df.index.freq and df.index.freq != freq:
        print(f"Resampling data to {freq}")
        df = df.resample(freq).last()
    
    # Calculate percentage changes
    pct_changes = df.pct_change()
    
    # Calculate volatility as absolute percentage change
    volatility = np.abs(pct_changes)
    
    return volatility

def calculate_volatility(df, method='pct_change', freq='1H'):
    """
    Calculate volatility using specified method
    
    Args:
        df: DataFrame with price data
        method: 'realized' for 5min RV or 'pct_change' for percentage changes
        freq: frequency for resampling
    
    Returns:
        DataFrame with volatility measures
    """
    if method == 'realized':
        freq_df = df.index.freq if df.index.freq else pd.infer_freq(df.index) 
        # Check if data is 5-minute data with time delta
        print(f'freq_df: {freq_df} into {freq}')
        # if freq_df and freq_df != '5min':
            # raise ValueError("Realized volatility requires 5-minute data")
        return calculate_realized_volatility(df, freq)
    elif method == 'pct_change':
        return calculate_pct_change_volatility(df, freq)
    else:
        raise ValueError("method must be either 'realized' or 'pct_change'")
