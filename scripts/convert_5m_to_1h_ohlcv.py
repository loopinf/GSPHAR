#!/usr/bin/env python
"""
Convert 5-minute close data to 1-hour OHLCV data for proper trading simulation.

This script takes the 5-minute close data and creates proper OHLCV bars for each hour,
which is essential for accurate order fill detection using high/low prices.
"""

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_5m_data(file_path):
    """
    Load 5-minute close data.

    Args:
        file_path (str): Path to the 5-minute data file

    Returns:
        pd.DataFrame: 5-minute close data with datetime index
    """
    logger.info(f"Loading 5-minute data from {file_path}")

    df = pd.read_parquet(file_path)

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    logger.info(f"Loaded 5-minute data: {df.shape}")
    logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")
    logger.info(f"Columns: {list(df.columns)}")

    return df


def create_synthetic_ohlcv_from_close(close_data, volatility_factor=0.02):
    """
    Create synthetic OHLCV data from close prices.

    Since we only have close prices, we'll create realistic OHLCV bars by:
    1. Using close prices as the base
    2. Creating realistic high/low based on typical intraday volatility
    3. Creating open prices from previous close

    Args:
        close_data (pd.Series): Close prices for one asset
        volatility_factor (float): Factor to determine high/low spread

    Returns:
        pd.DataFrame: OHLCV data with columns [open, high, low, close, volume]
    """

    # Calculate returns for volatility estimation
    returns = close_data.pct_change().dropna()

    # Estimate typical intraday volatility (use rolling std of returns)
    rolling_vol = returns.rolling(window=12, min_periods=1).std()  # 12 periods = 1 hour

    # Create OHLCV data
    ohlcv_data = []

    for i in range(len(close_data)):
        close_price = close_data.iloc[i]

        if i == 0:
            # First period: open = close
            open_price = close_price
            vol = volatility_factor
        else:
            # Open = previous close
            open_price = close_data.iloc[i-1]
            # Use estimated volatility, with minimum threshold
            if i < len(rolling_vol) and not pd.isna(rolling_vol.iloc[i]):
                vol = max(rolling_vol.iloc[i], volatility_factor)
            else:
                vol = volatility_factor

        # Create realistic high/low based on open and close
        high_low_range = vol * close_price

        # High is the maximum of open, close, plus some random variation
        high_price = max(open_price, close_price) + np.random.uniform(0, high_low_range)

        # Low is the minimum of open, close, minus some random variation
        low_price = min(open_price, close_price) - np.random.uniform(0, high_low_range)

        # Ensure logical constraints
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)

        # Synthetic volume (can be improved with actual volume data if available)
        volume = np.random.uniform(1000, 5000)

        ohlcv_data.append({
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })

    return pd.DataFrame(ohlcv_data, index=close_data.index)


def convert_5m_to_1h_ohlcv(df_5m):
    """
    Convert 5-minute close data to 1-hour OHLCV data.

    Args:
        df_5m (pd.DataFrame): 5-minute close data

    Returns:
        dict: Dictionary with asset names as keys and 1-hour OHLCV DataFrames as values
    """
    logger.info("Converting 5-minute data to 1-hour OHLCV...")

    ohlcv_dict = {}

    for asset in df_5m.columns:
        logger.info(f"Processing {asset}...")

        # Get 5-minute close data for this asset
        close_5m = df_5m[asset].dropna()

        # First, create synthetic OHLCV for 5-minute bars
        ohlcv_5m = create_synthetic_ohlcv_from_close(close_5m)

        # Then aggregate to 1-hour OHLCV
        ohlcv_1h = ohlcv_5m.resample('1H').agg({
            'open': 'first',    # First open in the hour
            'high': 'max',      # Highest high in the hour
            'low': 'min',       # Lowest low in the hour
            'close': 'last',    # Last close in the hour
            'volume': 'sum'     # Sum of volume in the hour
        }).dropna()

        ohlcv_dict[asset] = ohlcv_1h

        logger.info(f"  {asset}: {len(close_5m)} 5m bars -> {len(ohlcv_1h)} 1h bars")

    return ohlcv_dict


def save_ohlcv_data(ohlcv_dict, output_dir="data/ohlcv_1h"):
    """
    Save OHLCV data to files.

    Args:
        ohlcv_dict (dict): Dictionary of OHLCV DataFrames
        output_dir (str): Output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Saving OHLCV data to {output_dir}...")

    # Save individual asset files
    for asset, ohlcv_df in ohlcv_dict.items():
        file_path = os.path.join(output_dir, f"{asset}_1h_ohlcv.csv")
        ohlcv_df.to_csv(file_path)
        logger.info(f"  Saved {asset}: {file_path}")

    # Create combined files for each OHLC component
    all_assets = list(ohlcv_dict.keys())
    common_index = None

    # Find common time index
    for asset in all_assets:
        if common_index is None:
            common_index = ohlcv_dict[asset].index
        else:
            common_index = common_index.intersection(ohlcv_dict[asset].index)

    logger.info(f"Common time index: {len(common_index)} periods")

    # Create combined DataFrames for each OHLC component
    for component in ['open', 'high', 'low', 'close', 'volume']:
        combined_df = pd.DataFrame(index=common_index)

        for asset in all_assets:
            asset_data = ohlcv_dict[asset].loc[common_index, component]
            combined_df[asset] = asset_data

        # Save combined file
        output_file = os.path.join(output_dir, f"crypto_{component}_1h_38.csv")
        combined_df.to_csv(output_file)
        logger.info(f"  Saved combined {component}: {output_file}")

    return output_dir


def create_sample_ohlcv_for_testing():
    """
    Create a small sample OHLCV dataset for testing purposes.
    """
    logger.info("Creating sample OHLCV data for testing...")

    # Create sample data for 3 assets over 100 hours
    dates = pd.date_range('2024-01-01', periods=100, freq='H')
    assets = ['BTC', 'ETH', 'ADA']

    ohlcv_dict = {}

    for asset in assets:
        # Create realistic price movement
        np.random.seed(42 if asset == 'BTC' else 43 if asset == 'ETH' else 44)

        base_price = 100 if asset == 'BTC' else 50 if asset == 'ETH' else 1
        returns = np.random.normal(0, 0.02, len(dates))
        prices = base_price * np.exp(np.cumsum(returns))

        ohlcv_data = []
        for i, (date, close_price) in enumerate(zip(dates, prices)):
            if i == 0:
                open_price = close_price
            else:
                open_price = prices[i-1]

            # Create realistic high/low
            volatility = 0.01 * close_price
            high_price = max(open_price, close_price) + np.random.uniform(0, volatility)
            low_price = min(open_price, close_price) - np.random.uniform(0, volatility)

            # Ensure constraints
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)

            volume = np.random.uniform(1000, 5000)

            ohlcv_data.append({
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })

        ohlcv_dict[asset] = pd.DataFrame(ohlcv_data, index=dates)

    # Save sample data
    sample_dir = save_ohlcv_data(ohlcv_dict, "data/sample_ohlcv_1h")
    logger.info(f"Sample OHLCV data created in: {sample_dir}")

    return ohlcv_dict


def main():
    """
    Main function to convert 5-minute data to 1-hour OHLCV.
    """
    logger.info("Starting 5-minute to 1-hour OHLCV conversion...")

    # File paths
    input_file = "data/df_cl_5m.parquet"

    if not os.path.exists(input_file):
        logger.warning(f"Input file {input_file} not found. Creating sample data for testing...")
        create_sample_ohlcv_for_testing()
        return

    try:
        # Load 5-minute data
        df_5m = load_5m_data(input_file)

        # Convert to 1-hour OHLCV
        ohlcv_dict = convert_5m_to_1h_ohlcv(df_5m)

        # Save results
        output_dir = save_ohlcv_data(ohlcv_dict)

        logger.info("Conversion completed successfully!")
        logger.info(f"Output saved to: {output_dir}")

        # Print summary
        logger.info("\nSUMMARY:")
        logger.info(f"  Input: {df_5m.shape[0]} 5-minute bars, {df_5m.shape[1]} assets")

        total_1h_bars = sum(len(ohlcv_df) for ohlcv_df in ohlcv_dict.values())
        avg_1h_bars = total_1h_bars / len(ohlcv_dict) if ohlcv_dict else 0

        logger.info(f"  Output: ~{avg_1h_bars:.0f} 1-hour bars per asset, {len(ohlcv_dict)} assets")
        logger.info(f"  Files created in: {output_dir}")

    except Exception as e:
        logger.error(f"Error during conversion: {e}")
        raise


if __name__ == "__main__":
    main()
