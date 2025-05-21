#!/usr/bin/env python
"""
Test script for the trading strategy loss function.

This script demonstrates how the trading strategy loss function works
with different scenarios and visualizes the results.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the trading loss function
from src.trading_loss import TradingStrategyLoss, convert_pct_change_to_log_returns

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)


def generate_synthetic_data(n_samples=100, holding_period=24, volatility_range=(0.01, 0.1)):
    """
    Generate synthetic data for testing the loss function.

    Args:
        n_samples (int): Number of samples to generate
        holding_period (int): Number of periods to hold the position
        volatility_range (tuple): Range of volatility values to generate

    Returns:
        tuple: (predicted_volatility, log_returns)
    """
    # Generate random volatility predictions
    predicted_volatility = np.random.uniform(
        volatility_range[0], volatility_range[1], n_samples
    )

    # Generate random log returns for next period and holding period
    # We'll create some correlation between predicted volatility and actual returns
    # to simulate a somewhat accurate model

    # Next period returns (correlated with -volatility to simulate price drops)
    next_period_returns = -0.5 * predicted_volatility + np.random.normal(0, 0.02, n_samples)

    # Holding period returns (slightly mean-reverting)
    holding_period_returns = np.random.normal(0.0005, 0.01, (n_samples, holding_period))

    # Combine next period and holding period returns
    all_returns = np.column_stack([next_period_returns, holding_period_returns])

    # Convert to torch tensors
    predicted_volatility_tensor = torch.tensor(predicted_volatility, dtype=torch.float32)
    log_returns_tensor = torch.tensor(all_returns, dtype=torch.float32)

    return predicted_volatility_tensor, log_returns_tensor


def test_with_synthetic_data():
    """
    Test the trading strategy loss function with synthetic data.
    """
    # Generate synthetic data
    holding_period = 24
    vol_pred, log_returns = generate_synthetic_data(
        n_samples=1000, holding_period=holding_period
    )

    # Create the loss function
    trading_loss = TradingStrategyLoss(
        alpha=1.0, beta=1.0, gamma=2.0, holding_period=holding_period
    )

    # Calculate the loss
    loss = trading_loss(vol_pred, log_returns)

    print(f"Loss with synthetic data: {loss.item():.6f}")

    # Calculate the components of the loss for analysis
    log_entry_threshold = torch.log(1 - vol_pred + 1e-6)
    log_return_next = log_returns[:, 0]
    filled_orders = (log_return_next <= log_entry_threshold).float()

    # Calculate fill rate
    fill_rate = filled_orders.mean().item()
    print(f"Order fill rate: {fill_rate:.2%}")

    # Calculate average return for filled orders
    holding_returns = torch.sum(log_returns[:, 1:holding_period+1], dim=1)
    filled_returns = holding_returns[filled_orders > 0]
    if len(filled_returns) > 0:
        avg_return = filled_returns.mean().item()
        print(f"Average log return for filled orders: {avg_return:.4f}")
        print(f"Average percentage return for filled orders: {(torch.exp(filled_returns).mean().item() - 1):.4%}")

        # Calculate win rate
        win_rate = (filled_returns > 0).float().mean().item()
        print(f"Win rate for filled orders: {win_rate:.2%}")
    else:
        print("No orders were filled")


def test_with_real_data(crypto_data_file, symbol="BTCUSDT", start_date=None, end_date=None, min_data_threshold=0.8):
    """
    Test the trading strategy loss function with real cryptocurrency data.

    Args:
        crypto_data_file (str): Path to the cryptocurrency percentage change data file
        symbol (str): Symbol to use for testing
        start_date (str, optional): Start date for testing (YYYY-MM-DD)
        end_date (str, optional): End date for testing (YYYY-MM-DD)
        min_data_threshold (float): Minimum fraction of non-NaN values required to include a date (0.0-1.0)
    """
    # Load the data
    df = pd.read_csv(crypto_data_file, index_col=0, parse_dates=True)

    # Print initial data shape
    print(f"Initial data shape: {df.shape}")

    # Analyze data availability
    data_availability = df.notna().sum(axis=1) / df.shape[1]
    print(f"Data availability range: {data_availability.min():.2%} to {data_availability.max():.2%}")

    # Find the date when data availability exceeds the threshold
    if start_date is None:
        dates_with_sufficient_data = data_availability[data_availability >= min_data_threshold].index
        if len(dates_with_sufficient_data) > 0:
            auto_start_date = dates_with_sufficient_data[0]
            print(f"Automatically determined start date: {auto_start_date}")
            df = df[df.index >= auto_start_date]
        else:
            print(f"Warning: No dates found with at least {min_data_threshold:.0%} data availability")
    else:
        df = df[df.index >= start_date]

    if end_date:
        df = df[df.index <= end_date]

    # Print filtered data shape
    print(f"Filtered data shape: {df.shape}")

    # Check if the selected symbol has data
    if symbol not in df.columns:
        print(f"Error: Symbol {symbol} not found in data")
        return

    # Check for missing values in the selected symbol
    missing_values = df[symbol].isna().sum()
    if missing_values > 0:
        print(f"Warning: {missing_values} missing values found for {symbol} ({missing_values/len(df):.2%})")

        # Fill missing values with forward fill, then backward fill
        df[symbol] = df[symbol].fillna(method='ffill').fillna(method='bfill')
        print(f"Missing values filled with forward/backward fill")

    # Get the percentage changes for the symbol
    pct_changes = df[symbol].values

    # Convert to log returns
    log_returns = convert_pct_change_to_log_returns(pct_changes)

    # Create sequences of log returns for the loss function
    holding_period = 24  # 24 hours
    sequences = []

    for i in range(len(log_returns) - holding_period):
        sequences.append(log_returns[i:i+holding_period+1])

    # Convert to torch tensor
    log_returns_tensor = torch.tensor(np.array(sequences), dtype=torch.float32)

    # Generate some mock volatility predictions (in a real scenario, these would come from your model)
    # Here we'll use a simple rolling standard deviation of returns as a proxy
    rolling_std = pd.Series(pct_changes).rolling(window=24).std().values
    rolling_std = np.nan_to_num(rolling_std, nan=0.02)  # Replace NaNs with a default value

    # Use only the volatility values that align with our sequences
    vol_pred = torch.tensor(rolling_std[:len(sequences)], dtype=torch.float32)

    # Create the loss function
    trading_loss = TradingStrategyLoss(
        alpha=1.0, beta=1.0, gamma=2.0, holding_period=holding_period
    )

    # Calculate the loss
    loss = trading_loss(vol_pred, log_returns_tensor)

    print(f"\nLoss with real {symbol} data: {loss.item():.6f}")

    # Calculate the components of the loss for analysis
    log_entry_threshold = torch.log(1 - vol_pred + 1e-6)
    log_return_next = log_returns_tensor[:, 0]
    filled_orders = (log_return_next <= log_entry_threshold).float()

    # Calculate fill rate
    fill_rate = filled_orders.mean().item()
    print(f"Order fill rate: {fill_rate:.2%}")

    # Calculate average return for filled orders
    holding_returns = torch.sum(log_returns_tensor[:, 1:holding_period+1], dim=1)
    filled_returns = holding_returns[filled_orders > 0]
    if len(filled_returns) > 0:
        avg_return = filled_returns.mean().item()
        print(f"Average log return for filled orders: {avg_return:.4f}")
        print(f"Average percentage return for filled orders: {(torch.exp(filled_returns).mean().item() - 1):.4%}")

        # Calculate win rate
        win_rate = (filled_returns > 0).float().mean().item()
        print(f"Win rate for filled orders: {win_rate:.2%}")
    else:
        print("No orders were filled")

    # Visualize the results
    visualize_trading_strategy(
        df.index[:-holding_period],
        vol_pred.numpy(),
        log_return_next.numpy(),
        filled_orders.numpy(),
        holding_returns.numpy(),
        symbol,
        holding_period
    )


def visualize_trading_strategy(dates, volatility, next_returns, filled_orders, holding_returns, symbol, holding_period):
    """
    Visualize the trading strategy results.

    Args:
        dates (array-like): Dates for the data points
        volatility (array-like): Predicted volatility
        next_returns (array-like): Next period log returns
        filled_orders (array-like): Binary array indicating filled orders
        holding_returns (array-like): Holding period returns
        symbol (str): Symbol being analyzed
        holding_period (int): Holding period in hours
    """
    # Create output directory
    output_dir = Path("plots/trading_strategy_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to pandas Series for easier plotting
    dates = pd.to_datetime(dates)
    vol_series = pd.Series(volatility, index=dates)
    next_return_series = pd.Series(next_returns, index=dates)
    filled_series = pd.Series(filled_orders, index=dates)
    holding_return_series = pd.Series(holding_returns, index=dates)

    # Calculate entry thresholds
    clipped_volatility = np.clip(volatility, 0.0, 0.99)
    entry_threshold_series = pd.Series(np.log(1 - clipped_volatility), index=dates)

    # Calculate cumulative returns from the strategy
    strategy_returns = holding_return_series * filled_series
    cumulative_strategy_returns = np.exp(strategy_returns.cumsum()) - 1

    # Plot 1: Volatility predictions and next period returns
    plt.figure(figsize=(12, 6))
    plt.plot(vol_series.index, vol_series, label='Predicted Volatility', color='blue')
    plt.plot(next_return_series.index, next_return_series, label='Next Period Log Return', color='green', alpha=0.5)
    plt.plot(entry_threshold_series.index, entry_threshold_series, label='Entry Threshold', color='red', linestyle='--')

    # Highlight filled orders
    filled_dates = filled_series[filled_series > 0].index
    plt.scatter(filled_dates, next_return_series[filled_dates], color='red', s=30, label='Filled Orders')

    plt.title(f'Volatility Predictions and Next Period Returns for {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{symbol}_volatility_returns.png", dpi=300)

    # Plot 2: Holding period returns for filled orders
    plt.figure(figsize=(12, 6))

    # Create a mask for profitable and unprofitable trades
    profitable_mask = (holding_return_series > 0) & (filled_series > 0)
    unprofitable_mask = (holding_return_series <= 0) & (filled_series > 0)

    plt.bar(holding_return_series[profitable_mask].index,
            holding_return_series[profitable_mask],
            color='green', alpha=0.7, label='Profitable Trades')
    plt.bar(holding_return_series[unprofitable_mask].index,
            holding_return_series[unprofitable_mask],
            color='red', alpha=0.7, label='Unprofitable Trades')

    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title(f'Holding Period Returns for Filled Orders ({holding_period} hours) - {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Log Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{symbol}_holding_returns.png", dpi=300)

    # Plot 3: Cumulative strategy returns
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_strategy_returns.index, cumulative_strategy_returns * 100, label='Cumulative Strategy Returns (%)', color='blue')
    plt.title(f'Cumulative Strategy Returns for {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{symbol}_cumulative_returns.png", dpi=300)

    print(f"Visualizations saved to {output_dir}")


def main():
    """
    Main function to run the tests.
    """
    print("Testing trading strategy loss function with synthetic data...")
    test_with_synthetic_data()

    print("\nTesting trading strategy loss function with real data...")
    # Update the path to your percentage change data file
    crypto_data_file = "data/crypto_pct_change_1h_38_20200101_20250116.csv"

    # Set the minimum data threshold (percentage of non-NaN values required)
    min_data_threshold = 0.8  # 80% of symbols must have data

    # Test with different cryptocurrencies
    for symbol in ["BTCUSDT", "ETHUSDT", "LTCUSDT"]:
        print(f"\n{'='*50}")
        print(f"Testing with {symbol}")
        print(f"{'='*50}")
        test_with_real_data(
            crypto_data_file,
            symbol=symbol,
            min_data_threshold=min_data_threshold
        )


if __name__ == "__main__":
    main()
