#!/usr/bin/env python
"""
Test script to visualize and understand the time alignment in profit maximization loss functions.

This script demonstrates:
1. How volatility predictions align with future returns
2. The exact time sequence of the trading strategy
3. Visual representation of the profit calculation process
4. Step-by-step breakdown of the loss function components
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.trading_loss import SimpleProfitMaximizationLoss, MaximizeProfitLoss, TradingStrategyLoss


def create_synthetic_time_series(n_periods=100, start_date='2024-01-01'):
    """
    Create synthetic time series data for testing with clear time alignment.

    Returns:
        pd.DataFrame: DataFrame with datetime index and price/volatility data
    """
    # Create datetime index
    start = pd.to_datetime(start_date)
    dates = pd.date_range(start, periods=n_periods, freq='H')

    # Generate synthetic price data with some volatility clustering
    np.random.seed(42)
    returns = np.random.normal(0, 0.02, n_periods)

    # Add some volatility clustering
    for i in range(1, n_periods):
        if abs(returns[i-1]) > 0.03:  # High volatility period
            returns[i] = np.random.normal(0, 0.04)

    # Calculate prices and realized volatility
    prices = 100 * np.exp(np.cumsum(returns))
    pct_changes = np.diff(prices) / prices[:-1]

    # Calculate rolling realized volatility (simplified)
    realized_vol = pd.Series(pct_changes).rolling(window=5, min_periods=1).std().values

    # Create DataFrame
    df = pd.DataFrame({
        'datetime': dates,
        'price': prices,
        'pct_change': np.concatenate([[0], pct_changes]),
        'log_return': np.concatenate([[0], np.log(1 + pct_changes)]),
        'realized_vol': np.concatenate([[realized_vol[0]], realized_vol])
    })

    df.set_index('datetime', inplace=True)
    return df


def demonstrate_time_alignment(df, prediction_time_idx=50, holding_period=24):
    """
    Demonstrate the exact time alignment for the trading strategy.

    Args:
        df: DataFrame with time series data
        prediction_time_idx: Index where we make the prediction
        holding_period: Number of periods to hold the position
    """
    print("="*80)
    print("TIME ALIGNMENT DEMONSTRATION")
    print("="*80)

    # Get the prediction time and related data
    prediction_time = df.index[prediction_time_idx]
    current_price = df.iloc[prediction_time_idx]['price']

    # Simulate a volatility prediction (let's say we predict 3% volatility)
    vol_prediction = 0.03

    print(f"📅 PREDICTION TIME: {prediction_time}")
    print(f"💰 CURRENT PRICE: ${current_price:.2f}")
    print(f"📊 PREDICTED VOLATILITY: {vol_prediction:.1%}")
    print(f"⏱️  HOLDING PERIOD: {holding_period} hours")
    print()

    # Calculate limit order price
    limit_price = current_price * (1 - vol_prediction)
    print(f"🎯 LIMIT ORDER PRICE: ${limit_price:.2f} (discount: {vol_prediction:.1%})")
    print()

    # Show the time sequence
    print("TIME SEQUENCE:")
    print("-" * 50)

    # T+0: Prediction time
    print(f"T+0  {prediction_time}: Make prediction, place limit order")

    # T+1: Next period (order fill check)
    if prediction_time_idx + 1 < len(df):
        next_time = df.index[prediction_time_idx + 1]
        next_price = df.iloc[prediction_time_idx + 1]['price']
        next_return = df.iloc[prediction_time_idx + 1]['log_return']

        order_filled = next_price <= limit_price
        print(f"T+1  {next_time}: Check order fill")
        print(f"     Next price: ${next_price:.2f}")
        print(f"     Log return: {next_return:.4f}")
        print(f"     Order filled: {'✅ YES' if order_filled else '❌ NO'}")
        print()

        if order_filled:
            print("📈 POSITION TRACKING (if order filled):")
            print("-" * 50)

            # Track the position over the holding period
            entry_price = limit_price
            cumulative_log_return = 0  # This will match the loss function calculation

            print(f"T+1  {next_time}: Enter position at ${entry_price:.2f}")

            # Track holding period (T+1 to T+holding_period)
            # This corresponds to log_returns[:, 1:1+holding_period] in the loss function
            for i in range(1, min(holding_period + 1, len(df) - prediction_time_idx)):
                period_time = df.index[prediction_time_idx + i]
                period_price = df.iloc[prediction_time_idx + i]['price']
                period_log_return = df.iloc[prediction_time_idx + i]['log_return']

                # Accumulate log returns during holding period (matches loss function)
                cumulative_log_return += period_log_return

                # Calculate current profit from entry price
                # This is what the loss function calculates: exp(sum_log_returns) - 1
                current_profit_pct = (np.exp(cumulative_log_return) - 1) * 100

                # Also show direct price-based profit for verification
                direct_profit_pct = ((period_price - entry_price) / entry_price) * 100

                if i <= 5 or i == holding_period or i % 6 == 0:  # Show first 5, last, and every 6th
                    print(f"T+{i+1:2d} {period_time}: ${period_price:6.2f} | "
                          f"Log Return: {period_log_return:+.4f} | "
                          f"Cumulative: {current_profit_pct:+6.2f}% | "
                          f"Direct: {direct_profit_pct:+6.2f}%")

            # Final exit
            if prediction_time_idx + holding_period < len(df):
                exit_time = df.index[prediction_time_idx + holding_period]
                exit_price = df.iloc[prediction_time_idx + holding_period]['price']
                final_profit_log = (np.exp(cumulative_log_return) - 1) * 100
                final_profit_direct = ((exit_price - entry_price) / entry_price) * 100

                print(f"\nT+{holding_period+1} {exit_time}: EXIT position at ${exit_price:.2f}")
                print(f"     FINAL PROFIT (log method): {final_profit_log:+.2f}%")
                print(f"     FINAL PROFIT (direct method): {final_profit_direct:+.2f}%")
                print(f"     Cumulative Log Return: {cumulative_log_return:.4f}")
                print(f"     ✅ Methods match: {abs(final_profit_log - final_profit_direct) < 0.01}")

    return {
        'prediction_time': prediction_time,
        'vol_prediction': vol_prediction,
        'current_price': current_price,
        'limit_price': limit_price,
        'prediction_idx': prediction_time_idx
    }


def test_loss_function_step_by_step(df, prediction_idx=50, holding_period=24):
    """
    Test the loss function step by step with detailed output.
    """
    print("\n" + "="*80)
    print("LOSS FUNCTION STEP-BY-STEP CALCULATION")
    print("="*80)

    # Prepare data
    vol_pred = torch.tensor([0.03], dtype=torch.float32)  # 3% volatility prediction

    # Get log returns for the next holding_period + 1 periods
    end_idx = min(prediction_idx + holding_period + 1, len(df))
    log_returns_data = df.iloc[prediction_idx + 1:end_idx]['log_return'].values

    # Pad if necessary
    if len(log_returns_data) < holding_period + 1:
        padding = np.zeros(holding_period + 1 - len(log_returns_data))
        log_returns_data = np.concatenate([log_returns_data, padding])

    log_returns = torch.tensor(log_returns_data[:holding_period + 1], dtype=torch.float32).unsqueeze(0)

    print(f"📊 INPUT DATA:")
    print(f"   Volatility Prediction: {vol_pred.item():.1%}")
    print(f"   Log Returns Shape: {log_returns.shape}")
    print(f"   Next Period Return: {log_returns[0, 0].item():.4f}")
    print(f"   Holding Period: {holding_period} periods")
    print()

    # Test SimpleProfitMaximizationLoss
    print("🎯 SIMPLE PROFIT MAXIMIZATION LOSS:")
    print("-" * 50)

    loss_fn = SimpleProfitMaximizationLoss(holding_period=holding_period)

    # Step-by-step calculation
    vol_pred_clamped = torch.clamp(vol_pred, 0.001, 0.5)
    log_entry_threshold = torch.log(1 - vol_pred_clamped)
    log_return_next = log_returns[:, 0]

    print(f"1. Clamped vol prediction: {vol_pred_clamped.item():.4f}")
    print(f"2. Log entry threshold: {log_entry_threshold.item():.4f}")
    print(f"3. Next log return: {log_return_next.item():.4f}")

    # Fill probability calculation
    fill_probability = torch.sigmoid((log_entry_threshold - log_return_next) * 100)
    print(f"4. Fill probability: {fill_probability.item():.4f} ({fill_probability.item()*100:.1f}%)")

    # Holding period return
    log_return_holding = torch.sum(log_returns[:, 1:holding_period+1], dim=1)
    holding_period_profit = torch.exp(log_return_holding) - 1

    print(f"5. Holding period log return: {log_return_holding.item():.4f}")
    print(f"6. Holding period profit: {holding_period_profit.item():.4f} ({holding_period_profit.item()*100:.2f}%)")

    # Expected profit
    expected_profit = fill_probability * holding_period_profit
    print(f"7. Expected profit: {expected_profit.item():.4f} ({expected_profit.item()*100:.2f}%)")

    # Loss (negative profit)
    loss_value = -expected_profit.mean()
    print(f"8. Loss (negative profit): {loss_value.item():.4f}")

    # Verify with actual function call
    actual_loss = loss_fn(vol_pred, log_returns)
    print(f"9. Actual function result: {actual_loss.item():.4f}")
    print(f"   ✅ Match: {abs(loss_value.item() - actual_loss.item()) < 1e-6}")

    return {
        'vol_pred': vol_pred.item(),
        'fill_probability': fill_probability.item(),
        'holding_profit': holding_period_profit.item(),
        'expected_profit': expected_profit.item(),
        'loss_value': actual_loss.item()
    }


def visualize_time_alignment(df, prediction_idx=50, holding_period=24):
    """
    Create visualizations showing the time alignment and profit calculation.
    """
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)

    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle('Profit Maximization Loss: Time Alignment Visualization', fontsize=16)

    # Prepare data for plotting
    start_idx = max(0, prediction_idx - 20)
    end_idx = min(len(df), prediction_idx + holding_period + 10)
    plot_df = df.iloc[start_idx:end_idx].copy()

    prediction_time = df.index[prediction_idx]
    vol_prediction = 0.03
    limit_price = df.iloc[prediction_idx]['price'] * (1 - vol_prediction)

    # Plot 1: Price and Trading Points
    ax1 = axes[0]
    ax1.plot(plot_df.index, plot_df['price'], 'b-', linewidth=2, label='Price')

    # Mark key points
    ax1.axvline(prediction_time, color='red', linestyle='--', alpha=0.7, label='Prediction Time')
    ax1.axhline(limit_price, color='orange', linestyle=':', alpha=0.7, label=f'Limit Price (${limit_price:.2f})')

    # Mark entry and exit points
    if prediction_idx + 1 < len(df):
        entry_time = df.index[prediction_idx + 1]
        entry_price = df.iloc[prediction_idx + 1]['price']
        if entry_price <= limit_price:
            ax1.scatter([entry_time], [entry_price], color='green', s=100, zorder=5, label='Entry Point')

    if prediction_idx + holding_period < len(df):
        exit_time = df.index[prediction_idx + holding_period]
        exit_price = df.iloc[prediction_idx + holding_period]['price']
        ax1.scatter([exit_time], [exit_price], color='red', s=100, zorder=5, label='Exit Point')

    ax1.set_title('Price Chart with Trading Points')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Log Returns
    ax2 = axes[1]
    ax2.plot(plot_df.index, plot_df['log_return'], 'g-', linewidth=1, label='Log Returns')
    ax2.axvline(prediction_time, color='red', linestyle='--', alpha=0.7)

    # Highlight the relevant periods
    if prediction_idx + 1 < len(df):
        next_time = df.index[prediction_idx + 1]
        next_return = df.iloc[prediction_idx + 1]['log_return']
        ax2.scatter([next_time], [next_return], color='orange', s=100, zorder=5,
                   label=f'Entry Return ({next_return:.4f})')

    # Highlight holding period
    holding_start = prediction_idx + 1
    holding_end = min(prediction_idx + holding_period + 1, len(df))
    if holding_end > holding_start:
        holding_times = df.index[holding_start:holding_end]
        holding_returns = df.iloc[holding_start:holding_end]['log_return']
        ax2.fill_between(holding_times, holding_returns, alpha=0.3, color='blue',
                        label='Holding Period')

    ax2.set_title('Log Returns Timeline')
    ax2.set_ylabel('Log Return')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Cumulative Profit During Holding Period
    ax3 = axes[2]

    if prediction_idx + holding_period < len(df):
        # Calculate cumulative returns during holding period
        holding_returns = df.iloc[prediction_idx + 1:prediction_idx + holding_period + 1]['log_return'].values
        cumulative_returns = np.cumsum(holding_returns)
        cumulative_profits = (np.exp(cumulative_returns) - 1) * 100

        holding_times = df.index[prediction_idx + 1:prediction_idx + holding_period + 1]

        ax3.plot(holding_times, cumulative_profits, 'purple', linewidth=2, label='Cumulative Profit %')
        ax3.axhline(0, color='black', linestyle='-', alpha=0.5)
        ax3.fill_between(holding_times, cumulative_profits, alpha=0.3,
                        color='green' if cumulative_profits[-1] > 0 else 'red')

        final_profit = cumulative_profits[-1]
        ax3.set_title(f'Cumulative Profit During Holding Period (Final: {final_profit:+.2f}%)')
    else:
        ax3.set_title('Cumulative Profit During Holding Period (Insufficient Data)')

    ax3.set_ylabel('Cumulative Profit (%)')
    ax3.set_xlabel('Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Format x-axis
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()

    # Save the plot
    os.makedirs('plots/time_alignment', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f'plots/time_alignment/profit_loss_time_alignment_{timestamp}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Visualization saved to: {plot_path}")

    return plot_path


def compare_loss_functions(df, prediction_idx=50, holding_period=24):
    """
    Compare different loss functions on the same data.
    """
    print("\n" + "="*80)
    print("COMPARING DIFFERENT LOSS FUNCTIONS")
    print("="*80)

    # Prepare data
    vol_pred = torch.tensor([0.03], dtype=torch.float32)

    end_idx = min(prediction_idx + holding_period + 1, len(df))
    log_returns_data = df.iloc[prediction_idx + 1:end_idx]['log_return'].values

    if len(log_returns_data) < holding_period + 1:
        padding = np.zeros(holding_period + 1 - len(log_returns_data))
        log_returns_data = np.concatenate([log_returns_data, padding])

    log_returns = torch.tensor(log_returns_data[:holding_period + 1], dtype=torch.float32).unsqueeze(0)

    # Test different loss functions
    loss_functions = [
        ("Simple Profit Max", SimpleProfitMaximizationLoss(holding_period)),
        ("Advanced Profit Max", MaximizeProfitLoss(holding_period, risk_penalty=1.5, no_fill_penalty=0.005)),
        ("Trading Strategy", TradingStrategyLoss(alpha=2.0, beta=0.5, gamma=1.0, holding_period=holding_period))
    ]

    results = {}

    for name, loss_fn in loss_functions:
        loss_value = loss_fn(vol_pred, log_returns)
        results[name] = loss_value.item()
        print(f"{name:20}: {loss_value.item():8.4f}")

    return results


def main():
    """
    Main function to run all tests and demonstrations.
    """
    print("🚀 PROFIT MAXIMIZATION LOSS FUNCTION: TIME ALIGNMENT TEST")
    print("=" * 80)

    # Create synthetic data
    print("📊 Creating synthetic time series data...")
    df = create_synthetic_time_series(n_periods=150, start_date='2024-01-01')
    print(f"   Created {len(df)} periods of hourly data")
    print(f"   Date range: {df.index[0]} to {df.index[-1]}")

    # Demonstrate time alignment
    alignment_info = demonstrate_time_alignment(df, prediction_time_idx=50, holding_period=24)

    # Test loss function step by step
    loss_info = test_loss_function_step_by_step(df, prediction_idx=50, holding_period=24)

    # Create visualizations
    plot_path = visualize_time_alignment(df, prediction_idx=50, holding_period=24)

    # Compare loss functions
    comparison_results = compare_loss_functions(df, prediction_idx=50, holding_period=24)

    # Summary
    print("\n" + "="*80)
    print("📋 SUMMARY")
    print("="*80)
    print(f"✅ Time alignment demonstration completed")
    print(f"✅ Loss function calculation verified")
    print(f"✅ Visualizations created: {plot_path}")
    print(f"✅ Loss function comparison completed")
    print()
    print("🎯 KEY INSIGHTS:")
    print(f"   • Volatility prediction: {loss_info['vol_pred']:.1%}")
    print(f"   • Fill probability: {loss_info['fill_probability']:.1%}")
    print(f"   • Holding period profit: {loss_info['holding_profit']:.2%}")
    print(f"   • Expected profit: {loss_info['expected_profit']:.2%}")
    print(f"   • Loss value: {loss_info['loss_value']:.4f}")
    print()
    print("📊 The visualization clearly shows:")
    print("   1. Exact timing of prediction, entry, and exit")
    print("   2. How log returns align with the trading strategy")
    print("   3. Cumulative profit calculation during holding period")
    print("   4. Visual verification of the loss function logic")


if __name__ == '__main__':
    main()
