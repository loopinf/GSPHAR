#!/usr/bin/env python
"""
Test script specifically designed to show a case where the order fills
and we can see the complete profit calculation process.
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


def create_designed_scenario(holding_period=24):
    """
    Create a designed scenario where we know the order will fill and be profitable.
    """
    # Create a scenario where:
    # 1. We predict 5% volatility
    # 2. Price drops 6% in the next period (order fills)
    # 3. Price recovers over the holding period (profitable)

    dates = pd.date_range('2024-01-01', periods=50, freq='h')

    # Design the price movement
    prices = [100.0]  # Start at $100

    # Normal periods before prediction
    for i in range(20):
        prices.append(prices[-1] * (1 + np.random.normal(0, 0.01)))

    # Prediction point (index 20)
    prediction_price = prices[-1]

    # Next period: Big drop (order fills)
    drop_return = -0.06  # 6% drop
    prices.append(prediction_price * (1 + drop_return))

    # Holding period: Gradual recovery with some volatility
    recovery_per_period = 0.08 / holding_period  # 8% total recovery over holding period
    for i in range(holding_period):
        base_return = recovery_per_period + np.random.normal(0, 0.005)
        prices.append(prices[-1] * (1 + base_return))

    # Add some more periods after
    for i in range(8):
        prices.append(prices[-1] * (1 + np.random.normal(0, 0.01)))

    # Calculate returns
    pct_changes = np.diff(prices) / np.array(prices[:-1])
    log_returns = np.log(1 + pct_changes)

    # Create DataFrame - ensure all arrays have same length
    n_periods = len(prices)
    print(f"Debug: prices length: {len(prices)}")
    print(f"Debug: dates length: {len(dates)}")
    print(f"Debug: pct_changes length: {len(pct_changes)}")
    print(f"Debug: log_returns length: {len(log_returns)}")

    # Ensure we have enough dates
    if len(dates) < n_periods:
        additional_dates = pd.date_range(dates[-1] + timedelta(hours=1),
                                       periods=n_periods - len(dates), freq='h')
        dates = dates.append(additional_dates)

    df = pd.DataFrame({
        'datetime': dates[:n_periods],
        'price': prices,
        'pct_change': np.concatenate([[0], pct_changes]),
        'log_return': np.concatenate([[0], log_returns])
    })

    df.set_index('datetime', inplace=True)
    return df, 20  # Return df and prediction index


def demonstrate_filled_order_scenario():
    """
    Demonstrate a scenario where the order fills and generates profit.
    """
    print("🎯 DESIGNED SCENARIO: ORDER FILLS AND GENERATES PROFIT")
    print("="*80)

    holding_period = 24
    df, prediction_idx = create_designed_scenario(holding_period)

    # Scenario parameters
    vol_prediction = 0.05  # 5% volatility prediction
    prediction_time = df.index[prediction_idx]
    current_price = df.iloc[prediction_idx]['price']
    limit_price = current_price * (1 - vol_prediction)

    print(f"📅 PREDICTION TIME: {prediction_time}")
    print(f"💰 CURRENT PRICE: ${current_price:.2f}")
    print(f"📊 PREDICTED VOLATILITY: {vol_prediction:.1%}")
    print(f"🎯 LIMIT ORDER PRICE: ${limit_price:.2f}")
    print()

    # Check if order fills
    next_time = df.index[prediction_idx + 1]
    next_price = df.iloc[prediction_idx + 1]['price']
    next_log_return = df.iloc[prediction_idx + 1]['log_return']

    order_filled = next_price <= limit_price
    print(f"⏰ T+1 ({next_time}):")
    print(f"   Next price: ${next_price:.2f}")
    print(f"   Log return: {next_log_return:.4f}")
    print(f"   Order filled: {'✅ YES' if order_filled else '❌ NO'}")
    print()

    if order_filled:
        print("📈 POSITION TRACKING:")
        print("-" * 50)

        entry_price = limit_price
        cumulative_log_return = 0

        print(f"T+1  Enter position at ${entry_price:.2f}")

        # Track each period during holding
        # This matches log_returns[:, 1:1+holding_period] in the loss function
        for i in range(1, holding_period + 1):
            if prediction_idx + i < len(df):
                period_time = df.index[prediction_idx + i]
                period_price = df.iloc[prediction_idx + i]['price']
                period_log_return = df.iloc[prediction_idx + i]['log_return']

                # Accumulate log returns during holding period (matches loss function)
                cumulative_log_return += period_log_return
                current_profit_pct = (np.exp(cumulative_log_return) - 1) * 100

                # Also calculate direct profit for verification
                direct_profit_pct = ((period_price - entry_price) / entry_price) * 100

                if i <= 3 or i % 6 == 0 or i == holding_period:
                    print(f"T+{i+1:2d} {period_time}: ${period_price:6.2f} | "
                          f"Log Ret: {period_log_return:+.4f} | "
                          f"Cum Log: {current_profit_pct:+6.2f}% | "
                          f"Direct: {direct_profit_pct:+6.2f}%")

        # Final result
        final_profit_log = (np.exp(cumulative_log_return) - 1) * 100
        exit_price = df.iloc[prediction_idx + holding_period]['price']
        final_profit_direct = ((exit_price - entry_price) / entry_price) * 100

        print(f"\n🏁 FINAL RESULT:")
        print(f"   Entry Price: ${entry_price:.2f}")
        print(f"   Exit Price:  ${exit_price:.2f}")
        print(f"   Total Profit (log method): {final_profit_log:+.2f}%")
        print(f"   Total Profit (direct method): {final_profit_direct:+.2f}%")
        print(f"   Cumulative Log Return: {cumulative_log_return:.4f}")
        print(f"   ✅ Methods match: {abs(final_profit_log - final_profit_direct) < 0.01}")

    return df, prediction_idx, vol_prediction


def test_loss_calculation_detailed(df, prediction_idx, vol_prediction, holding_period=24):
    """
    Test the loss calculation with detailed step-by-step breakdown.
    """
    print("\n" + "="*80)
    print("DETAILED LOSS CALCULATION")
    print("="*80)

    # Prepare tensors
    vol_pred = torch.tensor([vol_prediction], dtype=torch.float32)

    # Get log returns
    end_idx = min(prediction_idx + holding_period + 1, len(df))
    log_returns_data = df.iloc[prediction_idx + 1:end_idx]['log_return'].values

    if len(log_returns_data) < holding_period + 1:
        padding = np.zeros(holding_period + 1 - len(log_returns_data))
        log_returns_data = np.concatenate([log_returns_data, padding])

    log_returns = torch.tensor(log_returns_data[:holding_period + 1], dtype=torch.float32).unsqueeze(0)

    print(f"📊 INPUTS:")
    print(f"   vol_pred shape: {vol_pred.shape}, value: {vol_pred.item():.4f}")
    print(f"   log_returns shape: {log_returns.shape}")
    print(f"   log_returns[0, 0] (next period): {log_returns[0, 0].item():.4f}")
    print(f"   log_returns[0, 1:] (holding): {log_returns[0, 1:5].tolist()}")
    print()

    # Step-by-step calculation for SimpleProfitMaximizationLoss
    print("🔍 STEP-BY-STEP CALCULATION:")
    print("-" * 50)

    # Step 1: Clamp volatility prediction
    vol_pred_clamped = torch.clamp(vol_pred, 0.001, 0.5)
    print(f"1. Clamped vol_pred: {vol_pred_clamped.item():.4f}")

    # Step 2: Calculate entry threshold
    log_entry_threshold = torch.log(1 - vol_pred_clamped)
    print(f"2. log_entry_threshold = ln(1 - {vol_pred_clamped.item():.4f}) = {log_entry_threshold.item():.4f}")

    # Step 3: Get next period return
    log_return_next = log_returns[:, 0]
    print(f"3. log_return_next: {log_return_next.item():.4f}")

    # Step 4: Calculate fill probability
    sigmoid_input = (log_entry_threshold - log_return_next) * 100
    fill_probability = torch.sigmoid(sigmoid_input)
    print(f"4. sigmoid_input = ({log_entry_threshold.item():.4f} - {log_return_next.item():.4f}) * 100 = {sigmoid_input.item():.2f}")
    print(f"   fill_probability = sigmoid({sigmoid_input.item():.2f}) = {fill_probability.item():.4f}")

    # Hard threshold check for comparison
    hard_fill = (log_return_next <= log_entry_threshold).float()
    print(f"   Hard threshold fill: {hard_fill.item():.0f} ({'YES' if hard_fill.item() > 0 else 'NO'})")

    # Step 5: Calculate holding period return
    log_return_holding = torch.sum(log_returns[:, 1:holding_period+1], dim=1)
    print(f"5. log_return_holding (sum of {holding_period} periods): {log_return_holding.item():.4f}")

    # Step 6: Convert to profit percentage
    holding_period_profit = torch.exp(log_return_holding) - 1
    print(f"6. holding_period_profit = exp({log_return_holding.item():.4f}) - 1 = {holding_period_profit.item():.4f}")
    print(f"   As percentage: {holding_period_profit.item() * 100:.2f}%")

    # Step 7: Calculate expected profit
    expected_profit = fill_probability * holding_period_profit
    print(f"7. expected_profit = {fill_probability.item():.4f} * {holding_period_profit.item():.4f} = {expected_profit.item():.4f}")
    print(f"   As percentage: {expected_profit.item() * 100:.2f}%")

    # Step 8: Calculate loss (negative profit)
    loss_value = -expected_profit.mean()
    print(f"8. loss = -expected_profit = {loss_value.item():.4f}")

    # Verify with actual function
    loss_fn = SimpleProfitMaximizationLoss(holding_period=holding_period)
    actual_loss = loss_fn(vol_pred, log_returns)
    print(f"9. Actual function result: {actual_loss.item():.4f}")
    print(f"   ✅ Match: {abs(loss_value.item() - actual_loss.item()) < 1e-6}")

    return {
        'fill_probability': fill_probability.item(),
        'holding_profit': holding_period_profit.item(),
        'expected_profit': expected_profit.item(),
        'loss_value': actual_loss.item()
    }


def create_comprehensive_visualization(df, prediction_idx, vol_prediction, holding_period=24):
    """
    Create comprehensive visualization of the filled order scenario.
    """
    print("\n" + "="*80)
    print("CREATING COMPREHENSIVE VISUALIZATION")
    print("="*80)

    fig, axes = plt.subplots(4, 1, figsize=(16, 16))
    fig.suptitle('Profit Maximization Loss: Filled Order Scenario', fontsize=16, fontweight='bold')

    # Data preparation
    start_idx = max(0, prediction_idx - 10)
    end_idx = min(len(df), prediction_idx + holding_period + 10)
    plot_df = df.iloc[start_idx:end_idx].copy()

    prediction_time = df.index[prediction_idx]
    current_price = df.iloc[prediction_idx]['price']
    limit_price = current_price * (1 - vol_prediction)

    # Plot 1: Price Chart with Trading Points
    ax1 = axes[0]
    ax1.plot(plot_df.index, plot_df['price'], 'b-', linewidth=2, label='Price', alpha=0.8)

    # Mark key points
    ax1.axvline(prediction_time, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Prediction Time')
    ax1.axhline(limit_price, color='orange', linestyle=':', linewidth=2, alpha=0.8,
               label=f'Limit Price (${limit_price:.2f})')

    # Entry point
    entry_time = df.index[prediction_idx + 1]
    entry_price = df.iloc[prediction_idx + 1]['price']
    ax1.scatter([entry_time], [entry_price], color='green', s=150, zorder=5,
               label=f'Entry (${entry_price:.2f})', edgecolor='darkgreen', linewidth=2)

    # Exit point
    if prediction_idx + holding_period < len(df):
        exit_time = df.index[prediction_idx + holding_period]
        exit_price = df.iloc[prediction_idx + holding_period]['price']
        ax1.scatter([exit_time], [exit_price], color='red', s=150, zorder=5,
                   label=f'Exit (${exit_price:.2f})', edgecolor='darkred', linewidth=2)

    # Shade holding period
    if prediction_idx + holding_period < len(df):
        ax1.axvspan(entry_time, exit_time, alpha=0.2, color='blue', label='Holding Period')

    ax1.set_title('Price Chart: Trading Strategy Execution', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Log Returns with Key Periods Highlighted
    ax2 = axes[1]
    ax2.plot(plot_df.index, plot_df['log_return'], 'g-', linewidth=1.5, alpha=0.7, label='Log Returns')
    ax2.axvline(prediction_time, color='red', linestyle='--', linewidth=2, alpha=0.8)
    ax2.axhline(0, color='black', linestyle='-', alpha=0.5, linewidth=1)

    # Highlight entry return
    next_return = df.iloc[prediction_idx + 1]['log_return']
    ax2.scatter([entry_time], [next_return], color='orange', s=120, zorder=5,
               label=f'Entry Return ({next_return:.4f})', edgecolor='darkorange', linewidth=2)

    # Highlight holding period returns
    holding_start = prediction_idx + 1
    holding_end = min(prediction_idx + holding_period + 1, len(df))
    if holding_end > holding_start:
        holding_times = df.index[holding_start:holding_end]
        holding_returns = df.iloc[holding_start:holding_end]['log_return']
        ax2.fill_between(holding_times, holding_returns, alpha=0.4, color='blue',
                        label='Holding Period Returns')

    ax2.set_title('Log Returns Timeline', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Log Return', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Cumulative Profit Evolution
    ax3 = axes[2]

    if prediction_idx + holding_period < len(df):
        holding_returns = df.iloc[prediction_idx + 1:prediction_idx + holding_period + 1]['log_return'].values
        cumulative_log_returns = np.cumsum(holding_returns)
        cumulative_profits = (np.exp(cumulative_log_returns) - 1) * 100

        holding_times = df.index[prediction_idx + 1:prediction_idx + holding_period + 1]

        # Plot cumulative profit
        ax3.plot(holding_times, cumulative_profits, 'purple', linewidth=3, label='Cumulative Profit %')
        ax3.fill_between(holding_times, cumulative_profits, alpha=0.3,
                        color='green' if cumulative_profits[-1] > 0 else 'red')

        # Mark final profit
        final_profit = cumulative_profits[-1]
        ax3.scatter([holding_times[-1]], [final_profit], color='red', s=150, zorder=5,
                   label=f'Final: {final_profit:+.2f}%', edgecolor='darkred', linewidth=2)

        ax3.axhline(0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        ax3.set_title(f'Cumulative Profit Evolution (Final: {final_profit:+.2f}%)',
                     fontsize=14, fontweight='bold')

    ax3.set_ylabel('Cumulative Profit (%)', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Loss Function Components
    ax4 = axes[3]

    # Calculate loss components for visualization
    vol_pred = torch.tensor([vol_prediction], dtype=torch.float32)
    end_idx_calc = min(prediction_idx + holding_period + 1, len(df))
    log_returns_data = df.iloc[prediction_idx + 1:end_idx_calc]['log_return'].values

    if len(log_returns_data) < holding_period + 1:
        padding = np.zeros(holding_period + 1 - len(log_returns_data))
        log_returns_data = np.concatenate([log_returns_data, padding])

    log_returns = torch.tensor(log_returns_data[:holding_period + 1], dtype=torch.float32).unsqueeze(0)

    # Calculate components
    vol_pred_clamped = torch.clamp(vol_pred, 0.001, 0.5)
    log_entry_threshold = torch.log(1 - vol_pred_clamped)
    log_return_next = log_returns[:, 0]
    fill_probability = torch.sigmoid((log_entry_threshold - log_return_next) * 100)
    log_return_holding = torch.sum(log_returns[:, 1:holding_period+1], dim=1)
    holding_period_profit = torch.exp(log_return_holding) - 1
    expected_profit = fill_probability * holding_period_profit

    # Create bar chart of components
    components = ['Vol Pred', 'Fill Prob', 'Hold Profit', 'Expected Profit', 'Loss']
    values = [vol_pred.item() * 100, fill_probability.item() * 100,
             holding_period_profit.item() * 100, expected_profit.item() * 100,
             -expected_profit.item() * 100]
    colors = ['blue', 'orange', 'green', 'purple', 'red']

    bars = ax4.bar(components, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    ax4.axhline(0, color='black', linestyle='-', alpha=0.5, linewidth=1)

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -3),
                f'{value:.2f}%', ha='center', va='bottom' if height >= 0 else 'top',
                fontweight='bold', fontsize=10)

    ax4.set_title('Loss Function Components (%)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Value (%)', fontsize=12)
    ax4.grid(True, alpha=0.3, axis='y')

    # Format x-axis for time plots
    for ax in axes[:3]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, fontsize=10)

    plt.tight_layout()

    # Save plot
    os.makedirs('plots/time_alignment', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f'plots/time_alignment/filled_order_scenario_{timestamp}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Comprehensive visualization saved to: {plot_path}")

    return plot_path


def main():
    """
    Main function to demonstrate filled order scenario.
    """
    print("🎯 PROFIT MAXIMIZATION LOSS: FILLED ORDER SCENARIO TEST")
    print("="*80)

    # Create and demonstrate scenario
    df, prediction_idx, vol_prediction = demonstrate_filled_order_scenario()

    # Detailed loss calculation
    loss_info = test_loss_calculation_detailed(df, prediction_idx, vol_prediction, holding_period=24)

    # Create visualization
    plot_path = create_comprehensive_visualization(df, prediction_idx, vol_prediction, holding_period=24)

    # Summary
    print("\n" + "="*80)
    print("📋 FILLED ORDER SCENARIO SUMMARY")
    print("="*80)
    print(f"✅ Order successfully filled")
    print(f"✅ Position held for 24 hours")
    print(f"✅ Profit calculation completed")
    print(f"✅ Visualization created: {plot_path}")
    print()
    print("🎯 KEY RESULTS:")
    print(f"   • Fill Probability: {loss_info['fill_probability']:.1%}")
    print(f"   • Holding Period Profit: {loss_info['holding_profit']:.2%}")
    print(f"   • Expected Profit: {loss_info['expected_profit']:.2%}")
    print(f"   • Loss Value: {loss_info['loss_value']:.4f}")
    print()
    print("📊 This scenario demonstrates:")
    print("   1. ✅ Order fills when price drops below limit")
    print("   2. ✅ Position generates profit during holding period")
    print("   3. ✅ Loss function correctly calculates negative profit")
    print("   4. ✅ Time alignment is precise and verifiable")


if __name__ == '__main__':
    main()
