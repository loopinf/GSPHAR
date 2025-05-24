#!/usr/bin/env python
"""
Test to demonstrate the CORRECT profit calculation for the trading strategy.

This shows how the profit should be calculated relative to the entry price (limit price),
not using the log returns from the original time series.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.trading_loss import SimpleProfitMaximizationLoss


def demonstrate_correct_profit_calculation():
    """
    Demonstrate the correct way to calculate profit for the trading strategy.
    """
    print("🎯 CORRECT PROFIT CALCULATION DEMONSTRATION")
    print("="*80)

    # Create a simple scenario
    dates = pd.date_range('2024-01-01', periods=30, freq='h')

    # Simple price scenario
    prices = [100.0]  # Start at $100

    # Add some price movements - design to show the issue
    price_changes = [0.02, -0.08, -0.05, 0.03, 0.02, 0.04, 0.02, -0.01, 0.03, 0.01,
                    0.02, 0.01, -0.01, 0.02, 0.01, 0.03, -0.02, 0.01, 0.02, 0.01,
                    0.01, 0.02, -0.01, 0.01, 0.02, 0.01, 0.01, 0.01, 0.01]

    for change in price_changes:
        prices.append(prices[-1] * (1 + change))

    # Calculate log returns from the price series
    pct_changes = np.diff(prices) / np.array(prices[:-1])
    log_returns = np.log(1 + pct_changes)

    df = pd.DataFrame({
        'datetime': dates[:len(prices)],
        'price': prices,
        'pct_change': np.concatenate([[0], pct_changes]),
        'log_return': np.concatenate([[0], log_returns])
    })
    df.set_index('datetime', inplace=True)

    # Trading scenario
    prediction_idx = 2  # Change to index 2 where we have the big drop
    vol_prediction = 0.04  # 4% volatility prediction (smaller than the 5% drop)
    holding_period = 10

    prediction_time = df.index[prediction_idx]
    current_price = df.iloc[prediction_idx]['price']
    limit_price = current_price * (1 - vol_prediction)

    print(f"📅 PREDICTION TIME: {prediction_time}")
    print(f"💰 CURRENT PRICE: ${current_price:.2f}")
    print(f"📊 PREDICTED VOLATILITY: {vol_prediction:.1%}")
    print(f"🎯 LIMIT ORDER PRICE: ${limit_price:.2f}")
    print()

    # Check order fill
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
        print("📊 PROFIT CALCULATION COMPARISON:")
        print("-" * 60)

        # Method 1: INCORRECT - Using log returns from data
        print("❌ INCORRECT METHOD (current loss function):")
        log_returns_data = df.iloc[prediction_idx + 1:prediction_idx + 1 + holding_period]['log_return'].values
        cumulative_log_return_wrong = np.sum(log_returns_data)
        profit_wrong = (np.exp(cumulative_log_return_wrong) - 1) * 100
        print(f"   Using log returns from data: {log_returns_data[:5]}")
        print(f"   Cumulative log return: {cumulative_log_return_wrong:.4f}")
        print(f"   Calculated profit: {profit_wrong:.2f}%")
        print()

        # Method 2: CORRECT - Calculate from entry price
        print("✅ CORRECT METHOD:")
        entry_price = limit_price
        exit_price = df.iloc[prediction_idx + holding_period]['price']
        profit_correct = ((exit_price - entry_price) / entry_price) * 100

        # Also calculate the correct log return
        correct_log_return = np.log(exit_price / entry_price)
        profit_from_log = (np.exp(correct_log_return) - 1) * 100

        print(f"   Entry price: ${entry_price:.2f}")
        print(f"   Exit price: ${exit_price:.2f}")
        print(f"   Direct profit calculation: {profit_correct:.2f}%")
        print(f"   Correct log return: {correct_log_return:.4f}")
        print(f"   Profit from correct log return: {profit_from_log:.2f}%")
        print(f"   ✅ Methods match: {abs(profit_correct - profit_from_log) < 0.01}")
        print()

        print("🔍 WHY THE DIFFERENCE:")
        print("-" * 60)
        print("❌ Wrong approach:")
        print("   - Uses log returns between consecutive periods in original data")
        print("   - These returns don't account for our actual entry price")
        print("   - Entry price (limit) ≠ market price at entry time")
        print()
        print("✅ Correct approach:")
        print("   - Calculate return from actual entry price (limit price)")
        print("   - Entry price is the discounted price we pay")
        print("   - This is the actual profit we would realize")
        print()

        # Show the price evolution
        print("📈 PRICE EVOLUTION DURING HOLDING:")
        print("-" * 60)
        print(f"{'Time':<20} {'Price':<10} {'Profit from Entry':<15}")
        print("-" * 60)

        for i in range(holding_period + 1):
            if prediction_idx + i < len(df):
                time = df.index[prediction_idx + i]
                price = df.iloc[prediction_idx + i]['price']

                if i == 0:
                    print(f"{time} ${price:6.2f}    (Prediction time)")
                elif i == 1:
                    profit_at_entry = ((price - entry_price) / entry_price) * 100
                    print(f"{time} ${price:6.2f}    {profit_at_entry:+6.2f}% (Entry)")
                else:
                    profit_current = ((price - entry_price) / entry_price) * 100
                    print(f"{time} ${price:6.2f}    {profit_current:+6.2f}%")

    return df, prediction_idx, vol_prediction, holding_period


def create_corrected_loss_function():
    """
    Create a corrected version of the loss function that properly calculates profit.
    """
    print("\n" + "="*80)
    print("CORRECTED LOSS FUNCTION IMPLEMENTATION")
    print("="*80)

    class CorrectedProfitMaximizationLoss(torch.nn.Module):
        """
        Corrected version that properly calculates profit from entry price.
        """

        def __init__(self, holding_period=24):
            super().__init__()
            self.holding_period = holding_period

        def forward(self, vol_pred, prices, prediction_idx):
            """
            Calculate loss using actual prices and entry price.

            Args:
                vol_pred: Volatility prediction tensor [batch_size]
                prices: Price tensor [batch_size, sequence_length]
                prediction_idx: Index where prediction is made
            """
            batch_size = vol_pred.shape[0]

            # Clamp volatility prediction
            vol_pred = torch.clamp(vol_pred, 0.001, 0.5)

            # Calculate entry price (limit price)
            current_price = prices[:, prediction_idx]
            entry_price = current_price * (1 - vol_pred)

            # Check if order fills
            next_price = prices[:, prediction_idx + 1]
            order_fills = (next_price <= entry_price).float()

            # Use smooth approximation for gradients
            price_ratio = next_price / entry_price
            fill_probability = torch.sigmoid((1.0 - price_ratio) * 100)

            # Calculate holding period profit
            exit_price = prices[:, prediction_idx + self.holding_period]
            holding_profit = (exit_price - entry_price) / entry_price

            # Expected profit
            expected_profit = fill_probability * holding_profit

            return -expected_profit.mean()

    print("✅ Corrected loss function created!")
    print("Key differences:")
    print("1. Uses actual prices instead of log returns")
    print("2. Calculates profit from entry price (limit price)")
    print("3. Properly accounts for the discount in entry price")
    print("4. Maintains gradient flow with smooth approximations")

    return CorrectedProfitMaximizationLoss


def test_both_approaches(df, prediction_idx, vol_prediction, holding_period):
    """
    Test both the original and corrected approaches side by side.
    """
    print("\n" + "="*80)
    print("COMPARING ORIGINAL VS CORRECTED APPROACHES")
    print("="*80)

    # Prepare data for original approach
    vol_pred = torch.tensor([vol_prediction], dtype=torch.float32)

    # Get log returns for original approach
    end_idx = min(prediction_idx + holding_period + 1, len(df))
    log_returns_data = df.iloc[prediction_idx + 1:end_idx]['log_return'].values

    if len(log_returns_data) < holding_period + 1:
        padding = np.zeros(holding_period + 1 - len(log_returns_data))
        log_returns_data = np.concatenate([log_returns_data, padding])

    log_returns = torch.tensor(log_returns_data[:holding_period + 1], dtype=torch.float32).unsqueeze(0)

    # Test original approach
    original_loss_fn = SimpleProfitMaximizationLoss(holding_period=holding_period)
    original_loss = original_loss_fn(vol_pred, log_returns)

    print(f"📊 ORIGINAL APPROACH:")
    print(f"   Loss value: {original_loss.item():.4f}")
    print(f"   Expected profit: {-original_loss.item():.4f} ({-original_loss.item()*100:.2f}%)")
    print()

    # Test corrected approach (manual calculation)
    current_price = df.iloc[prediction_idx]['price']
    entry_price = current_price * (1 - vol_prediction)
    next_price = df.iloc[prediction_idx + 1]['price']
    exit_price = df.iloc[prediction_idx + holding_period]['price']

    # Check if order fills
    order_fills = next_price <= entry_price
    fill_probability = 1.0 if order_fills else 0.0  # Simplified for demonstration

    # Calculate actual profit
    holding_profit = (exit_price - entry_price) / entry_price
    expected_profit = fill_probability * holding_profit
    corrected_loss = -expected_profit

    print(f"📊 CORRECTED APPROACH:")
    print(f"   Entry price: ${entry_price:.2f}")
    print(f"   Exit price: ${exit_price:.2f}")
    print(f"   Order fills: {order_fills}")
    print(f"   Holding profit: {holding_profit:.4f} ({holding_profit*100:.2f}%)")
    print(f"   Expected profit: {expected_profit:.4f} ({expected_profit*100:.2f}%)")
    print(f"   Loss value: {corrected_loss:.4f}")
    print()

    print(f"🔍 DIFFERENCE:")
    print(f"   Original expected profit: {-original_loss.item()*100:.2f}%")
    print(f"   Corrected expected profit: {expected_profit*100:.2f}%")
    print(f"   Difference: {(expected_profit + original_loss.item())*100:.2f} percentage points")

    return {
        'original_loss': original_loss.item(),
        'corrected_loss': corrected_loss,
        'original_profit': -original_loss.item(),
        'corrected_profit': expected_profit
    }


def main():
    """
    Main function to demonstrate the correct profit calculation.
    """
    print("🎯 CORRECT PROFIT CALCULATION FOR TRADING STRATEGY")
    print("="*80)

    # Demonstrate the issue
    df, prediction_idx, vol_prediction, holding_period = demonstrate_correct_profit_calculation()

    # Create corrected loss function
    corrected_loss_fn = create_corrected_loss_function()

    # Compare approaches
    comparison = test_both_approaches(df, prediction_idx, vol_prediction, holding_period)

    print("\n" + "="*80)
    print("📋 SUMMARY")
    print("="*80)
    print("🚨 ISSUE IDENTIFIED:")
    print("   The current loss function uses log returns from the original time series,")
    print("   but these don't account for the actual entry price (limit price).")
    print()
    print("✅ SOLUTION:")
    print("   Calculate profit directly from entry price to exit price.")
    print("   This gives the actual profit the trading strategy would realize.")
    print()
    print("📊 IMPACT:")
    print(f"   Original approach profit: {comparison['original_profit']*100:.2f}%")
    print(f"   Corrected approach profit: {comparison['corrected_profit']*100:.2f}%")
    print(f"   This affects model training and strategy evaluation!")


if __name__ == '__main__':
    main()
