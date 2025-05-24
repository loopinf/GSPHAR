#!/usr/bin/env python
"""
Test the impact of trading fees on optimal volatility strategy.

Compare strategies with and without fees to see how fees change the optimal approach.
"""

import torch
import numpy as np
import pandas as pd
import os
import sys

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.ohlcv_trading_dataset import load_ohlcv_trading_data
from src.ohlcv_trading_loss import OHLCVLongStrategyLoss, OHLCVSharpeRatioLoss


class SmallDataSubset(torch.utils.data.Dataset):
    """Same subset class as used in training."""
    
    def __init__(self, original_dataset, subset_size=1000):
        self.original_dataset = original_dataset
        self.subset_size = min(subset_size, len(original_dataset))
        
        total_size = len(original_dataset)
        step = max(1, total_size // subset_size)
        self.indices = list(range(0, total_size, step))[:subset_size]
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        return self.original_dataset[original_idx]


def analyze_fees_impact():
    """
    Analyze how trading fees affect the optimal volatility strategy.
    """
    print("🎯 TRADING FEES IMPACT ANALYSIS")
    print("=" * 60)
    
    # Load dataset
    volatility_file = "data/crypto_rv1h_38_20200822_20250116.csv"
    full_dataset, metadata = load_ohlcv_trading_data(
        volatility_file=volatility_file,
        lags=[1, 4, 24],
        holding_period=4,
        debug=False
    )
    
    # Create small subset for analysis
    subset_size = 200
    small_dataset = SmallDataSubset(full_dataset, subset_size=subset_size)
    
    print(f"Analyzing {len(small_dataset)} samples...")
    
    # Collect OHLCV data for analysis
    all_current_prices = []
    all_next_lows = []
    all_exit_prices = []
    
    for i in range(len(small_dataset)):
        sample = small_dataset[i]
        ohlcv_data = sample['ohlcv_data']
        
        # Average across assets for this sample
        current_prices = ohlcv_data[:, 0, 3]  # All assets, T+0, close
        next_lows = ohlcv_data[:, 1, 2]       # All assets, T+1, low
        exit_prices = ohlcv_data[:, 4, 3]     # All assets, T+4, close
        
        all_current_prices.extend(current_prices.tolist())
        all_next_lows.extend(next_lows.tolist())
        all_exit_prices.extend(exit_prices.tolist())
    
    # Convert to numpy for analysis
    all_current_prices = np.array(all_current_prices)
    all_next_lows = np.array(all_next_lows)
    all_exit_prices = np.array(all_exit_prices)
    
    print(f"Total trades analyzed: {len(all_current_prices)}")
    
    # Test different fee scenarios
    fee_scenarios = [
        ("No Fees", 0.0),
        ("Binance Futures (0.04%)", 0.0004),
        ("High Fees (0.1%)", 0.001),
        ("Very High Fees (0.2%)", 0.002)
    ]
    
    print(f"\n📊 OPTIMAL STRATEGY BY FEE LEVEL")
    print("=" * 60)
    
    results = {}
    
    for fee_name, total_fee_rate in fee_scenarios:
        print(f"\n{fee_name} (Total: {total_fee_rate*100:.2f}%):")
        print("-" * 40)
        
        # Test different volatility predictions
        vol_predictions = np.arange(0, 0.08, 0.002)  # 0% to 8% in 0.2% steps
        expected_profits = []
        fill_rates = []
        profitable_trades = []
        
        for vol_pred in vol_predictions:
            profits = []
            fills = []
            profitable_count = 0
            
            for i in range(len(all_current_prices)):
                current_price = all_current_prices[i]
                next_low = all_next_lows[i]
                exit_price = all_exit_prices[i]
                
                limit_price = current_price * (1 - vol_pred)
                order_fills = next_low <= limit_price
                
                if order_fills:
                    # Calculate profit with fees
                    gross_profit = (exit_price - limit_price) / limit_price
                    net_profit = gross_profit - total_fee_rate
                    
                    profits.append(net_profit)
                    fills.append(1)
                    
                    if net_profit > 0:
                        profitable_count += 1
                else:
                    profits.append(0)
                    fills.append(0)
            
            expected_profit = np.mean(profits)
            fill_rate = np.mean(fills)
            profitable_rate = profitable_count / len(profits) if len(profits) > 0 else 0
            
            expected_profits.append(expected_profit)
            fill_rates.append(fill_rate)
            profitable_trades.append(profitable_rate)
        
        # Find optimal strategy
        best_idx = np.argmax(expected_profits)
        best_vol_pred = vol_predictions[best_idx]
        best_profit = expected_profits[best_idx]
        best_fill_rate = fill_rates[best_idx]
        best_profitable_rate = profitable_trades[best_idx]
        
        # Store results
        results[fee_name] = {
            'optimal_vol_pred': best_vol_pred,
            'expected_profit': best_profit,
            'fill_rate': best_fill_rate,
            'profitable_rate': best_profitable_rate,
            'total_fee_rate': total_fee_rate
        }
        
        print(f"  Optimal volatility prediction: {best_vol_pred*100:.1f}%")
        print(f"  Expected profit per trade: {best_profit*100:+.3f}%")
        print(f"  Fill rate: {best_fill_rate*100:.1f}%")
        print(f"  Profitable trades: {best_profitable_rate*100:.1f}%")
        
        # Show some other interesting points
        zero_vol_profit = expected_profits[0]
        zero_vol_fill_rate = fill_rates[0]
        zero_vol_profitable = profitable_trades[0]
        
        print(f"  0% volatility strategy:")
        print(f"    Expected profit: {zero_vol_profit*100:+.3f}%")
        print(f"    Fill rate: {zero_vol_fill_rate*100:.1f}%")
        print(f"    Profitable trades: {zero_vol_profitable*100:.1f}%")
    
    # Comparison table
    print(f"\n📋 COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Fee Scenario':<25} {'Optimal Vol%':<12} {'Expected Profit%':<16} {'Fill Rate%':<12} {'Profitable%':<12}")
    print("-" * 80)
    
    for fee_name, result in results.items():
        print(f"{fee_name:<25} {result['optimal_vol_pred']*100:>10.1f}% "
              f"{result['expected_profit']*100:>14.3f}% "
              f"{result['fill_rate']*100:>10.1f}% "
              f"{result['profitable_rate']*100:>10.1f}%")
    
    # Analysis
    print(f"\n💡 KEY INSIGHTS")
    print("=" * 40)
    
    no_fees_vol = results["No Fees"]["optimal_vol_pred"]
    binance_fees_vol = results["Binance Futures (0.04%)"]["optimal_vol_pred"]
    
    print(f"1. Fee Impact on Strategy:")
    print(f"   No fees optimal: {no_fees_vol*100:.1f}%")
    print(f"   Binance fees optimal: {binance_fees_vol*100:.1f}%")
    print(f"   Difference: {(binance_fees_vol - no_fees_vol)*100:+.1f}%")
    
    no_fees_profit = results["No Fees"]["expected_profit"]
    binance_fees_profit = results["Binance Futures (0.04%)"]["expected_profit"]
    
    print(f"\n2. Profit Impact:")
    print(f"   No fees profit: {no_fees_profit*100:+.3f}%")
    print(f"   Binance fees profit: {binance_fees_profit*100:+.3f}%")
    print(f"   Profit reduction: {(binance_fees_profit - no_fees_profit)*100:.3f}%")
    
    # Check if 0% volatility is still optimal with fees
    zero_vol_optimal = []
    for fee_name, result in results.items():
        if result['optimal_vol_pred'] == 0.0:
            zero_vol_optimal.append(fee_name)
    
    print(f"\n3. Zero Volatility Strategy:")
    if zero_vol_optimal:
        print(f"   Still optimal with: {', '.join(zero_vol_optimal)}")
    else:
        print(f"   No longer optimal with any fee level tested")
    
    print(f"\n4. Minimum Profitable Trade Size:")
    for fee_name, result in results.items():
        min_profit_needed = result['total_fee_rate']
        print(f"   {fee_name}: Need >{min_profit_needed*100:.2f}% gross profit to break even")


def test_model_with_fees():
    """
    Test how the model behaves with different fee levels.
    """
    print(f"\n🤖 MODEL BEHAVIOR WITH DIFFERENT FEES")
    print("=" * 60)
    
    # Create simple test data
    batch_size, n_assets, holding_period = 1, 5, 4
    time_periods = holding_period + 1
    
    # Create OHLCV data with realistic price movements
    ohlcv_data = torch.zeros(batch_size, n_assets, time_periods, 5)
    
    base_prices = [100.0, 200.0, 50.0, 300.0, 150.0]
    
    for a in range(n_assets):
        for t in range(time_periods):
            base_price = base_prices[a] * (1 + 0.005 * t)  # Small uptrend
            
            # Create realistic OHLCV
            open_price = base_price
            high_price = base_price * 1.015  # 1.5% high
            low_price = base_price * 0.98    # 2% drop
            close_price = base_price * 1.008  # 0.8% gain
            volume = 1000.0
            
            ohlcv_data[0, a, t] = torch.tensor([open_price, high_price, low_price, close_price, volume])
    
    # Test different volatility predictions
    vol_predictions = [0.0, 0.01, 0.015, 0.02, 0.03]  # 0%, 1%, 1.5%, 2%, 3%
    
    # Test different fee levels
    fee_levels = [0.0, 0.0002, 0.0004, 0.001]  # 0%, 0.02%, 0.04%, 0.1%
    
    print(f"Testing {len(vol_predictions)} volatility predictions with {len(fee_levels)} fee levels...")
    
    for fee_rate in fee_levels:
        print(f"\nFee Level: {fee_rate*100:.2f}% per trade ({fee_rate*2*100:.2f}% total)")
        print("-" * 50)
        
        loss_fn = OHLCVLongStrategyLoss(holding_period=holding_period, trading_fee=fee_rate)
        
        for vol_pred_val in vol_predictions:
            vol_pred = torch.full((batch_size, n_assets, 1), vol_pred_val, dtype=torch.float32)
            
            # Calculate loss and metrics
            loss = loss_fn(vol_pred, ohlcv_data)
            metrics = loss_fn.calculate_metrics(vol_pred, ohlcv_data)
            
            print(f"  Vol {vol_pred_val*100:4.1f}%: "
                  f"Loss={loss.item():+8.6f}, "
                  f"Fill={metrics['fill_rate']*100:5.1f}%, "
                  f"Profit={metrics['avg_profit_when_filled']*100:+6.3f}%")


if __name__ == "__main__":
    analyze_fees_impact()
    test_model_with_fees()
