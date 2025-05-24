#!/usr/bin/env python
"""
Analyze the data subset used in training to understand why volatility converged to 0.
"""

import torch
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.ohlcv_trading_dataset import load_ohlcv_trading_data


class SmallDataSubset(torch.utils.data.Dataset):
    """Same subset class as used in training."""
    
    def __init__(self, original_dataset, subset_size=1000):
        self.original_dataset = original_dataset
        self.subset_size = min(subset_size, len(original_dataset))
        
        # Take evenly spaced samples across the dataset
        total_size = len(original_dataset)
        step = max(1, total_size // subset_size)
        self.indices = list(range(0, total_size, step))[:subset_size]
        
        print(f"Created subset: {len(self.indices)} samples from {total_size} total")
        print(f"Step size: {step}")
        print(f"First few indices: {self.indices[:10]}")
        print(f"Last few indices: {self.indices[-10:]}")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        return self.original_dataset[original_idx]


def analyze_subset_data():
    """
    Analyze the actual data used in training.
    """
    print("🔍 ANALYZING TRAINING DATA SUBSET")
    print("=" * 60)
    
    # Load full dataset
    volatility_file = "data/crypto_rv1h_38_20200822_20250116.csv"
    full_dataset, metadata = load_ohlcv_trading_data(
        volatility_file=volatility_file,
        lags=[1, 4, 24],
        holding_period=4,
        debug=False
    )
    
    print(f"Full dataset size: {len(full_dataset)}")
    print(f"Assets: {len(metadata['assets'])}")
    print(f"Date range: {metadata['date_range']}")
    
    # Create same subset as training
    subset_size = 300
    small_dataset = SmallDataSubset(full_dataset, subset_size=subset_size)
    
    # Analyze subset samples
    print(f"\n📊 SUBSET ANALYSIS")
    print("=" * 40)
    
    # Get time indices for subset
    time_indices = []
    for i in range(len(small_dataset)):
        sample_info = full_dataset.get_sample_info(small_dataset.indices[i])
        time_indices.append(sample_info['prediction_time'])
    
    print(f"Subset time range: {min(time_indices)} to {max(time_indices)}")
    print(f"Time span: {max(time_indices) - min(time_indices)}")
    
    # Analyze a few samples in detail
    print(f"\n🎯 DETAILED SAMPLE ANALYSIS")
    print("=" * 40)
    
    sample_analyses = []
    
    for i in range(min(5, len(small_dataset))):
        sample = small_dataset[i]
        sample_info = full_dataset.get_sample_info(small_dataset.indices[i])
        
        # Extract data
        ohlcv_data = sample['ohlcv_data']  # [assets, time_periods, 5]
        vol_targets = sample['vol_targets']  # [assets, 1]
        
        print(f"\nSample {i+1}:")
        print(f"  Time: {sample_info['prediction_time']}")
        print(f"  OHLCV shape: {ohlcv_data.shape}")
        print(f"  Vol targets shape: {vol_targets.shape}")
        
        # Analyze OHLCV data for first few assets
        for asset_idx in range(min(3, ohlcv_data.shape[0])):
            asset_name = metadata['assets'][asset_idx]
            
            # Get OHLCV for this asset: [time_periods, 5]
            asset_ohlcv = ohlcv_data[asset_idx]  # [5, 5] - 5 time periods, 5 OHLC components
            
            current_price = asset_ohlcv[0, 3].item()  # Close at T+0
            next_low = asset_ohlcv[1, 2].item()      # Low at T+1
            exit_price = asset_ohlcv[4, 3].item()    # Close at T+4 (holding period end)
            
            # Calculate what would happen with different volatility predictions
            vol_predictions = [0.0, 0.01, 0.02, 0.05]  # 0%, 1%, 2%, 5%
            
            print(f"    {asset_name}:")
            print(f"      Current price: ${current_price:.2f}")
            print(f"      Next low: ${next_low:.2f}")
            print(f"      Exit price: ${exit_price:.2f}")
            print(f"      Market drop: {((current_price - next_low) / current_price * 100):.2f}%")
            print(f"      4h return: {((exit_price - current_price) / current_price * 100):.2f}%")
            
            print(f"      Volatility prediction analysis:")
            for vol_pred in vol_predictions:
                limit_price = current_price * (1 - vol_pred)
                order_fills = next_low <= limit_price
                
                if order_fills:
                    profit = (exit_price - limit_price) / limit_price
                    print(f"        {vol_pred*100:4.1f}%: Limit=${limit_price:.2f}, FILLS, Profit={profit*100:+5.2f}%")
                else:
                    print(f"        {vol_pred*100:4.1f}%: Limit=${limit_price:.2f}, NO FILL")
        
        # Store analysis
        sample_analyses.append({
            'time': sample_info['prediction_time'],
            'ohlcv_data': ohlcv_data,
            'vol_targets': vol_targets
        })
    
    # Aggregate analysis
    print(f"\n📈 AGGREGATE ANALYSIS")
    print("=" * 40)
    
    all_current_prices = []
    all_next_lows = []
    all_exit_prices = []
    all_market_drops = []
    all_4h_returns = []
    
    for i in range(min(50, len(small_dataset))):  # Analyze first 50 samples
        sample = small_dataset[i]
        ohlcv_data = sample['ohlcv_data']
        
        # Average across assets for this sample
        current_prices = ohlcv_data[:, 0, 3]  # All assets, T+0, close
        next_lows = ohlcv_data[:, 1, 2]       # All assets, T+1, low
        exit_prices = ohlcv_data[:, 4, 3]     # All assets, T+4, close
        
        market_drops = (current_prices - next_lows) / current_prices
        returns_4h = (exit_prices - current_prices) / current_prices
        
        all_current_prices.extend(current_prices.tolist())
        all_next_lows.extend(next_lows.tolist())
        all_exit_prices.extend(exit_prices.tolist())
        all_market_drops.extend(market_drops.tolist())
        all_4h_returns.extend(returns_4h.tolist())
    
    # Convert to numpy for analysis
    all_market_drops = np.array(all_market_drops)
    all_4h_returns = np.array(all_4h_returns)
    
    print(f"Market drops (T+0 to T+1 low):")
    print(f"  Mean: {all_market_drops.mean()*100:.3f}%")
    print(f"  Std:  {all_market_drops.std()*100:.3f}%")
    print(f"  Min:  {all_market_drops.min()*100:.3f}%")
    print(f"  Max:  {all_market_drops.max()*100:.3f}%")
    print(f"  Median: {np.median(all_market_drops)*100:.3f}%")
    
    print(f"\n4-hour returns:")
    print(f"  Mean: {all_4h_returns.mean()*100:.3f}%")
    print(f"  Std:  {all_4h_returns.std()*100:.3f}%")
    print(f"  Min:  {all_4h_returns.min()*100:.3f}%")
    print(f"  Max:  {all_4h_returns.max()*100:.3f}%")
    print(f"  Median: {np.median(all_4h_returns)*100:.3f}%")
    
    # Analyze why 0% volatility might be optimal
    print(f"\n🎯 WHY 0% VOLATILITY MIGHT BE OPTIMAL")
    print("=" * 50)
    
    # Calculate expected profit for different volatility predictions
    vol_predictions = np.arange(0, 0.1, 0.005)  # 0% to 10% in 0.5% steps
    expected_profits = []
    fill_rates = []
    
    for vol_pred in vol_predictions:
        profits = []
        fills = []
        
        for i in range(len(all_current_prices)):
            current_price = all_current_prices[i]
            next_low = all_next_lows[i]
            exit_price = all_exit_prices[i]
            
            limit_price = current_price * (1 - vol_pred)
            order_fills = next_low <= limit_price
            
            if order_fills:
                profit = (exit_price - limit_price) / limit_price
                profits.append(profit)
                fills.append(1)
            else:
                profits.append(0)
                fills.append(0)
        
        expected_profit = np.mean(profits)
        fill_rate = np.mean(fills)
        
        expected_profits.append(expected_profit)
        fill_rates.append(fill_rate)
        
        if vol_pred in [0.0, 0.01, 0.02, 0.05]:
            print(f"Vol pred {vol_pred*100:4.1f}%: Fill rate {fill_rate*100:5.1f}%, Expected profit {expected_profit*100:+6.3f}%")
    
    # Find optimal volatility prediction
    best_idx = np.argmax(expected_profits)
    best_vol_pred = vol_predictions[best_idx]
    best_profit = expected_profits[best_idx]
    best_fill_rate = fill_rates[best_idx]
    
    print(f"\n🏆 OPTIMAL STRATEGY:")
    print(f"  Best volatility prediction: {best_vol_pred*100:.1f}%")
    print(f"  Expected profit: {best_profit*100:+.3f}%")
    print(f"  Fill rate: {best_fill_rate*100:.1f}%")
    
    # Check if 0% is actually optimal
    zero_vol_profit = expected_profits[0]
    zero_vol_fill_rate = fill_rates[0]
    
    print(f"\n📊 0% VOLATILITY PERFORMANCE:")
    print(f"  Expected profit: {zero_vol_profit*100:+.3f}%")
    print(f"  Fill rate: {zero_vol_fill_rate*100:.1f}%")
    print(f"  Is 0% optimal? {'YES' if best_vol_pred == 0.0 else 'NO'}")
    
    if best_vol_pred == 0.0:
        print(f"\n💡 EXPLANATION:")
        print(f"  The model converged to 0% because it IS the optimal strategy!")
        print(f"  In this data subset, buying at current price (0% discount)")
        print(f"  gives better expected returns than trying to 'buy the dip'.")
        print(f"  This suggests the market was generally trending upward")
        print(f"  during the time periods in our subset.")
    else:
        print(f"\n💡 EXPLANATION:")
        print(f"  The model should have learned {best_vol_pred*100:.1f}% volatility,")
        print(f"  but converged to 0% instead. This suggests a training issue.")


if __name__ == "__main__":
    analyze_subset_data()
