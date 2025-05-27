#!/usr/bin/env python
"""
Investigate why the fill rate is so high (98.8%) - this seems unrealistic.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.ohlcv_trading_dataset import load_ohlcv_trading_data
from src.models.flexible_gsphar import FlexibleGSPHAR


def investigate_fill_rate_issue():
    """Investigate why fill rate is unrealistically high."""
    print("🔍 INVESTIGATING HIGH FILL RATE ISSUE")
    print("=" * 60)
    
    # Load the fixed model
    model_path = "models/fixed_stage1_model_20250524_202400.pt"
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    test_indices = checkpoint['test_indices']
    
    # Load volatility data for correlation matrix
    vol_df = pd.read_csv("data/crypto_rv1h_38_20200822_20250116.csv", index_col=0, parse_dates=True)
    symbols = vol_df.columns.tolist()
    corr_sample = vol_df.iloc[-1000:] if len(vol_df) > 1000 else vol_df
    corr_matrix = corr_sample.corr().values
    A = (corr_matrix + corr_matrix.T) / 2
    
    # Create model
    model = FlexibleGSPHAR(
        lags=[1, 4, 24],
        output_dim=1,
        filter_size=38,
        A=A
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load dataset
    dataset, _ = load_ohlcv_trading_data(
        volatility_file="data/crypto_rv1h_38_20200822_20250116.csv",
        lags=[1, 4, 24],
        holding_period=4,
        debug=False
    )
    
    # Analyze first 10 periods in detail
    test_sample_indices = test_indices[:10]
    
    print(f"📊 DETAILED ANALYSIS OF FIRST {len(test_sample_indices)} PERIODS")
    print("=" * 60)
    
    all_analysis = []
    
    with torch.no_grad():
        for period_idx, idx in enumerate(test_sample_indices):
            sample = dataset[idx]
            sample_info = dataset.get_sample_info(idx)
            
            x_lags = [x.unsqueeze(0) for x in sample['x_lags']]
            vol_pred = model(*x_lags)
            vol_pred_np = vol_pred.squeeze().cpu().numpy()
            
            ohlcv_data = sample['ohlcv_data'].numpy()
            timestamp = sample_info['prediction_time']
            
            print(f"\n📅 PERIOD {period_idx + 1}: {timestamp}")
            print("-" * 50)
            
            period_fills = 0
            period_total = 0
            
            # Analyze each symbol in detail
            for asset_idx, symbol in enumerate(symbols[:5]):  # First 5 symbols for detailed view
                # Get prices
                open_price = ohlcv_data[asset_idx, 0, 0]      # T+0 OPEN
                current_low = ohlcv_data[asset_idx, 0, 2]     # T+0 LOW
                current_high = ohlcv_data[asset_idx, 0, 1]    # T+0 HIGH
                current_close = ohlcv_data[asset_idx, 0, 3]   # T+0 CLOSE
                
                # Get volatility prediction
                asset_vol_pred = vol_pred_np[asset_idx]
                
                # Calculate limit order price
                limit_price = open_price * (1 - asset_vol_pred)
                
                # Check if order fills
                order_fills = current_low <= limit_price
                
                # Calculate price movements
                intraday_drop = (open_price - current_low) / open_price * 100
                vol_pred_pct = asset_vol_pred * 100
                
                print(f"  {symbol}:")
                print(f"    Vol Pred: {vol_pred_pct:.2f}%")
                print(f"    Open: ${open_price:.4f}")
                print(f"    Low:  ${current_low:.4f}")
                print(f"    Limit: ${limit_price:.4f}")
                print(f"    Intraday Drop: {intraday_drop:.2f}%")
                print(f"    Order Fills: {'YES' if order_fills else 'NO'}")
                print(f"    Reason: {'Low <= Limit' if order_fills else 'Low > Limit'}")
                
                if order_fills:
                    period_fills += 1
                period_total += 1
                
                # Store for analysis
                all_analysis.append({
                    'period': period_idx + 1,
                    'symbol': symbol,
                    'vol_pred_pct': vol_pred_pct,
                    'open_price': open_price,
                    'current_low': current_low,
                    'limit_price': limit_price,
                    'intraday_drop': intraday_drop,
                    'order_fills': order_fills,
                    'gap_to_fill': (limit_price - current_low) / open_price * 100
                })
            
            # Count all symbols for this period
            all_fills = 0
            all_total = 0
            for asset_idx in range(len(symbols)):
                open_price = ohlcv_data[asset_idx, 0, 0]
                current_low = ohlcv_data[asset_idx, 0, 2]
                asset_vol_pred = vol_pred_np[asset_idx]
                limit_price = open_price * (1 - asset_vol_pred)
                order_fills = current_low <= limit_price
                
                if order_fills:
                    all_fills += 1
                all_total += 1
            
            period_fill_rate = all_fills / all_total
            print(f"\n  📊 PERIOD SUMMARY:")
            print(f"    Fill Rate: {period_fill_rate:.1%} ({all_fills}/{all_total})")
            print(f"    Avg Vol Pred: {np.mean(vol_pred_np)*100:.2f}%")
    
    # Overall analysis
    print(f"\n" + "="*60)
    print("📊 OVERALL FILL RATE ANALYSIS")
    print("="*60)
    
    df = pd.DataFrame(all_analysis)
    
    # Statistics
    overall_fill_rate = df['order_fills'].mean()
    avg_vol_pred = df['vol_pred_pct'].mean()
    avg_intraday_drop = df['intraday_drop'].mean()
    
    print(f"Overall Fill Rate: {overall_fill_rate:.1%}")
    print(f"Average Vol Prediction: {avg_vol_pred:.2f}%")
    print(f"Average Intraday Drop: {avg_intraday_drop:.2f}%")
    
    # Check if vol predictions are too low
    print(f"\n🔍 POTENTIAL ISSUES:")
    
    if avg_vol_pred < 1.0:  # Less than 1%
        print(f"❌ Vol predictions too low: {avg_vol_pred:.2f}%")
        print(f"   This makes limit orders very close to open price")
        print(f"   Even small intraday drops will fill orders")
    
    if avg_intraday_drop > avg_vol_pred:
        print(f"❌ Intraday drops ({avg_intraday_drop:.2f}%) > Vol predictions ({avg_vol_pred:.2f}%)")
        print(f"   Market is more volatile than model predicts")
        print(f"   This causes artificially high fill rates")
    
    # Check distribution
    filled_orders = df[df['order_fills'] == True]
    unfilled_orders = df[df['order_fills'] == False]
    
    print(f"\n📊 FILL ANALYSIS:")
    print(f"Filled orders: {len(filled_orders)}")
    print(f"Unfilled orders: {len(unfilled_orders)}")
    
    if len(filled_orders) > 0:
        print(f"Filled orders - Avg vol pred: {filled_orders['vol_pred_pct'].mean():.2f}%")
        print(f"Filled orders - Avg intraday drop: {filled_orders['intraday_drop'].mean():.2f}%")
    
    if len(unfilled_orders) > 0:
        print(f"Unfilled orders - Avg vol pred: {unfilled_orders['vol_pred_pct'].mean():.2f}%")
        print(f"Unfilled orders - Avg intraday drop: {unfilled_orders['intraday_drop'].mean():.2f}%")
    
    # Check realistic expectations
    print(f"\n🎯 REALISTIC EXPECTATIONS:")
    print(f"For crypto markets:")
    print(f"  - Typical intraday volatility: 2-5%")
    print(f"  - Reasonable fill rate: 20-40%")
    print(f"  - Our vol predictions: {avg_vol_pred:.2f}%")
    print(f"  - Our fill rate: {overall_fill_rate:.1%}")
    
    if overall_fill_rate > 0.8:  # > 80%
        print(f"\n🚨 CONCLUSION: FILL RATE TOO HIGH")
        print(f"Likely causes:")
        print(f"1. Vol predictions too low ({avg_vol_pred:.2f}% vs typical 2-5%)")
        print(f"2. Model under-predicting market volatility")
        print(f"3. Limit orders too close to market price")
        print(f"4. Strategy exploiting model miscalibration")
    else:
        print(f"\n✅ Fill rate appears reasonable")
    
    return df


def compare_with_realistic_volatility():
    """Compare model predictions with realistic volatility expectations."""
    print(f"\n🔍 COMPARING WITH REALISTIC VOLATILITY")
    print("=" * 60)
    
    # Load actual realized volatility data
    vol_df = pd.read_csv("data/crypto_rv1h_38_20200822_20250116.csv", index_col=0, parse_dates=True)
    
    # Get recent data (test period)
    test_start_date = "2024-05-20"
    recent_vol = vol_df[vol_df.index >= test_start_date]
    
    if len(recent_vol) > 0:
        print(f"📊 ACTUAL REALIZED VOLATILITY (Test Period):")
        print(f"Mean: {recent_vol.mean().mean():.4f} ({recent_vol.mean().mean()*100:.2f}%)")
        print(f"Std: {recent_vol.std().mean():.4f} ({recent_vol.std().mean()*100:.2f}%)")
        print(f"Min: {recent_vol.min().min():.4f} ({recent_vol.min().min()*100:.2f}%)")
        print(f"Max: {recent_vol.max().max():.4f} ({recent_vol.max().max()*100:.2f}%)")
        
        # Compare with model predictions
        model_pred_mean = 0.0082  # 0.82% from our results
        actual_vol_mean = recent_vol.mean().mean()
        
        print(f"\n📊 MODEL vs ACTUAL:")
        print(f"Model predictions: {model_pred_mean*100:.2f}%")
        print(f"Actual volatility: {actual_vol_mean*100:.2f}%")
        print(f"Ratio: {model_pred_mean/actual_vol_mean:.2f}x")
        
        if model_pred_mean < actual_vol_mean * 0.5:
            print(f"🚨 Model severely under-predicting volatility!")
            print(f"This explains the high fill rate")
        elif model_pred_mean > actual_vol_mean * 2:
            print(f"🚨 Model severely over-predicting volatility!")
        else:
            print(f"✅ Model predictions in reasonable range")
    else:
        print(f"❌ No recent volatility data found")


def main():
    """Main investigation function."""
    print("🔍 FILL RATE INVESTIGATION")
    print("=" * 80)
    
    # Investigate fill rate
    df = investigate_fill_rate_issue()
    
    # Compare with realistic expectations
    compare_with_realistic_volatility()
    
    print(f"\n🎯 SUMMARY:")
    print(f"The 98.8% fill rate is likely due to:")
    print(f"1. Model under-predicting volatility (0.82% vs actual)")
    print(f"2. Limit orders too close to market price")
    print(f"3. Strategy exploiting model miscalibration")
    print(f"4. Need better volatility prediction calibration")


if __name__ == "__main__":
    main()
