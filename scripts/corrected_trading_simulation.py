#!/usr/bin/env python
"""
Corrected trading simulation based on user's timeline:
1. Vol prediction using data up to T-1
2. Place order using T+0 open price
3. Check fills using T+0 low
4. Exit using holding period close
"""

import torch
import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.ohlcv_trading_dataset import load_ohlcv_trading_data
from src.models.flexible_gsphar import FlexibleGSPHAR


def corrected_trading_simulation(predictions, holding_period=4, position_size_per_asset=100, trading_fee=0.0002):
    """
    Corrected trading simulation using user's timeline.
    
    Timeline:
    1. Vol prediction using data up to T-1
    2. Place order at T+0 open (with 1-2 min delay)
    3. Check fills using T+0 low
    4. Exit at T+holding_period close
    """
    print("🔧 CORRECTED TRADING SIMULATION")
    print("=" * 60)
    print("Timeline:")
    print("1. Vol prediction using data up to T-1")
    print("2. Place order using T+0 open price (07:01-07:02)")
    print("3. Check fills using T+0 low (07:02-07:59)")
    print("4. Exit using holding period close")
    print()
    
    trading_results = []
    cumulative_pnl = 0.0
    total_fee_rate = 2 * trading_fee  # Buy + Sell fees
    
    for i, pred in enumerate(predictions[:10]):  # Test first 10 for comparison
        timestamp = pred['timestamp']
        vol_predictions = pred['vol_predictions']
        ohlcv_data = pred['ohlcv_data']  # [assets, time_periods, 5]
        
        print(f"\n📊 PERIOD {i+1}: {timestamp}")
        
        # Portfolio-level results for this period
        period_pnl = 0.0
        period_trades = 0
        period_filled = 0
        
        # Trade each asset
        n_assets = ohlcv_data.shape[0]
        
        for asset_idx in range(min(3, n_assets)):  # Test first 3 assets
            # CORRECTED IMPLEMENTATION
            open_price = ohlcv_data[asset_idx, 0, 0]      # T+0 OPEN (available at 07:00)
            current_low = ohlcv_data[asset_idx, 0, 2]     # T+0 LOW (07:00-07:59)
            exit_price = ohlcv_data[asset_idx, holding_period, 3]  # T+holding_period CLOSE
            
            # Get volatility prediction for this asset
            asset_vol_pred = vol_predictions[asset_idx]
            
            # Calculate limit order price (placed at 07:01-07:02)
            limit_price = open_price * (1 - asset_vol_pred)
            
            # Check if order fills during T+0 (07:02-07:59)
            # Conservative assumption: if T+0 low <= limit_price, order fills
            order_fills = current_low <= limit_price
            
            print(f"  Asset {asset_idx}:")
            print(f"    Open price: ${open_price:.2f}")
            print(f"    T+0 Low: ${current_low:.2f}")
            print(f"    Vol pred: {asset_vol_pred:.3f} ({asset_vol_pred*100:.1f}%)")
            print(f"    Limit price: ${limit_price:.2f}")
            print(f"    Order fills: {order_fills}")
            
            # Calculate trade result
            if order_fills:
                # Calculate profit with fees
                gross_profit_pct = (exit_price - limit_price) / limit_price
                net_profit_pct = gross_profit_pct - total_fee_rate
                
                # Calculate PnL for this asset
                asset_pnl = net_profit_pct * position_size_per_asset
                period_pnl += asset_pnl
                period_filled += 1
                
                print(f"    Exit price: ${exit_price:.2f}")
                print(f"    Gross profit: {gross_profit_pct*100:.2f}%")
                print(f"    Net profit: {net_profit_pct*100:.2f}%")
                print(f"    Asset PnL: ${asset_pnl:.2f}")
            
            period_trades += 1
        
        # Update cumulative PnL
        cumulative_pnl += period_pnl
        
        print(f"  Period PnL: ${period_pnl:.2f}")
        print(f"  Cumulative PnL: ${cumulative_pnl:.2f}")
        print(f"  Fill rate: {period_filled/period_trades:.1%}")
    
    return cumulative_pnl


def compare_implementations():
    """Compare original vs corrected implementation."""
    print("\n🔍 COMPARING IMPLEMENTATIONS")
    print("=" * 60)
    
    # Load model and data
    model_path = "models/two_stage_model_20250524_132116.pt"
    
    if not os.path.exists(model_path):
        print("❌ Model file not found")
        return
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    metadata = checkpoint['metadata']
    parameters = checkpoint['parameters']
    
    # Load volatility data for correlation matrix
    vol_df = pd.read_csv("data/crypto_rv1h_38_20200822_20250116.csv", index_col=0, parse_dates=True)
    corr_sample = vol_df.iloc[-1000:] if len(vol_df) > 1000 else vol_df
    corr_matrix = corr_sample.corr().values
    A = (corr_matrix + corr_matrix.T) / 2
    
    # Create model
    model = FlexibleGSPHAR(
        lags=parameters['lags'],
        output_dim=1,
        filter_size=len(metadata['assets']),
        A=A
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load dataset
    dataset, _ = load_ohlcv_trading_data(
        volatility_file="data/crypto_rv1h_38_20200822_20250116.csv",
        lags=parameters['lags'],
        holding_period=parameters['holding_period'],
        debug=False
    )
    
    # Generate a few predictions
    predictions = []
    with torch.no_grad():
        for i in range(10):
            sample = dataset[i]
            sample_info = dataset.get_sample_info(i)
            
            x_lags = [x.unsqueeze(0) for x in sample['x_lags']]
            vol_pred = model(*x_lags)
            vol_pred_np = vol_pred.squeeze().cpu().numpy()
            
            prediction_data = {
                'timestamp': sample_info['prediction_time'],
                'vol_predictions': vol_pred_np,
                'ohlcv_data': sample['ohlcv_data'].numpy(),
            }
            predictions.append(prediction_data)
    
    # Run corrected simulation
    corrected_pnl = corrected_trading_simulation(
        predictions=predictions,
        holding_period=parameters['holding_period'],
        position_size_per_asset=100,
        trading_fee=0.0002
    )
    
    print(f"\n🎯 RESULTS COMPARISON:")
    print(f"Corrected implementation (10 periods): ${corrected_pnl:.2f}")
    print(f"Original implementation (2000 periods): $47,699.50")
    print()
    print("Note: Direct comparison not possible due to different period counts")
    print("But we can see if the corrected approach changes fill rates significantly")


def analyze_timeline_validity():
    """Analyze if the corrected timeline is valid."""
    print("\n✅ TIMELINE VALIDITY ANALYSIS")
    print("=" * 60)
    
    print("🔍 USER'S PROPOSED TIMELINE:")
    print("1. ~06:59: Vol prediction using data up to 06:59 ✅")
    print("   - Uses only historical data")
    print("   - No look-ahead bias")
    print()
    print("2. 07:01-07:02: Place order using T+0 open ✅")
    print("   - Open price available at 07:00")
    print("   - 1-2 minute delay realistic")
    print("   - No look-ahead bias")
    print()
    print("3. 07:02-07:59: Check fills using T+0 low ✅")
    print("   - Order placed at 07:01")
    print("   - Check subsequent price action")
    print("   - No look-ahead bias")
    print()
    print("4. Exit at holding period close ✅")
    print("   - Standard backtesting approach")
    print("   - No look-ahead bias")
    print()
    
    print("🎯 CONCLUSION:")
    print("✅ User's timeline is VALID and has NO look-ahead bias")
    print("✅ More realistic than using T+1 low")
    print("✅ Proper temporal causality maintained")
    print()
    print("🔧 IMPLEMENTATION CHANGE NEEDED:")
    print("❌ Current: Uses T+0 close + T+1 low")
    print("✅ Correct: Uses T+0 open + T+0 low")
    print()
    print("📊 IMPACT ON RESULTS:")
    print("- May reduce fill rates (more realistic)")
    print("- May reduce profits (less favorable entry)")
    print("- But still need to fix 100% data leakage issue!")


def main():
    """Main function."""
    print("🔧 CORRECTED TRADING SIMULATION ANALYSIS")
    print("=" * 80)
    
    compare_implementations()
    analyze_timeline_validity()
    
    print(f"\n🎯 FINAL ASSESSMENT:")
    print("=" * 40)
    print("✅ User's timeline is correct and has no look-ahead bias")
    print("✅ Current implementation should be updated")
    print("❌ But main issue remains: 100% training/testing overlap")
    print()
    print("Priority order:")
    print("1. 🚨 Fix data leakage (retrain with proper split)")
    print("2. 🔧 Update trading simulation timeline")
    print("3. 📊 Test on out-of-sample data")


if __name__ == "__main__":
    main()
