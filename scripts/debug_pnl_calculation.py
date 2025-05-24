#!/usr/bin/env python
"""
Debug script to identify bugs in PnL calculation.

This script will:
1. Load the trained model and check its actual predictions
2. Analyze a few sample trades in detail
3. Verify the trading logic step by step
4. Identify potential bugs causing unrealistic results
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


def load_and_inspect_model():
    """Load the trained model and inspect its predictions."""
    print("🔍 LOADING AND INSPECTING TRAINED MODEL")
    print("=" * 60)
    
    model_path = "models/two_stage_model_20250524_132116.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return None, None, None
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    metadata = checkpoint['metadata']
    parameters = checkpoint['parameters']
    
    print(f"✅ Model loaded successfully")
    print(f"   Stage 1 history: {len(checkpoint['stage1_history']['train_loss'])} epochs")
    print(f"   Stage 2 history: {len(checkpoint['stage2_history']['train_loss'])} epochs")
    print(f"   Parameters: {parameters}")
    
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
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, metadata, parameters


def test_model_predictions(model, parameters):
    """Test the model predictions on a few samples."""
    print("\n🎯 TESTING MODEL PREDICTIONS")
    print("=" * 60)
    
    # Load dataset
    dataset, _ = load_ohlcv_trading_data(
        volatility_file="data/crypto_rv1h_38_20200822_20250116.csv",
        lags=parameters['lags'],
        holding_period=parameters['holding_period'],
        debug=False
    )
    
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Test on first 10 samples
    with torch.no_grad():
        for i in range(min(10, len(dataset))):
            sample = dataset[i]
            sample_info = dataset.get_sample_info(i)
            
            # Get model inputs
            x_lags = [x.unsqueeze(0) for x in sample['x_lags']]
            
            # Make prediction
            vol_pred = model(*x_lags)  # [1, assets, 1]
            vol_pred_np = vol_pred.squeeze().cpu().numpy()
            
            print(f"\nSample {i}:")
            print(f"  Timestamp: {sample_info['prediction_time']}")
            print(f"  Vol predictions (first 5 assets): {vol_pred_np[:5]}")
            print(f"  Vol prediction range: {vol_pred_np.min():.6f} to {vol_pred_np.max():.6f}")
            print(f"  Average vol prediction: {vol_pred_np.mean():.6f}")
            
            # Check if predictions are reasonable
            if vol_pred_np.mean() > 0.1:  # > 10%
                print(f"  ⚠️  WARNING: Very high volatility predictions!")
            elif vol_pred_np.mean() < 0.001:  # < 0.1%
                print(f"  ⚠️  WARNING: Very low volatility predictions!")
            else:
                print(f"  ✅ Reasonable volatility predictions")


def debug_single_trade(model, parameters):
    """Debug a single trade in detail."""
    print("\n🔍 DEBUGGING SINGLE TRADE IN DETAIL")
    print("=" * 60)
    
    # Load dataset
    dataset, _ = load_ohlcv_trading_data(
        volatility_file="data/crypto_rv1h_38_20200822_20250116.csv",
        lags=parameters['lags'],
        holding_period=parameters['holding_period'],
        debug=False
    )
    
    # Use sample index 100 for debugging
    sample_idx = 100
    sample = dataset[sample_idx]
    sample_info = dataset.get_sample_info(sample_idx)
    
    print(f"Debugging sample {sample_idx}")
    print(f"Timestamp: {sample_info['prediction_time']}")
    
    # Get model prediction
    with torch.no_grad():
        x_lags = [x.unsqueeze(0) for x in sample['x_lags']]
        vol_pred = model(*x_lags)
        vol_pred_np = vol_pred.squeeze().cpu().numpy()
    
    print(f"Vol predictions shape: {vol_pred_np.shape}")
    print(f"Average vol prediction: {vol_pred_np.mean():.6f}")
    
    # Get OHLCV data
    ohlcv_data = sample['ohlcv_data'].numpy()  # [assets, time_periods, 5]
    print(f"OHLCV data shape: {ohlcv_data.shape}")
    
    # Debug first asset in detail
    asset_idx = 0
    print(f"\n📊 ASSET {asset_idx} DETAILED ANALYSIS:")
    
    # Get prices
    current_price = ohlcv_data[asset_idx, 0, 3]  # Close at T+0
    next_low = ohlcv_data[asset_idx, 1, 2]       # Low at T+1
    exit_price = ohlcv_data[asset_idx, parameters['holding_period'], 3]  # Close at T+holding_period
    
    print(f"  Current price (T+0): ${current_price:.4f}")
    print(f"  Next low (T+1): ${next_low:.4f}")
    print(f"  Exit price (T+{parameters['holding_period']}): ${exit_price:.4f}")
    
    # Get volatility prediction
    asset_vol_pred = vol_pred_np[asset_idx]
    print(f"  Vol prediction: {asset_vol_pred:.6f} ({asset_vol_pred*100:.2f}%)")
    
    # Calculate limit order price
    limit_price = current_price * (1 - asset_vol_pred)
    print(f"  Limit price: ${limit_price:.4f}")
    print(f"  Discount: {((current_price - limit_price) / current_price * 100):.2f}%")
    
    # Check if order fills
    order_fills = next_low <= limit_price
    print(f"  Order fills: {order_fills}")
    print(f"  Fill logic: {next_low:.4f} <= {limit_price:.4f} = {order_fills}")
    
    if order_fills:
        # Calculate profit
        gross_profit_pct = (exit_price - limit_price) / limit_price
        trading_fee = 0.0002
        total_fee_rate = 2 * trading_fee
        net_profit_pct = gross_profit_pct - total_fee_rate
        
        print(f"  Gross profit %: {gross_profit_pct*100:.2f}%")
        print(f"  Trading fees: {total_fee_rate*100:.2f}%")
        print(f"  Net profit %: {net_profit_pct*100:.2f}%")
        
        position_size = 100
        asset_pnl = net_profit_pct * position_size
        print(f"  Asset PnL: ${asset_pnl:.2f}")
        
        # Check if this is realistic
        if net_profit_pct > 0.05:  # > 5% profit
            print(f"  ⚠️  WARNING: Very high profit per trade!")
        elif net_profit_pct < -0.05:  # > 5% loss
            print(f"  ⚠️  WARNING: Very high loss per trade!")
        else:
            print(f"  ✅ Reasonable profit/loss")
    else:
        print(f"  No trade executed")


def analyze_market_conditions(parameters):
    """Analyze the market conditions during the test period."""
    print("\n📈 ANALYZING MARKET CONDITIONS")
    print("=" * 60)
    
    # Load OHLCV data directly
    ohlcv_files = [
        "data/ohlcv_1h/BTCUSDT_1h.csv",
        "data/ohlcv_1h/ETHUSDT_1h.csv",
        "data/ohlcv_1h/ADAUSDT_1h.csv"
    ]
    
    for file_path in ohlcv_files:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            
            # Filter to test period (Aug-Nov 2020)
            test_start = "2020-08-23"
            test_end = "2020-11-14"
            test_df = df.loc[test_start:test_end]
            
            if len(test_df) > 0:
                asset_name = os.path.basename(file_path).split('_')[0]
                start_price = test_df['close'].iloc[0]
                end_price = test_df['close'].iloc[-1]
                total_return = (end_price - start_price) / start_price * 100
                
                print(f"{asset_name}:")
                print(f"  Start price: ${start_price:.2f}")
                print(f"  End price: ${end_price:.2f}")
                print(f"  Total return: {total_return:.1f}%")
                print(f"  Periods: {len(test_df)}")
                
                # Check for unusual price movements
                daily_returns = test_df['close'].pct_change().dropna()
                max_daily_gain = daily_returns.max() * 100
                max_daily_loss = daily_returns.min() * 100
                avg_volatility = daily_returns.std() * 100
                
                print(f"  Max daily gain: {max_daily_gain:.1f}%")
                print(f"  Max daily loss: {max_daily_loss:.1f}%")
                print(f"  Avg volatility: {avg_volatility:.1f}%")
                
                if total_return > 100:  # > 100% gain
                    print(f"  🚀 BULL MARKET: Very strong gains!")
                elif total_return < -50:  # > 50% loss
                    print(f"  📉 BEAR MARKET: Significant losses!")
                else:
                    print(f"  📊 NORMAL MARKET: Moderate movements")
                print()


def check_for_data_leakage():
    """Check for potential data leakage issues."""
    print("\n🚨 CHECKING FOR DATA LEAKAGE")
    print("=" * 60)
    
    # Load dataset with debug mode
    dataset, _ = load_ohlcv_trading_data(
        volatility_file="data/crypto_rv1h_38_20200822_20250116.csv",
        lags=[1, 4, 24],
        holding_period=4,
        debug=True  # Enable debug mode
    )
    
    # Check a few samples for timeline correctness
    for i in [0, 100, 500]:
        if i < len(dataset):
            sample = dataset[i]
            sample_info = dataset.get_sample_info(i)
            
            print(f"\nSample {i} timeline check:")
            print(f"  Prediction time: {sample_info['prediction_time']}")
            print(f"  Actual index: {sample_info['actual_idx']}")
            
            # Check OHLCV timeline
            ohlcv_data = sample['ohlcv_data']
            print(f"  OHLCV shape: {ohlcv_data.shape}")
            
            # Verify no future data leakage
            vol_targets = sample['vol_targets']
            print(f"  Vol targets shape: {vol_targets.shape}")
            print(f"  Vol targets (first 3): {vol_targets[:3].flatten()}")


def main():
    """Main debugging function."""
    print("🐛 PnL CALCULATION DEBUG ANALYSIS")
    print("=" * 80)
    
    # Load and inspect model
    model, metadata, parameters = load_and_inspect_model()
    
    if model is None:
        print("❌ Cannot proceed without model")
        return
    
    # Test model predictions
    test_model_predictions(model, parameters)
    
    # Debug single trade
    debug_single_trade(model, parameters)
    
    # Analyze market conditions
    analyze_market_conditions(parameters)
    
    # Check for data leakage
    check_for_data_leakage()
    
    print("\n🎯 DEBUG ANALYSIS COMPLETED")
    print("Check the output above for potential issues!")


if __name__ == "__main__":
    main()
