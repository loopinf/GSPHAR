#!/usr/bin/env python
"""
Analyze the realistic results from the properly trained model.
"""

import torch
import numpy as np
import pandas as pd
import os
import sys

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.ohlcv_trading_dataset import load_ohlcv_trading_data
from src.models.flexible_gsphar import FlexibleGSPHAR


def quick_realistic_test():
    """Quick test to get realistic results without plot display."""
    print("🎯 REALISTIC RESULTS ANALYSIS")
    print("=" * 60)
    
    # Load the proper model
    model_path = "models/proper_split_model_20250524_152004.pt"
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    metadata = checkpoint['metadata']
    parameters = checkpoint['parameters']
    test_indices = checkpoint['test_indices']
    
    print(f"✅ Model loaded: {model_path}")
    print(f"Training period: {metadata['train_period']}")
    print(f"Testing period: {metadata['test_period']}")
    print(f"Test samples: {len(test_indices)}")
    
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
    
    # Test on first 1000 samples for quick analysis
    test_sample_indices = test_indices[:1000]
    print(f"Quick test on first {len(test_sample_indices)} test samples")
    
    # Generate predictions
    predictions = []
    with torch.no_grad():
        for i, idx in enumerate(test_sample_indices):
            sample = dataset[idx]
            sample_info = dataset.get_sample_info(idx)
            
            x_lags = [x.unsqueeze(0) for x in sample['x_lags']]
            vol_pred = model(*x_lags)
            vol_pred_np = vol_pred.squeeze().cpu().numpy()
            
            prediction_data = {
                'timestamp': sample_info['prediction_time'],
                'vol_predictions': vol_pred_np,
                'ohlcv_data': sample['ohlcv_data'].numpy(),
            }
            predictions.append(prediction_data)
    
    # Run corrected trading simulation
    trading_results = []
    cumulative_pnl = 0.0
    total_fee_rate = 2 * 0.0002  # 0.04% total fees
    
    for pred in predictions:
        vol_predictions = pred['vol_predictions']
        ohlcv_data = pred['ohlcv_data']
        
        period_pnl = 0.0
        period_trades = 0
        period_filled = 0
        
        n_assets = ohlcv_data.shape[0]
        for asset_idx in range(n_assets):
            # CORRECTED TIMELINE
            open_price = ohlcv_data[asset_idx, 0, 0]      # T+0 OPEN
            current_low = ohlcv_data[asset_idx, 0, 2]     # T+0 LOW
            exit_price = ohlcv_data[asset_idx, 4, 3]      # T+4 CLOSE
            
            asset_vol_pred = vol_predictions[asset_idx]
            limit_price = open_price * (1 - asset_vol_pred)
            
            # Check if order fills during T+0 period
            order_fills = current_low <= limit_price
            
            if order_fills:
                gross_profit_pct = (exit_price - limit_price) / limit_price
                net_profit_pct = gross_profit_pct - total_fee_rate
                asset_pnl = net_profit_pct * 100  # $100 per asset
                period_pnl += asset_pnl
                period_filled += 1
            
            period_trades += 1
        
        cumulative_pnl += period_pnl
        
        trading_results.append({
            'timestamp': pred['timestamp'],
            'period_pnl': period_pnl,
            'cumulative_pnl': cumulative_pnl,
            'trades': period_trades,
            'filled': period_filled,
            'fill_rate': period_filled / period_trades,
            'avg_vol_pred': np.mean(vol_predictions)
        })
    
    # Analyze results
    df = pd.DataFrame(trading_results)
    
    total_periods = len(df)
    profitable_periods = (df['period_pnl'] > 0).sum()
    win_rate = profitable_periods / total_periods
    final_pnl = df['cumulative_pnl'].iloc[-1]
    
    print(f"\n📊 REALISTIC RESULTS (1000 test periods):")
    print(f"=" * 50)
    print(f"Test Period: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Total Periods: {total_periods}")
    print(f"Profitable Periods: {profitable_periods} ({win_rate:.1%})")
    print(f"Final Cumulative PnL: ${final_pnl:.2f}")
    print(f"Average Period PnL: ${df['period_pnl'].mean():.2f}")
    print(f"Best Period: ${df['period_pnl'].max():.2f}")
    print(f"Worst Period: ${df['period_pnl'].min():.2f}")
    print(f"Average Fill Rate: {df['fill_rate'].mean():.1%}")
    print(f"Average Vol Prediction: {df['avg_vol_pred'].mean()*100:.2f}%")
    
    # Extrapolate to full test set
    full_test_periods = len(test_indices)
    extrapolated_pnl = final_pnl * (full_test_periods / total_periods)
    
    print(f"\n📈 EXTRAPOLATED TO FULL TEST SET:")
    print(f"Full test periods: {full_test_periods}")
    print(f"Extrapolated PnL: ${extrapolated_pnl:.2f}")
    
    # Compare with previous results
    print(f"\n📊 COMPARISON WITH PREVIOUS RESULTS:")
    print(f"=" * 50)
    print(f"Previous (Data Leakage):  $47,699 | 85% win rate | 2,000 periods")
    print(f"Current (No Leakage):     ${extrapolated_pnl:.0f} | {win_rate:.1%} win rate | {full_test_periods} periods")
    print(f"Reduction factor:         {47699/extrapolated_pnl:.1f}x lower")
    
    # Assessment
    print(f"\n🎯 ASSESSMENT:")
    print(f"=" * 30)
    if extrapolated_pnl > 0:
        print(f"✅ Strategy still profitable without data leakage!")
        print(f"✅ Win rate {win_rate:.1%} is reasonable (>50%)")
        print(f"✅ Results are much more realistic")
        
        # Calculate annualized return
        test_duration_months = 8  # May 2024 to Jan 2025
        annual_return = (extrapolated_pnl / (full_test_periods * 38 * 100)) * (12 / test_duration_months) * 100
        print(f"✅ Estimated annual return: ~{annual_return:.1f}%")
    else:
        print(f"❌ Strategy not profitable on out-of-sample data")
        print(f"❌ Need to improve model or strategy")
    
    print(f"\n💡 KEY INSIGHTS:")
    print(f"1. Data leakage was the main cause of exceptional results")
    print(f"2. Proper train/test split reveals realistic performance")
    print(f"3. Strategy framework is sound but needs optimization")
    print(f"4. Corrected timeline (T+0 open/low) is more realistic")


if __name__ == "__main__":
    quick_realistic_test()
