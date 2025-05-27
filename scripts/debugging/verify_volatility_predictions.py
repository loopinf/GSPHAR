#!/usr/bin/env python
"""
Verify that volatility predictions are actually different for each symbol
and check for any remaining issues from our previous problems.
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


def verify_predictions():
    """Verify volatility predictions are working correctly."""
    print("🔍 VERIFYING VOLATILITY PREDICTIONS")
    print("=" * 60)
    
    # Load the improved model
    model_path = "models/improved_model_20250524_172018.pt"
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    metadata = checkpoint['metadata']
    parameters = checkpoint['parameters']
    test_indices = checkpoint['test_indices']
    
    # Get symbol names
    vol_df = pd.read_csv("data/crypto_rv1h_38_20200822_20250116.csv", index_col=0, parse_dates=True)
    symbols = vol_df.columns.tolist()
    
    # Load volatility data for correlation matrix
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
    
    print(f"✅ Model loaded, analyzing {len(symbols)} symbols")
    
    # Test multiple periods to check for issues
    test_sample_indices = test_indices[:5]
    
    all_predictions = []
    
    with torch.no_grad():
        for period_idx, idx in enumerate(test_sample_indices):
            sample = dataset[idx]
            sample_info = dataset.get_sample_info(idx)
            
            x_lags = [x.unsqueeze(0) for x in sample['x_lags']]
            vol_pred = model(*x_lags)
            vol_pred_np = vol_pred.squeeze().cpu().numpy()
            
            timestamp = sample_info['prediction_time']
            
            print(f"\n📅 PERIOD {period_idx + 1}: {timestamp}")
            print("-" * 50)
            
            # Check if all predictions are the same (red flag!)
            unique_predictions = np.unique(vol_pred_np)
            print(f"Unique predictions: {len(unique_predictions)}")
            print(f"Min prediction: {vol_pred_np.min():.6f} ({vol_pred_np.min()*100:.2f}%)")
            print(f"Max prediction: {vol_pred_np.max():.6f} ({vol_pred_np.max()*100:.2f}%)")
            print(f"Mean prediction: {vol_pred_np.mean():.6f} ({vol_pred_np.mean()*100:.2f}%)")
            print(f"Std prediction: {vol_pred_np.std():.6f} ({vol_pred_np.std()*100:.2f}%)")
            
            if len(unique_predictions) == 1:
                print("🚨 WARNING: All symbols have IDENTICAL predictions!")
                print("This suggests the model is not learning symbol-specific patterns")
            elif len(unique_predictions) < 5:
                print("⚠️  WARNING: Very few unique predictions")
                print("Model may be converging to similar values")
            else:
                print("✅ Good: Multiple unique predictions per symbol")
            
            # Show individual symbol predictions
            print(f"\nSample symbol predictions:")
            for i in range(min(10, len(symbols))):
                print(f"  {symbols[i]}: {vol_pred_np[i]:.6f} ({vol_pred_np[i]*100:.2f}%)")
            
            # Store for analysis
            for i, symbol in enumerate(symbols):
                all_predictions.append({
                    'period': period_idx + 1,
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'vol_pred': vol_pred_np[i]
                })
    
    # Overall analysis
    print(f"\n" + "="*60)
    print("📊 OVERALL PREDICTION ANALYSIS")
    print("="*60)
    
    pred_df = pd.DataFrame(all_predictions)
    
    # Check for symbol-specific differences
    symbol_stats = pred_df.groupby('symbol')['vol_pred'].agg(['mean', 'std', 'min', 'max']).round(6)
    
    print(f"Symbol-level statistics:")
    print(f"Mean across symbols: {symbol_stats['mean'].mean():.6f}")
    print(f"Std across symbols: {symbol_stats['mean'].std():.6f}")
    print(f"Range: {symbol_stats['mean'].min():.6f} to {symbol_stats['mean'].max():.6f}")
    
    if symbol_stats['mean'].std() < 0.0001:
        print("🚨 MAJOR ISSUE: All symbols have nearly identical predictions!")
        print("The model is NOT learning symbol-specific patterns")
    else:
        print("✅ Good: Symbols have different average predictions")
    
    # Check for time-varying predictions
    time_stats = pred_df.groupby('period')['vol_pred'].agg(['mean', 'std']).round(6)
    print(f"\nTime-varying statistics:")
    print(time_stats)
    
    if time_stats['mean'].std() < 0.0001:
        print("🚨 MAJOR ISSUE: Predictions don't vary over time!")
        print("The model is predicting constant values")
    else:
        print("✅ Good: Predictions vary over time")
    
    # Check what we're actually predicting vs using
    print(f"\n🎯 PREDICTION TARGET VERIFICATION:")
    print("=" * 50)
    
    # Get actual volatility targets for comparison
    sample = dataset[test_indices[0]]
    vol_targets = sample['vol_targets'].numpy()
    
    print(f"Volatility targets (what we should predict):")
    print(f"  Mean: {vol_targets.mean():.6f} ({vol_targets.mean()*100:.2f}%)")
    print(f"  Std: {vol_targets.std():.6f} ({vol_targets.std()*100:.2f}%)")
    print(f"  Range: {vol_targets.min():.6f} to {vol_targets.max():.6f}")
    
    print(f"\nModel predictions (what we actually predict):")
    all_preds = pred_df['vol_pred'].values
    print(f"  Mean: {all_preds.mean():.6f} ({all_preds.mean()*100:.2f}%)")
    print(f"  Std: {all_preds.std():.6f} ({all_preds.std()*100:.2f}%)")
    print(f"  Range: {all_preds.min():.6f} to {all_preds.max():.6f}")
    
    # Check if we're in the right ballpark
    target_mean = vol_targets.mean()
    pred_mean = all_preds.mean()
    
    if abs(target_mean - pred_mean) / target_mean > 0.5:  # >50% difference
        print("🚨 WARNING: Predictions very different from targets!")
        print("Model may not be learning the right patterns")
    else:
        print("✅ Good: Predictions in reasonable range vs targets")
    
    return pred_df, symbol_stats


def check_previous_issues():
    """Check for our previous specific issues."""
    print(f"\n🔍 CHECKING FOR PREVIOUS ISSUES")
    print("=" * 60)
    
    issues_found = []
    
    # Issue 1: Zero predictions
    pred_df, symbol_stats = verify_predictions()
    
    if (pred_df['vol_pred'] == 0).any():
        issues_found.append("❌ Zero predictions found")
    else:
        print("✅ No zero predictions")
    
    # Issue 2: All identical predictions
    if symbol_stats['mean'].std() < 0.0001:
        issues_found.append("❌ All symbols have identical predictions")
    else:
        print("✅ Symbols have different predictions")
    
    # Issue 3: Unrealistic values
    if pred_df['vol_pred'].max() > 0.1:  # >10%
        issues_found.append("❌ Unrealistically high predictions")
    elif pred_df['vol_pred'].min() < 0.0001:  # <0.01%
        issues_found.append("❌ Unrealistically low predictions")
    else:
        print("✅ Predictions in realistic range")
    
    # Issue 4: No time variation
    time_stats = pred_df.groupby('period')['vol_pred'].mean()
    if time_stats.std() < 0.0001:
        issues_found.append("❌ No time variation in predictions")
    else:
        print("✅ Predictions vary over time")
    
    print(f"\n🎯 ISSUE SUMMARY:")
    if issues_found:
        print("🚨 ISSUES FOUND:")
        for issue in issues_found:
            print(f"  {issue}")
    else:
        print("✅ NO MAJOR ISSUES DETECTED!")
        print("Model appears to be working correctly")


if __name__ == "__main__":
    check_previous_issues()
