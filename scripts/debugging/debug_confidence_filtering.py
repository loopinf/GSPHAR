#!/usr/bin/env python
"""
Debug why the confidence filtering is removing all trades.
Analyze the confidence scores and thresholds.
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


def debug_confidence_scores():
    """Debug confidence scoring to understand why all trades are filtered out."""
    print("🔍 DEBUGGING CONFIDENCE FILTERING")
    print("=" * 60)
    
    # Load model
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
    
    # Analyze first 5 periods
    test_periods = test_indices[:5]
    
    all_vol_preds = []
    all_confidence_scores = []
    
    with torch.no_grad():
        for period_idx, idx in enumerate(test_periods):
            sample = dataset[idx]
            sample_info = dataset.get_sample_info(idx)
            
            x_lags = [x.unsqueeze(0) for x in sample['x_lags']]
            vol_pred = model(*x_lags)
            vol_pred_np = vol_pred.squeeze().cpu().numpy()
            
            print(f"\n📅 PERIOD {period_idx + 1}: {sample_info['prediction_time']}")
            print("-" * 50)
            
            # Calculate confidence scores (same logic as in selective strategy)
            # Method 1: Based on prediction magnitude
            magnitude_confidence = np.clip(vol_pred_np / 0.05, 0, 1)  # Normalize to 5%
            
            # Method 2: Based on prediction consistency across assets
            pred_std = np.std(vol_pred_np)
            consistency_confidence = np.exp(-pred_std * 10)
            
            # Method 3: Default accuracy confidence
            accuracy_confidence = 0.5
            
            # Combine confidence measures
            combined_confidence = (
                0.4 * magnitude_confidence + 
                0.3 * consistency_confidence + 
                0.3 * accuracy_confidence
            )
            
            print(f"📊 VOLATILITY PREDICTIONS:")
            print(f"  Mean: {vol_pred_np.mean():.4f} ({vol_pred_np.mean()*100:.2f}%)")
            print(f"  Std: {vol_pred_np.std():.4f} ({vol_pred_np.std()*100:.2f}%)")
            print(f"  Min: {vol_pred_np.min():.4f} ({vol_pred_np.min()*100:.2f}%)")
            print(f"  Max: {vol_pred_np.max():.4f} ({vol_pred_np.max()*100:.2f}%)")
            
            print(f"\n📊 CONFIDENCE COMPONENTS:")
            print(f"  Magnitude confidence mean: {magnitude_confidence.mean():.4f}")
            print(f"  Consistency confidence: {consistency_confidence:.4f}")
            print(f"  Accuracy confidence: {accuracy_confidence:.4f}")
            
            print(f"\n📊 COMBINED CONFIDENCE:")
            print(f"  Mean: {combined_confidence.mean():.4f}")
            print(f"  Std: {combined_confidence.std():.4f}")
            print(f"  Min: {combined_confidence.min():.4f}")
            print(f"  Max: {combined_confidence.max():.4f}")
            
            # Check filtering thresholds
            vol_threshold_015 = (vol_pred_np >= 0.015).sum()
            vol_threshold_020 = (vol_pred_np >= 0.020).sum()
            vol_threshold_025 = (vol_pred_np >= 0.025).sum()
            
            conf_threshold_07 = (combined_confidence >= 0.7).sum()
            conf_threshold_08 = (combined_confidence >= 0.8).sum()
            conf_threshold_09 = (combined_confidence >= 0.9).sum()
            
            print(f"\n🔍 FILTERING ANALYSIS:")
            print(f"  Vol >= 1.5%: {vol_threshold_015}/38 assets")
            print(f"  Vol >= 2.0%: {vol_threshold_020}/38 assets")
            print(f"  Vol >= 2.5%: {vol_threshold_025}/38 assets")
            print(f"  Confidence >= 0.7: {conf_threshold_07}/38 assets")
            print(f"  Confidence >= 0.8: {conf_threshold_08}/38 assets")
            print(f"  Confidence >= 0.9: {conf_threshold_09}/38 assets")
            
            # Combined filtering
            combined_filter_07_015 = ((vol_pred_np >= 0.015) & (combined_confidence >= 0.7)).sum()
            combined_filter_08_020 = ((vol_pred_np >= 0.020) & (combined_confidence >= 0.8)).sum()
            
            print(f"\n🎯 COMBINED FILTERING:")
            print(f"  Vol >= 1.5% AND Conf >= 0.7: {combined_filter_07_015}/38 assets")
            print(f"  Vol >= 2.0% AND Conf >= 0.8: {combined_filter_08_020}/38 assets")
            
            # Store for overall analysis
            all_vol_preds.extend(vol_pred_np)
            all_confidence_scores.extend(combined_confidence)
    
    # Overall analysis
    print(f"\n" + "="*60)
    print("📊 OVERALL ANALYSIS")
    print("="*60)
    
    all_vol_preds = np.array(all_vol_preds)
    all_confidence_scores = np.array(all_confidence_scores)
    
    print(f"📊 ALL VOLATILITY PREDICTIONS:")
    print(f"  Mean: {all_vol_preds.mean():.4f} ({all_vol_preds.mean()*100:.2f}%)")
    print(f"  Std: {all_vol_preds.std():.4f} ({all_vol_preds.std()*100:.2f}%)")
    print(f"  25th percentile: {np.percentile(all_vol_preds, 25):.4f} ({np.percentile(all_vol_preds, 25)*100:.2f}%)")
    print(f"  75th percentile: {np.percentile(all_vol_preds, 75):.4f} ({np.percentile(all_vol_preds, 75)*100:.2f}%)")
    print(f"  95th percentile: {np.percentile(all_vol_preds, 95):.4f} ({np.percentile(all_vol_preds, 95)*100:.2f}%)")
    
    print(f"\n📊 ALL CONFIDENCE SCORES:")
    print(f"  Mean: {all_confidence_scores.mean():.4f}")
    print(f"  Std: {all_confidence_scores.std():.4f}")
    print(f"  25th percentile: {np.percentile(all_confidence_scores, 25):.4f}")
    print(f"  75th percentile: {np.percentile(all_confidence_scores, 75):.4f}")
    print(f"  95th percentile: {np.percentile(all_confidence_scores, 95):.4f}")
    
    # Suggest better thresholds
    print(f"\n🎯 SUGGESTED THRESHOLDS:")
    
    # For volatility: use percentiles
    vol_50th = np.percentile(all_vol_preds, 50)
    vol_75th = np.percentile(all_vol_preds, 75)
    vol_90th = np.percentile(all_vol_preds, 90)
    
    print(f"  Vol thresholds:")
    print(f"    50th percentile (moderate): {vol_50th:.4f} ({vol_50th*100:.2f}%)")
    print(f"    75th percentile (selective): {vol_75th:.4f} ({vol_75th*100:.2f}%)")
    print(f"    90th percentile (very selective): {vol_90th:.4f} ({vol_90th*100:.2f}%)")
    
    # For confidence: use percentiles
    conf_50th = np.percentile(all_confidence_scores, 50)
    conf_75th = np.percentile(all_confidence_scores, 75)
    conf_90th = np.percentile(all_confidence_scores, 90)
    
    print(f"  Confidence thresholds:")
    print(f"    50th percentile (moderate): {conf_50th:.4f}")
    print(f"    75th percentile (selective): {conf_75th:.4f}")
    print(f"    90th percentile (very selective): {conf_90th:.4f}")
    
    # Test realistic combinations
    print(f"\n🧪 REALISTIC COMBINATIONS:")
    
    combinations = [
        ("Moderate", vol_50th, conf_50th),
        ("Selective", vol_75th, conf_75th),
        ("Very Selective", vol_90th, conf_90th),
        ("Custom 1", 0.008, 0.4),  # 0.8% vol, 0.4 confidence
        ("Custom 2", 0.010, 0.5),  # 1.0% vol, 0.5 confidence
        ("Custom 3", 0.012, 0.6),  # 1.2% vol, 0.6 confidence
    ]
    
    for name, vol_thresh, conf_thresh in combinations:
        vol_pass = (all_vol_preds >= vol_thresh).sum()
        conf_pass = (all_confidence_scores >= conf_thresh).sum()
        combined_pass = ((all_vol_preds >= vol_thresh) & (all_confidence_scores >= conf_thresh)).sum()
        
        print(f"  {name}: Vol>={vol_thresh:.3f}, Conf>={conf_thresh:.3f}")
        print(f"    → {combined_pass}/{len(all_vol_preds)} trades ({combined_pass/len(all_vol_preds)*100:.1f}%)")
    
    return all_vol_preds, all_confidence_scores


def test_realistic_selective_strategy():
    """Test selective strategy with realistic thresholds."""
    print(f"\n" + "="*60)
    print("🧪 TESTING REALISTIC SELECTIVE STRATEGY")
    print("="*60)
    
    # Use the debug results to set realistic thresholds
    all_vol_preds, all_confidence_scores = debug_confidence_scores()
    
    # Set realistic thresholds based on data distribution
    vol_threshold = np.percentile(all_vol_preds, 75)  # Top 25% volatility predictions
    conf_threshold = np.percentile(all_confidence_scores, 50)  # Above median confidence
    
    print(f"\n🎯 USING REALISTIC THRESHOLDS:")
    print(f"  Volatility threshold: {vol_threshold:.4f} ({vol_threshold*100:.2f}%)")
    print(f"  Confidence threshold: {conf_threshold:.4f}")
    
    # Expected pass rate
    expected_pass_rate = ((all_vol_preds >= vol_threshold) & (all_confidence_scores >= conf_threshold)).mean()
    print(f"  Expected pass rate: {expected_pass_rate:.1%}")
    
    return vol_threshold, conf_threshold


if __name__ == "__main__":
    vol_threshold, conf_threshold = test_realistic_selective_strategy()
