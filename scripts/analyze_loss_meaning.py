#!/usr/bin/env python
"""
Analyze what the loss values mean in practical terms.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import sys
from sklearn.preprocessing import StandardScaler

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.flexible_dataloader import FlexibleTimeSeriesDataset
from src.models.flexible_gsphar import FlexibleGSPHAR

def main():
    """Analyze what the loss values mean in practical terms."""
    
    # Load data (same as training)
    rv_file = "data/crypto_rv1h_38_20200822_20250116.csv"
    
    print("Loading realized volatility data...")
    rv_df = pd.read_csv(rv_file, index_col=0, parse_dates=True)
    
    # Split data (same as training)
    train_ratio = 0.8
    train_size = int(len(rv_df) * train_ratio)
    
    rv_train = rv_df.iloc[:train_size]
    rv_val = rv_df.iloc[train_size:]
    
    print("Original data statistics (before standardization):")
    print(f"Training data - Mean: {rv_train.mean().mean():.6f}, Std: {rv_train.std().mean():.6f}")
    print(f"Training data - Min: {rv_train.min().min():.6f}, Max: {rv_train.max().max():.6f}")
    print(f"Validation data - Mean: {rv_val.mean().mean():.6f}, Std: {rv_val.std().mean():.6f}")
    print(f"Validation data - Min: {rv_val.min().min():.6f}, Max: {rv_val.max().max():.6f}")
    
    # Standardize the data (same as training)
    scaler = StandardScaler()
    rv_train_scaled = pd.DataFrame(
        scaler.fit_transform(rv_train),
        index=rv_train.index,
        columns=rv_train.columns
    )
    rv_val_scaled = pd.DataFrame(
        scaler.transform(rv_val),
        index=rv_val.index,
        columns=rv_val.columns
    )
    
    print("\nStandardized data statistics:")
    print(f"Training data - Mean: {rv_train_scaled.mean().mean():.6f}, Std: {rv_train_scaled.std().mean():.6f}")
    print(f"Validation data - Mean: {rv_val_scaled.mean().mean():.6f}, Std: {rv_val_scaled.std().mean():.6f}")
    
    # Our model achieved these losses on standardized data:
    mse_loss = 0.510090
    mae_loss = 0.512629
    rmse_loss = 0.714206
    
    print("\n" + "="*60)
    print("LOSS INTERPRETATION:")
    print("="*60)
    
    print(f"\nOn STANDARDIZED data:")
    print(f"MSE Loss: {mse_loss:.6f}")
    print(f"MAE Loss: {mae_loss:.6f}")
    print(f"RMSE Loss: {rmse_loss:.6f}")
    
    # Convert back to original scale
    # Since we used StandardScaler, we can estimate the original scale errors
    original_std = rv_train.std().mean()
    original_mean = rv_train.mean().mean()
    
    # RMSE in original scale (approximately)
    rmse_original_scale = rmse_loss * original_std
    mae_original_scale = mae_loss * original_std
    
    print(f"\nApproximate errors in ORIGINAL scale:")
    print(f"RMSE: {rmse_original_scale:.6f}")
    print(f"MAE: {mae_original_scale:.6f}")
    
    print(f"\nFor context:")
    print(f"Original data mean: {original_mean:.6f}")
    print(f"Original data std: {original_std:.6f}")
    print(f"RMSE as % of mean: {(rmse_original_scale/original_mean)*100:.2f}%")
    print(f"RMSE as % of std: {(rmse_original_scale/original_std)*100:.2f}%")
    
    # Model performance interpretation
    correlation = 0.468366  # From previous evaluation
    
    print(f"\n" + "="*60)
    print("MODEL PERFORMANCE INTERPRETATION:")
    print("="*60)
    
    print(f"Correlation between predictions and targets: {correlation:.3f}")
    
    if correlation > 0.4:
        print("✓ GOOD: Model shows moderate to good correlation with actual volatility")
    elif correlation > 0.2:
        print("⚠ FAIR: Model shows weak correlation with actual volatility")
    else:
        print("✗ POOR: Model shows very weak correlation with actual volatility")
    
    if rmse_loss < 1.0:
        print("✓ GOOD: RMSE on standardized data is less than 1 standard deviation")
    else:
        print("⚠ FAIR: RMSE on standardized data is greater than 1 standard deviation")
    
    # What this means for trading
    print(f"\n" + "="*60)
    print("TRADING IMPLICATIONS:")
    print("="*60)
    
    print("Since this model predicts realized volatility (RV), the predictions represent:")
    print("- How much the price is expected to move (volatility) in the next period")
    print("- Higher RV = more price movement expected")
    print("- Lower RV = less price movement expected")
    
    print(f"\nWith RMSE of {rmse_original_scale:.6f} in original scale:")
    print("- This is the typical prediction error for volatility forecasts")
    print("- For a trading strategy, this error affects how accurately we can")
    print("  predict whether our limit orders will be filled")
    
    # Sample interpretation
    print(f"\n" + "="*60)
    print("EXAMPLE INTERPRETATION:")
    print("="*60)
    
    sample_rv_values = [0.01, 0.05, 0.10, 0.20]
    print("If the model predicts these RV values, it means:")
    for rv in sample_rv_values:
        percentage = rv * 100
        print(f"- RV = {rv:.3f} → Expect ~{percentage:.1f}% price movement in next hour")
    
    print(f"\nWith our model's RMSE of {rmse_original_scale:.6f}:")
    print(f"- Prediction errors are typically ±{rmse_original_scale*100:.2f}% in volatility terms")
    print("- This affects the reliability of our trading signals")


if __name__ == '__main__':
    main()
