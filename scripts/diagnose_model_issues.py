#!/usr/bin/env python
"""
Systematic diagnosis of model training issues.
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.ohlcv_trading_dataset import load_ohlcv_trading_data
from src.models.flexible_gsphar import FlexibleGSPHAR


def diagnose_training_data():
    """Diagnose what the model should be learning from training data."""
    print("🔍 DIAGNOSING TRAINING DATA")
    print("=" * 60)

    # Load dataset
    dataset, metadata = load_ohlcv_trading_data(
        volatility_file="data/crypto_rv1h_38_20200822_20250116.csv",
        lags=[1, 4, 24],
        holding_period=4,
        debug=False
    )

    # Get training indices (first 70%)
    total_samples = len(dataset)
    train_size = int(total_samples * 0.7)
    train_indices = list(range(0, train_size))

    print(f"Training samples: {len(train_indices)}")

    # Analyze training targets
    train_targets = []
    train_timestamps = []

    for i in range(0, min(1000, len(train_indices))):  # Sample 1000 for analysis
        idx = train_indices[i]
        sample = dataset[idx]
        sample_info = dataset.get_sample_info(idx)

        vol_targets = sample['vol_targets'].numpy()
        train_targets.append(vol_targets)
        train_timestamps.append(sample_info['prediction_time'])

    train_targets = np.array(train_targets)  # [samples, assets]

    print(f"\n📊 TRAINING TARGET ANALYSIS:")
    print(f"Shape: {train_targets.shape}")
    print(f"Mean: {train_targets.mean():.6f} ({train_targets.mean()*100:.2f}%)")
    print(f"Std: {train_targets.std():.6f} ({train_targets.std()*100:.2f}%)")
    print(f"Min: {train_targets.min():.6f} ({train_targets.min()*100:.2f}%)")
    print(f"Max: {train_targets.max():.6f} ({train_targets.max()*100:.2f}%)")

    # Check temporal variation
    temporal_means = train_targets.mean(axis=1)  # Mean across assets for each time
    print(f"\n📈 TEMPORAL VARIATION:")
    print(f"Temporal mean std: {temporal_means.std():.6f}")
    print(f"Temporal range: {temporal_means.min():.6f} to {temporal_means.max():.6f}")

    if temporal_means.std() < 0.001:
        print("🚨 WARNING: Very low temporal variation in targets!")
    else:
        print("✅ Good temporal variation in targets")

    # Check asset variation
    asset_means = train_targets.mean(axis=0)  # Mean across time for each asset
    print(f"\n🏢 ASSET VARIATION:")
    print(f"Asset mean std: {asset_means.std():.6f}")
    print(f"Asset range: {asset_means.min():.6f} to {asset_means.max():.6f}")

    if asset_means.std() < 0.001:
        print("🚨 WARNING: Very low asset variation in targets!")
    else:
        print("✅ Good asset variation in targets")

    return train_targets, train_timestamps


def diagnose_model_architecture():
    """Diagnose the model architecture and initialization."""
    print(f"\n🧠 DIAGNOSING MODEL ARCHITECTURE")
    print("=" * 60)

    # Load volatility data for correlation matrix
    vol_df = pd.read_csv("data/crypto_rv1h_38_20200822_20250116.csv", index_col=0, parse_dates=True)
    corr_sample = vol_df.iloc[-1000:] if len(vol_df) > 1000 else vol_df
    corr_matrix = corr_sample.corr().values
    A = (corr_matrix + corr_matrix.T) / 2

    # Create fresh model
    model = FlexibleGSPHAR(
        lags=[1, 4, 24],
        output_dim=1,
        filter_size=38,
        A=A
    )

    print(f"Model parameters:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params}")
    print(f"  Trainable parameters: {trainable_params}")

    # Test forward pass
    print(f"\n🔄 TESTING FORWARD PASS:")

    # Create dummy input
    batch_size = 4
    dummy_inputs = []
    for lag in [1, 4, 24]:
        dummy_input = torch.randn(batch_size, 38, lag)
        dummy_inputs.append(dummy_input)

    with torch.no_grad():
        output = model(*dummy_inputs)
        print(f"Input shapes: {[x.shape for x in dummy_inputs]}")
        print(f"Output shape: {output.shape}")
        print(f"Output range: {output.min():.6f} to {output.max():.6f}")
        print(f"Output mean: {output.mean():.6f}")
        print(f"Output std: {output.std():.6f}")

    # Check if model produces reasonable initial outputs
    if output.std() < 0.001:
        print("🚨 WARNING: Very low output variance - model may be stuck")
    else:
        print("✅ Good output variance")

    return model


def diagnose_trained_model():
    """Diagnose the trained model behavior."""
    print(f"\n🎓 DIAGNOSING TRAINED MODEL")
    print("=" * 60)

    # Load the trained model
    model_path = "models/improved_model_20250524_172018.pt"
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    # Check training history
    stage1_history = checkpoint['stage1_history']
    stage2_history = checkpoint['stage2_history']

    print(f"📈 STAGE 1 TRAINING HISTORY:")
    print(f"  Epochs completed: {len(stage1_history['train_loss'])}")
    print(f"  Final train loss: {stage1_history['train_loss'][-1]:.6f}")
    print(f"  Final val loss: {stage1_history['val_loss'][-1]:.6f}")
    print(f"  Loss reduction: {stage1_history['train_loss'][0]:.6f} → {stage1_history['train_loss'][-1]:.6f}")

    print(f"\n📈 STAGE 2 TRAINING HISTORY:")
    print(f"  Epochs completed: {len(stage2_history['train_loss'])}")
    print(f"  Final train loss: {stage2_history['train_loss'][-1]:.6f}")
    print(f"  Final val loss: {stage2_history['val_loss'][-1]:.6f}")

    # Check if Stage 1 actually learned anything
    stage1_loss_reduction = (stage1_history['train_loss'][0] - stage1_history['train_loss'][-1]) / stage1_history['train_loss'][0]
    print(f"\nStage 1 loss reduction: {stage1_loss_reduction:.1%}")

    if stage1_loss_reduction < 0.1:  # Less than 10% improvement
        print("🚨 WARNING: Stage 1 barely improved - model may not be learning")
    else:
        print("✅ Stage 1 showed good improvement")

    # Load the model and test on training data
    vol_df = pd.read_csv("data/crypto_rv1h_38_20200822_20250116.csv", index_col=0, parse_dates=True)
    corr_sample = vol_df.iloc[-1000:] if len(vol_df) > 1000 else vol_df
    corr_matrix = corr_sample.corr().values
    A = (corr_matrix + corr_matrix.T) / 2

    model = FlexibleGSPHAR(
        lags=[1, 4, 24],
        output_dim=1,
        filter_size=38,
        A=A
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Test on training data
    dataset, _ = load_ohlcv_trading_data(
        volatility_file="data/crypto_rv1h_38_20200822_20250116.csv",
        lags=[1, 4, 24],
        holding_period=4,
        debug=False
    )

    train_size = int(len(dataset) * 0.7)
    train_indices = list(range(0, min(100, train_size)))  # Test first 100 training samples

    train_predictions = []
    train_targets = []

    with torch.no_grad():
        for idx in train_indices:
            sample = dataset[idx]
            x_lags = [x.unsqueeze(0) for x in sample['x_lags']]
            vol_targets = sample['vol_targets'].numpy()

            vol_pred = model(*x_lags)
            vol_pred_np = vol_pred.squeeze().cpu().numpy()

            train_predictions.append(vol_pred_np)
            train_targets.append(vol_targets.squeeze())

    train_predictions = np.array(train_predictions)
    train_targets = np.array(train_targets)

    print(f"\n📊 TRAINING DATA PERFORMANCE:")
    print(f"Target mean: {train_targets.mean():.6f} ({train_targets.mean()*100:.2f}%)")
    print(f"Prediction mean: {train_predictions.mean():.6f} ({train_predictions.mean()*100:.2f}%)")
    print(f"Target std: {train_targets.std():.6f}")
    print(f"Prediction std: {train_predictions.std():.6f}")

    # Calculate MSE on training data
    mse = np.mean((train_predictions - train_targets) ** 2)
    print(f"Training MSE: {mse:.6f}")

    # Check if model is just predicting mean
    mean_baseline_mse = np.mean((train_targets.mean() - train_targets) ** 2)
    print(f"Mean baseline MSE: {mean_baseline_mse:.6f}")

    if mse > mean_baseline_mse * 0.9:
        print("🚨 WARNING: Model barely better than predicting mean!")
    else:
        print("✅ Model learned better than mean baseline")

    return train_predictions, train_targets


def diagnose_loss_function():
    """Diagnose the trading loss function behavior."""
    print(f"\n💰 DIAGNOSING TRADING LOSS FUNCTION")
    print("=" * 60)

    from src.ohlcv_trading_loss import OHLCVLongStrategyLoss

    # Create loss function
    trading_loss_fn = OHLCVLongStrategyLoss(holding_period=4, trading_fee=0.0002)

    # Test with different volatility predictions
    test_vol_preds = [0.001, 0.01, 0.02, 0.05, 0.1]  # 0.1% to 10%

    # Create dummy OHLCV data
    batch_size = 1
    n_assets = 38
    n_periods = 5

    # Simulate market data
    base_price = 100.0
    ohlcv_data = torch.zeros(batch_size, n_assets, n_periods, 5)

    for asset in range(n_assets):
        for period in range(n_periods):
            # Simulate some price movement
            price_change = np.random.normal(0, 0.02)  # 2% volatility
            current_price = base_price * (1 + price_change)

            # OHLCV: Open, High, Low, Close, Volume
            ohlcv_data[0, asset, period, 0] = current_price  # Open
            ohlcv_data[0, asset, period, 1] = current_price * 1.01  # High
            ohlcv_data[0, asset, period, 2] = current_price * 0.99  # Low
            ohlcv_data[0, asset, period, 3] = current_price  # Close
            ohlcv_data[0, asset, period, 4] = 1000  # Volume

    print(f"Testing loss function with different volatility predictions:")

    for vol_pred in test_vol_preds:
        vol_pred_tensor = torch.full((batch_size, n_assets, 1), vol_pred)

        with torch.no_grad():
            loss = trading_loss_fn(vol_pred_tensor, ohlcv_data)
            print(f"  Vol pred {vol_pred:.3f} ({vol_pred*100:.1f}%): Loss = {loss.item():.6f}")

    # Check if loss function is working properly
    print(f"\n🔍 Loss function analysis:")
    print(f"  Negative loss = profitable (good)")
    print(f"  Positive loss = unprofitable (bad)")
    print(f"  Loss should vary with volatility predictions")


def main():
    """Main diagnosis function."""
    print("🔍 SYSTEMATIC MODEL DIAGNOSIS")
    print("=" * 80)

    # Step 1: Diagnose training data
    train_targets, train_timestamps = diagnose_training_data()

    # Step 2: Diagnose model architecture
    fresh_model = diagnose_model_architecture()

    # Step 3: Diagnose trained model
    train_predictions, train_targets = diagnose_trained_model()

    # Step 4: Diagnose loss function
    diagnose_loss_function()

    print(f"\n🎯 DIAGNOSIS SUMMARY:")
    print("=" * 40)

    # Identify main issues
    issues = []

    # Check prediction vs target mismatch
    pred_mean = train_predictions.mean()
    target_mean = train_targets.mean()
    if abs(pred_mean - target_mean) / target_mean > 0.5:
        issues.append(f"❌ Prediction/target mismatch: {pred_mean:.4f} vs {target_mean:.4f}")

    # Check prediction variance
    if train_predictions.std() < 0.001:
        issues.append("❌ Very low prediction variance")

    # Check temporal variation
    temporal_pred_std = train_predictions.mean(axis=1).std()
    if temporal_pred_std < 0.001:
        issues.append("❌ No temporal variation in predictions")

    if issues:
        print("🚨 CRITICAL ISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")

        print(f"\n🔧 RECOMMENDED FIXES:")
        print("1. Retrain Stage 1 with lower learning rate")
        print("2. Add proper normalization/scaling")
        print("3. Check loss function implementation")
        print("4. Increase model capacity or regularization")
    else:
        print("✅ No critical issues found")


if __name__ == "__main__":
    main()
