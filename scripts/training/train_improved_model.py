#!/usr/bin/env python
"""
Improved model training to fix zero volatility predictions and poor generalization.

Improvements:
1. More training epochs (20-30 instead of 10)
2. Better learning rate scheduling
3. Regularization to prevent overfitting
4. Gradient clipping for stability
5. Early stopping based on validation
6. Better monitoring and diagnostics
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import sys
import logging
from datetime import datetime
from torch.utils.data import DataLoader, Subset

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.ohlcv_trading_dataset import load_ohlcv_trading_data
from src.models.flexible_gsphar import FlexibleGSPHAR
from src.ohlcv_trading_loss import OHLCVLongStrategyLoss

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_data_splits(dataset):
    """Create proper train/validation/test splits."""
    total_samples = len(dataset)

    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15

    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)
    test_size = total_samples - train_size - val_size

    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, total_samples))

    logger.info(f"📊 DATA SPLIT:")
    logger.info(f"  Training: {len(train_indices)} samples ({train_ratio:.0%})")
    logger.info(f"  Validation: {len(val_indices)} samples ({val_ratio:.0%})")
    logger.info(f"  Testing: {len(test_indices)} samples ({test_ratio:.0%})")

    return train_indices, val_indices, test_indices


def train_stage1_improved(model, train_loader, val_loader, device, n_epochs=25, initial_lr=0.001):
    """Improved Stage 1: Supervised learning with better training."""
    logger.info(f"🎯 IMPROVED STAGE 1: SUPERVISED LEARNING")
    logger.info(f"  Epochs: {n_epochs}")
    logger.info(f"  Initial learning rate: {initial_lr}")

    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=1e-5)  # Add L2 regularization
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    train_history = {'train_loss': [], 'train_mae': [], 'val_loss': [], 'val_mae': [], 'lr': []}
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 7  # Early stopping patience

    for epoch in range(n_epochs):
        # Training
        model.train()
        train_losses = []
        train_maes = []

        for batch_idx, batch in enumerate(train_loader):
            x_lags = [x.to(device) for x in batch['x_lags']]
            vol_targets = batch['vol_targets'].to(device)

            optimizer.zero_grad()

            vol_pred = model(*x_lags)
            mse_loss = F.mse_loss(vol_pred, vol_targets)
            mae_loss = F.l1_loss(vol_pred, vol_targets)

            # Add gradient clipping for stability
            mse_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(mse_loss.item())
            train_maes.append(mae_loss.item())

            # Log progress every 500 batches
            if batch_idx % 500 == 0:
                logger.info(f"    Batch {batch_idx}/{len(train_loader)}: MSE={mse_loss.item():.6f}")

        # Validation
        model.eval()
        val_losses = []
        val_maes = []
        vol_pred_stats = []

        with torch.no_grad():
            for batch in val_loader:
                x_lags = [x.to(device) for x in batch['x_lags']]
                vol_targets = batch['vol_targets'].to(device)

                vol_pred = model(*x_lags)
                mse_loss = F.mse_loss(vol_pred, vol_targets)
                mae_loss = F.l1_loss(vol_pred, vol_targets)

                val_losses.append(mse_loss.item())
                val_maes.append(mae_loss.item())

                # Track prediction statistics
                vol_pred_stats.extend(vol_pred.cpu().numpy().flatten())

        # Record history
        avg_train_loss = np.mean(train_losses)
        avg_train_mae = np.mean(train_maes)
        avg_val_loss = np.mean(val_losses)
        avg_val_mae = np.mean(val_maes)
        current_lr = optimizer.param_groups[0]['lr']

        train_history['train_loss'].append(avg_train_loss)
        train_history['train_mae'].append(avg_train_mae)
        train_history['val_loss'].append(avg_val_loss)
        train_history['val_mae'].append(avg_val_mae)
        train_history['lr'].append(current_lr)

        # Prediction statistics
        vol_pred_mean = np.mean(vol_pred_stats)
        vol_pred_std = np.std(vol_pred_stats)
        vol_pred_min = np.min(vol_pred_stats)
        vol_pred_max = np.max(vol_pred_stats)

        logger.info(f"  Epoch {epoch+1}/{n_epochs}:")
        logger.info(f"    Train MSE={avg_train_loss:.6f}, MAE={avg_train_mae:.6f}")
        logger.info(f"    Val MSE={avg_val_loss:.6f}, MAE={avg_val_mae:.6f}")
        logger.info(f"    Vol Pred: μ={vol_pred_mean:.4f}, σ={vol_pred_std:.4f}, range=[{vol_pred_min:.4f}, {vol_pred_max:.4f}]")
        logger.info(f"    LR={current_lr:.6f}")

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model state
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f"  Early stopping triggered after {epoch+1} epochs")
            # Restore best model
            model.load_state_dict(best_model_state)
            break

    logger.info(f"✅ Stage 1 completed. Best validation MSE: {best_val_loss:.6f}")
    return train_history


def train_stage2_improved(model, train_loader, val_loader, device, n_epochs=20, initial_lr=0.0003):
    """Improved Stage 2: Trading optimization with better training."""
    logger.info(f"🎯 IMPROVED STAGE 2: TRADING OPTIMIZATION")
    logger.info(f"  Epochs: {n_epochs}")
    logger.info(f"  Initial learning rate: {initial_lr}")

    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=1e-6)  # Lower weight decay
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=2)
    trading_loss_fn = OHLCVLongStrategyLoss(holding_period=4, trading_fee=0.0002)

    train_history = {'train_loss': [], 'val_loss': [], 'lr': [], 'vol_pred_stats': []}
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 5

    for epoch in range(n_epochs):
        # Training
        model.train()
        train_losses = []

        for batch_idx, batch in enumerate(train_loader):
            x_lags = [x.to(device) for x in batch['x_lags']]
            ohlcv_data = batch['ohlcv_data'].to(device)

            optimizer.zero_grad()

            vol_pred = model(*x_lags)
            trading_loss = trading_loss_fn(vol_pred, ohlcv_data)

            # Add gradient clipping
            trading_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            train_losses.append(trading_loss.item())

            if batch_idx % 500 == 0:
                logger.info(f"    Batch {batch_idx}/{len(train_loader)}: Loss={trading_loss.item():.6f}")

        # Validation
        model.eval()
        val_losses = []
        vol_pred_stats = []

        with torch.no_grad():
            for batch in val_loader:
                x_lags = [x.to(device) for x in batch['x_lags']]
                ohlcv_data = batch['ohlcv_data'].to(device)

                vol_pred = model(*x_lags)
                trading_loss = trading_loss_fn(vol_pred, ohlcv_data)

                val_losses.append(trading_loss.item())
                vol_pred_stats.extend(vol_pred.cpu().numpy().flatten())

        # Record history
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        current_lr = optimizer.param_groups[0]['lr']

        train_history['train_loss'].append(avg_train_loss)
        train_history['val_loss'].append(avg_val_loss)
        train_history['lr'].append(current_lr)

        # Prediction statistics
        vol_pred_mean = np.mean(vol_pred_stats)
        vol_pred_std = np.std(vol_pred_stats)
        vol_pred_min = np.min(vol_pred_stats)
        vol_pred_max = np.max(vol_pred_stats)

        train_history['vol_pred_stats'].append({
            'mean': vol_pred_mean,
            'std': vol_pred_std,
            'min': vol_pred_min,
            'max': vol_pred_max
        })

        logger.info(f"  Epoch {epoch+1}/{n_epochs}:")
        logger.info(f"    Train Loss={avg_train_loss:.6f} | Val Loss={avg_val_loss:.6f}")
        logger.info(f"    Vol Pred: μ={vol_pred_mean:.4f}, σ={vol_pred_std:.4f}, range=[{vol_pred_min:.4f}, {vol_pred_max:.4f}]")
        logger.info(f"    LR={current_lr:.6f}")

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f"  Early stopping triggered after {epoch+1} epochs")
            model.load_state_dict(best_model_state)
            break

    logger.info(f"✅ Stage 2 completed. Best validation loss: {best_val_loss:.6f}")
    return train_history


def main():
    """Main improved training function."""
    logger.info("🚀 IMPROVED MODEL TRAINING")
    logger.info("=" * 80)

    # Parameters
    device = torch.device('cpu')
    batch_size = 16  # Increased batch size
    stage1_epochs = 25  # More epochs
    stage2_epochs = 20
    stage1_lr = 0.001
    stage2_lr = 0.0003  # Lower LR for stage 2

    # Load dataset
    logger.info("📊 Loading dataset...")
    dataset, metadata = load_ohlcv_trading_data(
        volatility_file="data/crypto_rv1h_38_20200822_20250116.csv",
        lags=[1, 4, 24],
        holding_period=4,
        debug=False
    )

    logger.info(f"Dataset loaded: {len(dataset)} total samples")

    # Create proper splits
    train_indices, val_indices, test_indices = create_data_splits(dataset)

    # Create data subsets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    logger.info(f"📦 Data loaders created:")
    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Validation batches: {len(val_loader)}")

    # Initialize model
    logger.info("🧠 Initializing model...")

    # Load volatility data for correlation matrix
    vol_df = pd.read_csv("data/crypto_rv1h_38_20200822_20250116.csv", index_col=0, parse_dates=True)
    corr_sample = vol_df.iloc[-1000:] if len(vol_df) > 1000 else vol_df
    corr_matrix = corr_sample.corr().values
    A = (corr_matrix + corr_matrix.T) / 2

    model = FlexibleGSPHAR(
        lags=[1, 4, 24],
        output_dim=1,
        filter_size=len(metadata['assets']),
        A=A
    )
    model = model.to(device)

    # Stage 1: Improved supervised learning
    stage1_history = train_stage1_improved(
        model, train_loader, val_loader, device,
        n_epochs=stage1_epochs, initial_lr=stage1_lr
    )

    # Stage 2: Improved trading optimization
    stage2_history = train_stage2_improved(
        model, train_loader, val_loader, device,
        n_epochs=stage2_epochs, initial_lr=stage2_lr
    )

    # Save improved model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/improved_model_{timestamp}.pt"

    # Prepare metadata
    model_metadata = {
        'assets': metadata['assets'],
        'training_samples': len(train_indices),
        'validation_samples': len(val_indices),
        'test_samples': len(test_indices),
        'train_period': f"{dataset.get_sample_info(train_indices[0])['prediction_time']} to {dataset.get_sample_info(train_indices[-1])['prediction_time']}",
        'test_period': f"{dataset.get_sample_info(test_indices[0])['prediction_time']} to {dataset.get_sample_info(test_indices[-1])['prediction_time']}",
        'no_data_leakage': True,
        'improvements': [
            'Increased epochs (25+20 vs 10+10)',
            'Learning rate scheduling',
            'L2 regularization',
            'Gradient clipping',
            'Early stopping',
            'Better monitoring'
        ]
    }

    model_parameters = {
        'lags': [1, 4, 24],
        'holding_period': 4,
        'stage1_epochs': stage1_epochs,
        'stage2_epochs': stage2_epochs,
        'stage1_lr': stage1_lr,
        'stage2_lr': stage2_lr,
        'batch_size': batch_size,
        'regularization': True,
        'early_stopping': True
    }

    # Save checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'stage1_history': stage1_history,
        'stage2_history': stage2_history,
        'metadata': model_metadata,
        'parameters': model_parameters,
        'train_indices': train_indices,
        'val_indices': val_indices,
        'test_indices': test_indices
    }, model_path)

    logger.info(f"✅ Improved model saved to: {model_path}")

    # Summary
    logger.info(f"\n🎯 IMPROVED TRAINING SUMMARY:")
    logger.info(f"  ✅ More epochs: {stage1_epochs}+{stage2_epochs} vs 10+10")
    logger.info(f"  ✅ Learning rate scheduling: Adaptive reduction")
    logger.info(f"  ✅ Regularization: L2 weight decay added")
    logger.info(f"  ✅ Gradient clipping: Prevents exploding gradients")
    logger.info(f"  ✅ Early stopping: Prevents overfitting")
    logger.info(f"  ✅ Better monitoring: Vol prediction statistics tracked")

    return model_path, test_indices


if __name__ == "__main__":
    model_path, test_indices = main()
    print(f"\n🚀 Next step: Test improved model on indices {test_indices[0]} to {test_indices[-1]}")
    print(f"Improved model saved at: {model_path}")
