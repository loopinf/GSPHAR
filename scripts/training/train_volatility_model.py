#!/usr/bin/env python
"""
Complete Volatility Model Training Process

This script demonstrates the full two-stage training process:
1. Stage 1: Supervised learning for volatility prediction
2. Stage 2: Trading optimization for profit maximization
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import sys
import logging
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.flexible_gsphar import FlexibleGSPHAR
from src.data.ohlcv_trading_dataset import load_ohlcv_trading_data
from src.ohlcv_trading_loss import OHLCVLongStrategyLoss
from torch.utils.data import DataLoader, Subset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_data_splits(dataset, train_ratio=0.7, val_ratio=0.15):
    """Create proper chronological data splits."""
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)

    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, total_size))

    logger.info(f"Data splits: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")
    return train_indices, val_indices, test_indices


def train_stage1_volatility_prediction(model, train_loader, val_loader, device, n_epochs=25, lr=0.001):
    """
    Stage 1: Train model to predict volatility accurately using MSE loss.

    This stage ensures the model learns fundamental volatility patterns
    before optimizing for trading profits.
    """
    logger.info("🎯 STAGE 1: VOLATILITY PREDICTION TRAINING")
    logger.info(f"  Epochs: {n_epochs}")
    logger.info(f"  Learning rate: {lr}")
    logger.info(f"  Objective: Learn volatility patterns with MSE loss")

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=3)

    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 7

    for epoch in range(n_epochs):
        # Training
        model.train()
        train_losses = []

        for batch_idx, batch in enumerate(train_loader):
            x_lags = [x.to(device) for x in batch['x_lags']]
            vol_targets = batch['vol_targets'].to(device)

            optimizer.zero_grad()
            vol_pred = model(*x_lags)

            # MSE loss for supervised learning
            mse_loss = nn.MSELoss()(vol_pred, vol_targets)

            mse_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(mse_loss.item())

            if batch_idx % 100 == 0:
                logger.info(f"    Epoch {epoch+1}, Batch {batch_idx}: MSE={mse_loss.item():.6f}")

        # Validation
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in val_loader:
                x_lags = [x.to(device) for x in batch['x_lags']]
                vol_targets = batch['vol_targets'].to(device)

                vol_pred = model(*x_lags)
                mse_loss = nn.MSELoss()(vol_pred, vol_targets)
                val_losses.append(mse_loss.item())

        # Calculate averages
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['lr'].append(current_lr)

        logger.info(f"  Epoch {epoch+1}/{n_epochs}: Train MSE={avg_train_loss:.6f}, Val MSE={avg_val_loss:.6f}, LR={current_lr:.6f}")

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'models/best_volatility_model_stage1.pt')
            logger.info(f"    ✅ New best validation MSE: {best_val_loss:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"    Early stopping triggered after {epoch+1} epochs")
                break

    logger.info(f"✅ Stage 1 completed. Best validation MSE: {best_val_loss:.6f}")
    return history


def train_stage2_trading_optimization(model, train_loader, val_loader, device, n_epochs=20, lr=0.0003):
    """
    Stage 2: Fine-tune model for trading profit optimization.

    This stage uses the pre-trained volatility model and optimizes it
    for actual trading performance using profit-based loss.
    """
    logger.info("🎯 STAGE 2: TRADING OPTIMIZATION")
    logger.info(f"  Epochs: {n_epochs}")
    logger.info(f"  Learning rate: {lr} (lower for fine-tuning)")
    logger.info(f"  Objective: Optimize for trading profits")

    # Load best model from stage 1
    model.load_state_dict(torch.load('models/best_volatility_model_stage1.pt'))

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=2)
    trading_loss_fn = OHLCVLongStrategyLoss(holding_period=4, trading_fee=0.0002)

    history = {'train_loss': [], 'val_loss': [], 'lr': [], 'metrics': []}
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

            # Trading loss for profit optimization
            trading_loss = trading_loss_fn(vol_pred, ohlcv_data)

            trading_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            train_losses.append(trading_loss.item())

            if batch_idx % 100 == 0:
                metrics = trading_loss_fn.calculate_metrics(vol_pred, ohlcv_data)
                logger.info(f"    Epoch {epoch+1}, Batch {batch_idx}: "
                           f"Loss={trading_loss.item():.6f}, "
                           f"Fill={metrics['fill_rate']:.3f}, "
                           f"Profit={metrics['avg_profit_when_filled']:.4f}")

        # Validation
        model.eval()
        val_losses = []
        val_metrics_list = []

        with torch.no_grad():
            for batch in val_loader:
                x_lags = [x.to(device) for x in batch['x_lags']]
                ohlcv_data = batch['ohlcv_data'].to(device)

                vol_pred = model(*x_lags)
                trading_loss = trading_loss_fn(vol_pred, ohlcv_data)
                val_losses.append(trading_loss.item())

                metrics = trading_loss_fn.calculate_metrics(vol_pred, ohlcv_data)
                val_metrics_list.append(metrics)

        # Calculate averages
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        current_lr = optimizer.param_groups[0]['lr']

        # Average metrics
        avg_metrics = {}
        if val_metrics_list:
            for key in val_metrics_list[0].keys():
                avg_metrics[key] = np.mean([m[key] for m in val_metrics_list])

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['lr'].append(current_lr)
        history['metrics'].append(avg_metrics)

        logger.info(f"  Epoch {epoch+1}/{n_epochs}: "
                   f"Train Loss={avg_train_loss:.6f}, Val Loss={avg_val_loss:.6f}")
        if avg_metrics:
            logger.info(f"    Metrics: Fill={avg_metrics['fill_rate']:.3f}, "
                       f"Profit={avg_metrics['avg_profit_when_filled']:.4f}, "
                       f"Vol={avg_metrics['avg_vol_pred']:.4f}")

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'models/best_volatility_model_stage2.pt')
            logger.info(f"    ✅ New best validation loss: {best_val_loss:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"    Early stopping triggered after {epoch+1} epochs")
                break

    logger.info(f"✅ Stage 2 completed. Best validation loss: {best_val_loss:.6f}")
    return history


def main():
    """Main training function demonstrating the complete two-stage process."""
    logger.info("🚀 COMPLETE VOLATILITY MODEL TRAINING")
    logger.info("=" * 80)

    # Parameters
    device = torch.device('cpu')
    batch_size = 16
    stage1_epochs = 25  # More epochs for foundation
    stage2_epochs = 20  # Fine-tuning epochs
    stage1_lr = 0.001   # Higher LR for initial learning
    stage2_lr = 0.0003  # Lower LR for fine-tuning

    logger.info(f"Training Parameters:")
    logger.info(f"  Device: {device}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Stage 1: {stage1_epochs} epochs, LR={stage1_lr}")
    logger.info(f"  Stage 2: {stage2_epochs} epochs, LR={stage2_lr}")

    # Load dataset
    logger.info("📊 Loading dataset...")
    dataset, metadata = load_ohlcv_trading_data(
        volatility_file="data/crypto_rv1h_38_20200822_20250116.csv",
        lags=[1, 4, 24],
        holding_period=4,
        debug=False  # Use full dataset
    )

    logger.info(f"Dataset loaded: {len(dataset)} samples")
    logger.info(f"Assets: {len(metadata['assets'])}")

    # Create data splits
    train_indices, val_indices, test_indices = create_data_splits(dataset)

    # Create data subsets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    logger.info("🧠 Initializing GSPHAR model...")

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
    ).to(device)

    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Create models directory
    os.makedirs('models', exist_ok=True)

    # Stage 1: Volatility Prediction Training
    stage1_history = train_stage1_volatility_prediction(
        model, train_loader, val_loader, device,
        n_epochs=stage1_epochs, lr=stage1_lr
    )

    # Stage 2: Trading Optimization
    stage2_history = train_stage2_trading_optimization(
        model, train_loader, val_loader, device,
        n_epochs=stage2_epochs, lr=stage2_lr
    )

    # Save final model with complete metadata
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = f"models/volatility_model_complete_{timestamp}.pt"

    torch.save({
        'model_state_dict': model.state_dict(),
        'stage1_history': stage1_history,
        'stage2_history': stage2_history,
        'metadata': {
            'assets': metadata['assets'],
            'training_samples': len(train_indices),
            'validation_samples': len(val_indices),
            'test_samples': len(test_indices),
            'two_stage_training': True,
            'stage1_objective': 'Volatility prediction (MSE)',
            'stage2_objective': 'Trading profit optimization'
        },
        'parameters': {
            'lags': [1, 4, 24],
            'holding_period': 4,
            'stage1_epochs': stage1_epochs,
            'stage2_epochs': stage2_epochs,
            'stage1_lr': stage1_lr,
            'stage2_lr': stage2_lr,
            'batch_size': batch_size
        }
    }, final_model_path)

    logger.info(f"✅ Complete model saved to: {final_model_path}")

    # Training Summary
    logger.info("\n" + "="*80)
    logger.info("🎯 TRAINING SUMMARY")
    logger.info("="*80)

    logger.info(f"Stage 1 (Volatility Prediction):")
    logger.info(f"  Best validation MSE: {min(stage1_history['val_loss']):.6f}")

    logger.info(f"Stage 2 (Trading Optimization):")
    logger.info(f"  Best validation loss: {min(stage2_history['val_loss']):.6f}")

    if stage2_history['metrics']:
        final_metrics = stage2_history['metrics'][-1]
        logger.info(f"  Final trading metrics:")
        for key, value in final_metrics.items():
            logger.info(f"    {key}: {value:.4f}")

    return final_model_path, test_indices


if __name__ == "__main__":
    model_path, test_indices = main()
    print(f"\n✅ Training completed!")
    print(f"Model saved: {model_path}")
    print(f"Test indices: {test_indices[0]} to {test_indices[-1]}")
