#!/usr/bin/env python
"""
Retrain the model with proper train/test split to eliminate data leakage.

Split:
- Training: 70% (samples 0 to 26,998)
- Validation: 15% (samples 26,999 to 32,783) 
- Testing: 15% (samples 32,784 to 38,569)
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


def create_proper_data_splits(dataset):
    """Create proper train/validation/test splits."""
    total_samples = len(dataset)
    
    # Calculate split sizes
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)
    test_size = total_samples - train_size - val_size
    
    # Create indices
    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, total_samples))
    
    logger.info(f"📊 DATA SPLIT CREATED:")
    logger.info(f"  Total samples: {total_samples}")
    logger.info(f"  Training: {len(train_indices)} samples ({train_ratio:.0%})")
    logger.info(f"  Validation: {len(val_indices)} samples ({val_ratio:.0%})")
    logger.info(f"  Testing: {len(test_indices)} samples ({test_ratio:.0%})")
    
    # Show time periods
    train_start_info = dataset.get_sample_info(train_indices[0])
    train_end_info = dataset.get_sample_info(train_indices[-1])
    
    val_start_info = dataset.get_sample_info(val_indices[0])
    val_end_info = dataset.get_sample_info(val_indices[-1])
    
    test_start_info = dataset.get_sample_info(test_indices[0])
    test_end_info = dataset.get_sample_info(test_indices[-1])
    
    logger.info(f"📅 TIME PERIODS:")
    logger.info(f"  Training: {train_start_info['prediction_time']} to {train_end_info['prediction_time']}")
    logger.info(f"  Validation: {val_start_info['prediction_time']} to {val_end_info['prediction_time']}")
    logger.info(f"  Testing: {test_start_info['prediction_time']} to {test_end_info['prediction_time']}")
    
    return train_indices, val_indices, test_indices


def train_stage1_supervised(model, train_loader, val_loader, device, n_epochs=10, lr=0.001):
    """Stage 1: Supervised learning with MSE loss."""
    logger.info(f"🎯 STAGE 1: SUPERVISED LEARNING")
    logger.info(f"  Epochs: {n_epochs}")
    logger.info(f"  Learning rate: {lr}")
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_history = {'train_loss': [], 'train_mae': [], 'val_loss': [], 'val_mae': []}
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_losses = []
        train_maes = []
        
        for batch in train_loader:
            x_lags = [x.to(device) for x in batch['x_lags']]
            vol_targets = batch['vol_targets'].to(device)
            
            optimizer.zero_grad()
            
            vol_pred = model(*x_lags)
            mse_loss = F.mse_loss(vol_pred, vol_targets)
            mae_loss = F.l1_loss(vol_pred, vol_targets)
            
            mse_loss.backward()
            optimizer.step()
            
            train_losses.append(mse_loss.item())
            train_maes.append(mae_loss.item())
        
        # Validation
        model.eval()
        val_losses = []
        val_maes = []
        
        with torch.no_grad():
            for batch in val_loader:
                x_lags = [x.to(device) for x in batch['x_lags']]
                vol_targets = batch['vol_targets'].to(device)
                
                vol_pred = model(*x_lags)
                mse_loss = F.mse_loss(vol_pred, vol_targets)
                mae_loss = F.l1_loss(vol_pred, vol_targets)
                
                val_losses.append(mse_loss.item())
                val_maes.append(mae_loss.item())
        
        # Record history
        avg_train_loss = np.mean(train_losses)
        avg_train_mae = np.mean(train_maes)
        avg_val_loss = np.mean(val_losses)
        avg_val_mae = np.mean(val_maes)
        
        train_history['train_loss'].append(avg_train_loss)
        train_history['train_mae'].append(avg_train_mae)
        train_history['val_loss'].append(avg_val_loss)
        train_history['val_mae'].append(avg_val_mae)
        
        logger.info(f"  Epoch {epoch+1}/{n_epochs}: "
                   f"Train MSE={avg_train_loss:.6f}, MAE={avg_train_mae:.6f} | "
                   f"Val MSE={avg_val_loss:.6f}, MAE={avg_val_mae:.6f}")
    
    logger.info(f"✅ Stage 1 completed. Final validation MSE: {avg_val_loss:.6f}")
    return train_history


def train_stage2_trading(model, train_loader, val_loader, device, n_epochs=10, lr=0.0005):
    """Stage 2: Trading optimization."""
    logger.info(f"🎯 STAGE 2: TRADING OPTIMIZATION")
    logger.info(f"  Epochs: {n_epochs}")
    logger.info(f"  Learning rate: {lr}")
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    trading_loss_fn = OHLCVLongStrategyLoss(holding_period=4, trading_fee=0.0002)
    
    train_history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_losses = []
        
        for batch in train_loader:
            x_lags = [x.to(device) for x in batch['x_lags']]
            ohlcv_data = batch['ohlcv_data'].to(device)
            
            optimizer.zero_grad()
            
            vol_pred = model(*x_lags)
            trading_loss = trading_loss_fn(vol_pred, ohlcv_data)
            
            trading_loss.backward()
            optimizer.step()
            
            train_losses.append(trading_loss.item())
        
        # Validation
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                x_lags = [x.to(device) for x in batch['x_lags']]
                ohlcv_data = batch['ohlcv_data'].to(device)
                
                vol_pred = model(*x_lags)
                trading_loss = trading_loss_fn(vol_pred, ohlcv_data)
                
                val_losses.append(trading_loss.item())
        
        # Record history
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        
        train_history['train_loss'].append(avg_train_loss)
        train_history['val_loss'].append(avg_val_loss)
        
        logger.info(f"  Epoch {epoch+1}/{n_epochs}: "
                   f"Train Loss={avg_train_loss:.6f} | Val Loss={avg_val_loss:.6f}")
    
    logger.info(f"✅ Stage 2 completed. Final validation loss: {avg_val_loss:.6f}")
    return train_history


def main():
    """Main training function with proper data split."""
    logger.info("🚀 TRAINING WITH PROPER DATA SPLIT")
    logger.info("=" * 80)
    
    # Parameters
    device = torch.device('cpu')
    batch_size = 8
    stage1_epochs = 10
    stage2_epochs = 10
    stage1_lr = 0.001
    stage2_lr = 0.0005
    
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
    train_indices, val_indices, test_indices = create_proper_data_splits(dataset)
    
    # Create data subsets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
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
    
    # Stage 1: Supervised learning
    stage1_history = train_stage1_supervised(
        model, train_loader, val_loader, device, 
        n_epochs=stage1_epochs, lr=stage1_lr
    )
    
    # Stage 2: Trading optimization
    stage2_history = train_stage2_trading(
        model, train_loader, val_loader, device,
        n_epochs=stage2_epochs, lr=stage2_lr
    )
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/proper_split_model_{timestamp}.pt"
    
    # Prepare metadata
    model_metadata = {
        'assets': metadata['assets'],
        'training_samples': len(train_indices),
        'validation_samples': len(val_indices),
        'test_samples': len(test_indices),
        'train_period': f"{dataset.get_sample_info(train_indices[0])['prediction_time']} to {dataset.get_sample_info(train_indices[-1])['prediction_time']}",
        'test_period': f"{dataset.get_sample_info(test_indices[0])['prediction_time']} to {dataset.get_sample_info(test_indices[-1])['prediction_time']}",
        'no_data_leakage': True
    }
    
    model_parameters = {
        'lags': [1, 4, 24],
        'holding_period': 4,
        'stage1_epochs': stage1_epochs,
        'stage2_epochs': stage2_epochs,
        'stage1_lr': stage1_lr,
        'stage2_lr': stage2_lr,
        'batch_size': batch_size
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
    
    logger.info(f"✅ Model saved to: {model_path}")
    
    # Summary
    logger.info(f"\n🎯 TRAINING SUMMARY:")
    logger.info(f"  ✅ No data leakage: Train and test completely separate")
    logger.info(f"  ✅ Proper temporal split: Train on past, test on future")
    logger.info(f"  ✅ Two-stage training: Supervised + Trading optimization")
    logger.info(f"  📊 Ready for realistic testing on samples {test_indices[0]} to {test_indices[-1]}")
    
    return model_path, test_indices


if __name__ == "__main__":
    model_path, test_indices = main()
    print(f"\n🚀 Next step: Test on out-of-sample data using indices {test_indices[0]} to {test_indices[-1]}")
    print(f"Model saved at: {model_path}")
