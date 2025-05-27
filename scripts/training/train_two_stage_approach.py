#!/usr/bin/env python
"""
Two-Stage Training Approach for OHLCV Trading Strategy.

Stage 1: Supervised Learning - Train model to predict volatility accurately (MSE loss)
Stage 2: Trading Optimization - Fine-tune with trading profit loss

This approach ensures the model learns volatility patterns before optimizing for trading profits.
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
from tqdm import tqdm

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.ohlcv_trading_dataset import load_ohlcv_trading_data, create_ohlcv_dataloaders
from src.ohlcv_trading_loss import OHLCVLongStrategyLoss
from src.models.flexible_gsphar import FlexibleGSPHAR

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SmallDataSubset(torch.utils.data.Dataset):
    """Wrapper to create a small subset of the dataset."""
    
    def __init__(self, original_dataset, subset_size=1000):
        self.original_dataset = original_dataset
        self.subset_size = min(subset_size, len(original_dataset))
        
        total_size = len(original_dataset)
        step = max(1, total_size // subset_size)
        self.indices = list(range(0, total_size, step))[:subset_size]
        
        logger.info(f"Created subset: {len(self.indices)} samples from {total_size} total")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        return self.original_dataset[original_idx]


def train_stage1_supervised(model, train_loader, val_loader, device, n_epochs=10, learning_rate=0.001):
    """
    Stage 1: Supervised learning to predict volatility accurately.
    
    Args:
        model: GSPHAR model
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Training device
        n_epochs: Number of epochs for stage 1
        learning_rate: Learning rate for stage 1
        
    Returns:
        dict: Training history for stage 1
    """
    logger.info(f"🎯 STAGE 1: SUPERVISED VOLATILITY PREDICTION")
    logger.info(f"Epochs: {n_epochs}, Learning Rate: {learning_rate}")
    logger.info("=" * 60)
    
    # Create optimizer for stage 1
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_mse': [],
        'val_mse': [],
        'train_mae': [],
        'val_mae': []
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(n_epochs):
        logger.info(f"\nEpoch {epoch+1}/{n_epochs}")
        logger.info("-" * 40)
        
        # Training phase
        model.train()
        train_losses = []
        train_mses = []
        train_maes = []
        
        train_pbar = tqdm(train_loader, desc=f"Stage 1 Train {epoch+1}", leave=False)
        
        for batch_idx, batch in enumerate(train_pbar):
            x_lags = batch['x_lags']
            vol_targets = batch['vol_targets']
            
            # Move to device
            x_lags = [x.to(device) for x in x_lags]
            vol_targets = vol_targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            vol_pred = model(*x_lags)
            
            # MSE Loss for supervised learning
            mse_loss = F.mse_loss(vol_pred, vol_targets)
            mae_loss = F.l1_loss(vol_pred, vol_targets)
            
            # Backward pass
            if not torch.isnan(mse_loss) and not torch.isinf(mse_loss):
                mse_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_losses.append(mse_loss.item())
                train_mses.append(mse_loss.item())
                train_maes.append(mae_loss.item())
                
                # Update progress bar
                train_pbar.set_postfix({
                    'MSE': f"{mse_loss.item():.6f}",
                    'MAE': f"{mae_loss.item():.6f}",
                    'Vol_Range': f"{vol_pred.min().item():.4f}-{vol_pred.max().item():.4f}"
                })
        
        # Validation phase
        model.eval()
        val_losses = []
        val_mses = []
        val_maes = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Stage 1 Val {epoch+1}", leave=False)
            
            for batch in val_pbar:
                x_lags = batch['x_lags']
                vol_targets = batch['vol_targets']
                
                # Move to device
                x_lags = [x.to(device) for x in x_lags]
                vol_targets = vol_targets.to(device)
                
                # Forward pass
                vol_pred = model(*x_lags)
                
                # Calculate losses
                mse_loss = F.mse_loss(vol_pred, vol_targets)
                mae_loss = F.l1_loss(vol_pred, vol_targets)
                
                if not torch.isnan(mse_loss) and not torch.isinf(mse_loss):
                    val_losses.append(mse_loss.item())
                    val_mses.append(mse_loss.item())
                    val_maes.append(mae_loss.item())
                
                val_pbar.set_postfix({
                    'MSE': f"{mse_loss.item():.6f}",
                    'MAE': f"{mae_loss.item():.6f}"
                })
        
        # Calculate epoch averages
        avg_train_loss = np.mean(train_losses) if train_losses else float('inf')
        avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
        avg_train_mse = np.mean(train_mses) if train_mses else float('inf')
        avg_val_mse = np.mean(val_mses) if val_mses else float('inf')
        avg_train_mae = np.mean(train_maes) if train_maes else float('inf')
        avg_val_mae = np.mean(val_maes) if val_maes else float('inf')
        
        # Store history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_mse'].append(avg_train_mse)
        history['val_mse'].append(avg_val_mse)
        history['train_mae'].append(avg_train_mae)
        history['val_mae'].append(avg_val_mae)
        
        # Log epoch results
        logger.info(f"Train MSE: {avg_train_mse:.6f}, Val MSE: {avg_val_mse:.6f}")
        logger.info(f"Train MAE: {avg_train_mae:.6f}, Val MAE: {avg_val_mae:.6f}")
        
        # Check for improvement
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            logger.info(f"✅ New best validation MSE: {best_val_loss:.6f}")
    
    logger.info(f"\n🎯 STAGE 1 COMPLETED")
    logger.info(f"Best Validation MSE: {best_val_loss:.6f}")
    
    return history


def train_stage2_trading(model, train_loader, val_loader, device, n_epochs=10, learning_rate=0.0005):
    """
    Stage 2: Trading optimization using profit-based loss.
    
    Args:
        model: Pre-trained GSPHAR model from stage 1
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Training device
        n_epochs: Number of epochs for stage 2
        learning_rate: Learning rate for stage 2 (usually lower)
        
    Returns:
        dict: Training history for stage 2
    """
    logger.info(f"\n🎯 STAGE 2: TRADING PROFIT OPTIMIZATION")
    logger.info(f"Epochs: {n_epochs}, Learning Rate: {learning_rate}")
    logger.info("=" * 60)
    
    # Create optimizer for stage 2 (lower learning rate for fine-tuning)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create trading loss function
    trading_loss_fn = OHLCVLongStrategyLoss(holding_period=4, trading_fee=0.0002)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_metrics': [],
        'val_metrics': []
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(n_epochs):
        logger.info(f"\nEpoch {epoch+1}/{n_epochs}")
        logger.info("-" * 40)
        
        # Training phase
        model.train()
        train_losses = []
        train_metrics_list = []
        
        train_pbar = tqdm(train_loader, desc=f"Stage 2 Train {epoch+1}", leave=False)
        
        for batch_idx, batch in enumerate(train_pbar):
            x_lags = batch['x_lags']
            vol_targets = batch['vol_targets']
            ohlcv_data = batch['ohlcv_data']
            
            # Move to device
            x_lags = [x.to(device) for x in x_lags]
            vol_targets = vol_targets.to(device)
            ohlcv_data = ohlcv_data.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            vol_pred = model(*x_lags)
            
            # Trading loss
            trading_loss = trading_loss_fn(vol_pred, ohlcv_data)
            
            # Backward pass
            if not torch.isnan(trading_loss) and not torch.isinf(trading_loss):
                trading_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_losses.append(trading_loss.item())
                
                # Calculate metrics
                metrics = trading_loss_fn.calculate_metrics(vol_pred, ohlcv_data)
                train_metrics_list.append(metrics)
                
                # Update progress bar
                train_pbar.set_postfix({
                    'Loss': f"{trading_loss.item():.6f}",
                    'Fill': f"{metrics['fill_rate']:.3f}",
                    'Profit': f"{metrics['avg_profit_when_filled']:.4f}",
                    'Vol': f"{metrics['avg_vol_pred']:.4f}"
                })
        
        # Validation phase
        model.eval()
        val_losses = []
        val_metrics_list = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Stage 2 Val {epoch+1}", leave=False)
            
            for batch in val_pbar:
                x_lags = batch['x_lags']
                vol_targets = batch['vol_targets']
                ohlcv_data = batch['ohlcv_data']
                
                # Move to device
                x_lags = [x.to(device) for x in x_lags]
                vol_targets = vol_targets.to(device)
                ohlcv_data = ohlcv_data.to(device)
                
                # Forward pass
                vol_pred = model(*x_lags)
                trading_loss = trading_loss_fn(vol_pred, ohlcv_data)
                
                if not torch.isnan(trading_loss) and not torch.isinf(trading_loss):
                    val_losses.append(trading_loss.item())
                    
                    # Calculate metrics
                    metrics = trading_loss_fn.calculate_metrics(vol_pred, ohlcv_data)
                    val_metrics_list.append(metrics)
                
                val_pbar.set_postfix({
                    'Loss': f"{trading_loss.item():.6f}",
                    'Fill': f"{metrics['fill_rate']:.3f}",
                    'Profit': f"{metrics['avg_profit_when_filled']:.4f}"
                })
        
        # Calculate epoch averages
        avg_train_loss = np.mean(train_losses) if train_losses else float('inf')
        avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
        
        # Average metrics
        avg_train_metrics = {}
        avg_val_metrics = {}
        
        if train_metrics_list:
            for key in train_metrics_list[0].keys():
                avg_train_metrics[key] = np.mean([m[key] for m in train_metrics_list])
        
        if val_metrics_list:
            for key in val_metrics_list[0].keys():
                avg_val_metrics[key] = np.mean([m[key] for m in val_metrics_list])
        
        # Store history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_metrics'].append(avg_train_metrics)
        history['val_metrics'].append(avg_val_metrics)
        
        # Log epoch results
        logger.info(f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        if avg_train_metrics:
            logger.info(f"Train: Fill={avg_train_metrics['fill_rate']:.3f}, "
                       f"Profit={avg_train_metrics['avg_profit_when_filled']:.4f}, "
                       f"Vol={avg_train_metrics['avg_vol_pred']:.4f}")
        
        if avg_val_metrics:
            logger.info(f"Val: Fill={avg_val_metrics['fill_rate']:.3f}, "
                       f"Profit={avg_val_metrics['avg_profit_when_filled']:.4f}, "
                       f"Vol={avg_val_metrics['avg_vol_pred']:.4f}")
        
        # Check for improvement
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            logger.info(f"✅ New best validation loss: {best_val_loss:.6f}")
    
    logger.info(f"\n🎯 STAGE 2 COMPLETED")
    logger.info(f"Best Validation Loss: {best_val_loss:.6f}")
    
    return history


def main():
    """
    Main function for two-stage training approach.
    """
    logger.info("🎯 TWO-STAGE TRAINING APPROACH")
    logger.info("=" * 80)
    
    # Parameters
    volatility_file = "data/crypto_rv1h_38_20200822_20250116.csv"
    lags = [1, 4, 24]
    holding_period = 4
    subset_size = 500  # Small dataset for testing
    batch_size = 8
    device = torch.device('cpu')
    
    # Stage parameters
    stage1_epochs = 10
    stage1_lr = 0.001
    stage2_epochs = 10
    stage2_lr = 0.0005  # Lower learning rate for fine-tuning
    
    logger.info(f"Parameters:")
    logger.info(f"  Subset size: {subset_size}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Stage 1: {stage1_epochs} epochs, LR={stage1_lr}")
    logger.info(f"  Stage 2: {stage2_epochs} epochs, LR={stage2_lr}")
    
    # Load dataset
    logger.info("Loading OHLCV trading dataset...")
    full_dataset, metadata = load_ohlcv_trading_data(
        volatility_file=volatility_file,
        lags=lags,
        holding_period=holding_period,
        debug=False
    )
    
    # Create small subset
    small_dataset = SmallDataSubset(full_dataset, subset_size=subset_size)
    
    # Create dataloaders
    train_loader, val_loader, split_info = create_ohlcv_dataloaders(
        small_dataset, 
        train_ratio=0.8, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    logger.info(f"Data split: {split_info}")
    
    # Create model
    logger.info("Creating GSPHAR model...")
    
    # Load volatility data for correlation matrix
    vol_df = pd.read_csv(volatility_file, index_col=0, parse_dates=True)
    corr_sample = vol_df.iloc[-1000:] if len(vol_df) > 1000 else vol_df
    corr_matrix = corr_sample.corr().values
    A = (corr_matrix + corr_matrix.T) / 2
    
    model = FlexibleGSPHAR(
        lags=lags,
        output_dim=1,
        filter_size=len(metadata['assets']),
        A=A
    )
    model = model.to(device)
    
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Stage 1: Supervised Learning
    stage1_history = train_stage1_supervised(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        n_epochs=stage1_epochs,
        learning_rate=stage1_lr
    )
    
    # Stage 2: Trading Optimization
    stage2_history = train_stage2_trading(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        n_epochs=stage2_epochs,
        learning_rate=stage2_lr
    )
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("TWO-STAGE TRAINING COMPLETED")
    logger.info("="*80)
    
    logger.info(f"Stage 1 (Supervised):")
    logger.info(f"  Best MSE: {min(stage1_history['val_mse']):.6f}")
    logger.info(f"  Best MAE: {min(stage1_history['val_mae']):.6f}")
    
    logger.info(f"Stage 2 (Trading):")
    logger.info(f"  Best Loss: {min(stage2_history['val_loss']):.6f}")
    
    if stage2_history['val_metrics']:
        final_metrics = stage2_history['val_metrics'][-1]
        logger.info(f"  Final Metrics:")
        for key, value in final_metrics.items():
            logger.info(f"    {key}: {value:.4f}")
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/two_stage_model_{timestamp}.pt"
    os.makedirs("models", exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'stage1_history': stage1_history,
        'stage2_history': stage2_history,
        'metadata': metadata,
        'parameters': {
            'lags': lags,
            'holding_period': holding_period,
            'stage1_epochs': stage1_epochs,
            'stage2_epochs': stage2_epochs,
            'stage1_lr': stage1_lr,
            'stage2_lr': stage2_lr
        }
    }, model_path)
    
    logger.info(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
