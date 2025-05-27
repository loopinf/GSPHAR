#!/usr/bin/env python
"""
Train GSPHAR model with OHLCV-based long strategy loss function.

This script trains the model to maximize actual trading profits using:
1. OHLCV data for accurate order fill detection
2. 4-hour holding period
3. Long strategy (buy on predicted volatility drops)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import sys
import logging
from datetime import datetime
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.ohlcv_trading_dataset import load_ohlcv_trading_data, create_ohlcv_dataloaders
from src.ohlcv_trading_loss import OHLCVLongStrategyLoss, OHLCVAdvancedLongLoss
from src.models.flexible_gsphar import FlexibleGSPHAR

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_ohlcv_model(model, train_loader, val_loader, loss_fn, optimizer, device, 
                     n_epochs=10, patience=3, save_dir="models/ohlcv"):
    """
    Train model with OHLCV-based loss function.
    
    Args:
        model: GSPHAR model
        train_loader: Training data loader
        val_loader: Validation data loader
        loss_fn: OHLCV loss function
        optimizer: Optimizer
        device: Training device
        n_epochs: Number of epochs
        patience: Early stopping patience
        save_dir: Directory to save models
        
    Returns:
        dict: Training history
    """
    os.makedirs(save_dir, exist_ok=True)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_metrics': [],
        'val_metrics': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    logger.info(f"Starting training for {n_epochs} epochs...")
    
    for epoch in range(n_epochs):
        # Training phase
        model.train()
        train_losses = []
        train_metrics_list = []
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]", leave=False)
        
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
            
            # Calculate loss
            loss = loss_fn(vol_pred, ohlcv_data)
            
            # Backward pass
            if not torch.isnan(loss) and not torch.isinf(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_losses.append(loss.item())
                
                # Calculate metrics (no gradients)
                if hasattr(loss_fn, 'calculate_metrics'):
                    metrics = loss_fn.calculate_metrics(vol_pred, ohlcv_data)
                    train_metrics_list.append(metrics)
                
                # Update progress bar
                train_pbar.set_postfix({
                    'loss': f"{loss.item():.6f}",
                    'avg_loss': f"{np.mean(train_losses):.6f}"
                })
        
        # Validation phase
        model.eval()
        val_losses = []
        val_metrics_list = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Val]", leave=False)
            
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
                loss = loss_fn(vol_pred, ohlcv_data)
                
                if not torch.isnan(loss) and not torch.isinf(loss):
                    val_losses.append(loss.item())
                    
                    # Calculate metrics
                    if hasattr(loss_fn, 'calculate_metrics'):
                        metrics = loss_fn.calculate_metrics(vol_pred, ohlcv_data)
                        val_metrics_list.append(metrics)
                
                val_pbar.set_postfix({
                    'loss': f"{loss.item():.6f}",
                    'avg_loss': f"{np.mean(val_losses):.6f}"
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
        logger.info(f"Epoch {epoch+1}/{n_epochs}:")
        logger.info(f"  Train Loss: {avg_train_loss:.6f}")
        logger.info(f"  Val Loss: {avg_val_loss:.6f}")
        
        if avg_train_metrics:
            logger.info(f"  Train Metrics:")
            for key, value in avg_train_metrics.items():
                logger.info(f"    {key}: {value:.4f}")
        
        if avg_val_metrics:
            logger.info(f"  Val Metrics:")
            for key, value in avg_val_metrics.items():
                logger.info(f"    {key}: {value:.4f}")
        
        # Early stopping and model saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save best model
            model_path = os.path.join(save_dir, "best_ohlcv_long_model.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': avg_val_loss,
                'history': history
            }, model_path)
            logger.info(f"  Saved best model: {model_path}")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    return history


def main():
    """
    Main training function.
    """
    logger.info("Starting OHLCV Long Strategy Training")
    logger.info("=" * 60)
    
    # Parameters
    volatility_file = "data/crypto_rv1h_38_20200822_20250116.csv"
    lags = [1, 4, 24]
    holding_period = 4  # 4 hours
    n_epochs = 15
    batch_size = 32
    learning_rate = 0.001
    weight_decay = 0.0001
    device = torch.device('cpu')  # Use CPU as preferred
    
    logger.info(f"Parameters:")
    logger.info(f"  Holding period: {holding_period} hours")
    logger.info(f"  Lags: {lags}")
    logger.info(f"  Epochs: {n_epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Device: {device}")
    
    # Load dataset
    logger.info("Loading OHLCV trading dataset...")
    dataset, metadata = load_ohlcv_trading_data(
        volatility_file=volatility_file,
        lags=lags,
        holding_period=holding_period,
        debug=True
    )
    
    logger.info(f"Dataset metadata: {metadata}")
    
    # Create dataloaders
    train_loader, val_loader, split_info = create_ohlcv_dataloaders(
        dataset, 
        train_ratio=0.8, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    logger.info(f"Data split: {split_info}")
    
    # Create model
    logger.info("Creating GSPHAR model...")
    
    # Load volatility data to get correlation matrix
    vol_df = pd.read_csv(volatility_file, index_col=0, parse_dates=True)
    
    # Use subset of data for correlation calculation (more stable)
    corr_sample = vol_df.iloc[-5000:] if len(vol_df) > 5000 else vol_df
    corr_matrix = corr_sample.corr().values
    A = (corr_matrix + corr_matrix.T) / 2  # Ensure symmetry
    
    filter_size = len(metadata['assets'])
    output_dim = 1
    
    model = FlexibleGSPHAR(
        lags=lags,
        output_dim=output_dim,
        filter_size=filter_size,
        A=A
    )
    model = model.to(device)
    
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create loss function and optimizer
    logger.info("Setting up loss function and optimizer...")
    
    # Try simple loss first
    loss_fn = OHLCVLongStrategyLoss(holding_period=holding_period)
    
    optimizer = optim.Adam(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay
    )
    
    # Train model
    logger.info("Starting training...")
    history = train_ohlcv_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        n_epochs=n_epochs,
        patience=5
    )
    
    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/ohlcv_long_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save training history
    history_df = pd.DataFrame({
        'epoch': range(len(history['train_loss'])),
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss']
    })
    history_df.to_csv(f"{results_dir}/training_history.csv", index=False)
    
    # Save final model
    final_model_path = f"{results_dir}/final_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'metadata': metadata,
        'parameters': {
            'lags': lags,
            'holding_period': holding_period,
            'n_epochs': n_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate
        }
    }, final_model_path)
    
    logger.info("Training completed!")
    logger.info(f"Results saved to: {results_dir}")
    logger.info(f"Best validation loss: {min(history['val_loss']):.6f}")
    
    # Print final metrics
    if history['val_metrics']:
        final_metrics = history['val_metrics'][-1]
        logger.info("Final validation metrics:")
        for key, value in final_metrics.items():
            logger.info(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
