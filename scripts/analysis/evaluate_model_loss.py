#!/usr/bin/env python
"""
Evaluate the trained model and show actual loss values.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import sys
import logging
from sklearn.preprocessing import StandardScaler

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.flexible_dataloader import FlexibleTimeSeriesDataset
from src.models.flexible_gsphar import FlexibleGSPHAR
from src.trading_loss import TradingStrategyLoss, convert_pct_change_to_log_returns

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Evaluate the trained model and show loss values."""
    
    # Load data (same as training)
    rv_file = "data/crypto_rv1h_38_20200822_20250116.csv"
    pct_change_file = "data/crypto_pct_change_1h_38_trimmed.csv"
    lags = [1, 4, 24]
    batch_size = 32
    device = torch.device('cpu')
    holding_period = 24
    
    logger.info("Loading data...")
    rv_df = pd.read_csv(rv_file, index_col=0, parse_dates=True)
    pct_change_df = pd.read_csv(pct_change_file, index_col=0, parse_dates=True)
    
    # Ensure the indices match
    common_index = rv_df.index.intersection(pct_change_df.index)
    rv_df = rv_df.loc[common_index]
    pct_change_df = pct_change_df.loc[common_index]
    
    # Use all common symbols
    symbols = list(set(rv_df.columns).intersection(set(pct_change_df.columns)))
    rv_df = rv_df[symbols]
    pct_change_df = pct_change_df[symbols]
    
    logger.info(f"Data shape: {rv_df.shape}")
    
    # Split data (same as training)
    train_ratio = 0.8
    train_size = int(len(rv_df) * train_ratio)
    
    rv_train = rv_df.iloc[:train_size]
    rv_val = rv_df.iloc[train_size:]
    
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
    
    # Create validation dataset
    val_dataset = FlexibleTimeSeriesDataset(
        data=rv_val_scaled.values,
        lags=lags,
        horizon=1,
        debug=False
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    # Create model architecture (same as training)
    filter_size = len(symbols)
    output_dim = 1
    
    # Compute adjacency matrix
    corr_matrix = rv_train.corr().values
    A = (corr_matrix + corr_matrix.T) / 2
    
    model = FlexibleGSPHAR(
        lags=lags,
        output_dim=output_dim,
        filter_size=filter_size,
        A=A
    )
    
    # Load the trained model
    model_path = "models/gsphar_trading_loss_best.pt"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info(f"Loaded model from {model_path}")
    else:
        logger.error(f"Model file not found: {model_path}")
        return
    
    model = model.to(device)
    model.eval()
    
    # Create loss functions
    mse_criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss()
    
    # Evaluate on validation set
    total_mse_loss = 0.0
    total_mae_loss = 0.0
    total_samples = 0
    
    predictions = []
    targets = []
    
    logger.info("Evaluating model...")
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(val_loader):
            # The last element is the target
            y = batch_data[-1]
            x_lags = batch_data[:-1]
            
            # Move data to device
            x_lags = [x_lag.to(device) for x_lag in x_lags]
            y = y.to(device)
            
            # Forward pass
            y_pred = model(*x_lags)
            
            # Calculate losses
            mse_loss = mse_criterion(y_pred, y)
            mae_loss = mae_criterion(y_pred, y)
            
            total_mse_loss += mse_loss.item() * y.size(0)
            total_mae_loss += mae_loss.item() * y.size(0)
            total_samples += y.size(0)
            
            # Store predictions and targets for analysis
            predictions.append(y_pred.cpu().numpy())
            targets.append(y.cpu().numpy())
            
            if batch_idx < 5:  # Show first few batches
                logger.info(f"Batch {batch_idx+1}: MSE={mse_loss.item():.6f}, MAE={mae_loss.item():.6f}")
    
    # Calculate average losses
    avg_mse_loss = total_mse_loss / total_samples
    avg_mae_loss = total_mae_loss / total_samples
    rmse_loss = np.sqrt(avg_mse_loss)
    
    logger.info("="*50)
    logger.info("FINAL EVALUATION RESULTS:")
    logger.info(f"Average MSE Loss: {avg_mse_loss:.6f}")
    logger.info(f"Average MAE Loss: {avg_mae_loss:.6f}")
    logger.info(f"RMSE Loss: {rmse_loss:.6f}")
    logger.info("="*50)
    
    # Convert predictions and targets to numpy arrays
    all_predictions = np.concatenate(predictions, axis=0)
    all_targets = np.concatenate(targets, axis=0)
    
    # Calculate some statistics
    pred_mean = np.mean(all_predictions)
    pred_std = np.std(all_predictions)
    target_mean = np.mean(all_targets)
    target_std = np.std(all_targets)
    
    logger.info("PREDICTION STATISTICS:")
    logger.info(f"Predictions - Mean: {pred_mean:.6f}, Std: {pred_std:.6f}")
    logger.info(f"Targets - Mean: {target_mean:.6f}, Std: {target_std:.6f}")
    logger.info(f"Prediction Range: [{np.min(all_predictions):.6f}, {np.max(all_predictions):.6f}]")
    logger.info(f"Target Range: [{np.min(all_targets):.6f}, {np.max(all_targets):.6f}]")
    
    # Calculate correlation
    correlation = np.corrcoef(all_predictions.flatten(), all_targets.flatten())[0, 1]
    logger.info(f"Prediction-Target Correlation: {correlation:.6f}")
    
    # Show some sample predictions vs targets
    logger.info("\nSAMPLE PREDICTIONS vs TARGETS (first 10 samples, first symbol):")
    for i in range(min(10, len(all_predictions))):
        pred_val = all_predictions[i, 0, 0]  # First symbol, first output
        target_val = all_targets[i, 0, 0]   # First symbol, first output
        logger.info(f"Sample {i+1}: Pred={pred_val:.6f}, Target={target_val:.6f}, Diff={abs(pred_val-target_val):.6f}")


if __name__ == '__main__':
    main()
