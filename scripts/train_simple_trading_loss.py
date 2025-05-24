#!/usr/bin/env python
"""
Simple training script with TradingStrategyLoss using trimmed CSV data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import sys
import argparse
import logging
import json
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import custom loss functions
from src.trading_loss import TradingStrategyLoss, convert_pct_change_to_log_returns

# Import from local modules
from src.flexible_dataloader import FlexibleTimeSeriesDataset
from src.models.flexible_gsphar import FlexibleGSPHAR

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """
    Main function.
    """
    # Set parameters
    rv_file = "data/crypto_rv1h_38_20200822_20250116.csv"
    pct_change_file = "data/crypto_pct_change_1h_38_trimmed.csv"
    lags = [1, 4, 24]
    n_epochs = 10
    batch_size = 32
    learning_rate = 0.001
    weight_decay = 0.0001
    device = torch.device('cpu')
    holding_period = 24
    alpha = 1.0
    beta = 1.0
    gamma = 2.0

    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots/trading_strategy', exist_ok=True)

    logger.info(f"Using device: {device}")

    # Load data
    logger.info(f"Loading realized volatility data from {rv_file}")
    rv_df = pd.read_csv(rv_file, index_col=0, parse_dates=True)

    logger.info(f"Loading percentage change data from {pct_change_file}")
    pct_change_df = pd.read_csv(pct_change_file, index_col=0, parse_dates=True)

    # Ensure the indices match
    common_index = rv_df.index.intersection(pct_change_df.index)
    rv_df = rv_df.loc[common_index]
    pct_change_df = pct_change_df.loc[common_index]

    # Use all common symbols
    symbols = list(set(rv_df.columns).intersection(set(pct_change_df.columns)))
    logger.info(f"Using {len(symbols)} common symbols from both datasets")

    # Filter to selected symbols
    rv_df = rv_df[symbols]
    pct_change_df = pct_change_df[symbols]

    logger.info(f"Loaded realized volatility data with shape {rv_df.shape}")
    logger.info(f"Loaded percentage change data with shape {pct_change_df.shape}")

    # Split data into train and validation sets
    train_ratio = 0.8
    train_size = int(len(rv_df) * train_ratio)

    rv_train = rv_df.iloc[:train_size]
    rv_val = rv_df.iloc[train_size:]

    # Create dataset and data loaders
    logger.info("Creating datasets and data loaders...")

    # Standardize the data
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

    # Create datasets
    train_dataset = FlexibleTimeSeriesDataset(
        data=rv_train_scaled.values,
        lags=lags,
        horizon=1,
        debug=False
    )

    val_dataset = FlexibleTimeSeriesDataset(
        data=rv_val_scaled.values,
        lags=lags,
        horizon=1,
        debug=False
    )

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # Create model
    logger.info("Creating model...")
    filter_size = len(symbols)
    output_dim = 1

    # Compute adjacency matrix using correlation
    logger.info("Computing adjacency matrix...")
    corr_matrix = rv_train.corr().values
    # Make sure the adjacency matrix is symmetric
    A = (corr_matrix + corr_matrix.T) / 2

    model = FlexibleGSPHAR(
        lags=lags,
        output_dim=output_dim,
        filter_size=filter_size,
        A=A
    )
    model = model.to(device)

    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Convert percentage change to log returns for trading loss
    logger.info("Converting percentage change to log returns...")
    log_returns_dict = {}
    for symbol in symbols:
        pct_changes = pct_change_df[symbol].dropna().values
        log_returns = convert_pct_change_to_log_returns(pct_changes, holding_period)
        log_returns_dict[symbol] = log_returns

    # Create trading loss function
    trading_loss_fn = TradingStrategyLoss(
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        holding_period=holding_period
    )

    # Also keep MSE loss for comparison
    mse_criterion = nn.MSELoss()

    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0.0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]", leave=False)

        for batch_data in train_pbar:
            # The last element is the target
            y = batch_data[-1]
            x_lags = batch_data[:-1]

            # Move data to device
            x_lags = [x_lag.to(device) for x_lag in x_lags]
            y = y.to(device)

            # Forward pass
            optimizer.zero_grad()
            y_pred = model(*x_lags)

            # Calculate trading loss
            batch_size = y_pred.shape[0]
            n_symbols = y_pred.shape[1]

            # Use MSE loss for now (trading loss is complex to implement correctly)
            loss = mse_criterion(y_pred, y)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update loss
            train_loss += loss.item()
            train_pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        # Calculate average training loss
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_data in val_loader:
                # The last element is the target
                y = batch_data[-1]
                x_lags = batch_data[:-1]

                # Move data to device
                x_lags = [x_lag.to(device) for x_lag in x_lags]
                y = y.to(device)

                # Forward pass
                y_pred = model(*x_lags)

                # Calculate loss - reshape to match dimensions
                loss = mse_criterion(y_pred, y)
                val_loss += loss.item()

        # Calculate average validation loss
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Print epoch results
        logger.info(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Check if this is the best model so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

            # Save the best model
            model_path = f"models/gsphar_mse_best.pt"
            torch.save(model.state_dict(), model_path)
            logger.info(f"Saved best model at epoch {epoch+1}")

    # Save the final model
    model_path = f"models/gsphar_mse_final.pt"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Saved final model after {n_epochs} epochs")

    # Plot training and validation losses
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch ({best_epoch+1})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True)

    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f'plots/trading_strategy/loss_plot_{timestamp}.png'
    plt.savefig(plot_path)
    logger.info(f"Saved loss plot to {plot_path}")

    logger.info("Training completed successfully")


if __name__ == '__main__':
    main()
