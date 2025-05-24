#!/usr/bin/env python
"""
Small data test for OHLCV long strategy training.

This script trains on a small subset of data to quickly test the implementation
and see how the loss function behaves.
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

        # Take evenly spaced samples across the dataset
        total_size = len(original_dataset)
        step = max(1, total_size // subset_size)
        self.indices = list(range(0, total_size, step))[:subset_size]

        logger.info(f"Created subset: {len(self.indices)} samples from {total_size} total")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        return self.original_dataset[original_idx]


def train_small_test(model, train_loader, val_loader, loss_fn, optimizer, device, n_epochs=5):
    """
    Train model on small dataset with detailed logging.
    """
    logger.info(f"Starting small data training for {n_epochs} epochs...")

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_metrics': [],
        'val_metrics': []
    }

    for epoch in range(n_epochs):
        logger.info(f"\n{'='*50}")
        logger.info(f"EPOCH {epoch+1}/{n_epochs}")
        logger.info(f"{'='*50}")

        # Training phase
        model.train()
        train_losses = []
        train_metrics_list = []

        logger.info("Training phase:")
        for batch_idx, batch in enumerate(train_loader):
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

            # Log detailed info for first few batches
            if batch_idx < 3:
                logger.info(f"  Batch {batch_idx+1}:")
                logger.info(f"    Vol pred range: {vol_pred.min().item():.4f} to {vol_pred.max().item():.4f}")
                logger.info(f"    Vol pred mean: {vol_pred.mean().item():.4f}")
                logger.info(f"    Loss: {loss.item():.6f}")

                # Calculate metrics
                metrics = loss_fn.calculate_metrics(vol_pred, ohlcv_data)
                logger.info(f"    Fill rate: {metrics['fill_rate']:.3f}")
                logger.info(f"    Avg profit when filled: {metrics['avg_profit_when_filled']:.4f}")
                logger.info(f"    Expected profit: {metrics['expected_profit']:.4f}")

            # Backward pass
            if not torch.isnan(loss) and not torch.isinf(loss):
                loss.backward()

                # Check gradients
                total_grad_norm = 0
                for param in model.parameters():
                    if param.grad is not None:
                        total_grad_norm += param.grad.data.norm(2).item() ** 2
                total_grad_norm = total_grad_norm ** 0.5

                if batch_idx < 3:
                    logger.info(f"    Gradient norm: {total_grad_norm:.6f}")

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_losses.append(loss.item())

                # Calculate metrics
                metrics = loss_fn.calculate_metrics(vol_pred, ohlcv_data)
                train_metrics_list.append(metrics)
            else:
                logger.warning(f"  Batch {batch_idx+1}: Invalid loss {loss.item()}")

        # Validation phase
        model.eval()
        val_losses = []
        val_metrics_list = []

        logger.info("Validation phase:")
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
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

                if batch_idx < 2:
                    logger.info(f"  Val Batch {batch_idx+1}:")
                    logger.info(f"    Vol pred range: {vol_pred.min().item():.4f} to {vol_pred.max().item():.4f}")
                    logger.info(f"    Loss: {loss.item():.6f}")

                if not torch.isnan(loss) and not torch.isinf(loss):
                    val_losses.append(loss.item())

                    # Calculate metrics
                    metrics = loss_fn.calculate_metrics(vol_pred, ohlcv_data)
                    val_metrics_list.append(metrics)

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

        # Log epoch summary
        logger.info(f"\nEPOCH {epoch+1} SUMMARY:")
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

    return history


def main():
    """
    Main function for small data test.
    """
    logger.info("🎯 OHLCV LONG STRATEGY - SMALL DATA TEST")
    logger.info("=" * 60)

    # Parameters
    volatility_file = "data/crypto_rv1h_38_20200822_20250116.csv"
    lags = [1, 4, 24]
    holding_period = 4  # 4 hours
    n_epochs = 5  # Small number for quick test
    batch_size = 8   # Small batch size
    learning_rate = 0.001
    device = torch.device('cpu')
    subset_size = 500  # Use small dataset for comparison

    logger.info(f"Parameters:")
    logger.info(f"  Subset size: {subset_size} samples")
    logger.info(f"  Holding period: {holding_period} hours")
    logger.info(f"  Epochs: {n_epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {learning_rate}")

    # Load dataset
    logger.info("Loading OHLCV trading dataset...")
    full_dataset, metadata = load_ohlcv_trading_data(
        volatility_file=volatility_file,
        lags=lags,
        holding_period=holding_period,
        debug=False  # Reduce logging for small test
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

    logger.info(f"Small dataset split: {split_info}")

    # Create model
    logger.info("Creating GSPHAR model...")

    # Load volatility data for correlation matrix
    vol_df = pd.read_csv(volatility_file, index_col=0, parse_dates=True)
    corr_sample = vol_df.iloc[-1000:] if len(vol_df) > 1000 else vol_df
    corr_matrix = corr_sample.corr().values
    A = (corr_matrix + corr_matrix.T) / 2

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

    # Create loss function and optimizer with Binance futures fees
    loss_fn = OHLCVLongStrategyLoss(holding_period=holding_period, trading_fee=0.0002)  # 0.02% Binance futures maker fee
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    logger.info("Loss function: OHLCVLongStrategyLoss with Binance futures fees (0.04% total)")

    # Train model
    logger.info("Starting training...")
    history = train_small_test(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        n_epochs=n_epochs
    )

    # Final summary
    logger.info("\n" + "="*60)
    logger.info("FINAL RESULTS")
    logger.info("="*60)

    logger.info(f"Training completed!")
    logger.info(f"Final train loss: {history['train_loss'][-1]:.6f}")
    logger.info(f"Final val loss: {history['val_loss'][-1]:.6f}")

    if history['val_metrics']:
        final_metrics = history['val_metrics'][-1]
        logger.info("Final validation metrics:")
        for key, value in final_metrics.items():
            logger.info(f"  {key}: {value:.4f}")

    # Show loss progression
    logger.info("\nLoss progression:")
    for i, (train_loss, val_loss) in enumerate(zip(history['train_loss'], history['val_loss'])):
        logger.info(f"  Epoch {i+1}: Train={train_loss:.6f}, Val={val_loss:.6f}")


if __name__ == "__main__":
    main()
