#!/usr/bin/env python
"""
Complete implementation of training with actual TradingStrategyLoss.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import sys
import logging
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.trading_loss import TradingStrategyLoss, convert_pct_change_to_log_returns
from src.models.flexible_gsphar import FlexibleGSPHAR

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradingDataset(torch.utils.data.Dataset):
    """
    Dataset that provides both volatility targets and log returns for TradingStrategyLoss.
    """
    def __init__(self, volatility_data, price_data, lags, holding_period, debug=False):
        self.volatility_data = volatility_data  # DataFrame
        self.price_data = price_data  # DataFrame
        self.lags = lags
        self.holding_period = holding_period
        self.debug = debug

        # Convert price data to log returns
        self.log_returns = self._calculate_log_returns()

        # Calculate valid indices (need future data for holding period)
        self.valid_indices = self._get_valid_indices()

        if self.debug:
            logger.info(f"Dataset created with {len(self.valid_indices)} valid samples")
            logger.info(f"Volatility data shape: {self.volatility_data.shape}")
            logger.info(f"Log returns shape: {self.log_returns.shape}")

    def _calculate_log_returns(self):
        """Calculate log returns from price data."""
        pct_changes = self.price_data.pct_change().fillna(0)
        # Clip extreme values to avoid log issues
        pct_changes = pct_changes.clip(-0.99, 10)
        log_returns = np.log(1 + pct_changes)
        return log_returns

    def _get_valid_indices(self):
        """Get indices where we have enough past and future data."""
        max_lag = max(self.lags)
        min_idx = max_lag
        max_idx = len(self.volatility_data) - self.holding_period - 1
        return list(range(min_idx, max_idx))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]

        # Get lag data for volatility prediction (match FlexibleTimeSeriesDataset format)
        x_lags = []
        for lag in self.lags:
            # Get sequence data for this lag: [lag_length, n_symbols]
            x_lag = self.volatility_data.iloc[actual_idx-lag:actual_idx].values

            # Pad if necessary
            if x_lag.shape[0] < lag:
                x_lag = np.pad(x_lag, ((lag - x_lag.shape[0], 0), (0, 0)), 'constant')

            # Transpose to get [n_symbols, lag_length]
            x_lag = x_lag.T

            x_lags.append(torch.tensor(x_lag, dtype=torch.float32))

        # Get volatility target
        vol_target = self.volatility_data.iloc[actual_idx].values
        vol_target = torch.tensor(vol_target, dtype=torch.float32).unsqueeze(-1)

        # Get log returns for trading loss (next + holding period)
        log_returns_data = self.log_returns.iloc[
            actual_idx + 1 : actual_idx + 1 + self.holding_period + 1
        ].values  # [holding_period + 1, n_symbols]

        log_returns_tensor = torch.tensor(
            log_returns_data.T, dtype=torch.float32
        )  # [n_symbols, holding_period + 1]

        return {
            'x_lags': x_lags,
            'vol_targets': vol_target,
            'log_returns': log_returns_tensor
        }


def train_with_actual_trading_loss(model, train_loader, val_loader, trading_loss_fn,
                                 optimizer, device, n_epochs, patience=5):
    """
    Training loop with actual TradingStrategyLoss.
    """
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0

    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_samples = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]", leave=False)

        for batch_idx, batch_data in enumerate(train_pbar):
            # Unpack batch data
            x_lags = batch_data['x_lags']
            vol_targets = batch_data['vol_targets']
            log_returns = batch_data['log_returns']

            # Move to device
            x_lags = [x.to(device) for x in x_lags]
            vol_targets = vol_targets.to(device)
            log_returns = log_returns.to(device)

            # Forward pass
            optimizer.zero_grad()
            vol_pred = model(*x_lags)  # [batch_size, n_symbols, 1]

            # Calculate trading loss
            batch_loss = 0.0
            batch_size, n_symbols = vol_pred.shape[:2]
            valid_samples = 0

            for i in range(batch_size):
                for j in range(n_symbols):
                    # Get prediction and returns for this sample and symbol
                    pred = vol_pred[i, j, 0].unsqueeze(0)  # [1]
                    returns = log_returns[i, j, :].unsqueeze(0)  # [1, holding_period+1]

                    # Calculate loss for this sample
                    try:
                        sample_loss = trading_loss_fn(pred, returns)
                        if not torch.isnan(sample_loss) and not torch.isinf(sample_loss):
                            batch_loss += sample_loss
                            valid_samples += 1
                    except Exception as e:
                        if batch_idx == 0:  # Only log once per epoch
                            logger.warning(f"Error in loss calculation: {e}")
                        continue

            if valid_samples > 0:
                # Average loss over valid samples
                batch_loss = batch_loss / valid_samples

                # Backward pass
                batch_loss.backward()

                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                train_loss += batch_loss.item()
                train_samples += 1

                train_pbar.set_postfix({"loss": f"{batch_loss.item():.6f}"})
            else:
                logger.warning(f"No valid samples in batch {batch_idx}")

        # Calculate average training loss
        if train_samples > 0:
            train_loss /= train_samples
            train_losses.append(train_loss)
        else:
            logger.error("No valid training samples in epoch")
            train_losses.append(float('inf'))

        # Validation
        model.eval()
        val_loss = 0.0
        val_samples = 0

        with torch.no_grad():
            for batch_data in val_loader:
                # Unpack batch data
                x_lags = batch_data['x_lags']
                vol_targets = batch_data['vol_targets']
                log_returns = batch_data['log_returns']

                # Move to device
                x_lags = [x.to(device) for x in x_lags]
                vol_targets = vol_targets.to(device)
                log_returns = log_returns.to(device)

                # Forward pass
                vol_pred = model(*x_lags)

                # Calculate trading loss
                batch_loss = 0.0
                batch_size, n_symbols = vol_pred.shape[:2]
                valid_samples = 0

                for i in range(batch_size):
                    for j in range(n_symbols):
                        pred = vol_pred[i, j, 0].unsqueeze(0)
                        returns = log_returns[i, j, :].unsqueeze(0)

                        try:
                            sample_loss = trading_loss_fn(pred, returns)
                            if not torch.isnan(sample_loss) and not torch.isinf(sample_loss):
                                batch_loss += sample_loss
                                valid_samples += 1
                        except:
                            continue

                if valid_samples > 0:
                    batch_loss = batch_loss / valid_samples
                    val_loss += batch_loss.item()
                    val_samples += 1

        # Calculate average validation loss
        if val_samples > 0:
            val_loss /= val_samples
            val_losses.append(val_loss)
        else:
            val_losses.append(float('inf'))

        # Print epoch results
        logger.info(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Check if this is the best model so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            # Save the best model
            model_path = f"models/gsphar_actual_trading_loss_best.pt"
            torch.save(model.state_dict(), model_path)
            logger.info(f"Saved best model at epoch {epoch+1}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    return model, train_losses, val_losses, best_epoch


def main():
    """Main function."""
    # Parameters
    rv_file = "data/crypto_rv1h_38_20200822_20250116.csv"
    pct_change_file = "data/crypto_pct_change_1h_38_trimmed.csv"
    lags = [1, 4, 24]
    n_epochs = 5  # Reduced for testing
    batch_size = 16  # Reduced for stability
    learning_rate = 0.0001  # Reduced for stability
    weight_decay = 0.0001
    device = torch.device('cpu')
    holding_period = 24
    alpha = 1.0
    beta = 0.1  # Reduced to make profit component less dominant
    gamma = 2.0

    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots/trading_strategy', exist_ok=True)

    logger.info(f"Using device: {device}")

    # Load data
    logger.info("Loading data...")
    rv_df = pd.read_csv(rv_file, index_col=0, parse_dates=True)
    pct_change_df = pd.read_csv(pct_change_file, index_col=0, parse_dates=True)

    # Ensure indices match and use common symbols
    common_index = rv_df.index.intersection(pct_change_df.index)
    rv_df = rv_df.loc[common_index]
    pct_change_df = pct_change_df.loc[common_index]

    symbols = list(set(rv_df.columns).intersection(set(pct_change_df.columns)))
    rv_df = rv_df[symbols]
    pct_change_df = pct_change_df[symbols]

    logger.info(f"Using {len(symbols)} symbols, data shape: {rv_df.shape}")

    # Split data
    train_ratio = 0.8
    train_size = int(len(rv_df) * train_ratio)

    rv_train = rv_df.iloc[:train_size]
    rv_val = rv_df.iloc[train_size:]
    pct_train = pct_change_df.iloc[:train_size]
    pct_val = pct_change_df.iloc[train_size:]

    # Standardize volatility data
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
    logger.info("Creating datasets...")
    train_dataset = TradingDataset(
        volatility_data=rv_train_scaled,
        price_data=pct_train,  # Use percentage change as proxy for price data
        lags=lags,
        holding_period=holding_period,
        debug=True
    )

    val_dataset = TradingDataset(
        volatility_data=rv_val_scaled,
        price_data=pct_val,
        lags=lags,
        holding_period=holding_period,
        debug=False
    )

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    # Create model
    logger.info("Creating model...")
    filter_size = len(symbols)
    output_dim = 1

    # Compute adjacency matrix
    corr_matrix = rv_train.corr().values
    A = (corr_matrix + corr_matrix.T) / 2

    model = FlexibleGSPHAR(lags=lags, output_dim=output_dim, filter_size=filter_size, A=A)
    model = model.to(device)

    # Create optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    trading_loss_fn = TradingStrategyLoss(alpha=alpha, beta=beta, gamma=gamma, holding_period=holding_period)

    # Train model
    logger.info("Training with actual TradingStrategyLoss...")
    model, train_losses, val_losses, best_epoch = train_with_actual_trading_loss(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        trading_loss_fn=trading_loss_fn,
        optimizer=optimizer,
        device=device,
        n_epochs=n_epochs,
        patience=3
    )

    # Save final model
    model_path = f"models/gsphar_actual_trading_loss_final.pt"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Saved final model")

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch ({best_epoch+1})')
    plt.xlabel('Epoch')
    plt.ylabel('Trading Strategy Loss')
    plt.title('Training with Actual TradingStrategyLoss')
    plt.legend()
    plt.grid(True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f'plots/trading_strategy/actual_trading_loss_{timestamp}.png'
    plt.savefig(plot_path)
    logger.info(f"Saved plot to {plot_path}")

    logger.info("Training with actual TradingStrategyLoss completed!")


if __name__ == '__main__':
    main()
