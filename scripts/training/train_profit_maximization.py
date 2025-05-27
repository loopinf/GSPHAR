#!/usr/bin/env python
"""
Train GSPHAR model with profit maximization loss functions.
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

from src.trading_loss import TradingStrategyLoss, MaximizeProfitLoss, SimpleProfitMaximizationLoss
from src.models.flexible_gsphar import FlexibleGSPHAR

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradingDataset(torch.utils.data.Dataset):
    """Dataset for profit maximization training."""
    
    def __init__(self, volatility_data, price_data, lags, holding_period, debug=False):
        self.volatility_data = volatility_data
        self.price_data = price_data
        self.lags = lags
        self.holding_period = holding_period
        self.debug = debug
        
        # Convert price data to log returns
        self.log_returns = self._calculate_log_returns()
        
        # Calculate valid indices
        self.valid_indices = self._get_valid_indices()
        
        if self.debug:
            logger.info(f"Dataset created with {len(self.valid_indices)} valid samples")
    
    def _calculate_log_returns(self):
        """Calculate log returns from price data."""
        pct_changes = self.price_data.pct_change().fillna(0)
        pct_changes = pct_changes.clip(-0.99, 10)
        log_returns = np.log(1 + pct_changes)
        return log_returns
    
    def _get_valid_indices(self):
        """Get valid indices for training."""
        max_lag = max(self.lags)
        min_idx = max_lag
        max_idx = len(self.volatility_data) - self.holding_period - 1
        return list(range(min_idx, max_idx))
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        
        # Get lag data for volatility prediction
        x_lags = []
        for lag in self.lags:
            x_lag = self.volatility_data.iloc[actual_idx-lag:actual_idx].values
            
            if x_lag.shape[0] < lag:
                x_lag = np.pad(x_lag, ((lag - x_lag.shape[0], 0), (0, 0)), 'constant')
            
            x_lag = x_lag.T
            x_lags.append(torch.tensor(x_lag, dtype=torch.float32))
        
        # Get volatility target
        vol_target = self.volatility_data.iloc[actual_idx].values
        vol_target = torch.tensor(vol_target, dtype=torch.float32).unsqueeze(-1)
        
        # Get log returns for profit calculation
        log_returns_data = self.log_returns.iloc[
            actual_idx + 1 : actual_idx + 1 + self.holding_period + 1
        ].values
        
        log_returns_tensor = torch.tensor(
            log_returns_data.T, dtype=torch.float32
        )
        
        return {
            'x_lags': x_lags,
            'vol_targets': vol_target,
            'log_returns': log_returns_tensor
        }


def train_with_profit_maximization(model, train_loader, val_loader, loss_fn, 
                                 optimizer, device, n_epochs, loss_name, patience=5):
    """Training loop with profit maximization loss."""
    
    train_losses = []
    val_losses = []
    train_profits = []  # Track actual profits
    val_profits = []
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_profit = 0.0
        train_samples = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]", leave=False)
        
        for batch_idx, batch_data in enumerate(train_pbar):
            x_lags = batch_data['x_lags']
            vol_targets = batch_data['vol_targets']
            log_returns = batch_data['log_returns']
            
            # Move to device
            x_lags = [x.to(device) for x in x_lags]
            vol_targets = vol_targets.to(device)
            log_returns = log_returns.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            vol_pred = model(*x_lags)
            
            # Calculate loss and profit
            batch_loss = 0.0
            batch_profit = 0.0
            batch_size, n_symbols = vol_pred.shape[:2]
            valid_samples = 0
            
            for i in range(batch_size):
                for j in range(n_symbols):
                    pred = vol_pred[i, j, 0].unsqueeze(0)
                    returns = log_returns[i, j, :].unsqueeze(0)
                    
                    try:
                        sample_loss = loss_fn(pred, returns)
                        
                        # Calculate actual profit for tracking
                        with torch.no_grad():
                            vol_pred_val = torch.clamp(torch.abs(pred), 0.001, 0.5)
                            log_entry_threshold = torch.log(1 - vol_pred_val)
                            log_return_next = returns[:, 0]
                            filled_orders = (log_return_next <= log_entry_threshold).float()
                            log_return_holding = torch.sum(returns[:, 1:25], dim=1)
                            profit = filled_orders * (torch.exp(log_return_holding) - 1)
                            batch_profit += profit.item()
                        
                        if not torch.isnan(sample_loss) and not torch.isinf(sample_loss):
                            batch_loss += sample_loss
                            valid_samples += 1
                    except Exception as e:
                        if batch_idx == 0:
                            logger.warning(f"Error in loss calculation: {e}")
                        continue
            
            if valid_samples > 0:
                batch_loss = batch_loss / valid_samples
                batch_profit = batch_profit / valid_samples
                
                # Backward pass
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += batch_loss.item()
                train_profit += batch_profit
                train_samples += 1
                
                train_pbar.set_postfix({
                    "loss": f"{batch_loss.item():.6f}",
                    "profit": f"{batch_profit:.4f}"
                })
        
        # Calculate average training metrics
        if train_samples > 0:
            train_loss /= train_samples
            train_profit /= train_samples
            train_losses.append(train_loss)
            train_profits.append(train_profit)
        else:
            train_losses.append(float('inf'))
            train_profits.append(0.0)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_profit = 0.0
        val_samples = 0
        
        with torch.no_grad():
            for batch_data in val_loader:
                x_lags = batch_data['x_lags']
                vol_targets = batch_data['vol_targets']
                log_returns = batch_data['log_returns']
                
                x_lags = [x.to(device) for x in x_lags]
                vol_targets = vol_targets.to(device)
                log_returns = log_returns.to(device)
                
                vol_pred = model(*x_lags)
                
                batch_loss = 0.0
                batch_profit = 0.0
                batch_size, n_symbols = vol_pred.shape[:2]
                valid_samples = 0
                
                for i in range(batch_size):
                    for j in range(n_symbols):
                        pred = vol_pred[i, j, 0].unsqueeze(0)
                        returns = log_returns[i, j, :].unsqueeze(0)
                        
                        try:
                            sample_loss = loss_fn(pred, returns)
                            
                            # Calculate actual profit
                            vol_pred_val = torch.clamp(torch.abs(pred), 0.001, 0.5)
                            log_entry_threshold = torch.log(1 - vol_pred_val)
                            log_return_next = returns[:, 0]
                            filled_orders = (log_return_next <= log_entry_threshold).float()
                            log_return_holding = torch.sum(returns[:, 1:25], dim=1)
                            profit = filled_orders * (torch.exp(log_return_holding) - 1)
                            batch_profit += profit.item()
                            
                            if not torch.isnan(sample_loss) and not torch.isinf(sample_loss):
                                batch_loss += sample_loss
                                valid_samples += 1
                        except:
                            continue
                
                if valid_samples > 0:
                    batch_loss = batch_loss / valid_samples
                    batch_profit = batch_profit / valid_samples
                    val_loss += batch_loss.item()
                    val_profit += batch_profit
                    val_samples += 1
        
        # Calculate average validation metrics
        if val_samples > 0:
            val_loss /= val_samples
            val_profit /= val_samples
            val_losses.append(val_loss)
            val_profits.append(val_profit)
        else:
            val_losses.append(float('inf'))
            val_profits.append(0.0)
        
        # Print epoch results
        logger.info(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss:.6f}, "
                   f"Train Profit: {train_profit:.4f}, Val Loss: {val_loss:.6f}, "
                   f"Val Profit: {val_profit:.4f}")
        
        # Check if this is the best model (lowest loss = highest profit)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            
            # Save the best model
            model_path = f"models/gsphar_{loss_name}_best.pt"
            torch.save(model.state_dict(), model_path)
            logger.info(f"Saved best model at epoch {epoch+1}")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    return model, train_losses, val_losses, train_profits, val_profits, best_epoch


def main():
    """Main function to train with different profit maximization loss functions."""
    
    # Parameters
    rv_file = "data/crypto_rv1h_38_20200822_20250116.csv"
    pct_change_file = "data/crypto_pct_change_1h_38_trimmed.csv"
    lags = [1, 4, 24]
    n_epochs = 10
    batch_size = 16
    learning_rate = 0.001  # Higher learning rate for profit maximization
    weight_decay = 0.0001
    device = torch.device('cpu')
    holding_period = 24
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots/profit_maximization', exist_ok=True)
    
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info("Loading data...")
    rv_df = pd.read_csv(rv_file, index_col=0, parse_dates=True)
    pct_change_df = pd.read_csv(pct_change_file, index_col=0, parse_dates=True)
    
    # Ensure indices match
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
        price_data=pct_train,
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
    
    # Define loss functions to test
    loss_functions = [
        {
            "name": "simple_profit_max",
            "loss_fn": SimpleProfitMaximizationLoss(holding_period=holding_period),
            "description": "Simple Profit Maximization (negative profit)"
        },
        {
            "name": "advanced_profit_max",
            "loss_fn": MaximizeProfitLoss(holding_period=holding_period, risk_penalty=1.5, no_fill_penalty=0.005),
            "description": "Advanced Profit Maximization (with risk management)"
        },
        {
            "name": "original_trading_loss",
            "loss_fn": TradingStrategyLoss(alpha=2.0, beta=0.5, gamma=1.0, holding_period=holding_period),
            "description": "Original Trading Strategy Loss (for comparison)"
        }
    ]
    
    results = {}
    
    # Train with each loss function
    for loss_config in loss_functions:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training with {loss_config['description']}")
        logger.info(f"{'='*60}")
        
        # Create fresh model
        filter_size = len(symbols)
        output_dim = 1
        corr_matrix = rv_train.corr().values
        A = (corr_matrix + corr_matrix.T) / 2
        
        model = FlexibleGSPHAR(lags=lags, output_dim=output_dim, filter_size=filter_size, A=A)
        model = model.to(device)
        
        # Create optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Train model
        model, train_losses, val_losses, train_profits, val_profits, best_epoch = train_with_profit_maximization(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_config["loss_fn"],
            optimizer=optimizer,
            device=device,
            n_epochs=n_epochs,
            loss_name=loss_config["name"],
            patience=3
        )
        
        # Save final model
        model_path = f"models/gsphar_{loss_config['name']}_final.pt"
        torch.save(model.state_dict(), model_path)
        
        # Store results
        results[loss_config["name"]] = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_profits": train_profits,
            "val_profits": val_profits,
            "best_epoch": best_epoch,
            "description": loss_config["description"]
        }
        
        logger.info(f"Completed training with {loss_config['name']}")
        logger.info(f"Best epoch: {best_epoch + 1}")
        logger.info(f"Final validation profit: {val_profits[-1]:.4f}")
    
    # Create comparison plots
    create_comparison_plots(results)
    
    logger.info("Profit maximization training completed!")


def create_comparison_plots(results):
    """Create comparison plots for different loss functions."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Profit Maximization Training Comparison', fontsize=16)
    
    # Plot 1: Training Loss
    ax1 = axes[0, 0]
    for name, data in results.items():
        ax1.plot(data["train_losses"], label=f"{name} (train)", linewidth=2)
        ax1.plot(data["val_losses"], label=f"{name} (val)", linestyle='--', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Training Profit
    ax2 = axes[0, 1]
    for name, data in results.items():
        ax2.plot(data["train_profits"], label=f"{name} (train)", linewidth=2)
        ax2.plot(data["val_profits"], label=f"{name} (val)", linestyle='--', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Average Profit per Trade')
    ax2.set_title('Training and Validation Profit')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Plot 3: Final Performance Comparison
    ax3 = axes[1, 0]
    names = list(results.keys())
    final_profits = [results[name]["val_profits"][-1] for name in names]
    colors = ['green' if p > 0 else 'red' for p in final_profits]
    
    bars = ax3.bar(range(len(names)), final_profits, color=colors, alpha=0.7)
    ax3.set_xticks(range(len(names)))
    ax3.set_xticklabels([name.replace('_', '\n') for name in names], rotation=0)
    ax3.set_ylabel('Final Validation Profit')
    ax3.set_title('Final Performance Comparison')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Add value labels on bars
    for bar, profit in zip(bars, final_profits):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + (0.0005 if height >= 0 else -0.0015),
                f'{profit:.4f}', ha='center', va='bottom' if height >= 0 else 'top')
    
    # Plot 4: Best Epoch Comparison
    ax4 = axes[1, 1]
    best_epochs = [results[name]["best_epoch"] + 1 for name in names]
    ax4.bar(range(len(names)), best_epochs, alpha=0.7)
    ax4.set_xticks(range(len(names)))
    ax4.set_xticklabels([name.replace('_', '\n') for name in names], rotation=0)
    ax4.set_ylabel('Best Epoch')
    ax4.set_title('Convergence Speed')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f'plots/profit_maximization/comparison_{timestamp}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved comparison plot to: {plot_path}")


if __name__ == '__main__':
    main()
