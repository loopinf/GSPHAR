#!/usr/bin/env python
"""
Compare different loss functions by training on various losses but validating on profit.
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


class MSELoss(nn.Module):
    """Standard Mean Squared Error Loss for volatility prediction."""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, vol_pred, vol_target):
        return self.mse(vol_pred.squeeze(), vol_target.squeeze())


class MAELoss(nn.Module):
    """Mean Absolute Error Loss for volatility prediction."""
    def __init__(self):
        super().__init__()
        self.mae = nn.L1Loss()
    
    def forward(self, vol_pred, vol_target):
        return self.mae(vol_pred.squeeze(), vol_target.squeeze())


class HuberLoss(nn.Module):
    """Huber Loss for robust volatility prediction."""
    def __init__(self, delta=1.0):
        super().__init__()
        self.huber = nn.SmoothL1Loss(beta=delta)
    
    def forward(self, vol_pred, vol_target):
        return self.huber(vol_pred.squeeze(), vol_target.squeeze())


class QuantileLoss(nn.Module):
    """Quantile Loss for asymmetric volatility prediction."""
    def __init__(self, quantile=0.5):
        super().__init__()
        self.quantile = quantile
    
    def forward(self, vol_pred, vol_target):
        vol_pred = vol_pred.squeeze()
        vol_target = vol_target.squeeze()
        errors = vol_target - vol_pred
        loss = torch.max(self.quantile * errors, (self.quantile - 1) * errors)
        return loss.mean()


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss that penalizes under-prediction more than over-prediction."""
    def __init__(self, alpha=2.0):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, vol_pred, vol_target):
        vol_pred = vol_pred.squeeze()
        vol_target = vol_target.squeeze()
        errors = vol_target - vol_pred
        
        # Penalize under-prediction (positive errors) more heavily
        loss = torch.where(errors >= 0, 
                          self.alpha * errors**2,  # Under-prediction penalty
                          errors**2)               # Over-prediction penalty
        return loss.mean()


class TradingDataset(torch.utils.data.Dataset):
    """Dataset that provides both volatility targets and log returns for profit calculation."""
    
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


def calculate_profit_metrics(model, data_loader, device):
    """Calculate profit metrics for a trained model."""
    model.eval()
    
    total_profit = 0.0
    total_samples = 0
    profitable_trades = 0
    filled_orders = 0
    
    with torch.no_grad():
        for batch_data in data_loader:
            x_lags = batch_data['x_lags']
            log_returns = batch_data['log_returns']
            
            x_lags = [x.to(device) for x in x_lags]
            log_returns = log_returns.to(device)
            
            vol_pred = model(*x_lags)
            
            batch_size, n_symbols = vol_pred.shape[:2]
            
            for i in range(batch_size):
                for j in range(n_symbols):
                    pred = vol_pred[i, j, 0].unsqueeze(0)
                    returns = log_returns[i, j, :].unsqueeze(0)
                    
                    try:
                        # Calculate actual profit
                        vol_pred_val = torch.clamp(torch.abs(pred), 0.001, 0.5)
                        log_entry_threshold = torch.log(1 - vol_pred_val)
                        log_return_next = returns[:, 0]
                        
                        # Check if order fills (using hard threshold for evaluation)
                        order_filled = (log_return_next <= log_entry_threshold).float()
                        
                        if order_filled.item() > 0:
                            filled_orders += 1
                            log_return_holding = torch.sum(returns[:, 1:25], dim=1)
                            profit = (torch.exp(log_return_holding) - 1).item()
                            total_profit += profit
                            
                            if profit > 0:
                                profitable_trades += 1
                        
                        total_samples += 1
                        
                    except Exception:
                        continue
    
    if total_samples == 0:
        return {
            'avg_profit': 0.0,
            'fill_rate': 0.0,
            'win_rate': 0.0,
            'total_samples': 0
        }
    
    fill_rate = filled_orders / total_samples if total_samples > 0 else 0.0
    win_rate = profitable_trades / filled_orders if filled_orders > 0 else 0.0
    avg_profit = total_profit / filled_orders if filled_orders > 0 else 0.0
    
    return {
        'avg_profit': avg_profit,
        'fill_rate': fill_rate,
        'win_rate': win_rate,
        'total_samples': total_samples,
        'filled_orders': filled_orders,
        'profitable_trades': profitable_trades
    }


def train_model_with_loss(model, train_loader, val_loader, loss_fn, loss_name, 
                         optimizer, device, n_epochs=5, patience=3):
    """Train a model with a specific loss function."""
    
    logger.info(f"Training with {loss_name}")
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_samples = 0
        
        for batch_data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False):
            x_lags = batch_data['x_lags']
            vol_targets = batch_data['vol_targets']
            log_returns = batch_data['log_returns']
            
            x_lags = [x.to(device) for x in x_lags]
            vol_targets = vol_targets.to(device)
            log_returns = log_returns.to(device)
            
            optimizer.zero_grad()
            vol_pred = model(*x_lags)
            
            # Calculate loss based on loss function type
            if hasattr(loss_fn, 'forward') and 'log_returns' in loss_fn.forward.__code__.co_varnames:
                # Trading loss functions that need log returns
                batch_loss = 0.0
                batch_size, n_symbols = vol_pred.shape[:2]
                valid_samples = 0
                
                for i in range(batch_size):
                    for j in range(n_symbols):
                        pred = vol_pred[i, j, 0].unsqueeze(0)
                        returns = log_returns[i, j, :].unsqueeze(0)
                        
                        try:
                            sample_loss = loss_fn(pred, returns)
                            if not torch.isnan(sample_loss) and not torch.isinf(sample_loss):
                                batch_loss += sample_loss
                                valid_samples += 1
                        except Exception:
                            continue
                
                if valid_samples > 0:
                    batch_loss = batch_loss / valid_samples
                else:
                    continue
            else:
                # Standard volatility prediction losses
                batch_loss = loss_fn(vol_pred, vol_targets)
            
            if not torch.isnan(batch_loss) and not torch.isinf(batch_loss):
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += batch_loss.item()
                train_samples += 1
        
        if train_samples > 0:
            train_loss /= train_samples
            train_losses.append(train_loss)
        else:
            train_losses.append(float('inf'))
        
        # Validation (similar logic)
        model.eval()
        val_loss = 0.0
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
                
                if hasattr(loss_fn, 'forward') and 'log_returns' in loss_fn.forward.__code__.co_varnames:
                    batch_loss = 0.0
                    batch_size, n_symbols = vol_pred.shape[:2]
                    valid_samples = 0
                    
                    for i in range(batch_size):
                        for j in range(n_symbols):
                            pred = vol_pred[i, j, 0].unsqueeze(0)
                            returns = log_returns[i, j, :].unsqueeze(0)
                            
                            try:
                                sample_loss = loss_fn(pred, returns)
                                if not torch.isnan(sample_loss) and not torch.isinf(sample_loss):
                                    batch_loss += sample_loss
                                    valid_samples += 1
                            except Exception:
                                continue
                    
                    if valid_samples > 0:
                        batch_loss = batch_loss / valid_samples
                        val_loss += batch_loss.item()
                        val_samples += 1
                else:
                    batch_loss = loss_fn(vol_pred, vol_targets)
                    if not torch.isnan(batch_loss) and not torch.isinf(batch_loss):
                        val_loss += batch_loss.item()
                        val_samples += 1
        
        if val_samples > 0:
            val_loss /= val_samples
            val_losses.append(val_loss)
        else:
            val_losses.append(float('inf'))
        
        logger.info(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    return model, train_losses, val_losses


def main():
    """Main comparison function."""
    
    # Parameters
    rv_file = "data/crypto_rv1h_38_20200822_20250116.csv"
    pct_change_file = "data/crypto_pct_change_1h_38_trimmed.csv"
    lags = [1, 4, 24]
    n_epochs = 6  # Reduced for faster comparison
    batch_size = 16
    learning_rate = 0.001
    weight_decay = 0.0001
    device = torch.device('cpu')
    holding_period = 24
    
    # Create directories
    os.makedirs('models/comparison', exist_ok=True)
    os.makedirs('plots/loss_comparison', exist_ok=True)
    
    logger.info(f"Using device: {device}")
    
    # Load and prepare data (same as before)
    logger.info("Loading data...")
    rv_df = pd.read_csv(rv_file, index_col=0, parse_dates=True)
    pct_change_df = pd.read_csv(pct_change_file, index_col=0, parse_dates=True)
    
    common_index = rv_df.index.intersection(pct_change_df.index)
    rv_df = rv_df.loc[common_index]
    pct_change_df = pct_change_df.loc[common_index]
    
    symbols = list(set(rv_df.columns).intersection(set(pct_change_df.columns)))
    rv_df = rv_df[symbols]
    pct_change_df = pct_change_df[symbols]
    
    logger.info(f"Using {len(symbols)} symbols, data shape: {rv_df.shape}")
    
    # Split and scale data
    train_ratio = 0.8
    train_size = int(len(rv_df) * train_ratio)
    
    rv_train = rv_df.iloc[:train_size]
    rv_val = rv_df.iloc[train_size:]
    pct_train = pct_change_df.iloc[:train_size]
    pct_val = pct_change_df.iloc[train_size:]
    
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
    
    # Create datasets and loaders
    train_dataset = TradingDataset(rv_train_scaled, pct_train, lags, holding_period, debug=True)
    val_dataset = TradingDataset(rv_val_scaled, pct_val, lags, holding_period, debug=False)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Define loss functions to compare
    loss_functions = [
        {"name": "MSE", "loss_fn": MSELoss(), "description": "Mean Squared Error"},
        {"name": "MAE", "loss_fn": MAELoss(), "description": "Mean Absolute Error"},
        {"name": "Huber", "loss_fn": HuberLoss(delta=0.1), "description": "Huber Loss"},
        {"name": "Asymmetric", "loss_fn": AsymmetricLoss(alpha=3.0), "description": "Asymmetric Loss"},
        {"name": "Simple_Profit", "loss_fn": SimpleProfitMaximizationLoss(holding_period), "description": "Simple Profit Max"},
        {"name": "Trading_Loss", "loss_fn": TradingStrategyLoss(2.0, 0.5, 1.0, holding_period), "description": "Trading Strategy Loss"}
    ]
    
    results = {}
    
    # Train and evaluate each loss function
    for loss_config in loss_functions:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training with {loss_config['description']}")
        logger.info(f"{'='*60}")
        
        # Create fresh model
        filter_size = len(symbols)
        corr_matrix = rv_train.corr().values
        A = (corr_matrix + corr_matrix.T) / 2
        
        model = FlexibleGSPHAR(lags=lags, output_dim=1, filter_size=filter_size, A=A)
        model = model.to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Train model
        trained_model, train_losses, val_losses = train_model_with_loss(
            model, train_loader, val_loader, loss_config["loss_fn"], 
            loss_config["name"], optimizer, device, n_epochs, patience=3
        )
        
        # Calculate profit metrics
        logger.info("Calculating profit metrics...")
        profit_metrics = calculate_profit_metrics(trained_model, val_loader, device)
        
        # Store results
        results[loss_config["name"]] = {
            "description": loss_config["description"],
            "train_losses": train_losses,
            "val_losses": val_losses,
            "profit_metrics": profit_metrics
        }
        
        # Save model
        model_path = f"models/comparison/gsphar_{loss_config['name']}.pt"
        torch.save(trained_model.state_dict(), model_path)
        
        logger.info(f"Results for {loss_config['name']}:")
        logger.info(f"  Avg Profit: {profit_metrics['avg_profit']:.4f} ({profit_metrics['avg_profit']*100:.2f}%)")
        logger.info(f"  Fill Rate: {profit_metrics['fill_rate']:.3f} ({profit_metrics['fill_rate']*100:.1f}%)")
        logger.info(f"  Win Rate: {profit_metrics['win_rate']:.3f} ({profit_metrics['win_rate']*100:.1f}%)")
    
    # Create summary
    create_summary_analysis(results)
    logger.info("Loss function comparison completed!")


def create_summary_analysis(results):
    """Create summary analysis and plots."""
    
    # Extract metrics
    loss_names = list(results.keys())
    avg_profits = [results[name]["profit_metrics"]["avg_profit"] for name in loss_names]
    fill_rates = [results[name]["profit_metrics"]["fill_rate"] for name in loss_names]
    win_rates = [results[name]["profit_metrics"]["win_rate"] for name in loss_names]
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Loss Function Comparison: Profit Performance', fontsize=16)
    
    # Plot 1: Average Profit
    ax1 = axes[0]
    colors = ['green' if p > 0 else 'red' for p in avg_profits]
    bars = ax1.bar(range(len(loss_names)), avg_profits, color=colors, alpha=0.7)
    ax1.set_xticks(range(len(loss_names)))
    ax1.set_xticklabels([name.replace('_', '\n') for name in loss_names], rotation=45)
    ax1.set_ylabel('Average Profit per Trade')
    ax1.set_title('Average Profit by Loss Function')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    for bar, profit in zip(bars, avg_profits):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + (0.001 if height >= 0 else -0.003),
                f'{profit:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)
    
    # Plot 2: Fill Rate
    ax2 = axes[1]
    ax2.bar(range(len(loss_names)), fill_rates, alpha=0.7, color='blue')
    ax2.set_xticks(range(len(loss_names)))
    ax2.set_xticklabels([name.replace('_', '\n') for name in loss_names], rotation=45)
    ax2.set_ylabel('Fill Rate')
    ax2.set_title('Order Fill Rate by Loss Function')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Win Rate
    ax3 = axes[2]
    ax3.bar(range(len(loss_names)), win_rates, alpha=0.7, color='orange')
    ax3.set_xticks(range(len(loss_names)))
    ax3.set_xticklabels([name.replace('_', '\n') for name in loss_names], rotation=45)
    ax3.set_ylabel('Win Rate')
    ax3.set_title('Win Rate by Loss Function')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f'plots/loss_comparison/loss_comparison_{timestamp}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved comparison plot to: {plot_path}")
    
    # Create and save summary table
    summary_df = pd.DataFrame({
        'Loss_Function': loss_names,
        'Description': [results[name]["description"] for name in loss_names],
        'Avg_Profit': avg_profits,
        'Fill_Rate': fill_rates,
        'Win_Rate': win_rates
    }).sort_values('Avg_Profit', ascending=False)
    
    summary_path = f'plots/loss_comparison/summary_{timestamp}.csv'
    summary_df.to_csv(summary_path, index=False)
    
    # Print ranking
    logger.info("\n" + "="*80)
    logger.info("LOSS FUNCTION RANKING BY PROFIT PERFORMANCE")
    logger.info("="*80)
    logger.info(f"{'Rank':<4} {'Loss Function':<15} {'Avg Profit':<12} {'Fill Rate':<10} {'Win Rate':<10}")
    logger.info("-" * 80)
    
    for i, (_, row) in enumerate(summary_df.iterrows(), 1):
        logger.info(f"{i:<4} {row['Loss_Function']:<15} {row['Avg_Profit']:<12.4f} "
                   f"{row['Fill_Rate']:<10.3f} {row['Win_Rate']:<10.3f}")
    
    return summary_df


if __name__ == '__main__':
    main()
