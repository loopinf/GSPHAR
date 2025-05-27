#!/usr/bin/env python
"""
Test Sharpe ratio loss function with small data.

Compare simple profit maximization vs Sharpe ratio optimization.
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

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.ohlcv_trading_dataset import load_ohlcv_trading_data, create_ohlcv_dataloaders
from src.ohlcv_trading_loss import OHLCVLongStrategyLoss, OHLCVSharpeRatioLoss, OHLCVAdvancedSharpeRatioLoss
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


def train_and_compare_losses(dataset, metadata, device='cpu'):
    """
    Train models with different loss functions and compare results.
    """
    logger.info("🎯 COMPARING LOSS FUNCTIONS")
    logger.info("=" * 60)
    
    # Common parameters
    lags = [1, 4, 24]
    batch_size = 8
    learning_rate = 0.001
    n_epochs = 3
    
    # Create model architecture components
    vol_df = pd.read_csv("data/crypto_rv1h_38_20200822_20250116.csv", index_col=0, parse_dates=True)
    corr_sample = vol_df.iloc[-1000:] if len(vol_df) > 1000 else vol_df
    corr_matrix = corr_sample.corr().values
    A = (corr_matrix + corr_matrix.T) / 2
    
    filter_size = len(metadata['assets'])
    output_dim = 1
    
    # Loss functions to test
    loss_functions = {
        'Simple Profit': OHLCVLongStrategyLoss(holding_period=4),
        'Sharpe Ratio': OHLCVSharpeRatioLoss(holding_period=4, risk_free_rate=0.02),  # 2% risk-free rate
        'Advanced Sharpe': OHLCVAdvancedSharpeRatioLoss(holding_period=4, risk_free_rate=0.02)
    }
    
    results = {}
    
    for loss_name, loss_fn in loss_functions.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"TRAINING WITH: {loss_name}")
        logger.info(f"{'='*50}")
        
        # Create fresh model for each loss function
        model = FlexibleGSPHAR(
            lags=lags,
            output_dim=output_dim,
            filter_size=filter_size,
            A=A
        ).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Create dataloaders
        train_loader, val_loader, split_info = create_ohlcv_dataloaders(
            dataset, train_ratio=0.8, batch_size=batch_size, shuffle=True
        )
        
        # Training loop
        history = {'train_loss': [], 'val_loss': [], 'train_metrics': [], 'val_metrics': []}
        
        for epoch in range(n_epochs):
            # Training
            model.train()
            train_losses = []
            train_metrics_list = []
            
            for batch in train_loader:
                x_lags = [x.to(device) for x in batch['x_lags']]
                vol_targets = batch['vol_targets'].to(device)
                ohlcv_data = batch['ohlcv_data'].to(device)
                
                optimizer.zero_grad()
                vol_pred = model(*x_lags)
                loss = loss_fn(vol_pred, ohlcv_data)
                
                if not torch.isnan(loss) and not torch.isinf(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    train_losses.append(loss.item())
                    
                    if hasattr(loss_fn, 'calculate_metrics'):
                        metrics = loss_fn.calculate_metrics(vol_pred, ohlcv_data)
                        train_metrics_list.append(metrics)
            
            # Validation
            model.eval()
            val_losses = []
            val_metrics_list = []
            
            with torch.no_grad():
                for batch in val_loader:
                    x_lags = [x.to(device) for x in batch['x_lags']]
                    vol_targets = batch['vol_targets'].to(device)
                    ohlcv_data = batch['ohlcv_data'].to(device)
                    
                    vol_pred = model(*x_lags)
                    loss = loss_fn(vol_pred, ohlcv_data)
                    
                    if not torch.isnan(loss) and not torch.isinf(loss):
                        val_losses.append(loss.item())
                        
                        if hasattr(loss_fn, 'calculate_metrics'):
                            metrics = loss_fn.calculate_metrics(vol_pred, ohlcv_data)
                            val_metrics_list.append(metrics)
            
            # Calculate averages
            avg_train_loss = np.mean(train_losses) if train_losses else float('inf')
            avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
            
            avg_train_metrics = {}
            avg_val_metrics = {}
            
            if train_metrics_list:
                for key in train_metrics_list[0].keys():
                    avg_train_metrics[key] = np.mean([m[key] for m in train_metrics_list])
            
            if val_metrics_list:
                for key in val_metrics_list[0].keys():
                    avg_val_metrics[key] = np.mean([m[key] for m in val_metrics_list])
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['train_metrics'].append(avg_train_metrics)
            history['val_metrics'].append(avg_val_metrics)
            
            logger.info(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.6f}, Val Loss={avg_val_loss:.6f}")
        
        # Store results
        results[loss_name] = {
            'history': history,
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1],
            'final_train_metrics': history['train_metrics'][-1] if history['train_metrics'] else {},
            'final_val_metrics': history['val_metrics'][-1] if history['val_metrics'] else {}
        }
        
        logger.info(f"Final {loss_name} Results:")
        logger.info(f"  Train Loss: {results[loss_name]['final_train_loss']:.6f}")
        logger.info(f"  Val Loss: {results[loss_name]['final_val_loss']:.6f}")
        
        if results[loss_name]['final_val_metrics']:
            logger.info("  Final Val Metrics:")
            for key, value in results[loss_name]['final_val_metrics'].items():
                logger.info(f"    {key}: {value:.4f}")
    
    return results


def compare_results(results):
    """
    Compare results across different loss functions.
    """
    logger.info("\n" + "="*80)
    logger.info("FINAL COMPARISON")
    logger.info("="*80)
    
    # Create comparison table
    comparison_data = []
    
    for loss_name, result in results.items():
        metrics = result['final_val_metrics']
        
        row = {
            'Loss Function': loss_name,
            'Final Loss': result['final_val_loss'],
            'Fill Rate': metrics.get('fill_rate', 0),
            'Avg Profit/Trade': metrics.get('avg_profit_when_filled', 0),
            'Overall Profit': metrics.get('avg_profit_overall', 0),
            'Vol Prediction': metrics.get('avg_vol_pred', 0)
        }
        
        # Add Sharpe-specific metrics
        if 'sharpe_ratio' in metrics:
            row['Sharpe Ratio'] = metrics['sharpe_ratio']
            row['Ann. Return'] = metrics['annualized_return']
            row['Ann. Volatility'] = metrics['annualized_volatility']
            row['Max Drawdown'] = metrics['max_drawdown']
        
        comparison_data.append(row)
    
    # Print comparison
    logger.info("PERFORMANCE COMPARISON:")
    logger.info("-" * 80)
    
    for row in comparison_data:
        logger.info(f"\n{row['Loss Function']}:")
        for key, value in row.items():
            if key != 'Loss Function':
                if isinstance(value, float):
                    logger.info(f"  {key}: {value:.4f}")
                else:
                    logger.info(f"  {key}: {value}")
    
    # Recommendations
    logger.info("\n" + "="*80)
    logger.info("RECOMMENDATIONS:")
    logger.info("="*80)
    
    # Find best Sharpe ratio
    sharpe_results = [(name, result) for name, result in results.items() 
                     if 'sharpe_ratio' in result['final_val_metrics']]
    
    if sharpe_results:
        best_sharpe = max(sharpe_results, 
                         key=lambda x: x[1]['final_val_metrics']['sharpe_ratio'])
        logger.info(f"🏆 Best Sharpe Ratio: {best_sharpe[0]} ({best_sharpe[1]['final_val_metrics']['sharpe_ratio']:.2f})")
    
    # Find best overall profit
    best_profit = max(results.items(), 
                     key=lambda x: x[1]['final_val_metrics'].get('avg_profit_overall', 0))
    logger.info(f"💰 Best Overall Profit: {best_profit[0]} ({best_profit[1]['final_val_metrics'].get('avg_profit_overall', 0):.4f})")


def main():
    """
    Main function to compare loss functions.
    """
    logger.info("🎯 SHARPE RATIO VS PROFIT MAXIMIZATION COMPARISON")
    logger.info("=" * 80)
    
    # Parameters
    volatility_file = "data/crypto_rv1h_38_20200822_20250116.csv"
    lags = [1, 4, 24]
    holding_period = 4
    subset_size = 300  # Small for quick comparison
    device = torch.device('cpu')
    
    logger.info(f"Parameters:")
    logger.info(f"  Subset size: {subset_size}")
    logger.info(f"  Holding period: {holding_period} hours")
    logger.info(f"  Device: {device}")
    
    # Load dataset
    full_dataset, metadata = load_ohlcv_trading_data(
        volatility_file=volatility_file,
        lags=lags,
        holding_period=holding_period,
        debug=False
    )
    
    # Create small subset
    small_dataset = SmallDataSubset(full_dataset, subset_size=subset_size)
    
    # Train and compare
    results = train_and_compare_losses(small_dataset, metadata, device)
    
    # Compare results
    compare_results(results)


if __name__ == "__main__":
    main()
