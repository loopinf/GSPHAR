#!/usr/bin/env python
"""
Train the flexible GSPHAR model with the custom trading strategy loss function.

This script extends the flexible GSPHAR training script to use the custom
trading strategy loss function that optimizes for the specific trading strategy.
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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


def load_data(rv_file, pct_change_file, symbols_file=None, symbols=None):
    """
    Load realized volatility data and percentage change data.
    
    Args:
        rv_file (str): Path to the realized volatility data file
        pct_change_file (str): Path to the percentage change data file
        symbols_file (str, optional): Path to the file containing symbols to use
        symbols (list, optional): List of symbols to use
        
    Returns:
        tuple: (rv_df, pct_change_df, symbols)
    """
    # Load realized volatility data
    logger.info(f"Loading realized volatility data from {rv_file}")
    rv_df = pd.read_csv(rv_file, index_col=0, parse_dates=True)
    
    # Load percentage change data
    logger.info(f"Loading percentage change data from {pct_change_file}")
    pct_change_df = pd.read_csv(pct_change_file, index_col=0, parse_dates=True)
    
    # Ensure the indices match
    common_index = rv_df.index.intersection(pct_change_df.index)
    rv_df = rv_df.loc[common_index]
    pct_change_df = pct_change_df.loc[common_index]
    
    # Select symbols
    if symbols_file:
        logger.info(f"Loading symbols from {symbols_file}")
        with open(symbols_file, 'r') as f:
            symbols = [line.strip() for line in f.readlines()]
        logger.info(f"Selected {len(symbols)} symbols from symbols file")
    elif symbols:
        logger.info(f"Using {len(symbols)} provided symbols")
    else:
        # Use all common symbols
        symbols = list(set(rv_df.columns).intersection(set(pct_change_df.columns)))
        logger.info(f"Using {len(symbols)} common symbols from both datasets")
    
    # Filter to selected symbols
    rv_df = rv_df[symbols]
    pct_change_df = pct_change_df[symbols]
    
    logger.info(f"Loaded realized volatility data with shape {rv_df.shape}")
    logger.info(f"Loaded percentage change data with shape {pct_change_df.shape}")
    
    return rv_df, pct_change_df, symbols


def prepare_data_for_trading_loss(rv_df, pct_change_df, holding_period=24):
    """
    Prepare data for the trading strategy loss function.
    
    Args:
        rv_df (pd.DataFrame): Realized volatility data
        pct_change_df (pd.DataFrame): Percentage change data
        holding_period (int): Number of periods to hold the position
        
    Returns:
        tuple: (log_returns_dict, date_indices)
    """
    # Convert percentage changes to log returns
    log_returns_df = np.log(1 + pct_change_df)
    
    # Create sequences of log returns for each symbol
    log_returns_dict = {}
    date_indices = {}
    
    for symbol in rv_df.columns:
        symbol_log_returns = log_returns_df[symbol].values
        sequences = []
        dates = []
        
        for i in range(len(symbol_log_returns) - holding_period):
            sequences.append(symbol_log_returns[i:i+holding_period+1])
            dates.append(log_returns_df.index[i])
        
        log_returns_dict[symbol] = np.array(sequences)
        date_indices[symbol] = dates
    
    return log_returns_dict, date_indices


def train_model_with_trading_loss(
    model, train_loader, val_loader, optimizer, trading_loss_fn, 
    device, n_epochs, log_returns_dict, patience=5
):
    """
    Train the model using the trading strategy loss function.
    
    Args:
        model (nn.Module): The model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        optimizer (Optimizer): Optimizer to use
        trading_loss_fn (nn.Module): Trading strategy loss function
        device (str): Device to use for training
        n_epochs (int): Number of epochs to train for
        log_returns_dict (dict): Dictionary of log returns sequences for each symbol
        patience (int): Number of epochs to wait for improvement before early stopping
        
    Returns:
        tuple: (model, train_losses, val_losses, best_epoch)
    """
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    # Training loop
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        # Create progress bar for training
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]", 
                          leave=False, ncols=100)
        
        for batch_idx, batch_data in enumerate(train_pbar):
            # The last element in batch_data is the target
            y = batch_data[-1]
            x_lags = batch_data[:-1]
            
            # Move data to device
            x_lags = [x_lag.to(device) for x_lag in x_lags]
            y = y.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            
            # Pass all inputs to the model
            y_pred = model(*x_lags, y)
            
            # Get the symbol indices for this batch
            symbol_indices = batch_data[-2]  # Assuming symbol indices are passed as the second-to-last element
            
            # Calculate trading loss
            batch_loss = 0.0
            batch_size = y_pred.shape[0]
            
            for i in range(batch_size):
                symbol_idx = symbol_indices[i].item()
                symbol = train_loader.dataset.symbols[symbol_idx]
                
                # Get log returns for this symbol and time step
                time_idx = batch_data[-3][i].item()  # Assuming time indices are passed as the third-to-last element
                if time_idx < len(log_returns_dict[symbol]):
                    log_returns = torch.tensor(log_returns_dict[symbol][time_idx], dtype=torch.float32).to(device)
                    
                    # Calculate trading loss for this sample
                    sample_loss = trading_loss_fn(y_pred[i:i+1], log_returns.unsqueeze(0))
                    batch_loss += sample_loss
            
            # Average the loss over the batch
            batch_loss = batch_loss / batch_size if batch_size > 0 else torch.tensor(0.0).to(device)
            
            # Backward pass
            batch_loss.backward()
            optimizer.step()
            
            # Update loss
            batch_loss_value = batch_loss.item()
            train_loss += batch_loss_value
            
            # Update progress bar
            train_pbar.set_postfix({"loss": f"{batch_loss_value:.6f}"})
        
        # Calculate average training loss
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        # Create progress bar for validation
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Val]", 
                        leave=False, ncols=100)
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(val_pbar):
                # The last element in batch_data is the target
                y = batch_data[-1]
                x_lags = batch_data[:-1]
                
                # Move data to device
                x_lags = [x_lag.to(device) for x_lag in x_lags]
                y = y.to(device)
                
                # Forward pass
                y_pred = model(*x_lags, y)
                
                # Get the symbol indices for this batch
                symbol_indices = batch_data[-2]  # Assuming symbol indices are passed as the second-to-last element
                
                # Calculate trading loss
                batch_loss = 0.0
                batch_size = y_pred.shape[0]
                
                for i in range(batch_size):
                    symbol_idx = symbol_indices[i].item()
                    symbol = val_loader.dataset.symbols[symbol_idx]
                    
                    # Get log returns for this symbol and time step
                    time_idx = batch_data[-3][i].item()  # Assuming time indices are passed as the third-to-last element
                    if time_idx < len(log_returns_dict[symbol]):
                        log_returns = torch.tensor(log_returns_dict[symbol][time_idx], dtype=torch.float32).to(device)
                        
                        # Calculate trading loss for this sample
                        sample_loss = trading_loss_fn(y_pred[i:i+1], log_returns.unsqueeze(0))
                        batch_loss += sample_loss
                
                # Average the loss over the batch
                batch_loss = batch_loss / batch_size if batch_size > 0 else torch.tensor(0.0).to(device)
                
                # Update loss
                batch_loss_value = batch_loss.item()
                val_loss += batch_loss_value
                
                # Update progress bar
                val_pbar.set_postfix({"loss": f"{batch_loss_value:.6f}"})
        
        # Calculate average validation loss
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Print epoch results
        logger.info(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Check if this is the best model so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            
            # Save the best model
            model_path = f"models/flexible_gsphar_trading_loss_best.pt"
            torch.save(model.state_dict(), model_path)
            logger.info(f"Saved best model at epoch {epoch+1}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Save the final model
    model_path = f"models/flexible_gsphar_trading_loss_final.pt"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Saved final model after {epoch+1} epochs")
    
    # Load the best model
    model.load_state_dict(torch.load(f"models/flexible_gsphar_trading_loss_best.pt"))
    logger.info(f"Loaded best model from epoch {best_epoch+1}")
    
    return model, train_losses, val_losses, best_epoch


def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(description='Train flexible GSPHAR model with trading strategy loss.')
    parser.add_argument('--rv-file', type=str, required=True,
                        help='Path to the realized volatility data file.')
    parser.add_argument('--pct-change-file', type=str, required=True,
                        help='Path to the percentage change data file.')
    parser.add_argument('--symbols-file', type=str, default=None,
                        help='Path to the file containing symbols to use.')
    parser.add_argument('--symbols', type=str, nargs='+', default=None,
                        help='List of symbols to use.')
    parser.add_argument('--lags', type=int, nargs='+', default=[1, 4, 24],
                        help='List of lags to use.')
    parser.add_argument('--n-epochs', type=int, default=10,
                        help='Number of epochs to train for.')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training.')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate for the optimizer.')
    parser.add_argument('--weight-decay', type=float, default=0.0001,
                        help='Weight decay for the optimizer.')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use for training.')
    parser.add_argument('--holding-period', type=int, default=24,
                        help='Number of periods to hold the position.')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Weight for the fill loss component.')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='Weight for the profit component.')
    parser.add_argument('--gamma', type=float, default=2.0,
                        help='Weight for the loss avoidance component.')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots/trading_strategy', exist_ok=True)
    
    # Set device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load data
    rv_df, pct_change_df, symbols = load_data(
        args.rv_file, args.pct_change_file, args.symbols_file, args.symbols
    )
    
    # Prepare data for trading loss
    log_returns_dict, date_indices = prepare_data_for_trading_loss(
        rv_df, pct_change_df, args.holding_period
    )
    
    # TODO: Implement the rest of the training pipeline
    # This would include:
    # 1. Creating the dataset and data loaders
    # 2. Creating the model
    # 3. Creating the optimizer and trading loss function
    # 4. Training the model
    # 5. Evaluating the model
    # 6. Visualizing the results
    
    logger.info("Training and evaluation completed successfully")


if __name__ == '__main__':
    main()
