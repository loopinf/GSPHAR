#!/usr/bin/env python
"""
Script to train the GSPHAR model on cryptocurrency data.
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from local modules
from src.models import GSPHAR
from src.utils import load_model, save_model
from src.dataloader import TimeSeriesDataset, create_dataloaders

# Create utils directory if it doesn't exist
os.makedirs(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src', 'utils'), exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(rv_file, symbols_file=None, top_n=None):
    """
    Load the realized volatility data for the selected symbols.

    Args:
        rv_file (str): Path to the realized volatility file.
        symbols_file (str, optional): Path to the file containing the symbols to use.
        top_n (int, optional): Number of top symbols to use if symbols_file is not provided.

    Returns:
        pd.DataFrame: DataFrame containing the realized volatility data for the selected symbols.
    """
    logger.info(f"Loading realized volatility data from {rv_file}")

    # Load realized volatility data
    if rv_file.endswith('.csv'):
        rv_df = pd.read_csv(rv_file, index_col=0, parse_dates=True)
    elif rv_file.endswith('.parquet'):
        rv_df = pd.read_parquet(rv_file)
    else:
        raise ValueError("Unsupported file format. Please use .csv or .parquet")

    # Select symbols
    if symbols_file is not None:
        logger.info(f"Loading symbols from {symbols_file}")
        with open(symbols_file, 'r') as f:
            symbols = [line.strip() for line in f.readlines()]

        # Filter to only include symbols that are in the data
        symbols = [s for s in symbols if s in rv_df.columns]
        logger.info(f"Selected {len(symbols)} symbols from symbols file")
    elif top_n is not None:
        # Select top_n symbols by volatility
        logger.info(f"Selecting top {top_n} symbols by volatility")
        mean_volatility = rv_df.mean()
        symbols = mean_volatility.sort_values(ascending=False).head(top_n).index.tolist()
        logger.info(f"Selected {len(symbols)} symbols by volatility")
    else:
        # Use all symbols
        symbols = rv_df.columns.tolist()
        logger.info(f"Using all {len(symbols)} symbols")

    # Filter data to only include selected symbols
    rv_df = rv_df[symbols]

    logger.info(f"Loaded realized volatility data with shape {rv_df.shape}")
    return rv_df


def prepare_data(rv_df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                sequence_length=22, horizon=1, batch_size=32, scale_data=True):
    """
    Prepare the data for training, validation, and testing.

    Args:
        rv_df (pd.DataFrame): DataFrame containing the realized volatility data.
        train_ratio (float): Ratio of data to use for training.
        val_ratio (float): Ratio of data to use for validation.
        test_ratio (float): Ratio of data to use for testing.
        sequence_length (int): Length of input sequences.
        horizon (int): Prediction horizon.
        batch_size (int): Batch size for dataloaders.
        scale_data (bool): Whether to scale the data.

    Returns:
        tuple: (train_loader, val_loader, test_loader, scaler, date_indices)
    """
    logger.info("Preparing data for training, validation, and testing")

    # Convert to numpy array
    data = rv_df.values

    # Scale data if requested
    if scale_data:
        logger.info("Scaling data")
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
    else:
        scaler = None

    # Ensure we have enough data for the sequence length
    min_required_length = max(1, 5, 22) + horizon
    if len(data) < min_required_length:
        raise ValueError(f"Not enough data. Need at least {min_required_length} samples.")

    # Split data into train, validation, and test sets
    n_samples = len(rv_df)
    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * val_ratio)

    # Ensure we have enough data for each set
    if train_size < min_required_length:
        raise ValueError(f"Not enough training data. Need at least {min_required_length} samples.")
    if val_size < min_required_length:
        raise ValueError(f"Not enough validation data. Need at least {min_required_length} samples.")
    if n_samples - train_size - val_size < min_required_length:
        raise ValueError(f"Not enough test data. Need at least {min_required_length} samples.")

    train_data = data[:train_size]
    val_data = data[train_size:train_size+val_size]
    test_data = data[train_size+val_size:]

    logger.info(f"Train data shape: {train_data.shape}")
    logger.info(f"Validation data shape: {val_data.shape}")
    logger.info(f"Test data shape: {test_data.shape}")

    # Create datasets
    train_dataset = TimeSeriesDataset(train_data, sequence_length, horizon)
    val_dataset = TimeSeriesDataset(val_data, sequence_length, horizon)
    test_dataset = TimeSeriesDataset(test_data, sequence_length, horizon)

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset, batch_size
    )

    # Create date indices for each set
    date_indices = {
        'train': rv_df.index[sequence_length:train_size],
        'val': rv_df.index[train_size+sequence_length:train_size+val_size],
        'test': rv_df.index[train_size+val_size+sequence_length:]
    }

    logger.info("Data preparation completed")
    return train_loader, val_loader, test_loader, scaler, date_indices


def create_adjacency_matrix(rv_df, method='correlation'):
    """
    Create an adjacency matrix for the GSPHAR model.

    Args:
        rv_df (pd.DataFrame): DataFrame containing the realized volatility data.
        method (str): Method to create the adjacency matrix.
            Options: 'correlation', 'spillover', 'ones'

    Returns:
        np.ndarray: Adjacency matrix.
    """
    logger.info(f"Creating adjacency matrix using method: {method}")

    n_symbols = rv_df.shape[1]

    if method == 'correlation':
        # Create adjacency matrix based on correlation
        A = rv_df.corr().abs().values

        # Set diagonal to 1
        np.fill_diagonal(A, 1.0)

    elif method == 'spillover':
        # Try to load spillover matrix
        spillover_file = 'data/top20_spillover_adj.csv'
        if os.path.exists(spillover_file):
            logger.info(f"Loading spillover matrix from {spillover_file}")
            spillover_df = pd.read_csv(spillover_file, index_col=0)

            # Ensure the symbols match
            common_symbols = list(set(rv_df.columns) & set(spillover_df.index))
            if len(common_symbols) == n_symbols:
                # Reorder to match rv_df
                spillover_df = spillover_df.loc[rv_df.columns, rv_df.columns]
                A = spillover_df.values
            else:
                logger.warning(f"Spillover matrix symbols don't match. Using correlation instead.")
                A = rv_df.corr().abs().values
                np.fill_diagonal(A, 1.0)
        else:
            logger.warning(f"Spillover file not found. Using correlation instead.")
            A = rv_df.corr().abs().values
            np.fill_diagonal(A, 1.0)

    elif method == 'ones':
        # Create adjacency matrix with all ones
        A = np.ones((n_symbols, n_symbols))

    else:
        raise ValueError("Unsupported method. Use 'correlation', 'spillover', or 'ones'")

    logger.info(f"Created adjacency matrix with shape {A.shape}")
    return A


def train_model(model, train_loader, val_loader, optimizer, criterion, device,
               n_epochs=10, patience=5, model_dir='models', model_name=None):
    """
    Train the GSPHAR model.

    Args:
        model (nn.Module): GSPHAR model.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        optimizer (Optimizer): Optimizer.
        criterion (Loss): Loss function.
        device (torch.device): Device to train on.
        n_epochs (int): Number of epochs to train for.
        patience (int): Patience for early stopping.
        model_dir (str): Directory to save models.
        model_name (str, optional): Name for the model. If None, a name will be generated.

    Returns:
        tuple: (model, train_losses, val_losses, best_epoch)
    """
    logger.info(f"Training model for {n_epochs} epochs")

    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)

    # Generate model name if not provided
    if model_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = f"gsphar_crypto_{timestamp}"

    # Initialize variables
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0

    # Move model to device
    model = model.to(device)

    # Training loop
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_idx, (x_lag1, x_lag5, x_lag22, y) in enumerate(train_loader):
            # Print input shapes and types for debugging (only for the first batch of the first epoch)
            if epoch == 0 and batch_idx == 0:
                logger.info(f"Input shapes:")
                logger.info(f"x_lag1: {x_lag1.shape}, type: {x_lag1.dtype}")
                logger.info(f"x_lag5: {x_lag5.shape}, type: {x_lag5.dtype}")
                logger.info(f"x_lag22: {x_lag22.shape}, type: {x_lag22.dtype}")
                logger.info(f"y: {y.shape}, type: {y.dtype}")

                # Print some sample values
                logger.info(f"Sample x_lag1 values: {x_lag1[0, :, 0][:5]}")
                logger.info(f"Sample x_lag5 values: {x_lag5[0, :, 0][:5]}")
                logger.info(f"Sample x_lag22 values: {x_lag22[0, :, 0][:5]}")
                logger.info(f"Sample y values: {y[0, :, 0][:5]}")

            # Move data to device
            x_lag1, x_lag5, x_lag22, y = x_lag1.to(device), x_lag5.to(device), x_lag22.to(device), y.to(device)

            # Forward pass
            optimizer.zero_grad()

            # Print model input shapes after moving to device (only for the first batch of the first epoch)
            if epoch == 0 and batch_idx == 0:
                logger.info(f"Model input shapes (after moving to device):")
                logger.info(f"x_lag1: {x_lag1.shape}, type: {x_lag1.dtype}")
                logger.info(f"x_lag5: {x_lag5.shape}, type: {x_lag5.dtype}")
                logger.info(f"x_lag22: {x_lag22.shape}, type: {x_lag22.dtype}")

            y_pred = model(x_lag1, x_lag5, x_lag22)

            # Print output shape (only for the first batch of the first epoch)
            if epoch == 0 and batch_idx == 0:
                logger.info(f"Model output shape: {y_pred.shape}, type: {y_pred.dtype}")
                logger.info(f"Target shape: {y.shape}, type: {y.dtype}")

            loss = criterion(y_pred, y)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update loss
            train_loss += loss.item()

        # Calculate average training loss
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (x_lag1, x_lag5, x_lag22, y) in enumerate(val_loader):
                # Move data to device
                x_lag1, x_lag5, x_lag22, y = x_lag1.to(device), x_lag5.to(device), x_lag22.to(device), y.to(device)

                # Forward pass
                y_pred = model(x_lag1, x_lag5, x_lag22)
                loss = criterion(y_pred, y)

                # Update loss
                val_loss += loss.item()

        # Calculate average validation loss
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Print progress
        logger.info(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Check if this is the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            # Save the best model
            save_model(f"{model_name}_best", model, best_loss_val=best_val_loss)
            logger.info(f"Saved best model at epoch {epoch+1}")
        else:
            patience_counter += 1

            # Check if we should stop early
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_model(f"{model_name}_epoch{epoch+1}", model, best_loss_val=val_loss)

    # Save the final model
    save_model(f"{model_name}_final", model, best_loss_val=val_loss)
    logger.info(f"Saved final model after {epoch+1} epochs")

    # Load the best model
    model, _ = load_model(f"{model_name}_best", model)
    logger.info(f"Loaded best model from epoch {best_epoch+1}")

    return model, train_losses, val_losses, best_epoch


def evaluate_model(model, test_loader, criterion, device, scaler=None, date_indices=None,
                  rv_df=None, output_dir='plots'):
    """
    Evaluate the GSPHAR model on the test set.

    Args:
        model (nn.Module): GSPHAR model.
        test_loader (DataLoader): Test data loader.
        criterion (Loss): Loss function.
        device (torch.device): Device to evaluate on.
        scaler (StandardScaler, optional): Scaler used to normalize the data.
        date_indices (dict, optional): Dictionary of date indices for each set.
        rv_df (pd.DataFrame, optional): Original realized volatility DataFrame.
        output_dir (str): Directory to save evaluation results.

    Returns:
        dict: Dictionary of evaluation metrics.
    """
    logger.info("Evaluating model on test set")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Move model to device
    model = model.to(device)

    # Evaluation
    model.eval()
    test_loss = 0.0
    predictions = []
    targets = []

    with torch.no_grad():
        for batch_idx, (x_lag1, x_lag5, x_lag22, y) in enumerate(test_loader):
            # Move data to device
            x_lag1, x_lag5, x_lag22, y = x_lag1.to(device), x_lag5.to(device), x_lag22.to(device), y.to(device)

            # Forward pass
            y_pred = model(x_lag1, x_lag5, x_lag22)
            loss = criterion(y_pred, y)

            # Update loss
            test_loss += loss.item()

            # Store predictions and targets
            predictions.append(y_pred.cpu().numpy())
            targets.append(y.cpu().numpy())

    # Calculate average test loss
    test_loss /= len(test_loader)
    logger.info(f"Test Loss: {test_loss:.6f}")

    # Concatenate predictions and targets
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)

    # Reshape predictions and targets for inverse transform
    # They are currently [batch_size, n_features, horizon]
    # We need to reshape to [batch_size * horizon, n_features]
    batch_size, n_features, horizon = predictions.shape
    predictions_reshaped = predictions.reshape(-1, n_features)
    targets_reshaped = targets.reshape(-1, n_features)

    # Inverse transform if scaler is provided
    if scaler is not None:
        predictions_reshaped = scaler.inverse_transform(predictions_reshaped)
        targets_reshaped = scaler.inverse_transform(targets_reshaped)

    # Reshape back to original shape
    predictions = predictions_reshaped.reshape(batch_size, n_features, horizon)
    targets = targets_reshaped.reshape(batch_size, n_features, horizon)

    # Calculate metrics
    metrics = {}
    metrics['test_loss'] = test_loss

    # Flatten predictions and targets for metric calculation
    predictions_flat = predictions.reshape(-1)
    targets_flat = targets.reshape(-1)

    metrics['mse'] = mean_squared_error(targets_flat, predictions_flat)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['mae'] = mean_absolute_error(targets_flat, predictions_flat)
    metrics['r2'] = r2_score(targets_flat, predictions_flat)

    # Print metrics
    logger.info(f"MSE: {metrics['mse']:.6f}")
    logger.info(f"RMSE: {metrics['rmse']:.6f}")
    logger.info(f"MAE: {metrics['mae']:.6f}")
    logger.info(f"R^2: {metrics['r2']:.6f}")

    # Save metrics
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # Create visualizations if date_indices and rv_df are provided
    if date_indices is not None and rv_df is not None:
        # Get test dates
        test_dates = date_indices['test']

        # Ensure we have the same number of dates as predictions
        if len(test_dates) > predictions.shape[0]:
            test_dates = test_dates[:predictions.shape[0]]

        # Create DataFrame with predictions and targets
        results_df = pd.DataFrame(index=test_dates)

        # Add predictions and targets for each symbol
        for i, symbol in enumerate(rv_df.columns):
            results_df[f"{symbol}_pred"] = predictions[:, i, 0]
            results_df[f"{symbol}_true"] = targets[:, i, 0]

        # Save results
        results_df.to_csv(os.path.join(output_dir, 'predictions.csv'))

        # Create plots for each symbol
        for i, symbol in enumerate(rv_df.columns):
            plt.figure(figsize=(12, 6))
            plt.plot(test_dates, targets[:, i, 0], label='True')
            plt.plot(test_dates, predictions[:, i, 0], label='Predicted')
            plt.title(f"{symbol} - True vs Predicted")
            plt.xlabel('Date')
            plt.ylabel('Realized Volatility')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{symbol}_prediction.png"), dpi=300)
            plt.close()

    logger.info("Evaluation completed")
    return metrics


def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(description='Train the GSPHAR model on cryptocurrency data.')
    parser.add_argument('--rv-file', type=str, default='data/crypto_rv1h_38_20200822_20250116.csv',
                        help='Path to the realized volatility file.')
    parser.add_argument('--symbols-file', type=str, default='data/universe/crypto_spillover_top20_to_others_symbols.txt',
                        help='Path to the file containing the symbols to use.')
    parser.add_argument('--top-n', type=int, default=None,
                        help='Number of top symbols to use if symbols-file is not provided.')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='Ratio of data to use for training.')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                        help='Ratio of data to use for validation.')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                        help='Ratio of data to use for testing.')
    parser.add_argument('--sequence-length', type=int, default=22,
                        help='Length of input sequences.')
    parser.add_argument('--horizon', type=int, default=1,
                        help='Prediction horizon.')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for dataloaders.')
    parser.add_argument('--scale-data', action='store_true', default=True,
                        help='Whether to scale the data.')
    parser.add_argument('--adjacency-method', type=str, default='correlation',
                        choices=['correlation', 'spillover', 'ones'],
                        help='Method to create the adjacency matrix.')
    parser.add_argument('--n-epochs', type=int, default=15,
                        help='Number of epochs to train for.')
    parser.add_argument('--patience', type=int, default=5,
                        help='Patience for early stopping.')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate for the optimizer.')
    parser.add_argument('--weight-decay', type=float, default=0.0001,
                        help='Weight decay for the optimizer.')
    parser.add_argument('--model-dir', type=str, default='models',
                        help='Directory to save models.')
    parser.add_argument('--output-dir', type=str, default='plots/gsphar_crypto',
                        help='Directory to save evaluation results.')
    parser.add_argument('--model-name', type=str, default=None,
                        help='Name for the model. If None, a name will be generated.')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda', 'mps'],
                        help='Device to train on.')

    args = parser.parse_args()

    # Set device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    logger.info(f"Using device: {device}")

    # Load data
    rv_df = load_data(args.rv_file, args.symbols_file, args.top_n)

    # Prepare data
    train_loader, val_loader, test_loader, scaler, date_indices = prepare_data(
        rv_df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        sequence_length=args.sequence_length,
        horizon=args.horizon,
        batch_size=args.batch_size,
        scale_data=args.scale_data
    )

    # Create adjacency matrix
    A = create_adjacency_matrix(rv_df, method=args.adjacency_method)

    # Create model
    model = GSPHAR(
        input_dim=3,  # 3 lags: lag1, lag5, lag22
        output_dim=args.horizon,
        filter_size=rv_df.shape[1],
        A=A
    )

    # Create optimizer and loss function
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    criterion = nn.MSELoss()

    # Train model
    model, train_losses, val_losses, best_epoch = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        n_epochs=args.n_epochs,
        patience=args.patience,
        model_dir=args.model_dir,
        model_name=args.model_name
    )

    # Evaluate model
    metrics = evaluate_model(
        model,
        test_loader,
        criterion,
        device,
        scaler=scaler,
        date_indices=date_indices,
        rv_df=rv_df,
        output_dir=args.output_dir
    )

    logger.info("Training and evaluation completed successfully")


if __name__ == '__main__':
    main()
