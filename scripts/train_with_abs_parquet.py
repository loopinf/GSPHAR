#!/usr/bin/env python
"""
Training script for GSPHAR using absolute percentage change data from parquet file.
This script loads the df_cl_5m.parquet file, calculates percentage changes,
takes the absolute value, and trains a GSPHAR model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import sys
import datetime
import shutil
import numpy as np
import pandas as pd

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from local modules
from config import settings
from src.models import GSPHAR
from src.training import GSPHARTrainer
from src.utils import load_model, save_model
from src.utils.device_utils import set_device_seeds
from src.training.custom_losses import WeightedMSELoss, AsymmetricMSELoss, ThresholdMSELoss, HybridLoss
from src.data import split_data, create_lagged_features, prepare_data_dict, create_dataloaders
from src.utils.graph_utils import compute_spillover_index


def load_parquet_and_process(filepath, resample='D', top_n=20, method='abs_pct_change'):
    """
    Load parquet file, calculate percentage changes, and take absolute value.

    Args:
        filepath (str): Path to the parquet file.
        resample (str): Frequency to resample the data to (e.g., 'D' for daily).
        top_n (int): Number of top symbols to select.
        method (str): Method to process the data ('abs_pct_change', 'pct_change', 'volatility').

    Returns:
        pd.DataFrame: Processed data.
    """
    try:
        # Load the parquet file
        print(f"Loading parquet file from {filepath}...")
        df = pd.read_parquet(filepath)
        print(f"Loaded data with shape {df.shape}")

        # Check if the dataframe has a datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            # Try to find a datetime column
            datetime_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
            if datetime_cols:
                print(f"Setting index to {datetime_cols[0]}")
                df[datetime_cols[0]] = pd.to_datetime(df[datetime_cols[0]])
                df.set_index(datetime_cols[0], inplace=True)
            else:
                raise ValueError("No datetime column found in the dataframe")

        # For this specific format, the symbols are already columns
        print(f"Found {len(df.columns)} symbols as columns")

        # Resample to the specified frequency
        if resample:
            print(f"Resampling to {resample} frequency...")
            df = df.resample(resample).last()

        # Calculate percentage changes for each symbol
        print("Calculating percentage changes...")
        pct_changes = df.pct_change().dropna()

        # Process according to the specified method
        if method == 'abs_pct_change':
            print("Taking absolute values of percentage changes...")
            processed_data = pct_changes.abs()
        elif method == 'pct_change':
            processed_data = pct_changes
        elif method == 'volatility':
            # Calculate rolling volatility (standard deviation)
            print("Calculating rolling volatility...")
            processed_data = pct_changes.rolling(window=5).std().dropna()
        else:
            raise ValueError(f"Unknown method: {method}")

        # Drop rows with all NaN values
        processed_data = processed_data.dropna(how='all')

        print(f"Processed data shape: {processed_data.shape}")
        print(f"Date range: {processed_data.index[0]} to {processed_data.index[-1]}")

        # Select top N symbols based on data availability
        if top_n > 0 and top_n < len(processed_data.columns):
            # Count non-NaN values for each symbol
            data_counts = processed_data.count()
            # Sort by data availability (descending)
            sorted_symbols = data_counts.sort_values(ascending=False).index.tolist()
            # Select top symbols
            selected_symbols = sorted_symbols[:top_n]

            print(f"Selected top {len(selected_symbols)} symbols based on data availability:")
            for i, symbol in enumerate(selected_symbols, 1):
                print(f"{i}. {symbol}: {data_counts[symbol]} data points")

            # Filter data to include only selected symbols
            processed_data = processed_data[selected_symbols]

        return processed_data

    except Exception as e:
        print(f"Error loading and processing parquet file: {e}")
        return None


def prepare_data_for_gsphar(data, train_ratio=0.7, h=1, look_back_window=22, batch_size=32):
    """
    Prepare data for GSPHAR model.

    Args:
        data (pd.DataFrame): Input data.
        train_ratio (float): Ratio for train/test split.
        h (int): Prediction horizon.
        look_back_window (int): Number of lagged observations to use.
        batch_size (int): Batch size for dataloaders.

    Returns:
        tuple: (dataloader_train, dataloader_test, train_dataset, test_dataset, market_indices_list, DY_adj)
    """
    # Split data
    train_dataset_raw, test_dataset_raw = split_data(data, train_ratio)

    # Get market indices
    market_indices_list = train_dataset_raw.columns.tolist()

    # Compute spillover index
    try:
        DY_adj = compute_spillover_index(train_dataset_raw, h, look_back_window, 0.0, standardized=True)
    except Exception as e:
        print(f"Error computing spillover index: {e}")
        print("Using a simple adjacency matrix instead.")
        # Create a simple adjacency matrix as fallback
        n = len(market_indices_list)
        DY_adj = np.ones((n, n))  # Full connectivity

    # Create lagged features
    train_dataset = create_lagged_features(train_dataset_raw, market_indices_list, h, look_back_window)
    test_dataset = create_lagged_features(test_dataset_raw, market_indices_list, h, look_back_window)

    # Prepare data dictionaries
    train_dict = prepare_data_dict(train_dataset, market_indices_list, look_back_window)
    test_dict = prepare_data_dict(test_dataset, market_indices_list, look_back_window)

    # Create dataloaders
    dataloader_train, dataloader_test = create_dataloaders(train_dict, test_dict, batch_size)

    return dataloader_train, dataloader_test, train_dataset, test_dataset, market_indices_list, DY_adj


def parse_args():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Train a GSPHAR model on absolute percentage change data from parquet file.')
    parser.add_argument('--data-file', type=str, default='data/df_cl_5m.parquet',
                        help='Path to the parquet file.')
    parser.add_argument('--resample', type=str, default='D',
                        help='Frequency to resample the data to (e.g., "D" for daily).')
    parser.add_argument('--method', type=str, default='abs_pct_change',
                        choices=['abs_pct_change', 'pct_change', 'volatility'],
                        help='Method to process the data.')
    parser.add_argument('--top-n', type=int, default=20,
                        help='Number of top symbols to select based on data availability.')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='Ratio for train/test split.')
    parser.add_argument('--horizon', type=int, default=1,
                        help='Prediction horizon.')
    parser.add_argument('--look-back', type=int, default=22,
                        help='Look-back window size.')
    parser.add_argument('--input-dim', type=int, default=3,
                        help='Input dimension.')
    parser.add_argument('--output-dim', type=int, default=1,
                        help='Output dimension.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size.')
    parser.add_argument('--patience', type=int, default=20,
                        help='Patience for early stopping.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed.')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use for training.')
    parser.add_argument('--tag', type=str, default='',
                        help='Add a custom tag to the model name.')

    # Custom loss function arguments
    parser.add_argument('--loss-fn', type=str, default='mse',
                        choices=['mse', 'weighted_mse', 'asymmetric_mse', 'threshold_mse', 'hybrid'],
                        help='Loss function to use for training.')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for weighted and threshold loss functions.')
    parser.add_argument('--weight-factor', type=float, default=5.0,
                        help='Weight factor for weighted loss function.')

    return parser.parse_args()


def main():
    """
    Main function.
    """
    # Parse arguments
    args = parse_args()

    # Set random seed for all devices
    set_device_seeds(seed=args.seed, device=args.device)

    # Load and process data
    data = load_parquet_and_process(
        args.data_file,
        resample=args.resample,
        top_n=args.top_n,
        method=args.method
    )

    if data is None:
        print("Failed to load and process data. Exiting.")
        return

    # Prepare data for GSPHAR
    print("Preparing data for GSPHAR...")
    dataloader_train, dataloader_test, train_dataset, test_dataset, market_indices_list, DY_adj = prepare_data_for_gsphar(
        data, args.train_ratio, args.horizon, args.look_back, args.batch_size
    )

    # Create model
    filter_size = len(market_indices_list)
    print(f"Creating GSPHAR model with filter_size={filter_size}...")
    model = GSPHAR(args.input_dim, args.output_dim, filter_size, DY_adj)

    # Create optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=len(dataloader_train),
        epochs=args.epochs,
        three_phase=True
    )

    # Select loss function based on arguments
    if args.loss_fn == 'mse':
        criterion = nn.MSELoss()
        print("Using standard MSE loss")
    elif args.loss_fn == 'weighted_mse':
        criterion = WeightedMSELoss(threshold=args.threshold, weight_factor=args.weight_factor)
        print(f"Using weighted MSE loss with threshold={args.threshold}, weight_factor={args.weight_factor}")
    elif args.loss_fn == 'asymmetric_mse':
        criterion = AsymmetricMSELoss(under_prediction_factor=args.weight_factor)
        print(f"Using asymmetric MSE loss with under_prediction_factor={args.weight_factor}")
    elif args.loss_fn == 'threshold_mse':
        thresholds = [0.2, 0.5, 1.0]
        weights = [1.0, 2.0, 5.0, 10.0]
        criterion = ThresholdMSELoss(thresholds=thresholds, weights=weights)
        print(f"Using threshold MSE loss with thresholds={thresholds}, weights={weights}")
    elif args.loss_fn == 'hybrid':
        criterion = HybridLoss(
            mse_weight=1.0,
            large_jump_weight=2.0,
            threshold=args.threshold,
            jump_factor=args.weight_factor
        )
        print(f"Using hybrid loss with threshold={args.threshold}, jump_factor={args.weight_factor}")
    else:
        criterion = nn.MSELoss()
        print("Using default MSE loss")

    # Create trainer
    trainer = GSPHARTrainer(
        model=model,
        device=args.device,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler
    )

    # Create a unique model name with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_model_name = f"GSPHAR_{filter_size}_h{args.horizon}_{args.method}"

    # Add loss function info to the model name
    if args.loss_fn != 'mse':
        base_model_name = f"{base_model_name}_{args.loss_fn}"

    # Add tag if provided
    if args.tag:
        base_model_name = f"{base_model_name}_{args.tag}"

    # Add timestamp
    model_save_name = f"{base_model_name}_{timestamp}"

    print(f"Model will be saved as: {model_save_name}")

    # Train the model
    print(f"Training model for {args.epochs} epochs with patience {args.patience}...")
    best_loss_val, _, _, train_loss_list, test_loss_list = trainer.train(
        dataloader_train=dataloader_train,
        dataloader_test=dataloader_test,
        num_epochs=args.epochs,
        patience=args.patience,
        model_save_name=model_save_name
    )

    # Create a final model name with validation loss
    final_model_name = f"{base_model_name}_best_val{best_loss_val:.4f}"

    # Create a "latest_best" model name that's easy to reference
    latest_best_name = f"{base_model_name}_latest_best"

    # Rename the model file to include validation loss
    model_path = os.path.join(settings.MODEL_DIR, f"{model_save_name}.pt")
    final_model_path = os.path.join(settings.MODEL_DIR, f"{final_model_name}.pt")
    latest_best_path = os.path.join(settings.MODEL_DIR, f"{latest_best_name}.pt")

    if os.path.exists(model_path):
        # Copy to the final model name with validation score
        shutil.copy(model_path, final_model_path)
        print(f"Saved best model as: {final_model_name}")

        # Create a symlink for "latest_best" that points to the final model
        # Remove existing symlink if it exists
        if os.path.exists(latest_best_path) or os.path.islink(latest_best_path):
            os.remove(latest_best_path)

        # Create a relative symlink
        os.symlink(os.path.basename(final_model_path), latest_best_path)
        print(f"Created symlink: {latest_best_name} -> {os.path.basename(final_model_path)}")

    print(f"Training completed. Best validation loss: {best_loss_val:.4f}")
    print(f"Best model saved as: {latest_best_name} (and as {final_model_name})")


if __name__ == '__main__':
    main()
