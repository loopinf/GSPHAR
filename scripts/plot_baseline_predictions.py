#!/usr/bin/env python
"""
Script to plot baseline predictions from the GSPHAR model trained with MSE loss.
This script loads the trained model, makes predictions on the test set,
and plots the time series of predictions versus actual values.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import argparse
import os
import sys
from pathlib import Path
import datetime

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from local modules
from config import settings
from src.models import GSPHAR
from src.utils import load_model
from scripts.load_pct_change_data import load_pct_change_data, prepare_data_for_gsphar


def load_top20_symbols(symbols_file='data/top20_symbols.txt'):
    """
    Load the list of top 20 symbols from a file.

    Args:
        symbols_file (str): Path to the file containing the symbols.

    Returns:
        list: List of symbols.
    """
    try:
        with open(symbols_file, 'r') as f:
            symbols = [line.strip() for line in f.readlines()]
        print(f"Loaded {len(symbols)} symbols from {symbols_file}")
        return symbols
    except Exception as e:
        print(f"Error loading symbols from {symbols_file}: {e}")
        return None


def load_model_and_predict(model_path, dataloader, device='cpu', filter_size=20):
    """
    Load a trained model and make predictions on the given dataloader.

    Args:
        model_path (str): Path to the trained model.
        dataloader (torch.utils.data.DataLoader): DataLoader for the test set.
        device (str): Device to use for inference.
        filter_size (int): Filter size for the model.

    Returns:
        tuple: (predictions, targets) as numpy arrays.
    """
    # Create a new model instance
    from src.models import GSPHAR
    from src.utils import load_model

    # Load the adjacency matrix
    try:
        adj_matrix = np.load('data/top20_spillover_adj.npy')
        print(f"Loaded adjacency matrix with shape {adj_matrix.shape}")
    except Exception as e:
        print(f"Error loading adjacency matrix: {e}")
        print("Using identity matrix as fallback")
        adj_matrix = np.eye(filter_size)

    # Create a new model instance
    model = GSPHAR(input_dim=3, output_dim=1, filter_size=filter_size, A=adj_matrix)

    # Load the trained weights
    model, _ = load_model(os.path.basename(model_path).replace('.pt', ''), model)

    model.to(device)
    model.eval()

    # Make predictions
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for x_lag1, x_lag5, x_lag22, y in dataloader:
            x_lag1 = x_lag1.to(device)
            x_lag5 = x_lag5.to(device)
            x_lag22 = x_lag22.to(device)
            y = y.to(device)

            # Forward pass
            outputs, _, _ = model(x_lag1, x_lag5, x_lag22)

            # Store predictions and targets
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    # Concatenate batches
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    return predictions, targets


def get_dates_from_test_dataset(test_dataset, dataloader_test):
    """
    Extract dates from the test dataset.

    Args:
        test_dataset (pd.DataFrame): Test dataset with DatetimeIndex.
        dataloader_test (torch.utils.data.DataLoader): Test dataloader.

    Returns:
        list: List of dates for the predictions.
    """
    # Count the total number of samples in the dataloader
    total_samples = 0
    for batch in dataloader_test:
        total_samples += batch[0].shape[0]

    # Get the dates from the test dataset
    dates = test_dataset.index[-total_samples:].tolist()

    return dates


def plot_predictions_vs_actual(predictions, targets, dates, symbols, output_dir='plots', prefix='baseline'):
    """
    Plot predictions versus actual values for each symbol.

    Args:
        predictions (np.ndarray): Model predictions.
        targets (np.ndarray): Ground truth values.
        dates (list): List of dates for the predictions.
        symbols (list): List of symbols.
        output_dir (str): Directory to save plots.
        prefix (str): Prefix for plot filenames.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Convert dates to datetime if they're not already
    if not isinstance(dates[0], datetime.datetime):
        dates = [pd.to_datetime(date) for date in dates]

    # Print shapes for debugging
    print(f"Predictions shape: {predictions.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Number of dates: {len(dates)}")
    print(f"Number of symbols: {len(symbols)}")

    # Ensure we have the right number of dates
    if len(dates) != predictions.shape[0]:
        print(f"Warning: Number of dates ({len(dates)}) doesn't match number of predictions ({predictions.shape[0]})")
        # Adjust dates if needed
        if len(dates) > predictions.shape[0]:
            dates = dates[-predictions.shape[0]:]
        else:
            # Pad with None if we have more predictions than dates
            dates = dates + [None] * (predictions.shape[0] - len(dates))

    # Create a figure for each symbol
    for i, symbol in enumerate(symbols):
        plt.figure(figsize=(12, 6))

        # Extract predictions and targets for this symbol
        # Handle different dimensions based on the actual shape
        if len(predictions.shape) == 3:
            symbol_predictions = predictions[:, i, 0]
            symbol_targets = targets[:, i, 0]
        else:
            # Assuming batch_size x num_symbols format
            symbol_predictions = predictions[:, i]
            symbol_targets = targets[:, i]

        # Plot predictions and targets
        plt.plot(dates, symbol_targets, label='Actual', color='blue', alpha=0.7)
        plt.plot(dates, symbol_predictions, label='Predicted', color='red', alpha=0.7)

        # Calculate and display metrics
        mse = np.mean((symbol_predictions - symbol_targets) ** 2)
        mae = np.mean(np.abs(symbol_predictions - symbol_targets))

        # Add title and labels
        plt.title(f'{symbol} - Predictions vs Actual (MSE: {mse:.4f}, MAE: {mae:.4f})')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()

        # Format x-axis to show dates nicely
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.gcf().autofmt_xdate()

        # Add grid
        plt.grid(True, alpha=0.3)

        # Save the plot
        plot_path = os.path.join(output_dir, f'{prefix}_{symbol}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot for {symbol} to {plot_path}")

        # Close the figure to free memory
        plt.close()

    # Create a summary plot with all symbols
    plt.figure(figsize=(15, 10))

    # Calculate average MSE and MAE across all symbols
    if len(predictions.shape) == 3:
        avg_mse = np.mean((predictions[:, :, 0] - targets[:, :, 0]) ** 2)
        avg_mae = np.mean(np.abs(predictions[:, :, 0] - targets[:, :, 0]))
        avg_predictions = np.mean(predictions[:, :, 0], axis=1)
        avg_targets = np.mean(targets[:, :, 0], axis=1)
    else:
        avg_mse = np.mean((predictions - targets) ** 2)
        avg_mae = np.mean(np.abs(predictions - targets))
        avg_predictions = np.mean(predictions, axis=1)
        avg_targets = np.mean(targets, axis=1)

    # Plot average predictions and targets
    plt.plot(dates, avg_targets, label='Average Actual', color='blue', linewidth=2)
    plt.plot(dates, avg_predictions, label='Average Predicted', color='red', linewidth=2)

    # Add title and labels
    plt.title(f'Average Across All Symbols (MSE: {avg_mse:.4f}, MAE: {avg_mae:.4f})')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()

    # Format x-axis to show dates nicely
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.gcf().autofmt_xdate()

    # Add grid
    plt.grid(True, alpha=0.3)

    # Save the plot
    plot_path = os.path.join(output_dir, f'{prefix}_average.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved average plot to {plot_path}")

    # Close the figure to free memory
    plt.close()


def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(description='Plot baseline predictions from the GSPHAR model.')
    parser.add_argument('--data-file', type=str, default='data/daily_pct_change_crypto.csv',
                        help='Path to the data file.')
    parser.add_argument('--symbols-file', type=str, default='data/top20_symbols.txt',
                        help='Path to the file containing the top 20 symbols.')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to the trained model. If None, will use the latest best model.')
    parser.add_argument('--horizon', type=int, default=1,
                        help='Prediction horizon.')
    parser.add_argument('--look-back', type=int, default=22,
                        help='Look-back window size.')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size.')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use for inference.')
    parser.add_argument('--output-dir', type=str, default='plots',
                        help='Directory to save plots.')
    parser.add_argument('--prefix', type=str, default='baseline',
                        help='Prefix for plot filenames.')

    args = parser.parse_args()

    # Load top 20 symbols
    top_symbols = load_top20_symbols(args.symbols_file)
    if top_symbols is None:
        print("Failed to load symbols. Exiting.")
        return

    # Load data
    print(f"Loading data from {args.data_file}...")
    data = load_pct_change_data(args.data_file)

    if data is None:
        print("Failed to load data. Exiting.")
        return

    # Filter data to include only top 20 symbols
    filtered_data = data[top_symbols].copy()
    print(f"Filtered data shape: {filtered_data.shape}")

    # Prepare data for GSPHAR
    print("Preparing data for GSPHAR...")
    dataloader_train, dataloader_test, train_dataset, test_dataset, market_indices_list, _ = prepare_data_for_gsphar(
        filtered_data, 0.7, args.horizon, args.look_back, args.batch_size
    )

    # Determine model path
    if args.model_path is None:
        model_path = os.path.join(settings.MODEL_DIR, f"GSPHAR_20_h{args.horizon}_top20_spillover_latest_best.pt")
    else:
        model_path = args.model_path

    print(f"Using model: {model_path}")

    # Load model and make predictions
    predictions, targets = load_model_and_predict(model_path, dataloader_test, args.device)

    # Get dates for the predictions
    dates = get_dates_from_test_dataset(test_dataset, dataloader_test)

    # Plot predictions versus actual values
    plot_predictions_vs_actual(predictions, targets, dates, top_symbols, args.output_dir, args.prefix)

    print("Plotting completed successfully.")


if __name__ == '__main__':
    main()
