#!/usr/bin/env python
"""
Evaluation script for GSPHAR.
This script evaluates a trained GSPHAR model on the test data.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import sys
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Add the project root directory to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Import from local modules
from config import settings
from src.data import load_data, split_data, create_lagged_features, prepare_data_dict, create_dataloaders
from src.models import GSPHAR
from src.utils import compute_spillover_index, load_model
from src.utils.device_utils import set_device_seeds


def parse_args():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Evaluate a trained GSPHAR model.')
    parser.add_argument('--data-file', type=str, default=settings.DATA_FILE,
                        help='Path to the data file.')
    parser.add_argument('--train-split', type=float, default=settings.TRAIN_SPLIT_RATIO,
                        help='Ratio for train/test split.')
    parser.add_argument('--horizon', type=int, default=settings.PREDICTION_HORIZON,
                        help='Prediction horizon.')
    parser.add_argument('--look-back', type=int, default=settings.LOOK_BACK_WINDOW,
                        help='Look-back window size.')
    parser.add_argument('--input-dim', type=int, default=settings.INPUT_DIM,
                        help='Input dimension.')
    parser.add_argument('--output-dim', type=int, default=settings.OUTPUT_DIM,
                        help='Output dimension.')
    parser.add_argument('--filter-size', type=int, default=settings.FILTER_SIZE,
                        help='Filter size.')
    parser.add_argument('--batch-size', type=int, default=settings.BATCH_SIZE,
                        help='Batch size.')
    parser.add_argument('--seed', type=int, default=settings.SEED,
                        help='Random seed.')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use for evaluation (cpu recommended for torch.complex).')
    parser.add_argument('--model-name', type=str, default=None,
                        help='Name of the model to evaluate. If None, use the default pattern.')
    parser.add_argument('--plot', action='store_true',
                        help='Whether to plot the predictions.')
    parser.add_argument('--save-results', action='store_true',
                        help='Whether to save the results to a CSV file.')
    return parser.parse_args()


def generate_predictions(model, dataloader, market_indices_list, device):
    """
    Generate predictions using the model.

    Args:
        model (nn.Module): GSPHAR model.
        dataloader (DataLoader): Test dataloader.
        market_indices_list (list): List of market indices.
        device (str): Device to use for evaluation.

    Returns:
        tuple: (predictions_df, actuals_df)
    """
    model.eval()
    all_predictions = []
    all_actuals = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating predictions"):
            x_lag1, x_lag5, x_lag22, y = batch

            # Move to device
            x_lag1 = x_lag1.to(device)
            x_lag5 = x_lag5.to(device)
            x_lag22 = x_lag22.to(device)

            # Generate predictions
            output, _, _ = model(x_lag1, x_lag5, x_lag22)

            # Store predictions and actuals
            all_predictions.append(output.cpu().numpy())
            all_actuals.append(y.numpy())

    # Concatenate all predictions and actuals
    all_predictions = np.vstack(all_predictions)
    all_actuals = np.vstack(all_actuals)

    # Create DataFrames
    pred_df = pd.DataFrame(all_predictions, columns=market_indices_list)
    actual_df = pd.DataFrame(all_actuals, columns=market_indices_list)

    return pred_df, actual_df


def calculate_metrics(predictions, actuals, market_indices):
    """
    Calculate performance metrics for each market index.

    Args:
        predictions (DataFrame): Predictions DataFrame.
        actuals (DataFrame): Actuals DataFrame.
        market_indices (list): List of market indices.

    Returns:
        dict: Dictionary of metrics for each market index.
    """
    metrics = {}

    for market_index in market_indices:
        y_pred = predictions[market_index]
        y_true = actuals[market_index]

        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)

        metrics[market_index] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae
        }

    return metrics


def plot_predictions(predictions, actuals, market_indices, save_dir=None):
    """
    Plot predictions for each market index.

    Args:
        predictions (DataFrame): Predictions DataFrame.
        actuals (DataFrame): Actuals DataFrame.
        market_indices (list): List of market indices.
        save_dir (str, optional): Directory to save plots. Defaults to None.
    """
    # Set plotting style
    plt.style.use('ggplot')
    sns.set(style="darkgrid")

    # Create save directory if it doesn't exist
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # Plot predictions for each market index
    for market_index in market_indices:
        plt.figure(figsize=(12, 6))

        plt.plot(actuals[market_index], 'k-', label='Actual')
        plt.plot(predictions[market_index], 'b-', label='GSPHAR')

        plt.title(f'Volatility Predictions for {market_index}')
        plt.xlabel('Time')
        plt.ylabel('Volatility')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, f'{market_index}_predictions.png'))
            plt.close()
        else:
            plt.show()


def main():
    """
    Main function.
    """
    # Parse arguments
    args = parse_args()

    # Set random seed
    set_device_seeds(seed=args.seed, device=args.device)

    # Load data
    print(f"Loading data from {args.data_file}...")
    data = load_data(args.data_file)

    # Split data
    print(f"Splitting data with ratio {args.train_split}...")
    train_dataset_raw, test_dataset_raw = split_data(data, args.train_split)

    # Get market indices
    market_indices_list = train_dataset_raw.columns.tolist()

    # Compute spillover index
    print(f"Computing spillover index with horizon {args.horizon} and lag {args.look_back}...")
    DY_adj = compute_spillover_index(train_dataset_raw, args.horizon, args.look_back, 0.0, standardized=True)

    # Create lagged features
    print("Creating lagged features...")
    train_dataset = create_lagged_features(train_dataset_raw, market_indices_list, args.horizon, args.look_back)
    test_dataset = create_lagged_features(test_dataset_raw, market_indices_list, args.horizon, args.look_back)

    # Prepare data dictionaries
    print("Preparing data dictionaries...")
    train_dict = prepare_data_dict(train_dataset, market_indices_list, args.look_back)
    test_dict = prepare_data_dict(test_dataset, market_indices_list, args.look_back)

    # Create dataloaders
    print(f"Creating dataloaders with batch size {args.batch_size}...")
    _, dataloader_test = create_dataloaders(train_dict, test_dict, args.batch_size)

    # Create model
    print("Creating model...")
    model = GSPHAR(args.input_dim, args.output_dim, args.filter_size, DY_adj)

    # Determine model name
    if args.model_name is None:
        model_save_name = settings.MODEL_SAVE_NAME_PATTERN.format(
            filter_size=args.filter_size,
            h=args.horizon
        )
    else:
        model_save_name = args.model_name

    # Load trained model
    print(f"Loading model {model_save_name}...")
    trained_model, mae_loss = load_model(model_save_name, model)
    print(f"Model loaded with MAE loss: {mae_loss:.4f}")

    # Move model to device
    trained_model.to(args.device)

    # Generate predictions
    print("Generating predictions...")
    pred_df, actual_df = generate_predictions(trained_model, dataloader_test, market_indices_list, args.device)

    # Calculate metrics
    print("Calculating metrics...")
    metrics = calculate_metrics(pred_df, actual_df, market_indices_list)

    # Display metrics
    print("\nModel Performance Metrics:")
    for market_index, market_metrics in metrics.items():
        print(f"\n{market_index}:")
        for metric_name, value in market_metrics.items():
            print(f"  {metric_name}: {value:.4f}")

    # Calculate average metrics across all market indices
    avg_mse = np.mean([metrics[market_index]['MSE'] for market_index in market_indices_list])
    avg_rmse = np.mean([metrics[market_index]['RMSE'] for market_index in market_indices_list])
    avg_mae = np.mean([metrics[market_index]['MAE'] for market_index in market_indices_list])

    print("\nAverage Metrics Across All Market Indices:")
    print(f"  MSE: {avg_mse:.4f}")
    print(f"  RMSE: {avg_rmse:.4f}")
    print(f"  MAE: {avg_mae:.4f}")

    # Plot predictions if requested
    if args.plot:
        print("\nPlotting predictions...")
        plot_predictions(pred_df, actual_df, market_indices_list, save_dir='plots')

    # Save results if requested
    if args.save_results:
        print("\nSaving results...")
        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)

        # Save predictions and actuals
        pred_df.to_csv(os.path.join(results_dir, f'{model_save_name}_predictions.csv'))
        actual_df.to_csv(os.path.join(results_dir, f'{model_save_name}_actuals.csv'))

        # Save metrics
        metrics_df = pd.DataFrame()
        for market_index, market_metrics in metrics.items():
            for metric_name, value in market_metrics.items():
                metrics_df.loc[market_index, metric_name] = value

        metrics_df.to_csv(os.path.join(results_dir, f'{model_save_name}_metrics.csv'))

        print(f"Results saved to {results_dir} directory.")

    print("\nEvaluation completed.")


if __name__ == '__main__':
    main()
