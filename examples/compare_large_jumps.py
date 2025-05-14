#!/usr/bin/env python
"""
Compare model performance on large jumps in volatility.

This script evaluates two models and compares their performance specifically
on large jumps in the target variable (volatility).
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import torch

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import load_data, split_data, IndexMappingDataset
from src.models import GSPHAR
from src.utils import compute_spillover_index, load_model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Compare model performance on large jumps')
    parser.add_argument('--model1', type=str, required=True,
                        help='First model to compare (standard MSE)')
    parser.add_argument('--model2', type=str, required=True,
                        help='Second model to compare (weighted MSE)')
    parser.add_argument('--horizon', type=int, default=1,
                        help='Prediction horizon')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for defining large jumps')
    parser.add_argument('--percentile', type=float, default=90,
                        help='Percentile threshold for defining large jumps (alternative to absolute threshold)')
    parser.add_argument('--use-percentile', action='store_true',
                        help='Use percentile instead of absolute threshold')
    return parser.parse_args()


def load_evaluation_data(horizon):
    """Load and prepare data for evaluation."""
    print("Loading data from data/rv5_sqrt_24.csv...")
    data = load_data('data/rv5_sqrt_24.csv')

    # Split data
    train_data, test_data = split_data(data, train_ratio=0.7)
    print(f"Data shape: {data.shape}")
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")

    # Get number of market indices
    n_indices = data.shape[1]
    print(f"Number of market indices: {n_indices}")

    # Compute spillover index for adjacency matrix
    print("Computing spillover index for adjacency matrix...")
    adj_matrix = compute_spillover_index(train_data, horizon=horizon, lag=22, scarcity_prop=0.0)

    # Create date-aware dataset
    print("\n=== Using IndexMappingDataset for evaluation ===")
    lag_list = [1, 5, 22]  # Lag values to use
    dataset = IndexMappingDataset(
        data=test_data,
        lag_list=lag_list,
        h=horizon
    )

    print(f"Date-aware dataset size: {len(dataset)}")
    print(f"First date in dataset: {dataset.get_date(0)}")
    print(f"Last date in dataset: {dataset.get_date(len(dataset)-1)}")

    return dataset, train_data


def evaluate_models(model1, model2, dataset, threshold, use_percentile, percentile, model1_name="standard", model2_name="custom"):
    """Evaluate both models and compare their performance on large jumps."""
    device = 'cpu'

    # Make predictions with both models
    all_predictions1 = []
    all_predictions2 = []
    all_actuals = []
    all_dates = []

    with torch.no_grad():
        for i in range(len(dataset)):
            # Get data
            x_lag1, x_lag5, x_lag22, y = dataset[i]

            # Add batch dimension
            x_lag1 = x_lag1.unsqueeze(0).to(device)
            x_lag5 = x_lag5.unsqueeze(0).to(device)
            x_lag22 = x_lag22.unsqueeze(0).to(device)

            # Make predictions
            pred1, _, _ = model1(x_lag1, x_lag5, x_lag22)
            pred2, _, _ = model2(x_lag1, x_lag5, x_lag22)

            # Get the date for this sample
            date = dataset.get_date(i)

            # Store results
            all_predictions1.append(pred1.squeeze().cpu().numpy())
            all_predictions2.append(pred2.squeeze().cpu().numpy())
            all_actuals.append(y.squeeze().cpu().numpy())
            all_dates.append(date)

    # Convert to arrays
    predictions1 = np.array(all_predictions1)
    predictions2 = np.array(all_predictions2)
    actuals = np.array(all_actuals)
    dates = np.array(all_dates)

    # Create DataFrame with dates as index
    results_df = pd.DataFrame({
        'date': dates
    })

    # Determine large jumps
    if use_percentile:
        # Calculate threshold based on percentile
        threshold = np.percentile(actuals.flatten(), percentile)
        print(f"Using {percentile}th percentile as threshold: {threshold:.4f}")
    else:
        print(f"Using absolute threshold: {threshold:.4f}")

    # Calculate errors
    errors1 = np.abs(predictions1 - actuals)
    errors2 = np.abs(predictions2 - actuals)

    # Identify large jumps
    large_jumps_mask = actuals > threshold

    # Count large jumps
    num_large_jumps = np.sum(large_jumps_mask)
    total_points = actuals.size
    print(f"\n=== Large Jump Analysis ===")
    print(f"Total data points: {total_points}")
    print(f"Number of large jumps (>{threshold:.4f}): {num_large_jumps} ({100*num_large_jumps/total_points:.2f}%)")

    # Calculate MAE for all points and large jumps
    mae1_all = np.mean(errors1)
    mae2_all = np.mean(errors2)

    # Calculate MAE for large jumps only
    if num_large_jumps > 0:
        mae1_large = np.mean(errors1[large_jumps_mask])
        mae2_large = np.mean(errors2[large_jumps_mask])
    else:
        mae1_large = 0
        mae2_large = 0

    print("\n=== Overall Performance ===")
    print(f"Model 1 (Standard MSE) - Overall MAE: {mae1_all:.4f}")
    print(f"Model 2 (Weighted MSE) - Overall MAE: {mae2_all:.4f}")
    print(f"Difference: {mae1_all - mae2_all:.4f} ({100*(mae1_all - mae2_all)/mae1_all:.2f}%)")

    print("\n=== Large Jump Performance ===")
    print(f"Model 1 (Standard MSE) - Large Jump MAE: {mae1_large:.4f}")
    print(f"Model 2 (Weighted MSE) - Large Jump MAE: {mae2_large:.4f}")
    print(f"Difference: {mae1_large - mae2_large:.4f} ({100*(mae1_large - mae2_large)/mae1_large:.2f}%)")

    # Calculate performance per market index
    market_indices = dataset.data.columns

    # Print sample predictions and actuals for debugging
    print("\n=== Sample Predictions and Actuals ===")
    print("First 5 predictions for first market index:")
    print(f"Market: {market_indices[0]}")
    print("Index | Actual  | Model1  | Model2")
    print("------|---------|---------|--------")
    for i in range(5):
        print(f"{i:5d} | {actuals[i, 0]:.6f} | {predictions1[i, 0]:.6f} | {predictions2[i, 0]:.6f}")

    print("\nLast 5 predictions for first market index:")
    for i in range(len(actuals)-5, len(actuals)):
        print(f"{i:5d} | {actuals[i, 0]:.6f} | {predictions1[i, 0]:.6f} | {predictions2[i, 0]:.6f}")

    print("\nStatistics for first market index:")
    print(f"Actuals - Min: {actuals[:, 0].min():.6f}, Max: {actuals[:, 0].max():.6f}, Mean: {actuals[:, 0].mean():.6f}")
    print(f"Model1  - Min: {predictions1[:, 0].min():.6f}, Max: {predictions1[:, 0].max():.6f}, Mean: {predictions1[:, 0].mean():.6f}")
    print(f"Model2  - Min: {predictions2[:, 0].min():.6f}, Max: {predictions2[:, 0].max():.6f}, Mean: {predictions2[:, 0].mean():.6f}")

    print("\n=== Performance by Market Index ===")
    print("Market Index | Standard MSE (All) | Weighted MSE (All) | % Improvement | Standard MSE (Large) | Weighted MSE (Large) | % Improvement")
    print("-------------|-------------------|-------------------|--------------|---------------------|---------------------|-------------")

    for i, market in enumerate(market_indices):
        # Calculate MAE for all points
        mae1_market_all = np.mean(errors1[:, i])
        mae2_market_all = np.mean(errors2[:, i])
        pct_improvement_all = 100 * (mae1_market_all - mae2_market_all) / mae1_market_all if mae1_market_all > 0 else 0

        # Calculate MAE for large jumps
        large_jumps_market = large_jumps_mask[:, i]
        if np.sum(large_jumps_market) > 0:
            mae1_market_large = np.mean(errors1[:, i][large_jumps_market])
            mae2_market_large = np.mean(errors2[:, i][large_jumps_market])
            pct_improvement_large = 100 * (mae1_market_large - mae2_market_large) / mae1_market_large if mae1_market_large > 0 else 0
        else:
            mae1_market_large = 0
            mae2_market_large = 0
            pct_improvement_large = 0

        print(f"{market:12} | {mae1_market_all:.4f} | {mae2_market_all:.4f} | {pct_improvement_all:+.2f}% | {mae1_market_large:.4f} | {mae2_market_large:.4f} | {pct_improvement_large:+.2f}%")

    # Create visualization
    create_comparison_plots(predictions1, predictions2, actuals, dates, market_indices, threshold, model1_name, model2_name)

    return predictions1, predictions2, actuals, dates


def create_comparison_plots(predictions1, predictions2, actuals, dates, market_indices, threshold, model1_name="standard", model2_name="custom"):
    """Create plots comparing model performance on large jumps."""
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)

    # Convert dates to datetime objects if they're strings
    if isinstance(dates[0], str):
        dates = [datetime.strptime(date, '%Y-%m-%d') for date in dates]

    # Create timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Select a few indices to plot
    indices_to_plot = [0, 10, 20]  # First, middle, and last indices

    for idx in indices_to_plot:
        market = market_indices[idx]

        # Get data for this market
        pred1 = predictions1[:, idx]
        pred2 = predictions2[:, idx]
        actual = actuals[:, idx]

        # Identify large jumps
        large_jumps = actual > threshold

        # Create figure
        plt.figure(figsize=(12, 8))

        # Plot all data
        plt.subplot(2, 1, 1)
        plt.plot(dates, actual, 'k-', label='Actual', linewidth=1.5)
        plt.plot(dates, pred1, 'b-', label=f'Standard MSE', linewidth=1)
        plt.plot(dates, pred2, 'r-', label=f'{model2_name}', linewidth=1)
        plt.axhline(y=threshold, color='g', linestyle='--', label=f'Threshold ({threshold:.2f})')
        plt.title(f'Comparison for {market} - All Data')
        plt.legend()
        plt.grid(True)

        # Plot only large jumps
        plt.subplot(2, 1, 2)
        if np.any(large_jumps):
            large_jump_dates = [date for i, date in enumerate(dates) if large_jumps[i]]
            large_jump_actual = actual[large_jumps]
            large_jump_pred1 = pred1[large_jumps]
            large_jump_pred2 = pred2[large_jumps]

            plt.plot(large_jump_dates, large_jump_actual, 'ko-', label='Actual', linewidth=1.5)
            plt.plot(large_jump_dates, large_jump_pred1, 'bo-', label=f'Standard MSE', linewidth=1)
            plt.plot(large_jump_dates, large_jump_pred2, 'ro-', label=f'{model2_name}', linewidth=1)
            plt.axhline(y=threshold, color='g', linestyle='--', label=f'Threshold ({threshold:.2f})')

            # Calculate errors
            mae1_large = np.mean(np.abs(large_jump_pred1 - large_jump_actual))
            mae2_large = np.mean(np.abs(large_jump_pred2 - large_jump_actual))
            improvement = 100 * (mae1_large - mae2_large) / mae1_large if mae1_large > 0 else 0

            plt.title(f'Large Jumps Only - Standard MAE: {mae1_large:.4f}, {model2_name} MAE: {mae2_large:.4f} ({improvement:+.2f}%)')
        else:
            plt.text(0.5, 0.5, 'No large jumps found', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
            plt.title('Large Jumps Only')

        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        # Save plot with loss function names in the filename
        plot_path = f'plots/large_jump_comparison_{market}_standard_vs_{model2_name}_{timestamp}.png'
        plt.savefig(plot_path)
        plt.close()

        print(f"Plot saved to {plot_path}")


def main():
    """Main function."""
    args = parse_args()

    # Load evaluation data
    dataset, train_data = load_evaluation_data(args.horizon)

    # Load models
    print("\n=== Loading Models ===")

    # Create models
    from src.models import GSPHAR

    # Get adjacency matrix from the train data
    adj_matrix = compute_spillover_index(train_data, horizon=args.horizon, lag=22, scarcity_prop=0.0)

    # Create model instances
    filter_size = 24  # Number of market indices
    input_dim = 3     # Standard for GSPHAR
    output_dim = 1    # Standard for GSPHAR

    # Create model instances
    model1 = GSPHAR(input_dim, output_dim, filter_size, adj_matrix)
    model2 = GSPHAR(input_dim, output_dim, filter_size, adj_matrix)

    # Load model weights
    print(f"Loading model 1 (Standard MSE): {args.model1}")
    model1, _ = load_model(args.model1, model1)
    model1.eval()

    print(f"Loading model 2 (Weighted MSE): {args.model2}")
    model2, _ = load_model(args.model2, model2)
    model2.eval()

    # Extract model names for better labeling
    model1_parts = args.model1.split('_')
    model2_parts = args.model2.split('_')

    # Try to extract the loss function name from the model name
    if "standard_mse" in args.model1:
        model1_name = "Standard MSE"
    elif "weighted_mse" in args.model1:
        model1_name = "Weighted MSE"
    elif "asymmetric_mse" in args.model1:
        model1_name = "Asymmetric MSE"
    elif "hybrid" in args.model1:
        model1_name = "Hybrid Loss"
    else:
        model1_name = "Standard MSE"

    if "standard_mse" in args.model2:
        model2_name = "Standard MSE"
    elif "weighted_mse" in args.model2:
        model2_name = "Weighted MSE"
    elif "asymmetric_mse" in args.model2:
        model2_name = "Asymmetric MSE"
    elif "hybrid" in args.model2:
        model2_name = "Hybrid Loss"
    else:
        model2_name = "Custom Loss"

    # Evaluate models
    evaluate_models(model1, model2, dataset, args.threshold, args.use_percentile, args.percentile, model1_name, model2_name)


if __name__ == "__main__":
    main()
