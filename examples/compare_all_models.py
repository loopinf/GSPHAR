#!/usr/bin/env python
"""
Compare all models for each market index.

This script loads multiple models and creates plots comparing their performance
on each market index, with a focus on large jumps in volatility.
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
    parser = argparse.ArgumentParser(description='Compare all models for each market index')
    parser.add_argument('--standard-model', type=str, required=True,
                        help='Path to the standard MSE model')
    parser.add_argument('--asymmetric-model', type=str, required=True,
                        help='Path to the asymmetric MSE model')
    parser.add_argument('--hybrid-model', type=str, required=True,
                        help='Path to the hybrid loss model')
    parser.add_argument('--weighted-model', type=str, default=None,
                        help='Path to the weighted MSE model (optional)')
    parser.add_argument('--horizon', type=int, default=1,
                        help='Prediction horizon')
    parser.add_argument('--threshold', type=float, default=1.0,
                        help='Threshold for large jumps')
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

    return dataset, train_data, adj_matrix


def make_predictions(model, dataset, device='cpu'):
    """Make predictions with a model."""
    all_predictions = []
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
            pred, _, _ = model(x_lag1, x_lag5, x_lag22)

            # Get the date for this sample
            date = dataset.get_date(i)

            # Store results
            all_predictions.append(pred.squeeze().cpu().numpy())
            all_actuals.append(y.squeeze().cpu().numpy())
            all_dates.append(date)

    # Convert to arrays
    predictions = np.array(all_predictions)
    actuals = np.array(all_actuals)
    dates = np.array(all_dates)

    return predictions, actuals, dates


def create_comparison_plots(predictions_dict, actuals, dates, market_indices, threshold):
    """Create plots comparing all models for each market index."""
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Convert dates to datetime objects if they're strings
    if isinstance(dates[0], str):
        dates = [datetime.strptime(date, '%Y-%m-%d') for date in dates]
    
    # Create timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define colors for each model
    colors = {
        'Standard MSE': 'blue',
        'Asymmetric MSE': 'red',
        'Hybrid Loss': 'green',
        'Weighted MSE': 'purple'
    }
    
    # Plot for each market index
    for idx, market in enumerate(market_indices):
        # Get data for this market
        actual = actuals[:, idx]
        
        # Identify large jumps
        large_jumps = actual > threshold
        
        # Create figure
        plt.figure(figsize=(14, 10))
        
        # Plot all data
        plt.subplot(2, 1, 1)
        plt.plot(dates, actual, 'k-', label='Actual', linewidth=1.5)
        
        # Plot predictions for each model
        for model_name, predictions in predictions_dict.items():
            pred = predictions[:, idx]
            plt.plot(dates, pred, '-', color=colors[model_name], label=model_name, linewidth=1)
        
        plt.axhline(y=threshold, color='gray', linestyle='--', label=f'Threshold ({threshold:.2f})')
        plt.title(f'Comparison for {market} - All Data')
        plt.legend(loc='best', frameon=True)
        plt.grid(True)
        
        # Plot only large jumps
        plt.subplot(2, 1, 2)
        if np.any(large_jumps):
            large_jump_dates = [date for i, date in enumerate(dates) if large_jumps[i]]
            large_jump_actual = actual[large_jumps]
            
            plt.plot(large_jump_dates, large_jump_actual, 'ko-', label='Actual', linewidth=1.5)
            
            # Plot predictions for each model
            mae_values = {}
            for model_name, predictions in predictions_dict.items():
                pred = predictions[:, idx]
                large_jump_pred = pred[large_jumps]
                plt.plot(large_jump_dates, large_jump_pred, 'o-', color=colors[model_name], label=model_name, linewidth=1)
                
                # Calculate MAE for large jumps
                mae = np.mean(np.abs(large_jump_pred - large_jump_actual))
                mae_values[model_name] = mae
            
            # Calculate improvements relative to standard MSE
            standard_mae = mae_values['Standard MSE']
            improvements = {model: 100 * (standard_mae - mae) / standard_mae for model, mae in mae_values.items() if model != 'Standard MSE'}
            
            # Create title with MAE values and improvements
            title = f'Large Jumps Only - MAEs: Standard={mae_values["Standard MSE"]:.4f}'
            for model, mae in mae_values.items():
                if model != 'Standard MSE':
                    title += f', {model}={mae:.4f} ({improvements[model]:+.2f}%)'
            
            plt.title(title)
            plt.axhline(y=threshold, color='gray', linestyle='--', label=f'Threshold ({threshold:.2f})')
        else:
            plt.text(0.5, 0.5, 'No large jumps found', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
            plt.title('Large Jumps Only')
        
        plt.legend(loc='best', frameon=True)
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = f'plots/all_models_comparison_{market}_{timestamp}.png'
        plt.savefig(plot_path)
        plt.close()
        
        print(f"Plot saved to {plot_path}")


def main():
    """Main function."""
    args = parse_args()
    
    # Load evaluation data
    dataset, train_data, adj_matrix = load_evaluation_data(args.horizon)
    
    # Create model instances
    filter_size = 24  # Number of market indices
    input_dim = 3     # Standard for GSPHAR
    output_dim = 1    # Standard for GSPHAR
    
    # Create model instances
    standard_model = GSPHAR(input_dim, output_dim, filter_size, adj_matrix)
    asymmetric_model = GSPHAR(input_dim, output_dim, filter_size, adj_matrix)
    hybrid_model = GSPHAR(input_dim, output_dim, filter_size, adj_matrix)
    weighted_model = None
    if args.weighted_model:
        weighted_model = GSPHAR(input_dim, output_dim, filter_size, adj_matrix)
    
    # Load model weights
    print(f"Loading Standard MSE model: {args.standard_model}")
    standard_model, _ = load_model(args.standard_model, standard_model)
    standard_model.eval()
    
    print(f"Loading Asymmetric MSE model: {args.asymmetric_model}")
    asymmetric_model, _ = load_model(args.asymmetric_model, asymmetric_model)
    asymmetric_model.eval()
    
    print(f"Loading Hybrid Loss model: {args.hybrid_model}")
    hybrid_model, _ = load_model(args.hybrid_model, hybrid_model)
    hybrid_model.eval()
    
    if weighted_model:
        print(f"Loading Weighted MSE model: {args.weighted_model}")
        weighted_model, _ = load_model(args.weighted_model, weighted_model)
        weighted_model.eval()
    
    # Make predictions with each model
    print("\n=== Making Predictions ===")
    print("Standard MSE model predictions...")
    standard_preds, actuals, dates = make_predictions(standard_model, dataset)
    
    print("Asymmetric MSE model predictions...")
    asymmetric_preds, _, _ = make_predictions(asymmetric_model, dataset)
    
    print("Hybrid Loss model predictions...")
    hybrid_preds, _, _ = make_predictions(hybrid_model, dataset)
    
    weighted_preds = None
    if weighted_model:
        print("Weighted MSE model predictions...")
        weighted_preds, _, _ = make_predictions(weighted_model, dataset)
    
    # Get market indices
    market_indices = dataset.data.columns
    
    # Create dictionary of predictions
    predictions_dict = {
        'Standard MSE': standard_preds,
        'Asymmetric MSE': asymmetric_preds,
        'Hybrid Loss': hybrid_preds
    }
    
    if weighted_preds is not None:
        predictions_dict['Weighted MSE'] = weighted_preds
    
    # Create comparison plots
    print("\n=== Creating Comparison Plots ===")
    create_comparison_plots(predictions_dict, actuals, dates, market_indices, args.threshold)
    
    print("\nComparison complete!")


if __name__ == "__main__":
    main()
