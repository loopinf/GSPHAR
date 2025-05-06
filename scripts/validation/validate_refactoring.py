#!/usr/bin/env python
"""
Validation script to compare the original GSPHAR implementation with the refactored version.
This script runs both implementations and compares their outputs.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Add the project root directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import from refactored modules
from config import settings
from src.data import load_data, split_data, create_lagged_features, prepare_data_dict, create_dataloaders
from src.data import LegacyGSPHAR_Dataset
from src.models import GSPHAR
from src.training import GSPHARTrainer
from src.utils import compute_spillover_index, load_model
from src.utils.device_utils import get_device, set_device_seeds

# Import original implementation (from tmp folder)
from tmp.d_GSPHAR import GSPHAR as GSPHAR_Original
from tmp.d_GSPHAR import GSPHAR_Dataset as GSPHAR_Dataset_Original
from tmp.d_GSPHAR import compute_spillover_index as compute_spillover_index_original
from tmp.d_GSPHAR import train_eval_model as train_eval_model_original
from tmp.d_GSPHAR import evaluate_model as evaluate_model_original


def run_original_implementation(data_file, h=5, num_epochs=5, lr=0.01):
    """
    Run the original GSPHAR implementation.

    Args:
        data_file (str): Path to the data file.
        h (int, optional): Prediction horizon. Defaults to 5.
        num_epochs (int, optional): Number of epochs. Defaults to 5.
        lr (float, optional): Learning rate. Defaults to 0.01.

    Returns:
        tuple: (trained_model, predictions, actual_values)
    """
    print("Running original implementation...")

    # Set random seed for reproducibility using our utility function
    set_device_seeds()

    # Load data
    data = pd.read_csv(data_file, index_col=0) * 100
    date_list = data.index.tolist()
    train_end_idx = int(len(date_list) * 0.7)
    train_dataset = data.iloc[0:train_end_idx, :]
    test_dataset = data.iloc[train_end_idx:, :]

    market_indices_list = train_dataset.columns.tolist()

    # Compute spillover index
    DY_adj = compute_spillover_index_original(train_dataset, h, 22, 0.0, standardized=True)

    # Create lagged features
    look_back_window = 22
    for market_index in market_indices_list:
        for lag in range(look_back_window):
            train_dataset[market_index + f'_{lag+1}'] = train_dataset[market_index].shift(lag+h)
            test_dataset[market_index + f'_{lag+1}'] = test_dataset[market_index].shift(lag+h)

    train_dataset = train_dataset.dropna()
    test_dataset = test_dataset.dropna()

    # Prepare data
    columns_lag1 = [x for x in train_dataset.columns.tolist() if x[-2:] == '_1']
    columns_lag5 = [x for x in train_dataset.columns.tolist() if (x[-2]=='_') and (float(x[-1]) in range(1,6))]
    columns_lag22 = [x for x in train_dataset.columns.tolist() if '_' in x]
    x_columns = columns_lag1 + columns_lag5 + columns_lag22
    y_columns = [x for x in train_dataset.columns.tolist() if x not in x_columns]
    row_index_order = market_indices_list
    column_index_order_5 = [f'lag_{i}' for i in range(1,6)]
    column_index_order_22 = [f'lag_{i}' for i in range(1,23)]

    # Create data dictionaries
    train_dict = {}
    for date in tqdm(train_dataset.index, desc="Preparing train data"):
        y = train_dataset.loc[date, y_columns]

        x_lag1 = train_dataset.loc[date, columns_lag1]
        new_index = [ind[:-2] for ind in x_lag1.index.tolist()]
        x_lag1.index = new_index

        x_lag5 = train_dataset.loc[date, columns_lag5]
        data_lag5 = {
            'Market': [index.split('_')[0] for index in x_lag5.index],
            'Lag': [f'lag_{index.split("_")[1]}' for index in x_lag5.index],
            'Value': x_lag5.values
        }
        df_lag5 = pd.DataFrame(data_lag5)
        df_lag5 = df_lag5.pivot(index='Market', columns='Lag', values='Value')

        x_lag22 = train_dataset.loc[date, columns_lag22]
        data_lag22 = {
            'Market': [index.split('_')[0] for index in x_lag22.index],
            'Lag': [f'lag_{index.split("_")[1]}' for index in x_lag22.index],
            'Value': x_lag22.values
        }
        df_lag22 = pd.DataFrame(data_lag22)
        df_lag22 = df_lag22.pivot(index='Market', columns='Lag', values='Value')

        x_lag1 = x_lag1.reindex(row_index_order)
        df_lag5 = df_lag5.reindex(row_index_order)
        df_lag22 = df_lag22.reindex(row_index_order)
        df_lag5 = df_lag5[column_index_order_5]
        df_lag22 = df_lag22[column_index_order_22]

        dfs_dict = {
            'y': y,
            'x_lag1': x_lag1,
            'x_lag5': df_lag5,
            'x_lag22': df_lag22
        }
        train_dict[date] = dfs_dict

    test_dict = {}
    for date in tqdm(test_dataset.index, desc="Preparing test data"):
        y = test_dataset.loc[date, y_columns]

        x_lag1 = test_dataset.loc[date, columns_lag1]
        new_index = [ind[:-2] for ind in x_lag1.index.tolist()]
        x_lag1.index = new_index

        x_lag5 = test_dataset.loc[date, columns_lag5]
        data_lag5 = {
            'Market': [index.split('_')[0] for index in x_lag5.index],
            'Lag': [f'lag_{index.split("_")[1]}' for index in x_lag5.index],
            'Value': x_lag5.values
        }
        df_lag5 = pd.DataFrame(data_lag5)
        df_lag5 = df_lag5.pivot(index='Market', columns='Lag', values='Value')

        x_lag22 = test_dataset.loc[date, columns_lag22]
        data_lag22 = {
            'Market': [index.split('_')[0] for index in x_lag22.index],
            'Lag': [f'lag_{index.split("_")[1]}' for index in x_lag22.index],
            'Value': x_lag22.values
        }
        df_lag22 = pd.DataFrame(data_lag22)
        df_lag22 = df_lag22.pivot(index='Market', columns='Lag', values='Value')

        x_lag1 = x_lag1.reindex(row_index_order)
        df_lag5 = df_lag5.reindex(row_index_order)
        df_lag22 = df_lag22.reindex(row_index_order)
        df_lag5 = df_lag5[column_index_order_5]
        df_lag22 = df_lag22[column_index_order_22]

        dfs_dict = {
            'y': y,
            'x_lag1': x_lag1,
            'x_lag5': df_lag5,
            'x_lag22': df_lag22
        }
        test_dict[date] = dfs_dict

    # Create dataset and dataloader
    dataset_train = GSPHAR_Dataset_Original(train_dict)
    dataset_test = GSPHAR_Dataset_Original(test_dict)

    dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=False)

    # Create and train model
    input_dim = 3
    output_dim = 1
    filter_size = 24

    GSPHAR_RV = GSPHAR_Original(input_dim, output_dim, filter_size, DY_adj)
    valid_loss, final_conv1d_lag5_weights, final_conv1d_lag22_weights = train_eval_model_original(
        GSPHAR_RV, dataloader_train, dataloader_test, num_epochs, lr
    )

    # Run inference
    y_hat_list = []
    y_list = []

    GSPHAR_RV.eval()

    with torch.no_grad():
        for x_lag1, x_lag5, x_lag22, y in dataloader_test:
            y_hat, _, _ = GSPHAR_RV(x_lag1, x_lag5, x_lag22)

            # Append the predicted and actual values to their respective lists
            y_hat_list.append(y_hat.cpu().numpy())
            y_list.append(y.cpu().numpy())

    y_hat_concatenated = np.concatenate(y_hat_list, axis=0)
    y_concatenated = np.concatenate(y_list, axis=0)

    rv_hat_GSPHAR_dynamic = pd.DataFrame(data=y_hat_concatenated, columns=market_indices_list)
    rv_true = pd.DataFrame(data=y_concatenated, columns=market_indices_list)

    return GSPHAR_RV, rv_hat_GSPHAR_dynamic, rv_true


def run_refactored_implementation(data_file, h=5, num_epochs=5, lr=0.01):
    """
    Run the refactored GSPHAR implementation.

    Args:
        data_file (str): Path to the data file.
        h (int, optional): Prediction horizon. Defaults to 5.
        num_epochs (int, optional): Number of epochs. Defaults to 5.
        lr (float, optional): Learning rate. Defaults to 0.01.

    Returns:
        tuple: (trained_model, predictions, actual_values)
    """
    print("Running refactored implementation...")

    # Set random seed for reproducibility using our utility function
    set_device_seeds()

    # Load data
    data = load_data(data_file)

    # Split data
    train_dataset_raw, test_dataset_raw = split_data(data, settings.TRAIN_SPLIT_RATIO)

    # Get market indices
    market_indices_list = train_dataset_raw.columns.tolist()

    # Compute spillover index
    DY_adj = compute_spillover_index(
        train_dataset_raw,
        h,
        settings.LOOK_BACK_WINDOW,
        0.0,
        standardized=True
    )

    # Create lagged features
    train_dataset = create_lagged_features(
        train_dataset_raw,
        market_indices_list,
        h,
        settings.LOOK_BACK_WINDOW
    )
    test_dataset = create_lagged_features(
        test_dataset_raw,
        market_indices_list,
        h,
        settings.LOOK_BACK_WINDOW
    )

    # Prepare data dictionaries
    train_dict = prepare_data_dict(train_dataset, market_indices_list, settings.LOOK_BACK_WINDOW)
    test_dict = prepare_data_dict(test_dataset, market_indices_list, settings.LOOK_BACK_WINDOW)

    # Create dataloaders
    dataloader_train, dataloader_test = create_dataloaders(train_dict, test_dict, settings.BATCH_SIZE)

    # Create model
    model = GSPHAR(settings.INPUT_DIM, settings.OUTPUT_DIM, settings.FILTER_SIZE, DY_adj)

    # Create optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        steps_per_epoch=len(dataloader_train),
        epochs=num_epochs,
        three_phase=True
    )

    # Create trainer with device from our utility function
    trainer = GSPHARTrainer(
        model=model,
        device=get_device(),
        criterion=torch.nn.MSELoss(),
        optimizer=optimizer,
        scheduler=scheduler
    )

    # Train model
    model_save_name = f"test_refactored_GSPHAR_{settings.FILTER_SIZE}_magnet_dynamic_h{h}"
    best_loss_val, _, _, train_loss_list, test_loss_list = trainer.train(
        dataloader_train=dataloader_train,
        dataloader_test=dataloader_test,
        num_epochs=num_epochs,
        patience=200,
        model_save_name=model_save_name
    )

    # Run inference
    y_hat_list = []
    y_list = []

    model.eval()

    with torch.no_grad():
        for x_lag1, x_lag5, x_lag22, y in dataloader_test:
            # Move data to device using our utility function
            device = get_device()
            x_lag1 = x_lag1.to(device)
            x_lag5 = x_lag5.to(device)
            x_lag22 = x_lag22.to(device)

            # Forward pass
            y_hat, _, _ = model(x_lag1, x_lag5, x_lag22)

            # Append the predicted and actual values to their respective lists
            y_hat_list.append(y_hat.cpu().numpy())
            y_list.append(y.cpu().numpy())

    # Concatenate the results
    y_hat_concatenated = np.concatenate(y_hat_list, axis=0)
    y_concatenated = np.concatenate(y_list, axis=0)

    # Create DataFrames
    rv_hat_GSPHAR_dynamic = pd.DataFrame(data=y_hat_concatenated, columns=market_indices_list)
    rv_true = pd.DataFrame(data=y_concatenated, columns=market_indices_list)

    return model, rv_hat_GSPHAR_dynamic, rv_true


def compare_results(original_predictions, refactored_predictions, market_indices_list):
    """
    Compare the predictions from the original and refactored implementations.

    Args:
        original_predictions (pd.DataFrame): Predictions from the original implementation.
        refactored_predictions (pd.DataFrame): Predictions from the refactored implementation.
        market_indices_list (list): List of market indices.

    Returns:
        pd.DataFrame: Comparison results.
    """
    print("Comparing results...")

    # Create a DataFrame to store the comparison results
    comparison_results = pd.DataFrame(index=market_indices_list)

    # Calculate mean absolute difference
    mean_abs_diff = []
    for market_index in market_indices_list:
        abs_diff = np.abs(original_predictions[market_index] - refactored_predictions[market_index]).mean()
        mean_abs_diff.append(abs_diff)

    comparison_results['Mean Absolute Difference'] = mean_abs_diff

    # Calculate correlation
    correlation = []
    for market_index in market_indices_list:
        corr = np.corrcoef(original_predictions[market_index], refactored_predictions[market_index])[0, 1]
        correlation.append(corr)

    comparison_results['Correlation'] = correlation

    # Calculate mean squared error
    mse = []
    for market_index in market_indices_list:
        mse_val = ((original_predictions[market_index] - refactored_predictions[market_index]) ** 2).mean()
        mse.append(mse_val)

    comparison_results['Mean Squared Error'] = mse

    # Calculate mean absolute percentage error
    mape = []
    for market_index in market_indices_list:
        mape_val = (np.abs((original_predictions[market_index] - refactored_predictions[market_index]) / original_predictions[market_index])).mean() * 100
        mape.append(mape_val)

    comparison_results['Mean Absolute Percentage Error (%)'] = mape

    return comparison_results


def plot_comparison(original_predictions, refactored_predictions, actual_values, market_indices_list, num_markets=5):
    """
    Plot the predictions from the original and refactored implementations.

    Args:
        original_predictions (pd.DataFrame): Predictions from the original implementation.
        refactored_predictions (pd.DataFrame): Predictions from the refactored implementation.
        actual_values (pd.DataFrame): Actual values.
        market_indices_list (list): List of market indices.
        num_markets (int, optional): Number of markets to plot. Defaults to 5.
    """
    print("Plotting comparison...")

    # Select a subset of markets to plot
    markets_to_plot = market_indices_list[:num_markets]

    # Create a figure with subplots
    fig, axes = plt.subplots(num_markets, 1, figsize=(12, 4 * num_markets))

    # Plot each market
    for i, market_index in enumerate(markets_to_plot):
        ax = axes[i] if num_markets > 1 else axes

        # Plot actual values
        ax.plot(actual_values[market_index], label='Actual', color='black', linestyle='-')

        # Plot original predictions
        ax.plot(original_predictions[market_index], label='Original', color='blue', linestyle='--')

        # Plot refactored predictions
        ax.plot(refactored_predictions[market_index], label='Refactored', color='red', linestyle=':')

        # Add labels and legend
        ax.set_title(f'Market: {market_index}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)

    # Adjust layout
    plt.tight_layout()

    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # Save the figure
    plt.savefig('results/comparison_plot.png')
    print("Plot saved as 'results/comparison_plot.png'")


def main():
    """
    Main function.
    """
    # Set parameters
    data_file = 'data/rv5_sqrt_24.csv'
    h = 5
    num_epochs = 4  # Increased number of epochs for better comparison
    lr = 0.01

    # Run original implementation
    original_model, original_predictions, original_actual_values = run_original_implementation(
        data_file, h, num_epochs, lr
    )

    # Run refactored implementation
    refactored_model, refactored_predictions, refactored_actual_values = run_refactored_implementation(
        data_file, h, num_epochs, lr
    )

    # Get market indices
    market_indices_list = original_predictions.columns.tolist()

    # Compare results
    comparison_results = compare_results(
        original_predictions, refactored_predictions, market_indices_list
    )

    # Print comparison results
    print("\nComparison Results:")
    print(comparison_results)

    # Calculate overall statistics
    print("\nOverall Statistics:")
    print(f"Mean Absolute Difference: {comparison_results['Mean Absolute Difference'].mean()}")
    print(f"Mean Correlation: {comparison_results['Correlation'].mean()}")
    print(f"Mean MSE: {comparison_results['Mean Squared Error'].mean()}")
    print(f"Mean MAPE: {comparison_results['Mean Absolute Percentage Error (%)'].mean()}%")

    # Plot comparison
    plot_comparison(
        original_predictions, refactored_predictions, original_actual_values, market_indices_list
    )

    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # Save comparison results
    comparison_results.to_csv('results/comparison_results.csv')
    print("Comparison results saved as 'results/comparison_results.csv'")


if __name__ == '__main__':
    main()
