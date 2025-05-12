#!/usr/bin/env python
"""
Example script demonstrating how to use both GSPHAR_Dataset for training and
IndexMappingDataset for evaluation.

This script shows how to:
1. Load and preprocess data
2. Create training dataloader with GSPHAR_Dataset
3. Create evaluation dataloader with IndexMappingDataset
4. Train the model
5. Generate date-aware predictions
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from local modules
from config import settings
from src.data import (
    load_data,
    split_data,
    create_lagged_features,
    prepare_data_dict,
    create_dataloaders,
    GSPHAR_Dataset,
    IndexMappingDataset,
    generate_index_mapped_predictions
)
from src.models import GSPHAR
from src.training import GSPHARTrainer
from src.utils import compute_spillover_index, save_model, load_model
from src.utils.device_utils import get_device, set_device_seeds


def main():
    """Main function demonstrating the dual dataset approach."""
    # Set random seeds for reproducibility
    set_device_seeds(42)

    # Configuration
    data_file = 'data/rv5_sqrt_24.csv'
    h = 5  # Prediction horizon
    look_back_window = 22  # Look-back window
    train_ratio = 0.7  # Train/test split ratio
    batch_size = 32  # Batch size
    num_epochs = 2  # Number of epochs (reduced for demonstration)
    patience = 1  # Patience for early stopping
    lr = 0.01  # Learning rate

    # Load data
    print(f"Loading data from {data_file}...")
    data = load_data(data_file)
    print(f"Data shape: {data.shape}")

    # Split data into train and test sets
    train_data, test_data = split_data(data, train_ratio)
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")

    # Get market indices
    market_indices_list = train_data.columns.tolist()
    print(f"Number of market indices: {len(market_indices_list)}")

    # Compute spillover index for adjacency matrix
    print("Computing spillover index for adjacency matrix...")
    DY_adj = compute_spillover_index(train_data, h, look_back_window, 0.0, standardized=True)

    # Ensure adjacency matrix is float32 to avoid MPS issues
    DY_adj = DY_adj.astype(np.float32)

    # Create lagged features
    print("Creating lagged features...")
    train_lagged = create_lagged_features(train_data, market_indices_list, h, look_back_window)
    test_lagged = create_lagged_features(test_data, market_indices_list, h, look_back_window)

    # =====================================================================
    # Approach 1: Using GSPHAR_Dataset for training
    # =====================================================================
    print("\n=== Using GSPHAR_Dataset for training ===")

    # Define column groups
    y_columns = [col for col in train_lagged.columns if '_' not in col]
    columns_lag1 = [x for x in train_lagged.columns if x.endswith('_1')]
    columns_lag5 = [x for x in train_lagged.columns if '_' in x and x.split('_')[-1].isdigit()
                   and int(x.split('_')[-1]) in range(1, 6)]
    columns_lag22 = [x for x in train_lagged.columns if '_' in x and x.split('_')[-1].isdigit()
                    and int(x.split('_')[-1]) in range(1, look_back_window + 1)]

    # Create datasets
    train_dataset = GSPHAR_Dataset(
        train_lagged, y_columns, columns_lag1, columns_lag5, columns_lag22, market_indices_list
    )
    test_dataset = GSPHAR_Dataset(
        test_lagged, y_columns, columns_lag1, columns_lag5, columns_lag22, market_indices_list
    )

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # =====================================================================
    # Approach 2: Using IndexMappingDataset for evaluation
    # =====================================================================
    print("\n=== Using IndexMappingDataset for evaluation ===")

    # Create date-aware dataset for evaluation
    lag_list = [1, 5, 22]  # Lag values to use
    date_aware_dataset = IndexMappingDataset(test_data, lag_list, h)
    date_aware_dataloader = DataLoader(date_aware_dataset, batch_size=batch_size, shuffle=False)

    print(f"Date-aware dataset size: {len(date_aware_dataset)}")
    print(f"First date in dataset: {date_aware_dataset.get_date(0)}")
    print(f"Last date in dataset: {date_aware_dataset.get_date(len(date_aware_dataset)-1)}")

    # =====================================================================
    # Model Training
    # =====================================================================
    print("\n=== Training Model ===")

    # Model parameters
    input_dim = 3  # For lag1, lag5, lag22 features
    output_dim = 1  # Single prediction per symbol
    filter_size = len(market_indices_list)  # Number of symbols/market indices

    # Create model
    model = GSPHAR(input_dim, output_dim, filter_size, DY_adj)

    # Ensure model parameters are float32
    for param in model.parameters():
        param.data = param.data.to(torch.float32)

    # Create optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        steps_per_epoch=len(train_dataloader),
        epochs=num_epochs,
        three_phase=True
    )

    # Create trainer
    # Force CPU usage as requested
    device = 'cpu'
    print(f"Using device: {device} (MPS/GPU disabled as requested)")

    trainer = GSPHARTrainer(
        model=model,
        device=device,
        criterion=nn.MSELoss(),
        optimizer=optimizer,
        scheduler=scheduler
    )

    # Train model (using minimal epochs for demonstration purposes)
    print(f"Training model for {num_epochs} epochs with patience {patience}...")
    print("Note: Using minimal epochs for demonstration. Increase for real training.")
    model_save_name = f"dual_dataset_example_h{h}"
    best_loss_val, _, _, train_loss_list, test_loss_list = trainer.train(
        dataloader_train=train_dataloader,
        dataloader_test=test_dataloader,
        num_epochs=num_epochs,
        patience=patience,
        model_save_name=model_save_name
    )

    # Load best model
    print(f"Loading best model {model_save_name}...")
    trained_model, mae_loss = load_model(model_save_name, model)

    print(f"Training completed. Best validation loss: {best_loss_val:.4f}")

    # =====================================================================
    # Date-aware Evaluation
    # =====================================================================
    print("\n=== Date-aware Evaluation ===")

    # Generate predictions with date awareness
    pred_df, actual_df = generate_index_mapped_predictions(
        trained_model, date_aware_dataloader, date_aware_dataset, market_indices_list
    )

    print(f"Predictions shape: {pred_df.shape}")
    print(f"Actuals shape: {actual_df.shape}")
    print(f"Predictions index type: {type(pred_df.index)}")
    print(f"First few prediction dates: {pred_df.index[:5]}")

    # Calculate MAE for each market index
    mae_per_index = np.abs(pred_df.values - actual_df.values).mean(axis=0)
    overall_mae = mae_per_index.mean()

    print(f"Overall MAE: {overall_mae:.4f}")
    print("MAE per market index:")
    for i, market in enumerate(market_indices_list):
        print(f"  {market}: {mae_per_index[i]:.4f}")

    # =====================================================================
    # Visualization
    # =====================================================================
    print("\n=== Visualization ===")

    # Plot predictions for a sample market index
    sample_index = market_indices_list[0]
    plt.figure(figsize=(12, 6))
    plt.plot(actual_df.index, actual_df[sample_index], 'k-', label='Actual')
    plt.plot(pred_df.index, pred_df[sample_index], 'b-', label='Predicted')
    plt.title(f'Volatility Predictions for {sample_index}')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot
    os.makedirs('plots', exist_ok=True)
    plot_file = f'plots/dual_dataset_predictions_{sample_index}.png'
    plt.savefig(plot_file)
    print(f"Plot saved to {plot_file}")

    print("\nDone!")


if __name__ == "__main__":
    main()
