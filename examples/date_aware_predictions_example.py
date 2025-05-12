"""
Example script demonstrating how to use the Index Mapping approach for date-aware predictions.
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Add the parent directory to the path to import from the GSPHAR package
sys.path.insert(0, os.path.abspath('..'))

# Import from local modules
from config import settings
from src.data import load_data, split_data
from src.models import GSPHAR
from src.data import IndexMappingDataset, create_index_mapping_dataloaders, generate_index_mapped_predictions

def main():
    """Main function to demonstrate date-aware predictions."""
    # Load data
    data_file = 'data/rv5_sqrt_24.csv'
    data = load_data(data_file)
    print(f"Data shape: {data.shape}")
    print(f"Data index type: {type(data.index)}")
    print(f"First few dates: {data.index[:5]}")

    # Split data into train and test sets
    train_data, test_data = split_data(data, test_size=0.2)
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")

    # Create dictionaries for dataloaders
    lag_list = [1, 5, 22]
    h = 5
    train_dict = {
        'data': train_data,
        'lag_list': lag_list,
        'h': h
    }
    test_dict = {
        'data': test_data,
        'lag_list': lag_list,
        'h': h
    }

    # Create dataloaders with index mapping
    batch_size = settings.BATCH_SIZE
    train_dataloader, test_dataloader, train_dataset, test_dataset = create_index_mapping_dataloaders(
        train_dict, test_dict, batch_size
    )

    # Create and initialize a model
    input_dim = train_data.shape[1]
    output_dim = train_data.shape[1]
    filter_size = 24
    DY_adj = torch.ones(input_dim, input_dim)  # Simple adjacency matrix

    model = GSPHAR(input_dim, output_dim, filter_size, DY_adj)

    # Try to load a trained model if available
    model_save_name = settings.MODEL_SAVE_NAME_PATTERN.format(
        filter_size=filter_size,
        h=h
    )
    model_path = os.path.join('models', f'{model_save_name}.tar')

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model: {model_save_name}")
    else:
        print(f"No trained model found at {model_path}. Using untrained model.")

    # Generate predictions with date awareness
    market_indices_list = data.columns.tolist()
    pred_df, actual_df = generate_index_mapped_predictions(
        model, test_dataloader, test_dataset, market_indices_list
    )

    print(f"Predictions shape: {pred_df.shape}")
    print(f"Actuals shape: {actual_df.shape}")
    print(f"Predictions index type: {type(pred_df.index)}")
    print(f"First few prediction dates: {pred_df.index[:5]}")
    print(f"Last few prediction dates: {pred_df.index[-5:]}")

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
    plt.savefig(f'plots/date_aware_predictions_{sample_index}.png')
    print(f"Plot saved to plots/date_aware_predictions_{sample_index}.png")

    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()
