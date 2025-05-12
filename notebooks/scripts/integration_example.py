"""
Example script showing how to use the date-aware dataset functionality.
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add the parent directory to the path to import from the GSPHAR package
sys.path.insert(0, os.path.abspath('..'))

# Import from local modules
from src.data import load_data, split_data
from src.data import IndexMappingDataset

def main():
    """Main function to demonstrate the date-aware dataset functionality."""
    # Load data
    data_file = '../data/rv5_sqrt_24.csv'
    data = load_data(data_file)
    print(f"Data shape: {data.shape}")

    # Split data into train and test sets
    train_data, test_data = split_data(data, train_ratio=0.8)

    # Create a date-aware dataset
    lag_list = [1, 5, 22]
    h = 5
    dataset = IndexMappingDataset(test_data, lag_list, h)

    # Print information about the dataset
    print(f"Dataset length: {len(dataset)}")
    print(f"First date in dataset: {dataset.get_date(0)}")
    print(f"Last date in dataset: {dataset.get_date(len(dataset)-1)}")

    # Get a sample from the dataset
    sample_idx = 0
    x_lag1, x_lag5, x_lag22, y = dataset[sample_idx]

    # Print information about the sample
    print(f"Sample index: {sample_idx}")
    print(f"Sample date: {dataset.get_date(sample_idx)}")
    print(f"x_lag1 shape: {x_lag1.shape}")
    print(f"x_lag5 shape: {x_lag5.shape}")
    print(f"x_lag22 shape: {x_lag22.shape}")
    print(f"y shape: {y.shape}")

    # Create a dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False)

    # Iterate through the dataloader
    batch_count = 0
    for batch in dataloader:
        batch_count += 1
        if batch_count == 1:
            # Print information about the first batch
            x_lag1, x_lag5, x_lag22, y = batch
            print(f"Batch {batch_count}:")
            print(f"  x_lag1 shape: {x_lag1.shape}")
            print(f"  x_lag5 shape: {x_lag5.shape}")
            print(f"  x_lag22 shape: {x_lag22.shape}")
            print(f"  y shape: {y.shape}")

    print(f"Total batches: {batch_count}")

    # Demonstrate date reconstruction
    indices = list(range(5))  # First 5 indices
    dates = dataset.get_dates(indices)

    print("Date reconstruction:")
    for i, date in zip(indices, dates):
        print(f"  Index {i}: {date}")

if __name__ == "__main__":
    main()
