"""
Data loading utilities for time series data.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class TimeSeriesDataset(Dataset):
    """
    Dataset for time series data.

    Args:
        data (np.ndarray): Time series data with shape (n_samples, n_features).
        seq_length (int): Length of input sequences.
        horizon (int): Prediction horizon.
    """
    def __init__(self, data, seq_length, horizon=1):
        self.data = data
        self.seq_length = seq_length
        self.horizon = horizon

        # Print data shape for debugging
        print(f"TimeSeriesDataset initialized with data shape: {data.shape}")
        print(f"Sequence length: {seq_length}, Horizon: {horizon}")

        # Calculate valid indices
        self.valid_idx = list(range(seq_length, len(data) - horizon + 1))
        print(f"Number of valid indices: {len(self.valid_idx)}")

    def __len__(self):
        return len(self.valid_idx)

    def __getitem__(self, idx):
        # Get the current index
        i = self.valid_idx[idx]

        # Print debug info for the first item
        debug = (idx == 0)
        if debug:
            print(f"Getting item at index {idx}, data index {i}")

        # Get the input sequences with proper dimensions for GSPHAR
        # For lag1, we need the last observation
        x_lag1 = self.data[i-1:i, :]
        # For lag5, we need the last 5 observations
        x_lag5 = self.data[i-5:i, :]
        # For lag22, we need the last 22 observations
        x_lag22 = self.data[i-22:i, :]

        if debug:
            print(f"Initial shapes - x_lag1: {x_lag1.shape}, x_lag5: {x_lag5.shape}, x_lag22: {x_lag22.shape}")

        # Pad sequences if necessary
        if x_lag1.shape[0] < 1:
            x_lag1 = np.pad(x_lag1, ((1 - x_lag1.shape[0], 0), (0, 0)), 'constant')
        if x_lag5.shape[0] < 5:
            x_lag5 = np.pad(x_lag5, ((5 - x_lag5.shape[0], 0), (0, 0)), 'constant')
        if x_lag22.shape[0] < 22:
            x_lag22 = np.pad(x_lag22, ((22 - x_lag22.shape[0], 0), (0, 0)), 'constant')

        # Get the target
        y = self.data[i:i+self.horizon, :]

        if debug:
            print(f"After padding - x_lag1: {x_lag1.shape}, x_lag5: {x_lag5.shape}, x_lag22: {x_lag22.shape}, y: {y.shape}")

        # Reshape to match GSPHAR input requirements
        # GSPHAR expects inputs of shape (batch_size, filter_size, input_dim)
        # where filter_size is the number of features (symbols)
        # and input_dim is the sequence length

        # Transpose to get (n_features, seq_length)
        x_lag1 = x_lag1.T
        x_lag5 = x_lag5.T
        x_lag22 = x_lag22.T

        if debug:
            print(f"After transpose - x_lag1: {x_lag1.shape}, x_lag5: {x_lag5.shape}, x_lag22: {x_lag22.shape}")

        # Add batch dimension to get (1, n_features, seq_length)
        x_lag1 = np.expand_dims(x_lag1, axis=0)
        x_lag5 = np.expand_dims(x_lag5, axis=0)
        x_lag22 = np.expand_dims(x_lag22, axis=0)

        # Transpose target to get (n_features, horizon)
        y = y.T
        # Add batch dimension to get (1, n_features, horizon)
        y = np.expand_dims(y, axis=0)

        if debug:
            print(f"After adding batch dim - x_lag1: {x_lag1.shape}, x_lag5: {x_lag5.shape}, x_lag22: {x_lag22.shape}, y: {y.shape}")

        # Convert to tensors
        x_lag1 = torch.tensor(x_lag1, dtype=torch.float32)
        x_lag5 = torch.tensor(x_lag5, dtype=torch.float32)
        x_lag22 = torch.tensor(x_lag22, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        # Remove the batch dimension since DataLoader will add it
        x_lag1 = x_lag1.squeeze(0)
        x_lag5 = x_lag5.squeeze(0)
        x_lag22 = x_lag22.squeeze(0)
        y = y.squeeze(0)

        if debug:
            print(f"Final shapes - x_lag1: {x_lag1.shape}, x_lag5: {x_lag5.shape}, x_lag22: {x_lag22.shape}, y: {y.shape}")

        return x_lag1, x_lag5, x_lag22, y


def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=32, num_workers=0):
    """
    Create dataloaders for training, validation, and testing.

    Args:
        train_dataset (Dataset): Training dataset.
        val_dataset (Dataset): Validation dataset.
        test_dataset (Dataset): Test dataset.
        batch_size (int): Batch size.
        num_workers (int): Number of workers for data loading.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader
