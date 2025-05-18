"""
Flexible data loading utilities for time series data with customizable lags.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class FlexibleTimeSeriesDataset(Dataset):
    """
    Dataset for time series data with customizable lags.
    
    Args:
        data (np.ndarray): Time series data with shape (n_samples, n_features).
        lags (list): List of lag values to use (e.g., [1, 4, 24]).
        horizon (int): Prediction horizon.
        debug (bool): Whether to print debug information.
    """
    def __init__(self, data, lags, horizon=1, debug=False):
        self.data = data
        self.lags = sorted(lags)  # Sort lags in ascending order
        self.horizon = horizon
        self.debug = debug
        
        # Calculate the maximum lag
        self.max_lag = max(self.lags)
        
        # Calculate valid indices
        self.valid_idx = list(range(self.max_lag, len(data) - horizon + 1))
        
        if debug:
            print(f"FlexibleTimeSeriesDataset initialized with data shape: {data.shape}")
            print(f"Lags: {self.lags}, Max lag: {self.max_lag}, Horizon: {horizon}")
            print(f"Number of valid indices: {len(self.valid_idx)}")
    
    def __len__(self):
        return len(self.valid_idx)
    
    def __getitem__(self, idx):
        # Get the current index
        i = self.valid_idx[idx]
        
        # Print debug info for the first item
        debug = self.debug and (idx == 0)
        if debug:
            print(f"Getting item at index {idx}, data index {i}")
        
        # Get the input sequences for each lag
        x_lags = []
        for lag in self.lags:
            x_lag = self.data[i-lag:i, :]
            
            # Pad sequences if necessary
            if x_lag.shape[0] < lag:
                x_lag = np.pad(x_lag, ((lag - x_lag.shape[0], 0), (0, 0)), 'constant')
            
            # Transpose to get (n_features, seq_length)
            x_lag = x_lag.T
            
            # Add batch dimension to get (1, n_features, seq_length)
            x_lag = np.expand_dims(x_lag, axis=0)
            
            x_lags.append(x_lag)
            
            if debug:
                print(f"Lag {lag} shape after processing: {x_lag.shape}")
        
        # Get the target
        y = self.data[i:i+self.horizon, :]
        
        # Transpose target to get (n_features, horizon)
        y = y.T
        # Add batch dimension to get (1, n_features, horizon)
        y = np.expand_dims(y, axis=0)
        
        if debug:
            print(f"Target shape after processing: {y.shape}")
        
        # Convert to tensors
        x_lags_tensors = []
        for x_lag in x_lags:
            x_lag_tensor = torch.tensor(x_lag, dtype=torch.float32)
            # Remove the batch dimension since DataLoader will add it
            x_lag_tensor = x_lag_tensor.squeeze(0)
            x_lags_tensors.append(x_lag_tensor)
        
        y = torch.tensor(y, dtype=torch.float32)
        # Remove the batch dimension since DataLoader will add it
        y = y.squeeze(0)
        
        if debug:
            for idx, lag in enumerate(self.lags):
                print(f"Final tensor shape for lag {lag}: {x_lags_tensors[idx].shape}")
            print(f"Final target shape: {y.shape}")
        
        return tuple(x_lags_tensors + [y])


def create_flexible_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=32, num_workers=0):
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
