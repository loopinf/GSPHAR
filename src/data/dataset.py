"""
Dataset classes for GSPHAR.
This module contains PyTorch Dataset classes for handling GSPHAR data.
"""

import torch
import numpy as np
from torch.utils.data import Dataset


class LegacyGSPHAR_Dataset(Dataset):
    """
    Legacy dataset class for GSPHAR model (original implementation).

    This class handles the data dictionary format used in the original GSPHAR model.
    It is maintained for backward compatibility and validation purposes.

    Note: This implementation requires preprocessing data into a specific dictionary format.
    For new code, use the GSPHAR_Dataset class instead, which works directly with DataFrames.
    """

    def __init__(self, data_dict):
        """
        Initialize the dataset.

        Args:
            data_dict (dict): Dictionary with data for each date.
        """
        self.dict = data_dict

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.dict.keys())

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (x_lag1_tensor, x_lag5_tensor, x_lag22_tensor, y_tensor)
        """
        date = list(self.dict.keys())[idx]
        dfs_dict = self.dict[date]
        y = dfs_dict['y'].values
        x_lag1 = dfs_dict['x_lag1'].values
        x_lag5 = dfs_dict['x_lag5'].values
        x_lag22 = dfs_dict['x_lag22'].values

        y_tensor = torch.tensor(y, dtype=torch.float32)
        x_lag1_tensor = torch.tensor(x_lag1, dtype=torch.float32)
        x_lag5_tensor = torch.tensor(x_lag5, dtype=torch.float32)
        x_lag22_tensor = torch.tensor(x_lag22, dtype=torch.float32)
        return x_lag1_tensor, x_lag5_tensor, x_lag22_tensor, y_tensor


class GSPHAR_Dataset(Dataset):
    """
    Standard dataset class for GSPHAR model.

    This class provides a direct approach to handling the data,
    working directly with DataFrames without requiring an intermediate dictionary structure.
    This implementation is more memory-efficient and flexible than the legacy version.
    """

    def __init__(self, dataset, y_cols, lag1_cols, lag5_cols, lag22_cols, market_indices):
        """
        Initialize the dataset.

        Args:
            dataset (pd.DataFrame): Input dataframe.
            y_cols (list): List of target columns.
            lag1_cols (list): List of lag-1 columns.
            lag5_cols (list): List of lag-5 columns.
            lag22_cols (list): List of lag-22 columns.
            market_indices (list): List of market indices.
        """
        self.dataset = dataset
        self.y_cols = y_cols
        self.lag1_cols = lag1_cols
        self.lag5_cols = lag5_cols
        self.lag22_cols = lag22_cols
        self.market_indices = market_indices
        self.n_markets = len(market_indices)

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (x_lag1_tensor, x_lag5_tensor, x_lag22_tensor, y_tensor)
        """
        date = self.dataset.index[idx]

        # Get y values
        y = self.dataset.loc[date, self.y_cols].values

        # Get lag1 values
        x_lag1 = self.dataset.loc[date, self.lag1_cols].values.reshape(-1)

        # Get lag5 values - create a properly shaped array
        x_lag5 = np.zeros((self.n_markets, 5))
        for i, market in enumerate(self.market_indices):
            for j in range(5):
                lag_col = f"{market}_{j+1}"
                if lag_col in self.lag5_cols:
                    x_lag5[i, j] = self.dataset.loc[date, lag_col]

        # Get lag22 values - create a properly shaped array
        x_lag22 = np.zeros((self.n_markets, 22))
        for i, market in enumerate(self.market_indices):
            for j in range(22):
                lag_col = f"{market}_{j+1}"
                if lag_col in self.lag22_cols:
                    x_lag22[i, j] = self.dataset.loc[date, lag_col]

        return (torch.tensor(x_lag1, dtype=torch.float32),
                torch.tensor(x_lag5, dtype=torch.float32),
                torch.tensor(x_lag22, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32))


# For backward compatibility
NewGSPHAR_Dataset = GSPHAR_Dataset
