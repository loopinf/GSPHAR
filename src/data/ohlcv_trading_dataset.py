#!/usr/bin/env python
"""
OHLCV Trading Dataset for profit maximization with proper order fill detection.

This dataset provides:
1. Volatility prediction inputs (lag features)
2. Volatility targets (RV data)
3. OHLCV data for accurate order fill detection using LOW prices
4. Proper time alignment for trading strategy simulation
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)


class OHLCVTradingDataset(Dataset):
    """
    Dataset for OHLCV-based trading strategy training.

    Provides volatility prediction features alongside OHLCV data for
    accurate order fill detection and profit calculation.
    """

    def __init__(self, volatility_data, ohlcv_tensor, time_index, assets,
                 lags, holding_period=4, debug=False):
        """
        Initialize the OHLCV trading dataset.

        Args:
            volatility_data (pd.DataFrame): Realized volatility data [time, assets]
            ohlcv_tensor (np.ndarray): OHLCV data [time, assets, 5] (O,H,L,C,V)
            time_index (pd.DatetimeIndex): Time index for alignment
            assets (list): List of asset names
            lags (list): List of lag values for features [1, 4, 24]
            holding_period (int): Holding period in hours (default: 4)
            debug (bool): Enable debug logging
        """
        self.volatility_data = volatility_data
        self.ohlcv_tensor = torch.tensor(ohlcv_tensor, dtype=torch.float32)
        self.time_index = time_index
        self.assets = assets
        self.lags = lags
        self.holding_period = holding_period
        self.debug = debug

        # Ensure data alignment
        self._align_data()

        # Calculate valid indices for training
        self.valid_indices = self._get_valid_indices()

        if self.debug:
            logger.info(f"OHLCVTradingDataset initialized:")
            logger.info(f"  Volatility data shape: {self.volatility_data.shape}")
            logger.info(f"  OHLCV tensor shape: {self.ohlcv_tensor.shape}")
            logger.info(f"  Assets: {len(self.assets)}")
            logger.info(f"  Lags: {self.lags}")
            logger.info(f"  Holding period: {self.holding_period}h")
            logger.info(f"  Valid samples: {len(self.valid_indices)}")

    def _align_data(self):
        """Ensure volatility data and OHLCV data are properly aligned."""

        # Check time alignment
        vol_index = self.volatility_data.index
        ohlcv_time_len = self.ohlcv_tensor.shape[0]

        if len(vol_index) != ohlcv_time_len:
            raise ValueError(f"Time dimension mismatch: volatility {len(vol_index)} vs OHLCV {ohlcv_time_len}")

        # Check asset alignment
        vol_assets = list(self.volatility_data.columns)
        if vol_assets != self.assets:
            # Reorder volatility data to match OHLCV asset order
            self.volatility_data = self.volatility_data[self.assets]
            if self.debug:
                logger.info("Reordered volatility data to match OHLCV asset order")

        if self.debug:
            logger.info("Data alignment verified")

    def _get_valid_indices(self):
        """Calculate valid indices for training samples."""
        max_lag = max(self.lags)
        min_idx = max_lag
        # Extra -1 for volatility target shift (we predict T+1, not T+0)
        max_idx = len(self.volatility_data) - self.holding_period - 2

        valid_indices = list(range(min_idx, max_idx))

        if self.debug:
            logger.info(f"Valid indices: {min_idx} to {max_idx-1} ({len(valid_indices)} samples)")
            logger.info(f"Look-ahead bias fix: predicting T+1 volatility using T-lag features")

        return valid_indices

    def __len__(self):
        """Return the number of valid samples."""
        return len(self.valid_indices)

    def __getitem__(self, idx):
        """
        Get a training sample.

        Args:
            idx (int): Sample index

        Returns:
            dict: Training sample with keys:
                - 'x_lags': List of lag tensors for volatility prediction
                - 'vol_targets': Volatility targets [n_assets, 1]
                - 'ohlcv_data': OHLCV sequence [n_assets, holding_period+1, 5]
                - 'prediction_idx': Index where prediction is made (for reference)
        """
        actual_idx = self.valid_indices[idx]

        # Get lag features for volatility prediction
        x_lags = []
        for lag in self.lags:
            # Get lag periods of volatility data
            lag_data = self.volatility_data.iloc[actual_idx-lag:actual_idx].values

            # Ensure we have the right number of periods
            if lag_data.shape[0] < lag:
                # Pad if necessary (shouldn't happen with valid indices)
                pad_size = lag - lag_data.shape[0]
                lag_data = np.pad(lag_data, ((pad_size, 0), (0, 0)), 'edge')

            # Transpose to [n_assets, lag_periods]
            lag_tensor = torch.tensor(lag_data.T, dtype=torch.float32)
            x_lags.append(lag_tensor)

        # Get volatility targets (NEXT period to avoid look-ahead bias)
        # We predict T+1 volatility using T-1, T-4, T-24 lags
        vol_targets = self.volatility_data.iloc[actual_idx + 1].values
        vol_targets = torch.tensor(vol_targets, dtype=torch.float32).unsqueeze(-1)  # [n_assets, 1]

        # Get OHLCV data for trading simulation
        # We need: current period + next period + holding period
        ohlcv_start = actual_idx
        ohlcv_end = actual_idx + self.holding_period + 1

        # Extract OHLCV sequence: [time_periods, n_assets, 5]
        ohlcv_sequence = self.ohlcv_tensor[ohlcv_start:ohlcv_end]

        # Transpose to [n_assets, time_periods, 5] for easier processing
        ohlcv_sequence = ohlcv_sequence.permute(1, 0, 2)

        return {
            'x_lags': x_lags,
            'vol_targets': vol_targets,
            'ohlcv_data': ohlcv_sequence,
            'prediction_idx': actual_idx
        }

    def get_sample_info(self, idx):
        """
        Get human-readable information about a sample.

        Args:
            idx (int): Sample index

        Returns:
            dict: Sample information
        """
        actual_idx = self.valid_indices[idx]
        prediction_time = self.time_index[actual_idx]

        return {
            'sample_idx': idx,
            'actual_idx': actual_idx,
            'prediction_time': prediction_time,
            'holding_period': self.holding_period,
            'assets': self.assets
        }


def load_ohlcv_trading_data(volatility_file, ohlcv_dir="data/preprocessed",
                           lags=[1, 4, 24], holding_period=4, debug=False):
    """
    Load and prepare OHLCV trading data.

    Args:
        volatility_file (str): Path to volatility CSV file
        ohlcv_dir (str): Directory containing preprocessed OHLCV data
        lags (list): Lag values for features
        holding_period (int): Holding period in hours
        debug (bool): Enable debug logging

    Returns:
        tuple: (dataset, metadata)
    """
    logger.info("Loading OHLCV trading data...")

    # Load volatility data
    volatility_data = pd.read_csv(volatility_file, index_col=0, parse_dates=True)
    logger.info(f"Loaded volatility data: {volatility_data.shape}")

    # Load OHLCV tensor
    ohlcv_tensor_file = f"{ohlcv_dir}/ohlcv_tensor.npy"
    ohlcv_tensor = np.load(ohlcv_tensor_file)
    logger.info(f"Loaded OHLCV tensor: {ohlcv_tensor.shape}")

    # Load time index
    time_index_file = f"{ohlcv_dir}/time_index.csv"
    time_df = pd.read_csv(time_index_file)
    time_index = pd.to_datetime(time_df['datetime'])
    logger.info(f"Loaded time index: {len(time_index)} periods")

    # Get asset names (from volatility data columns)
    assets = list(volatility_data.columns)
    logger.info(f"Assets: {len(assets)}")

    # Create dataset
    dataset = OHLCVTradingDataset(
        volatility_data=volatility_data,
        ohlcv_tensor=ohlcv_tensor,
        time_index=time_index,
        assets=assets,
        lags=lags,
        holding_period=holding_period,
        debug=debug
    )

    metadata = {
        'n_assets': len(assets),
        'n_samples': len(dataset),
        'lags': lags,
        'holding_period': holding_period,
        'date_range': (time_index.min(), time_index.max()),
        'assets': assets
    }

    logger.info(f"Created dataset with {len(dataset)} samples")

    return dataset, metadata


def create_ohlcv_dataloaders(dataset, train_ratio=0.8, batch_size=16, shuffle=True):
    """
    Create train/validation dataloaders from OHLCV dataset.

    Args:
        dataset (OHLCVTradingDataset): The dataset
        train_ratio (float): Ratio for train/validation split
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle training data

    Returns:
        tuple: (train_loader, val_loader, split_info)
    """
    from torch.utils.data import DataLoader, Subset

    # Calculate split indices
    n_samples = len(dataset)
    train_size = int(n_samples * train_ratio)

    # Create subsets (time-ordered split, no shuffling of indices)
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, n_samples))

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )

    split_info = {
        'total_samples': n_samples,
        'train_samples': len(train_indices),
        'val_samples': len(val_indices),
        'train_ratio': train_ratio,
        'batch_size': batch_size
    }

    logger.info(f"Created dataloaders:")
    logger.info(f"  Train: {len(train_indices)} samples, {len(train_loader)} batches")
    logger.info(f"  Validation: {len(val_indices)} samples, {len(val_loader)} batches")

    return train_loader, val_loader, split_info


if __name__ == "__main__":
    # Test the dataset
    logging.basicConfig(level=logging.INFO)

    # Load test data
    dataset, metadata = load_ohlcv_trading_data(
        volatility_file="data/crypto_rv1h_38_20200822_20250116.csv",
        holding_period=4,
        debug=True
    )

    # Test a sample
    sample = dataset[0]
    sample_info = dataset.get_sample_info(0)

    print("\nSample structure:")
    for key, value in sample.items():
        if isinstance(value, list):
            print(f"  {key}: list of {len(value)} tensors, shapes: {[x.shape for x in value]}")
        else:
            print(f"  {key}: {value.shape if hasattr(value, 'shape') else type(value)}")

    print(f"\nSample info: {sample_info}")

    # Create dataloaders
    train_loader, val_loader, split_info = create_ohlcv_dataloaders(dataset, batch_size=4)

    print(f"\nDataloader info: {split_info}")

    # Test batch
    batch = next(iter(train_loader))
    print(f"\nBatch structure:")
    for key, value in batch.items():
        if isinstance(value, list):
            print(f"  {key}: list of {len(value)} tensors, shapes: {[x.shape for x in value]}")
        else:
            print(f"  {key}: {value.shape if hasattr(value, 'shape') else type(value)}")
