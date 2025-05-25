#!/usr/bin/env python
"""
Fixed dataset that predicts downside movement potential instead of realized volatility.

Key Changes:
1. Target: Maximum downside movement in next period (open to low)
2. Prediction: How much price might drop (for limit order placement)
3. Realistic: Based on actual intraday price movements
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)


class DownsideMovementDataset(Dataset):
    """
    Dataset for predicting downside movement potential instead of realized volatility.

    Target: Maximum percentage drop from open to low in next period
    Use case: Determine optimal limit order discount for long strategy
    """

    def __init__(self, volatility_file, ohlcv_folder="data/ohlcv_1h", lags=[1, 4, 24],
                 holding_period=4, debug=False):
        """
        Initialize dataset with downside movement targets.

        Args:
            volatility_file: Path to volatility CSV (for features)
            ohlcv_folder: Folder containing OHLCV data
            lags: List of lag periods for features
            holding_period: How long to hold positions (for exit price)
            debug: Enable debug logging
        """
        self.volatility_file = volatility_file
        self.ohlcv_folder = ohlcv_folder
        self.lags = sorted(lags)
        self.holding_period = holding_period
        self.debug = debug

        if debug:
            logger.setLevel(logging.DEBUG)

        # Load and prepare data
        self._load_data()
        self._create_samples()

        logger.info(f"DownsideMovementDataset initialized:")
        logger.info(f"  Total samples: {len(self.samples)}")
        logger.info(f"  Assets: {len(self.assets)}")
        logger.info(f"  Lags: {self.lags}")
        logger.info(f"  Holding period: {self.holding_period}")
        logger.info(f"  Target: Downside movement (open to low)")

    def _load_data(self):
        """Load volatility and OHLCV data."""
        logger.info("Loading volatility data...")

        # Load volatility data (for features)
        self.vol_df = pd.read_csv(self.volatility_file, index_col=0, parse_dates=True)
        self.assets = self.vol_df.columns.tolist()

        logger.info(f"Loaded volatility data: {self.vol_df.shape}")
        logger.info(f"Assets: {len(self.assets)}")
        logger.info(f"Date range: {self.vol_df.index.min()} to {self.vol_df.index.max()}")

        # Load OHLCV data
        logger.info("Loading OHLCV data...")
        self.ohlcv_data = {}

        for asset in self.assets:
            ohlcv_file = f"{self.ohlcv_folder}/{asset}_1h_ohlcv.csv"
            try:
                df = pd.read_csv(ohlcv_file, index_col=0, parse_dates=True)
                # Check if columns need renaming
                if len(df.columns) >= 5:
                    df.columns = ['open', 'high', 'low', 'close', 'volume']
                self.ohlcv_data[asset] = df

                if self.debug:
                    logger.debug(f"Loaded {asset}: {df.shape}")

            except FileNotFoundError:
                logger.warning(f"OHLCV file not found for {asset}: {ohlcv_file}")
                continue

        logger.info(f"Loaded OHLCV data for {len(self.ohlcv_data)} assets")

        # Align timestamps
        self._align_timestamps()

    def _align_timestamps(self):
        """Align volatility and OHLCV timestamps."""
        logger.info("Aligning timestamps...")

        # Get common timestamps
        vol_timestamps = set(self.vol_df.index)
        ohlcv_timestamps = set()

        for asset, df in self.ohlcv_data.items():
            ohlcv_timestamps.update(df.index)

        # Find intersection
        common_timestamps = vol_timestamps.intersection(ohlcv_timestamps)
        common_timestamps = sorted(list(common_timestamps))

        logger.info(f"Common timestamps: {len(common_timestamps)}")
        logger.info(f"Range: {min(common_timestamps)} to {max(common_timestamps)}")

        # Filter data to common timestamps
        self.vol_df = self.vol_df.loc[common_timestamps]

        for asset in self.assets:
            if asset in self.ohlcv_data:
                self.ohlcv_data[asset] = self.ohlcv_data[asset].loc[
                    self.ohlcv_data[asset].index.isin(common_timestamps)
                ]

        self.timestamps = common_timestamps
        logger.info(f"Aligned to {len(self.timestamps)} timestamps")

    def _create_samples(self):
        """Create samples with downside movement targets."""
        logger.info("Creating samples with downside movement targets...")

        self.samples = []
        max_lag = max(self.lags)

        # Need enough history for lags and future for holding period
        start_idx = max_lag
        end_idx = len(self.timestamps) - self.holding_period

        logger.info(f"Sample range: {start_idx} to {end_idx} ({end_idx - start_idx} samples)")

        for i in range(start_idx, end_idx):
            current_time = self.timestamps[i]

            # Get prediction time (T+1)
            if i + 1 < len(self.timestamps):
                prediction_time = self.timestamps[i + 1]
            else:
                continue

            # Create sample
            sample = self._create_single_sample(i, current_time, prediction_time)

            if sample is not None:
                self.samples.append(sample)

                if self.debug and len(self.samples) <= 5:
                    logger.debug(f"Sample {len(self.samples)}: {current_time} -> {prediction_time}")

        logger.info(f"Created {len(self.samples)} samples")

    def _create_single_sample(self, time_idx, current_time, prediction_time):
        """Create a single sample with downside movement target."""

        # Get volatility features (lagged)
        vol_features = {}
        for lag in self.lags:
            lag_time = self.timestamps[time_idx - lag]
            vol_features[lag] = self.vol_df.loc[lag_time].values

        # Get OHLCV data for prediction period and holding period
        ohlcv_arrays = []
        downside_targets = []

        for asset in self.assets:
            if asset not in self.ohlcv_data:
                return None

            asset_ohlcv = self.ohlcv_data[asset]

            # Get OHLCV data for prediction period + holding period
            ohlcv_period = []

            for period_offset in range(self.holding_period + 1):
                period_time = self.timestamps[time_idx + 1 + period_offset]

                if period_time not in asset_ohlcv.index:
                    return None

                period_data = asset_ohlcv.loc[period_time]
                ohlcv_period.append([
                    period_data['open'],
                    period_data['high'],
                    period_data['low'],
                    period_data['close'],
                    period_data['volume']
                ])

            ohlcv_arrays.append(ohlcv_period)

            # Calculate downside movement target for T+1 period
            # Target: Maximum drop from open to low in next period
            next_period = ohlcv_period[0]  # T+1 period
            open_price = next_period[0]
            low_price = next_period[2]

            # Downside movement as percentage
            downside_movement = (open_price - low_price) / open_price
            downside_targets.append(max(0.0, downside_movement))  # Ensure non-negative

        return {
            'time_idx': time_idx,
            'current_time': current_time,
            'prediction_time': prediction_time,
            'vol_features': vol_features,
            'ohlcv_arrays': np.array(ohlcv_arrays),  # [assets, periods, ohlcv]
            'downside_targets': np.array(downside_targets)  # [assets]
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Get a sample for training."""
        sample = self.samples[idx]

        # Prepare volatility features for each lag
        x_lags = []
        for lag in self.lags:
            vol_features = torch.FloatTensor(sample['vol_features'][lag])
            x_lags.append(vol_features)

        # Prepare targets (downside movement)
        downside_targets = torch.FloatTensor(sample['downside_targets']).unsqueeze(-1)  # [assets, 1]

        # Prepare OHLCV data
        ohlcv_data = torch.FloatTensor(sample['ohlcv_arrays'])  # [assets, periods, 5]

        return {
            'x_lags': x_lags,
            'downside_targets': downside_targets,
            'ohlcv_data': ohlcv_data
        }

    def get_sample_info(self, idx):
        """Get metadata for a sample."""
        sample = self.samples[idx]
        return {
            'time_idx': sample['time_idx'],
            'current_time': sample['current_time'],
            'prediction_time': sample['prediction_time']
        }

    def get_assets(self):
        """Get list of assets."""
        return self.assets.copy()


def load_downside_movement_data(volatility_file, ohlcv_folder="data/ohlcv_1h",
                               lags=[1, 4, 24], holding_period=4, debug=False):
    """
    Load downside movement dataset.

    Returns:
        dataset: DownsideMovementDataset instance
        metadata: Dictionary with dataset information
    """
    logger.info("Loading downside movement dataset...")

    dataset = DownsideMovementDataset(
        volatility_file=volatility_file,
        ohlcv_folder=ohlcv_folder,
        lags=lags,
        holding_period=holding_period,
        debug=debug
    )

    metadata = {
        'assets': dataset.get_assets(),
        'total_samples': len(dataset),
        'lags': lags,
        'holding_period': holding_period,
        'target_type': 'downside_movement',
        'target_description': 'Maximum percentage drop from open to low in next period'
    }

    logger.info(f"Dataset loaded successfully:")
    logger.info(f"  Target: {metadata['target_description']}")
    logger.info(f"  Samples: {metadata['total_samples']}")
    logger.info(f"  Assets: {len(metadata['assets'])}")

    return dataset, metadata


if __name__ == "__main__":
    # Test the dataset
    logging.basicConfig(level=logging.INFO)

    print("🧪 TESTING DOWNSIDE MOVEMENT DATASET")
    print("=" * 60)

    # Load dataset
    dataset, metadata = load_downside_movement_data(
        volatility_file="data/crypto_rv1h_38_20200822_20250116.csv",
        lags=[1, 4, 24],
        holding_period=4,
        debug=True
    )

    print(f"\n📊 DATASET SUMMARY:")
    print(f"Total samples: {len(dataset)}")
    print(f"Assets: {len(metadata['assets'])}")
    print(f"Target: {metadata['target_description']}")

    # Test first few samples
    print(f"\n🔍 SAMPLE ANALYSIS:")
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        sample_info = dataset.get_sample_info(i)

        downside_targets = sample['downside_targets'].numpy().flatten()

        print(f"\nSample {i+1}: {sample_info['prediction_time']}")
        print(f"  Downside targets mean: {downside_targets.mean():.4f} ({downside_targets.mean()*100:.2f}%)")
        print(f"  Downside targets std: {downside_targets.std():.4f} ({downside_targets.std()*100:.2f}%)")
        print(f"  Downside targets range: {downside_targets.min():.4f} to {downside_targets.max():.4f}")
        print(f"  Non-zero targets: {(downside_targets > 0).sum()}/{len(downside_targets)}")

    print(f"\n✅ Dataset test completed!")
