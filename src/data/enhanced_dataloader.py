"""
Enhanced Data Loader with OHLCV and Technical Features.

This module provides comprehensive data loading capabilities including:
- OHLCV data loading and preprocessing
- Volume data integration
- Technical indicator calculation
- Multi-timeframe data alignment
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class EnhancedOHLCVDataLoader:
    """
    Enhanced data loader for OHLCV data with technical features.
    """

    def __init__(
        self,
        data_dir: str = "data",
        symbols: Optional[List[str]] = None,
        lookback_window: int = 48,  # Hours of historical data
        device: torch.device = torch.device('cpu')
    ):
        """
        Initialize enhanced OHLCV data loader.

        Args:
            data_dir: Directory containing OHLCV data
            symbols: List of symbols to load (if None, loads all available)
            lookback_window: Hours of historical data to maintain
            device: Device for tensor operations
        """
        self.data_dir = Path(data_dir)
        self.lookback_window = lookback_window
        self.device = device

        # Load OHLCV data
        self.ohlcv_data = self._load_ohlcv_data(symbols)
        self.symbols = list(self.ohlcv_data['close'].columns)
        self.n_assets = len(self.symbols)

        # Create time index
        self.time_index = self.ohlcv_data['close'].index

        logger.info(f"Loaded OHLCV data for {self.n_assets} assets")
        logger.info(f"Time range: {self.time_index[0]} to {self.time_index[-1]}")

    def _load_ohlcv_data(self, symbols: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Load OHLCV data from CSV files.

        Args:
            symbols: Optional list of symbols to filter

        Returns:
            Dictionary with OHLCV DataFrames
        """
        ohlcv_dir = self.data_dir / "ohlcv_1h"

        # Load combined OHLCV files
        ohlcv_files = {
            'open': 'crypto_open_1h_38.csv',
            'high': 'crypto_high_1h_38.csv',
            'low': 'crypto_low_1h_38.csv',
            'close': 'crypto_close_1h_38.csv',
            'volume': 'crypto_volume_1h_38.csv'
        }

        ohlcv_data = {}

        for data_type, filename in ohlcv_files.items():
            file_path = ohlcv_dir / filename

            if file_path.exists():
                df = pd.read_csv(file_path, index_col=0, parse_dates=True, dtype=np.float32)

                # Filter symbols if specified
                if symbols is not None:
                    available_symbols = [s for s in symbols if s in df.columns]
                    df = df[available_symbols]

                ohlcv_data[data_type] = df
                logger.info(f"Loaded {data_type} data: {df.shape}")
            else:
                logger.warning(f"File not found: {file_path}")

        # Ensure all OHLCV components have the same shape
        if ohlcv_data:
            reference_shape = ohlcv_data['close'].shape
            for data_type, df in ohlcv_data.items():
                if df.shape != reference_shape:
                    logger.warning(f"{data_type} shape {df.shape} != reference {reference_shape}")

        return ohlcv_data

    def get_ohlcv_window(
        self,
        end_time: Union[str, pd.Timestamp],
        window_hours: int = None
    ) -> Dict[str, torch.Tensor]:
        """
        Get OHLCV data for a specific time window.

        Args:
            end_time: End time for the window
            window_hours: Hours of data to retrieve (default: lookback_window)

        Returns:
            Dictionary of OHLCV tensors
        """
        if window_hours is None:
            window_hours = self.lookback_window

        # Convert end_time to timestamp with timezone awareness
        if isinstance(end_time, str):
            end_time = pd.Timestamp(end_time)

        # Ensure timezone compatibility
        if end_time.tz is None and self.time_index.tz is not None:
            end_time = end_time.tz_localize('UTC')
        elif end_time.tz is not None and self.time_index.tz is None:
            end_time = end_time.tz_localize(None)

        # Find the end index
        try:
            end_idx = self.time_index.get_loc(end_time)
        except KeyError:
            # Find nearest timestamp
            end_idx = self.time_index.get_indexer([end_time], method='nearest')[0]

        # Calculate start index
        start_idx = max(0, end_idx - window_hours + 1)

        # Extract data window
        ohlcv_tensors = {}
        for data_type, df in self.ohlcv_data.items():
            window_data = df.iloc[start_idx:end_idx + 1].values  # [time, assets]

            # Convert to tensor and add batch dimension
            tensor = torch.tensor(window_data.T, dtype=torch.float32, device=self.device)  # [assets, time]
            tensor = tensor.unsqueeze(0)  # [1, assets, time]

            ohlcv_tensors[data_type] = tensor

        return ohlcv_tensors

    def get_volume_data(
        self,
        end_time: Union[str, pd.Timestamp],
        window_hours: int = None
    ) -> torch.Tensor:
        """
        Get volume data for a specific time window.

        Args:
            end_time: End time for the window
            window_hours: Hours of data to retrieve

        Returns:
            Volume tensor [1, n_assets, time_steps]
        """
        ohlcv_data = self.get_ohlcv_window(end_time, window_hours)
        return ohlcv_data.get('volume', torch.empty(1, self.n_assets, 0, dtype=torch.float32, device=self.device))

    def get_price_data(
        self,
        end_time: Union[str, pd.Timestamp],
        window_hours: int = None,
        price_type: str = 'close'
    ) -> torch.Tensor:
        """
        Get price data for a specific time window.

        Args:
            end_time: End time for the window
            window_hours: Hours of data to retrieve
            price_type: Type of price ('open', 'high', 'low', 'close')

        Returns:
            Price tensor [1, n_assets, time_steps]
        """
        ohlcv_data = self.get_ohlcv_window(end_time, window_hours)
        return ohlcv_data.get(price_type, torch.empty(1, self.n_assets, 0, dtype=torch.float32, device=self.device))

    def get_current_prices(self, end_time: Union[str, pd.Timestamp]) -> torch.Tensor:
        """
        Get current prices for all assets.

        Args:
            end_time: Current time

        Returns:
            Current prices tensor [1, n_assets]
        """
        # Convert to timestamp with timezone handling
        if isinstance(end_time, str):
            end_time = pd.Timestamp(end_time)

        # Ensure timezone compatibility
        if end_time.tz is None and self.time_index.tz is not None:
            end_time = end_time.tz_localize('UTC')
        elif end_time.tz is not None and self.time_index.tz is None:
            end_time = end_time.tz_localize(None)

        price_data = self.get_price_data(end_time, window_hours=1, price_type='close')
        return price_data[..., -1]  # Latest price

    def create_enhanced_batch(
        self,
        end_times: List[Union[str, pd.Timestamp]],
        vol_pred_data: torch.Tensor,
        vol_history_data: torch.Tensor,
        window_hours: int = None
    ) -> Dict[str, torch.Tensor]:
        """
        Create a batch with enhanced OHLCV features.

        Args:
            end_times: List of end times for each batch item
            vol_pred_data: Volatility predictions [batch_size, n_assets, 1]
            vol_history_data: Volatility history [batch_size, n_assets, history_length]
            window_hours: Hours of OHLCV data to include

        Returns:
            Dictionary containing all data for enhanced model
        """
        batch_size = len(end_times)

        if window_hours is None:
            window_hours = self.lookback_window

        # Initialize batch containers
        batch_ohlcv = {key: [] for key in ['open', 'high', 'low', 'close', 'volume']}
        batch_current_prices = []

        # Process each time point
        for i, end_time in enumerate(end_times):
            # Get OHLCV data
            ohlcv_data = self.get_ohlcv_window(end_time, window_hours)

            for key in batch_ohlcv.keys():
                if key in ohlcv_data:
                    batch_ohlcv[key].append(ohlcv_data[key])
                else:
                    # Create empty tensor if data not available
                    empty_tensor = torch.zeros(1, self.n_assets, window_hours, dtype=torch.float32, device=self.device)
                    batch_ohlcv[key].append(empty_tensor)

            # Get current prices
            current_prices = self.get_current_prices(end_time)
            batch_current_prices.append(current_prices)

        # Stack batch data
        batch_data = {}
        for key, tensor_list in batch_ohlcv.items():
            if tensor_list:
                batch_data[key] = torch.cat(tensor_list, dim=0)  # [batch_size, n_assets, time]

        batch_data['current_prices'] = torch.cat(batch_current_prices, dim=0)  # [batch_size, n_assets]
        batch_data['vol_pred'] = vol_pred_data
        batch_data['vol_history'] = vol_history_data

        # Add timestamps for time-based features
        batch_data['timestamps'] = [pd.Timestamp(t) for t in end_times]

        return batch_data

    def get_symbol_info(self) -> Dict[str, any]:
        """
        Get information about loaded symbols.

        Returns:
            Dictionary with symbol information
        """
        return {
            'symbols': self.symbols,
            'n_assets': self.n_assets,
            'time_range': (self.time_index[0], self.time_index[-1]),
            'total_hours': len(self.time_index),
            'data_types': list(self.ohlcv_data.keys())
        }

    def validate_data_quality(self) -> Dict[str, any]:
        """
        Validate the quality of loaded OHLCV data.

        Returns:
            Dictionary with data quality metrics
        """
        quality_report = {}

        for data_type, df in self.ohlcv_data.items():
            # Check for missing values
            missing_count = df.isnull().sum().sum()
            missing_pct = (missing_count / df.size) * 100

            # Check for zero values (suspicious for prices)
            if data_type in ['open', 'high', 'low', 'close']:
                zero_count = (df == 0).sum().sum()
                zero_pct = (zero_count / df.size) * 100
            else:
                zero_count = zero_pct = 0

            # Check for negative values (suspicious for prices/volume)
            negative_count = (df < 0).sum().sum()
            negative_pct = (negative_count / df.size) * 100

            quality_report[data_type] = {
                'missing_values': missing_count,
                'missing_percentage': missing_pct,
                'zero_values': zero_count,
                'zero_percentage': zero_pct,
                'negative_values': negative_count,
                'negative_percentage': negative_pct,
                'shape': df.shape
            }

        return quality_report


def create_enhanced_dataloader(
    data_dir: str = "data",
    symbols: Optional[List[str]] = None,
    device: torch.device = torch.device('cpu')
) -> EnhancedOHLCVDataLoader:
    """
    Factory function to create enhanced data loader.

    Args:
        data_dir: Directory containing data
        symbols: Optional list of symbols
        device: Device for tensors

    Returns:
        Configured enhanced data loader
    """
    return EnhancedOHLCVDataLoader(
        data_dir=data_dir,
        symbols=symbols,
        device=device
    )
