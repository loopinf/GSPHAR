"""
Dataset for agent-based trading that includes volatility prediction history.

This dataset extends the OHLCV trading dataset to include historical volatility
predictions needed for the trading agent model.
"""

import torch
import torch.utils.data as data
import pandas as pd
import numpy as np
import os
import sys
from typing import List, Tuple, Dict, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


class AgentTradingDataset(data.Dataset):
    """
    Dataset for agent-based trading with volatility prediction history.
    
    This dataset provides:
    1. Lag features for volatility prediction (x_lags)
    2. Volatility targets for supervised learning
    3. Historical volatility predictions for agent input
    4. OHLCV data for trading simulation
    """
    
    def __init__(self, volatility_data, ohlcv_data, lags, holding_period=4, 
                 history_length=24, min_history_periods=48):
        """
        Initialize the agent trading dataset.
        
        Args:
            volatility_data (pd.DataFrame): Realized volatility data with datetime index
            ohlcv_data (pd.DataFrame): OHLCV data with datetime index
            lags (list): List of lag values for volatility prediction
            holding_period (int): Number of periods to hold position
            history_length (int): Length of volatility prediction history for agent
            min_history_periods (int): Minimum periods needed for history
        """
        self.volatility_data = volatility_data
        self.ohlcv_data = ohlcv_data
        self.lags = sorted(lags)
        self.holding_period = holding_period
        self.history_length = history_length
        self.min_history_periods = min_history_periods
        
        # Align data by datetime index
        self.common_index = volatility_data.index.intersection(ohlcv_data.index)
        self.volatility_data = volatility_data.loc[self.common_index]
        self.ohlcv_data = ohlcv_data.loc[self.common_index]
        
        # Convert OHLCV to tensor format [time, assets, 5]
        self.ohlcv_tensor = self._prepare_ohlcv_tensor()
        
        # Calculate valid indices (considering lags, history, and holding period)
        self.valid_indices = self._calculate_valid_indices()
        
        print(f"AgentTradingDataset initialized:")
        print(f"  Total periods: {len(self.common_index)}")
        print(f"  Valid samples: {len(self.valid_indices)}")
        print(f"  Assets: {len(self.volatility_data.columns)}")
        print(f"  History length: {history_length}")
        print(f"  Holding period: {holding_period}")
    
    def _prepare_ohlcv_tensor(self):
        """Convert OHLCV DataFrame to tensor format."""
        # Assuming OHLCV data has MultiIndex columns (asset, component)
        assets = self.ohlcv_data.columns.get_level_values(0).unique()
        components = ['open', 'high', 'low', 'close', 'volume']
        
        ohlcv_array = np.zeros((len(self.ohlcv_data), len(assets), 5))
        
        for i, asset in enumerate(assets):
            for j, component in enumerate(components):
                if (asset, component) in self.ohlcv_data.columns:
                    ohlcv_array[:, i, j] = self.ohlcv_data[(asset, component)].values
        
        return torch.tensor(ohlcv_array, dtype=torch.float32)
    
    def _calculate_valid_indices(self):
        """Calculate valid sample indices considering all constraints."""
        max_lag = max(self.lags)
        
        # Need enough periods for:
        # - Maximum lag for volatility prediction
        # - History length for agent
        # - Holding period for trading simulation
        min_start_idx = max(max_lag, self.min_history_periods)
        max_end_idx = len(self.volatility_data) - self.holding_period - 1
        
        valid_indices = list(range(min_start_idx, max_end_idx))
        return valid_indices
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        """
        Get a training sample.
        
        Returns:
            dict: Training sample with keys:
                - 'x_lags': List of lag tensors for volatility prediction
                - 'vol_targets': Volatility targets [n_assets, 1]
                - 'vol_pred_history': Historical volatility predictions [n_assets, history_length]
                - 'ohlcv_data': OHLCV sequence [n_assets, holding_period+1, 5]
                - 'prediction_idx': Index where prediction is made
        """
        actual_idx = self.valid_indices[idx]
        
        # Get lag features for volatility prediction
        x_lags = []
        for lag in self.lags:
            lag_data = self.volatility_data.iloc[actual_idx-lag:actual_idx].values
            
            if lag_data.shape[0] < lag:
                pad_size = lag - lag_data.shape[0]
                lag_data = np.pad(lag_data, ((pad_size, 0), (0, 0)), 'edge')
            
            lag_tensor = torch.tensor(lag_data.T, dtype=torch.float32)
            x_lags.append(lag_tensor)
        
        # Get volatility targets (T+1 to avoid look-ahead bias)
        vol_targets = self.volatility_data.iloc[actual_idx + 1].values
        vol_targets = torch.tensor(vol_targets, dtype=torch.float32).unsqueeze(-1)
        
        # Get historical volatility predictions for agent
        # For now, use actual historical volatility as proxy for predictions
        # In real training, this would be replaced with actual model predictions
        history_start = actual_idx - self.history_length
        history_end = actual_idx
        
        if history_start < 0:
            # Pad with edge values if not enough history
            vol_history = self.volatility_data.iloc[:actual_idx].values
            pad_size = self.history_length - vol_history.shape[0]
            vol_history = np.pad(vol_history, ((pad_size, 0), (0, 0)), 'edge')
        else:
            vol_history = self.volatility_data.iloc[history_start:history_end].values
        
        vol_pred_history = torch.tensor(vol_history.T, dtype=torch.float32)
        
        # Get OHLCV data for trading simulation
        ohlcv_start = actual_idx
        ohlcv_end = actual_idx + self.holding_period + 1
        ohlcv_sequence = self.ohlcv_tensor[ohlcv_start:ohlcv_end]
        ohlcv_sequence = ohlcv_sequence.permute(1, 0, 2)  # [n_assets, time_periods, 5]
        
        return {
            'x_lags': x_lags,
            'vol_targets': vol_targets,
            'vol_pred_history': vol_pred_history,
            'ohlcv_data': ohlcv_sequence,
            'prediction_idx': actual_idx
        }
    
    def get_sample_info(self, idx):
        """Get metadata for a sample."""
        actual_idx = self.valid_indices[idx]
        return {
            'prediction_time': self.common_index[actual_idx],
            'actual_idx': actual_idx
        }
    
    def update_vol_pred_history(self, predictions_dict):
        """
        Update the dataset with actual volatility predictions from a trained model.
        
        Args:
            predictions_dict (dict): Dictionary mapping indices to volatility predictions
        """
        # This method would be called after training the volatility model
        # to replace the proxy historical volatility with actual predictions
        self.vol_predictions = predictions_dict
        print(f"Updated dataset with {len(predictions_dict)} volatility predictions")


def load_agent_trading_data(volatility_file, ohlcv_folder=None, lags=[1, 4, 24], 
                           holding_period=4, history_length=24, debug=False):
    """
    Load and prepare data for agent-based trading.
    
    Args:
        volatility_file (str): Path to volatility CSV file
        ohlcv_folder (str): Path to OHLCV data folder (optional)
        lags (list): Lag values for volatility prediction
        holding_period (int): Holding period for trading
        history_length (int): Length of volatility history for agent
        debug (bool): If True, use subset of data for debugging
        
    Returns:
        tuple: (dataset, metadata)
    """
    print(f"Loading agent trading data...")
    print(f"  Volatility file: {volatility_file}")
    print(f"  OHLCV folder: {ohlcv_folder}")
    print(f"  Lags: {lags}")
    print(f"  Holding period: {holding_period}")
    print(f"  History length: {history_length}")
    
    # Load volatility data
    vol_df = pd.read_csv(volatility_file, index_col=0, parse_dates=True)
    
    if debug:
        vol_df = vol_df.iloc[-1000:]  # Use last 1000 samples for debugging
        print(f"  Debug mode: Using last {len(vol_df)} samples")
    
    # Load OHLCV data
    if ohlcv_folder is None:
        # Create synthetic OHLCV data for testing
        print("  Creating synthetic OHLCV data...")
        ohlcv_df = create_synthetic_ohlcv(vol_df)
    else:
        print(f"  Loading OHLCV data from {ohlcv_folder}...")
        ohlcv_df = load_ohlcv_from_folder(ohlcv_folder, vol_df.index)
    
    # Create dataset
    dataset = AgentTradingDataset(
        volatility_data=vol_df,
        ohlcv_data=ohlcv_df,
        lags=lags,
        holding_period=holding_period,
        history_length=history_length
    )
    
    # Metadata
    metadata = {
        'assets': list(vol_df.columns),
        'n_assets': len(vol_df.columns),
        'lags': lags,
        'holding_period': holding_period,
        'history_length': history_length,
        'total_samples': len(dataset),
        'date_range': (vol_df.index.min(), vol_df.index.max())
    }
    
    print(f"✅ Agent trading data loaded successfully")
    print(f"  Assets: {metadata['n_assets']}")
    print(f"  Samples: {metadata['total_samples']}")
    print(f"  Date range: {metadata['date_range'][0]} to {metadata['date_range'][1]}")
    
    return dataset, metadata


def create_synthetic_ohlcv(vol_df):
    """Create synthetic OHLCV data for testing purposes."""
    assets = vol_df.columns
    dates = vol_df.index
    
    # Create MultiIndex columns for OHLCV
    components = ['open', 'high', 'low', 'close', 'volume']
    columns = pd.MultiIndex.from_product([assets, components])
    
    ohlcv_df = pd.DataFrame(index=dates, columns=columns)
    
    # Generate synthetic price data
    np.random.seed(42)  # For reproducibility
    
    for asset in assets:
        # Start with base price
        base_price = 100.0
        prices = [base_price]
        
        # Generate price series using volatility
        for i in range(1, len(dates)):
            vol = vol_df.loc[dates[i-1], asset] if not pd.isna(vol_df.loc[dates[i-1], asset]) else 0.02
            return_shock = np.random.normal(0, vol)
            new_price = prices[-1] * (1 + return_shock)
            prices.append(max(new_price, 0.01))  # Ensure positive prices
        
        # Create OHLCV from price series
        for i, date in enumerate(dates):
            close_price = prices[i]
            
            # Generate intraday high/low based on volatility
            vol = vol_df.loc[date, asset] if not pd.isna(vol_df.loc[date, asset]) else 0.02
            high_factor = 1 + np.random.uniform(0, vol)
            low_factor = 1 - np.random.uniform(0, vol)
            
            open_price = close_price * np.random.uniform(0.99, 1.01)
            high_price = close_price * high_factor
            low_price = close_price * low_factor
            volume = np.random.uniform(1000, 10000)
            
            ohlcv_df.loc[date, (asset, 'open')] = open_price
            ohlcv_df.loc[date, (asset, 'high')] = high_price
            ohlcv_df.loc[date, (asset, 'low')] = low_price
            ohlcv_df.loc[date, (asset, 'close')] = close_price
            ohlcv_df.loc[date, (asset, 'volume')] = volume
    
    return ohlcv_df.astype(float)


def load_ohlcv_from_folder(folder_path, date_index):
    """Load OHLCV data from folder structure."""
    # This would implement loading from actual OHLCV files
    # For now, return synthetic data
    print("  Note: Using synthetic OHLCV data (implement actual loading as needed)")
    return create_synthetic_ohlcv(pd.DataFrame(index=date_index, columns=['BTC', 'ETH']))
