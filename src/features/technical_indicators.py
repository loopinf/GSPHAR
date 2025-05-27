"""
Advanced Technical Indicators for Trading Agent Feature Engineering.

This module provides comprehensive technical analysis features including:
- Price momentum indicators
- Volume analysis
- Volatility regime detection
- Market microstructure features
- Cross-asset correlation features
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TechnicalIndicatorEngine:
    """
    Comprehensive technical indicator calculation engine.

    Computes various technical indicators from OHLCV data to enhance
    trading agent decision making.
    """

    def __init__(self, lookback_periods: List[int] = [1, 4, 12, 24, 48]):
        """
        Initialize the technical indicator engine.

        Args:
            lookback_periods: List of lookback periods for indicators (in hours)
        """
        self.lookback_periods = lookback_periods
        self.max_lookback = max(lookback_periods)

    def calculate_momentum_features(self, prices: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calculate price momentum features.

        Args:
            prices: [batch_size, n_assets, time_steps] price tensor

        Returns:
            Dictionary of momentum features
        """
        features = {}

        # Price returns for different periods
        for period in self.lookback_periods:
            if prices.shape[-1] > period:
                returns = (prices[..., -1] / prices[..., -1-period] - 1.0)
                features[f'return_{period}h'] = returns

                # Momentum strength (absolute return)
                features[f'momentum_{period}h'] = torch.abs(returns)

                # Momentum direction (sign of return)
                features[f'direction_{period}h'] = torch.sign(returns)

        # Rolling volatility of returns
        if prices.shape[-1] > 24:
            returns_1h = prices[..., 1:] / prices[..., :-1] - 1.0
            rolling_vol = torch.std(returns_1h[..., -24:], dim=-1)
            features['rolling_volatility_24h'] = rolling_vol

        # Price acceleration (second derivative)
        if prices.shape[-1] > 2:
            price_diff1 = prices[..., 1:] - prices[..., :-1]
            price_diff2 = price_diff1[..., 1:] - price_diff1[..., :-1]
            features['price_acceleration'] = price_diff2[..., -1]

        return features

    def calculate_volume_features(self, volumes: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calculate volume-based features.

        Args:
            volumes: [batch_size, n_assets, time_steps] volume tensor

        Returns:
            Dictionary of volume features
        """
        features = {}

        # Volume momentum
        for period in self.lookback_periods:
            if volumes.shape[-1] > period:
                vol_ratio = volumes[..., -1] / (volumes[..., -1-period:].mean(dim=-1) + 1e-8)
                features[f'volume_ratio_{period}h'] = vol_ratio

        # Volume trend
        if volumes.shape[-1] > 12:
            recent_vol = volumes[..., -6:].mean(dim=-1)
            past_vol = volumes[..., -12:-6].mean(dim=-1)
            features['volume_trend'] = recent_vol / (past_vol + 1e-8)

        # Volume volatility
        if volumes.shape[-1] > 24:
            vol_std = torch.std(volumes[..., -24:], dim=-1)
            vol_mean = torch.mean(volumes[..., -24:], dim=-1)
            features['volume_volatility'] = vol_std / (vol_mean + 1e-8)

        return features

    def calculate_ohlc_features(self, ohlc: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Calculate OHLC-specific features.

        Args:
            ohlc: Dictionary with 'open', 'high', 'low', 'close' tensors

        Returns:
            Dictionary of OHLC features
        """
        features = {}

        open_prices = ohlc['open']
        high_prices = ohlc['high']
        low_prices = ohlc['low']
        close_prices = ohlc['close']

        # Current candle features
        features['open_close_ratio'] = close_prices[..., -1] / (open_prices[..., -1] + 1e-8)
        features['high_low_ratio'] = high_prices[..., -1] / (low_prices[..., -1] + 1e-8)
        features['close_high_ratio'] = close_prices[..., -1] / (high_prices[..., -1] + 1e-8)
        features['close_low_ratio'] = close_prices[..., -1] / (low_prices[..., -1] + 1e-8)

        # Intraday range
        features['intraday_range'] = (high_prices[..., -1] - low_prices[..., -1]) / (close_prices[..., -1] + 1e-8)

        # Gap features
        if open_prices.shape[-1] > 1:
            features['gap'] = (open_prices[..., -1] - close_prices[..., -2]) / (close_prices[..., -2] + 1e-8)

        # Wick analysis
        upper_wick = high_prices[..., -1] - torch.max(open_prices[..., -1], close_prices[..., -1])
        lower_wick = torch.min(open_prices[..., -1], close_prices[..., -1]) - low_prices[..., -1]
        body_size = torch.abs(close_prices[..., -1] - open_prices[..., -1])

        features['upper_wick_ratio'] = upper_wick / (body_size + 1e-8)
        features['lower_wick_ratio'] = lower_wick / (body_size + 1e-8)

        return features

    def calculate_volatility_regime(self, volatility_history: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Detect volatility regime features.

        Args:
            volatility_history: [batch_size, n_assets, time_steps] volatility predictions

        Returns:
            Dictionary of volatility regime features
        """
        features = {}

        if volatility_history.shape[-1] > 24:
            # Current vs historical volatility
            current_vol = volatility_history[..., -1]
            historical_vol = volatility_history[..., -24:].mean(dim=-1)
            features['vol_regime'] = current_vol / (historical_vol + 1e-8)

            # Volatility trend
            recent_vol = volatility_history[..., -6:].mean(dim=-1)
            past_vol = volatility_history[..., -12:-6].mean(dim=-1)
            features['vol_trend'] = recent_vol / (past_vol + 1e-8)

            # Volatility persistence
            vol_changes = volatility_history[..., 1:] - volatility_history[..., :-1]
            features['vol_persistence'] = torch.std(vol_changes[..., -12:], dim=-1)

        return features

    def calculate_time_features(self, timestamps: Optional[pd.DatetimeIndex] = None) -> Dict[str, torch.Tensor]:
        """
        Calculate time-based features.

        Args:
            timestamps: Optional datetime index for time-based features

        Returns:
            Dictionary of time features
        """
        features = {}

        if timestamps is not None:
            # Hour of day (crypto markets are 24/7)
            hour = timestamps[-1].hour
            features['hour_sin'] = torch.tensor(np.sin(2 * np.pi * hour / 24), dtype=torch.float32)
            features['hour_cos'] = torch.tensor(np.cos(2 * np.pi * hour / 24), dtype=torch.float32)

            # Day of week
            day = timestamps[-1].weekday()
            features['day_sin'] = torch.tensor(np.sin(2 * np.pi * day / 7), dtype=torch.float32)
            features['day_cos'] = torch.tensor(np.cos(2 * np.pi * day / 7), dtype=torch.float32)

        return features

    def calculate_cross_asset_features(self, prices: torch.Tensor, volumes: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calculate cross-asset correlation and market-wide features.

        Args:
            prices: [batch_size, n_assets, time_steps] price tensor
            volumes: [batch_size, n_assets, time_steps] volume tensor

        Returns:
            Dictionary of cross-asset features
        """
        features = {}

        if prices.shape[-1] > 12:
            # Calculate returns
            returns = prices[..., 1:] / prices[..., :-1] - 1.0
            recent_returns = returns[..., -12:]  # Last 12 hours

            # Market-wide momentum (equal-weighted)
            market_momentum = recent_returns.mean(dim=1)  # [batch_size, time_steps]
            features['market_momentum'] = market_momentum[..., -1]  # Latest

            # Asset vs market correlation
            if recent_returns.shape[-1] > 6:
                asset_returns = recent_returns[..., -6:]  # [batch_size, n_assets, 6]
                market_returns = asset_returns.mean(dim=1, keepdim=True)  # [batch_size, 1, 6]

                # Simple correlation approximation
                asset_centered = asset_returns - asset_returns.mean(dim=-1, keepdim=True)
                market_centered = market_returns - market_returns.mean(dim=-1, keepdim=True)

                correlation = (asset_centered * market_centered).mean(dim=-1)
                features['market_correlation'] = correlation.squeeze()

        # Volume concentration (how concentrated is trading in few assets)
        if volumes.shape[-1] > 0:
            total_volume = volumes[..., -1].sum(dim=1, keepdim=True)  # [batch_size, 1]
            volume_share = volumes[..., -1] / (total_volume + 1e-8)  # [batch_size, n_assets]
            features['volume_concentration'] = volume_share

        return features


class FeatureProcessor(nn.Module):
    """
    Neural network module to process and combine technical indicators.
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 64, output_dim: int = 32):
        """
        Initialize feature processor.

        Args:
            feature_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output feature dimension
        """
        super().__init__()

        self.feature_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Process features through neural network.

        Args:
            features: [batch_size, n_assets, feature_dim] feature tensor

        Returns:
            Processed features [batch_size, n_assets, output_dim]
        """
        return self.feature_net(features)


def combine_features(feature_dict: Dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    """
    Combine multiple feature tensors into a single tensor.

    Args:
        feature_dict: Dictionary of feature tensors
        device: Target device for tensors

    Returns:
        Combined feature tensor [batch_size, n_assets, total_features]
    """
    feature_list = []
    target_batch_size = None
    target_n_assets = None

    # First pass: determine target dimensions
    for key, tensor in feature_dict.items():
        tensor = tensor.to(device)

        if tensor.dim() >= 2:
            if target_batch_size is None:
                target_batch_size = tensor.shape[0]
                target_n_assets = tensor.shape[1] if tensor.dim() > 1 else 1
            break

    # Default dimensions if no multi-dimensional tensors found
    if target_batch_size is None:
        target_batch_size = 1
        target_n_assets = 1

    # Second pass: reshape all tensors to match target dimensions
    for key, tensor in feature_dict.items():
        tensor = tensor.to(device, dtype=torch.float32)

        # Handle different tensor shapes
        if tensor.dim() == 0:  # scalar -> [batch_size, n_assets, 1]
            tensor = tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            tensor = tensor.expand(target_batch_size, target_n_assets, 1)
        elif tensor.dim() == 1:  # [n_assets] or [batch_size] -> [batch_size, n_assets, 1]
            if tensor.shape[0] == target_n_assets:
                # Assume it's [n_assets]
                tensor = tensor.unsqueeze(0).unsqueeze(-1)
                tensor = tensor.expand(target_batch_size, -1, -1)
            else:
                # Assume it's [batch_size]
                tensor = tensor.unsqueeze(1).unsqueeze(-1)
                tensor = tensor.expand(-1, target_n_assets, -1)
        elif tensor.dim() == 2:  # [batch_size, n_assets] -> [batch_size, n_assets, 1]
            tensor = tensor.unsqueeze(-1)
        elif tensor.dim() == 3:  # Already correct shape
            pass
        else:
            # Flatten extra dimensions
            tensor = tensor.view(target_batch_size, target_n_assets, -1)

        # Ensure correct batch and asset dimensions
        if tensor.shape[0] != target_batch_size or tensor.shape[1] != target_n_assets:
            # Try to broadcast or reshape
            if tensor.shape[0] == 1 and tensor.shape[1] == target_n_assets:
                tensor = tensor.expand(target_batch_size, -1, -1)
            elif tensor.shape[0] == target_batch_size and tensor.shape[1] == 1:
                tensor = tensor.expand(-1, target_n_assets, -1)
            else:
                # Skip incompatible tensors with warning
                print(f"Warning: Skipping feature '{key}' with incompatible shape {tensor.shape}")
                continue

        feature_list.append(tensor)

    if feature_list:
        return torch.cat(feature_list, dim=-1)
    else:
        # Return empty tensor if no features
        return torch.empty(target_batch_size, target_n_assets, 0, dtype=torch.float32, device=device)
