"""
Enhanced Trading Agent Model with Advanced Feature Engineering.

This model incorporates technical indicators, market microstructure features,
and multi-timeframe analysis for sophisticated trading decisions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

from src.features.technical_indicators import TechnicalIndicatorEngine, FeatureProcessor, combine_features

logger = logging.getLogger(__name__)


class EnhancedTradingAgentModel(nn.Module):
    """
    Enhanced trading agent with comprehensive feature engineering.

    Features:
    - Technical indicators (momentum, volume, OHLC patterns)
    - Volatility regime detection
    - Cross-asset correlation analysis
    - Time-based patterns
    - Multi-timeframe analysis
    """

    def __init__(
        self,
        n_assets: int = 38,
        vol_history_length: int = 24,
        feature_dim: int = 128,
        hidden_dim: int = 256,
        num_conv_layers: int = 3,
        dropout_rate: float = 0.1,
        init_with_vol_pred: bool = True
    ):
        """
        Initialize enhanced trading agent.

        Args:
            n_assets: Number of assets to trade
            vol_history_length: Length of volatility history
            feature_dim: Dimension of processed features
            hidden_dim: Hidden layer dimension
            num_conv_layers: Number of convolutional layers
            dropout_rate: Dropout rate for regularization
            init_with_vol_pred: Whether to use vol_pred initialization
        """
        super().__init__()

        self.n_assets = n_assets
        self.vol_history_length = vol_history_length
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.init_with_vol_pred = init_with_vol_pred

        # Technical indicator engine
        self.indicator_engine = TechnicalIndicatorEngine()

        # Feature processing networks
        self.volatility_processor = FeatureProcessor(
            feature_dim=vol_history_length + 1,  # current + history
            hidden_dim=64,
            output_dim=32
        )

        self.technical_processor = FeatureProcessor(
            feature_dim=feature_dim,  # Will be determined dynamically
            hidden_dim=128,
            output_dim=64
        )

        # Main processing network
        total_feature_dim = 32 + 64  # vol features + technical features

        # Convolutional layers for temporal pattern recognition
        self.conv_layers = nn.ModuleList()
        in_channels = 1
        out_channels = 32

        for i in range(num_conv_layers):
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                    groups=1  # Regular convolution for feature mixing
                )
            )
            in_channels = out_channels
            out_channels = min(out_channels * 2, 128)

        # Global pooling and dense layers
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.dense_layers = nn.Sequential(
            nn.Linear(in_channels + total_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 64)
        )

        # Output heads
        self.ratio_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )

        self.direction_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output between 0 and 1 (0=short, 1=long)
        )

        # Initialize feature dimension tracker
        self.technical_feature_dim = None

    def extract_technical_features(
        self,
        ohlc_data: Optional[Dict[str, torch.Tensor]] = None,
        volume_data: Optional[torch.Tensor] = None,
        vol_pred_history: Optional[torch.Tensor] = None,
        timestamps: Optional[List] = None
    ) -> torch.Tensor:
        """
        Extract comprehensive technical features.

        Args:
            ohlc_data: Dictionary with OHLC data
            volume_data: Volume data tensor
            vol_pred_history: Volatility prediction history
            timestamps: Timestamp information

        Returns:
            Combined technical features tensor
        """
        all_features = {}

        # Price momentum features
        if ohlc_data is not None and 'close' in ohlc_data:
            momentum_features = self.indicator_engine.calculate_momentum_features(
                ohlc_data['close']
            )
            all_features.update(momentum_features)

            # OHLC pattern features
            ohlc_features = self.indicator_engine.calculate_ohlc_features(ohlc_data)
            all_features.update(ohlc_features)

            # Cross-asset features
            if volume_data is not None:
                cross_features = self.indicator_engine.calculate_cross_asset_features(
                    ohlc_data['close'], volume_data
                )
                all_features.update(cross_features)

        # Volume features
        if volume_data is not None:
            volume_features = self.indicator_engine.calculate_volume_features(volume_data)
            all_features.update(volume_features)

        # Volatility regime features
        if vol_pred_history is not None:
            vol_regime_features = self.indicator_engine.calculate_volatility_regime(
                vol_pred_history
            )
            all_features.update(vol_regime_features)

        # Time features
        if timestamps is not None:
            time_features = self.indicator_engine.calculate_time_features(timestamps)
            all_features.update(time_features)

        # Combine all features
        if all_features:
            device = next(self.parameters()).device
            combined_features = combine_features(all_features, device)

            # Update feature dimension if not set
            if self.technical_feature_dim is None:
                self.technical_feature_dim = combined_features.shape[-1]
                # Reinitialize technical processor with correct dimension
                self.technical_processor = FeatureProcessor(
                    feature_dim=self.technical_feature_dim,
                    hidden_dim=128,
                    output_dim=64
                ).to(device)

            return combined_features
        else:
            # Return empty features if no data available
            device = next(self.parameters()).device
            batch_size = vol_pred_history.shape[0] if vol_pred_history is not None else 1
            return torch.zeros(batch_size, self.n_assets, 0, device=device)

    def forward(
        self,
        vol_pred: torch.Tensor,
        vol_pred_history: torch.Tensor,
        ohlc_data: Optional[Dict[str, torch.Tensor]] = None,
        volume_data: Optional[torch.Tensor] = None,
        timestamps: Optional[List] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with enhanced features.

        Args:
            vol_pred: Current volatility prediction [batch_size, n_assets, 1]
            vol_pred_history: Historical predictions [batch_size, n_assets, history_length]
            ohlc_data: Optional OHLC data dictionary
            volume_data: Optional volume data
            timestamps: Optional timestamp information

        Returns:
            Tuple of (ratio, direction) predictions
        """
        batch_size = vol_pred.shape[0]

        # Process volatility features
        vol_input = torch.cat([vol_pred, vol_pred_history], dim=2)  # [batch, assets, time]
        vol_features = self.volatility_processor(vol_input)  # [batch, assets, 32]

        # Extract technical features
        technical_features = self.extract_technical_features(
            ohlc_data=ohlc_data,
            volume_data=volume_data,
            vol_pred_history=vol_pred_history,
            timestamps=timestamps
        )

        # Process technical features if available
        if technical_features.shape[-1] > 0:
            tech_features = self.technical_processor(technical_features)  # [batch, assets, 64]
        else:
            # Use zero features if no technical data
            device = vol_pred.device
            tech_features = torch.zeros(batch_size, self.n_assets, 64, device=device)

        # Combine volatility and technical features
        combined_features = torch.cat([vol_features, tech_features], dim=-1)  # [batch, assets, 96]

        # Apply convolutional layers for temporal patterns
        # Reshape for conv1d: [batch * assets, 1, features]
        conv_input = combined_features.view(-1, 1, combined_features.shape[-1])

        conv_output = conv_input
        for conv_layer in self.conv_layers:
            conv_output = F.relu(conv_layer(conv_output))

        # Global pooling
        pooled = self.global_pool(conv_output).squeeze(-1)  # [batch * assets, channels]

        # Reshape back and combine with original features
        pooled = pooled.view(batch_size, self.n_assets, -1)  # [batch, assets, channels]

        # Combine pooled conv features with original combined features
        final_features = torch.cat([pooled, combined_features], dim=-1)

        # Dense processing
        dense_output = self.dense_layers(final_features)  # [batch, assets, 64]

        # Generate outputs
        ratio_raw = self.ratio_head(dense_output).squeeze(-1)  # [batch, assets]
        direction_raw = self.direction_head(dense_output).squeeze(-1)  # [batch, assets]

        # Scale ratio to reasonable range (0.85 to 1.0)
        ratio = 0.85 + 0.15 * torch.sigmoid(ratio_raw)

        # Convert direction to probability (0-1 range)
        direction_prob = torch.sigmoid(direction_raw)

        return ratio, direction_prob

    def forward_with_vol_pred_init(
        self,
        vol_pred: torch.Tensor,
        vol_pred_history: torch.Tensor,
        vol_multiplier: float = 0.4,
        alpha: float = 0.8,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with vol_pred initialization strategy.

        Args:
            vol_pred: Current volatility prediction
            vol_pred_history: Historical predictions
            vol_multiplier: Multiplier for volatility-based strategy
            alpha: Blending factor (higher = more vol_pred strategy)
            **kwargs: Additional arguments for forward pass

        Returns:
            Tuple of (ratio, direction) predictions
        """
        # Get network predictions
        ratio_network, direction_network = self.forward(
            vol_pred, vol_pred_history, **kwargs
        )

        # Calculate vol_pred baseline strategy
        vol_pred_squeezed = vol_pred.squeeze(-1)  # [batch_size, n_assets]
        ratio_base = 1.0 - (vol_pred_squeezed * vol_multiplier)
        ratio_base = torch.clamp(ratio_base, 0.92, 0.998)

        # Blend baseline strategy with network predictions
        final_ratio = alpha * ratio_base + (1 - alpha) * ratio_network

        return final_ratio, direction_network

    def get_trading_signals(
        self,
        vol_pred: torch.Tensor,
        vol_pred_history: torch.Tensor,
        current_price: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Get comprehensive trading signals with enhanced features.

        Args:
            vol_pred: Current volatility prediction
            vol_pred_history: Historical predictions
            current_price: Current market prices
            **kwargs: Additional data (OHLC, volume, etc.)

        Returns:
            Dictionary of trading signals
        """
        # Use vol_pred initialization if enabled
        if self.init_with_vol_pred:
            ratio, direction = self.forward_with_vol_pred_init(
                vol_pred, vol_pred_history, **kwargs
            )
        else:
            ratio, direction = self.forward(
                vol_pred, vol_pred_history, **kwargs
            )

        # Calculate limit prices (differentiable)
        # Use soft blending instead of hard thresholding
        is_long_prob = direction  # This is now a probability (0-1)
        is_short_prob = 1 - direction

        # Soft limit price calculation
        long_limit_price = current_price * ratio  # Long: buy below current price
        short_limit_price = current_price * (2 - ratio)  # Short: sell above current price

        limit_price = is_long_prob * long_limit_price + is_short_prob * short_limit_price

        return {
            'ratio': ratio,
            'direction': direction,
            'limit_price': limit_price,
            'is_long': is_long_prob,  # Now a probability, not binary
            'is_short': is_short_prob
        }
