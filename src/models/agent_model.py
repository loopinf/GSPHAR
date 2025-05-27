"""
Agent model for deciding trading ratios and directions based on volatility predictions.

The agent takes volatility predictions and their history to decide:
1. Ratio: How much discount/premium to apply to current price
2. Direction: Whether to go long (1) or short (0)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TradingAgentModel(nn.Module):
    """
    Trading agent that decides order placement ratios and directions.

    Inputs:
        - vol_pred: Current volatility prediction [batch_size, n_assets, 1]
        - vol_pred_history: Historical volatility predictions [batch_size, n_assets, history_length]

    Outputs:
        - ratio: Order placement ratio [batch_size, n_assets, 1] (0 to 1)
        - direction: Trading direction [batch_size, n_assets, 1] (0=short, 1=long)
    """

    def __init__(self, n_assets, history_length=24, hidden_dim=64, dropout=0.1,
                 init_with_vol_pred=True):
        """
        Initialize the trading agent model.

        Args:
            n_assets (int): Number of assets/symbols
            history_length (int): Length of volatility prediction history (default: 24 hours)
            hidden_dim (int): Hidden dimension for dense layers
            dropout (float): Dropout rate for regularization
            init_with_vol_pred (bool): Initialize to mimic previous vol_pred strategy
        """
        super(TradingAgentModel, self).__init__()

        self.n_assets = n_assets
        self.history_length = history_length
        self.hidden_dim = hidden_dim
        self.init_with_vol_pred = init_with_vol_pred

        # Input dimension: current vol_pred (1) + history (history_length)
        self.input_dim = 1 + history_length

        # 1D Convolutional layers for temporal pattern extraction
        self.conv1 = nn.Conv1d(
            in_channels=n_assets,
            out_channels=n_assets * 2,
            kernel_size=3,
            padding=1,
            groups=n_assets  # Depthwise convolution
        )

        self.conv2 = nn.Conv1d(
            in_channels=n_assets * 2,
            out_channels=n_assets * 4,
            kernel_size=3,
            padding=1,
            groups=n_assets  # Depthwise convolution
        )

        self.conv3 = nn.Conv1d(
            in_channels=n_assets * 4,
            out_channels=n_assets * 2,
            kernel_size=3,
            padding=1,
            groups=n_assets  # Depthwise convolution
        )

        # Global average pooling to reduce temporal dimension
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Dense layers for final decision making
        self.dense1 = nn.Linear(n_assets * 2, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, hidden_dim // 2)

        # Output heads
        self.ratio_head = nn.Linear(hidden_dim // 2, n_assets)
        self.direction_head = nn.Linear(hidden_dim // 2, n_assets)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(n_assets * 2)
        self.bn2 = nn.BatchNorm1d(n_assets * 4)
        self.bn3 = nn.BatchNorm1d(n_assets * 2)

        # Initialize weights to mimic vol_pred strategy if requested
        if self.init_with_vol_pred:
            self._initialize_with_vol_pred_strategy()

    def forward(self, vol_pred, vol_pred_history):
        """
        Forward pass of the trading agent.

        Args:
            vol_pred: Current volatility prediction [batch_size, n_assets, 1]
            vol_pred_history: Historical predictions [batch_size, n_assets, history_length]

        Returns:
            tuple: (ratio, direction)
                - ratio: [batch_size, n_assets, 1] values in (0, 1)
                - direction: [batch_size, n_assets, 1] values in (0, 1)
        """
        batch_size = vol_pred.shape[0]

        # Clamp inputs to prevent extreme values
        vol_pred = torch.clamp(vol_pred, 1e-6, 1.0)
        vol_pred_history = torch.clamp(vol_pred_history, 1e-6, 1.0)

        # Concatenate current prediction with history
        # vol_pred: [batch_size, n_assets, 1]
        # vol_pred_history: [batch_size, n_assets, history_length]
        combined_input = torch.cat([vol_pred, vol_pred_history], dim=2)  # [batch_size, n_assets, 1+history_length]

        # Apply 1D convolutions for temporal pattern extraction
        x = self.conv1(combined_input)  # [batch_size, n_assets*2, input_dim]
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x)  # [batch_size, n_assets*4, input_dim]
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv3(x)  # [batch_size, n_assets*2, input_dim]
        x = self.bn3(x)
        x = F.relu(x)

        # Global average pooling to get fixed-size representation
        x = self.global_avg_pool(x)  # [batch_size, n_assets*2, 1]
        x = x.squeeze(-1)  # [batch_size, n_assets*2]

        # Dense layers for decision making
        x = self.dense1(x)  # [batch_size, hidden_dim]
        x = F.relu(x)
        x = self.dropout(x)

        x = self.dense2(x)  # [batch_size, hidden_dim//2]
        x = F.relu(x)
        x = self.dropout(x)

        # Output heads
        ratio_logits = self.ratio_head(x)  # [batch_size, n_assets]
        direction_logits = self.direction_head(x)  # [batch_size, n_assets]

        # Apply activations with clamping to prevent extreme values
        # Ratio: sigmoid to get values in (0.85, 0.99) for more realistic order prices
        ratio = torch.sigmoid(ratio_logits) * 0.14 + 0.85  # Scale to [0.85, 0.99]
        ratio = ratio.unsqueeze(-1)  # [batch_size, n_assets, 1]

        # Direction: sigmoid to get values in (0, 1) where 0=short, 1=long
        direction = torch.sigmoid(direction_logits).unsqueeze(-1)  # [batch_size, n_assets, 1]

        return ratio, direction

    def _initialize_with_vol_pred_strategy(self):
        """
        Initialize the agent to mimic the previous vol_pred strategy.

        Previous strategy: limit_price = current_price * (1 - vol_pred)
        This means: ratio = 1 - vol_pred, direction = long (1.0)
        """
        print("🎯 Initializing agent with vol_pred strategy...")

        # Initialize direction head to favor long positions
        with torch.no_grad():
            # Direction head: bias towards long positions
            self.direction_head.bias.fill_(2.0)  # sigmoid(2.0) ≈ 0.88 (mostly long)
            self.direction_head.weight.fill_(0.1)  # Small weights for gradual learning

            # Ratio head: initialize with small weights
            # We'll handle vol_pred integration in the forward pass
            self.ratio_head.weight.fill_(0.1)
            self.ratio_head.bias.fill_(0.0)

        print("✅ Agent initialized to favor long positions with vol_pred-based ratios")

    def forward_with_vol_pred_init(self, vol_pred, vol_pred_history, vol_multiplier=0.4):
        """
        Forward pass that incorporates vol_pred initialization strategy.

        This version uses vol_pred directly for ratio calculation while still
        allowing the network to learn modifications.

        Args:
            vol_multiplier (float): Multiplier for volatility impact (default: 0.4)
                                  Lower values = less conservative pricing
        """
        # Clamp inputs to prevent extreme values
        vol_pred = torch.clamp(vol_pred, 1e-6, 0.2)  # Limit vol_pred to reasonable range
        vol_pred_history = torch.clamp(vol_pred_history, 1e-6, 0.2)

        # Get network outputs (learned modifications)
        ratio_network, direction_network = self.forward(vol_pred, vol_pred_history)

        # Combine vol_pred strategy with learned modifications
        # IMPROVED: Use smaller volatility multiplier for less conservative pricing
        vol_pred_squeezed = vol_pred.squeeze(-1)  # [batch_size, n_assets]
        ratio_base = 1.0 - (vol_pred_squeezed * vol_multiplier)  # Reduced impact

        # IMPROVED: Less conservative range for better fill rates
        ratio_base = torch.clamp(ratio_base, 0.92, 0.998)

        # Combine base strategy with learned modifications
        ratio_network_squeezed = ratio_network.squeeze(-1)

        # Weighted combination: start with vol_pred, gradually learn modifications
        alpha = 0.8  # Weight for vol_pred strategy (start conservative)
        final_ratio = alpha * ratio_base + (1 - alpha) * ratio_network_squeezed
        final_ratio = torch.clamp(final_ratio, 0.92, 0.998)

        # Direction: use network output (but initialized to favor longs)
        final_direction = direction_network

        return final_ratio.unsqueeze(-1), final_direction

    def get_trading_signals(self, vol_pred, vol_pred_history, current_price):
        """
        Get trading signals including limit prices for both long and short positions.

        Args:
            vol_pred: Current volatility prediction [batch_size, n_assets, 1]
            vol_pred_history: Historical predictions [batch_size, n_assets, history_length]
            current_price: Current market prices [batch_size, n_assets]

        Returns:
            dict: Trading signals with keys:
                - 'ratio': Order placement ratio
                - 'direction': Trading direction (0=short, 1=long)
                - 'limit_price': Calculated limit order prices
                - 'is_long': Boolean mask for long positions
                - 'is_short': Boolean mask for short positions
        """
        # Use vol_pred initialization if enabled, otherwise use regular forward
        if self.init_with_vol_pred:
            ratio, direction = self.forward_with_vol_pred_init(vol_pred, vol_pred_history)
        else:
            ratio, direction = self.forward(vol_pred, vol_pred_history)

        # Calculate limit prices based on direction and ratio
        # Long: buy at current_price * ratio (discount when ratio < 1)
        # Short: sell at current_price * (2 - ratio) (premium when ratio < 1)

        ratio_squeezed = ratio.squeeze(-1)  # [batch_size, n_assets]
        direction_squeezed = direction.squeeze(-1)  # [batch_size, n_assets]

        # Long positions: limit_price = current_price * ratio
        long_limit_price = current_price * ratio_squeezed

        # Short positions: limit_price = current_price * (2 - ratio)
        short_limit_price = current_price * (2 - ratio_squeezed)

        # Choose limit price based on direction
        # Use direction as a soft selector (can be between 0 and 1)
        limit_price = direction_squeezed * long_limit_price + (1 - direction_squeezed) * short_limit_price

        # Create boolean masks for easier interpretation
        is_long = direction_squeezed > 0.5
        is_short = direction_squeezed <= 0.5

        return {
            'ratio': ratio,
            'direction': direction,
            'limit_price': limit_price,
            'is_long': is_long,
            'is_short': is_short
        }


class SimpleTradingAgent(nn.Module):
    """
    Simplified trading agent for initial testing and comparison.

    Uses a simpler architecture with just dense layers.
    """

    def __init__(self, n_assets, history_length=24, hidden_dim=32):
        super(SimpleTradingAgent, self).__init__()

        self.n_assets = n_assets
        self.history_length = history_length

        # Input dimension: current vol_pred (1) + history (history_length)
        input_dim = (1 + history_length) * n_assets

        # Simple dense network
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, n_assets * 2)  # ratio + direction for each asset
        )

    def forward(self, vol_pred, vol_pred_history):
        """
        Forward pass of the simple trading agent.

        Args:
            vol_pred: Current volatility prediction [batch_size, n_assets, 1]
            vol_pred_history: Historical predictions [batch_size, n_assets, history_length]

        Returns:
            tuple: (ratio, direction)
        """
        batch_size = vol_pred.shape[0]

        # Concatenate and flatten
        combined_input = torch.cat([vol_pred, vol_pred_history], dim=2)  # [batch_size, n_assets, 1+history_length]
        flattened_input = combined_input.view(batch_size, -1)  # [batch_size, n_assets*(1+history_length)]

        # Forward pass
        output = self.network(flattened_input)  # [batch_size, n_assets*2]

        # Split into ratio and direction
        ratio_logits = output[:, :self.n_assets]  # [batch_size, n_assets]
        direction_logits = output[:, self.n_assets:]  # [batch_size, n_assets]

        # Apply activations
        ratio = torch.sigmoid(ratio_logits).unsqueeze(-1)  # [batch_size, n_assets, 1]
        direction = torch.sigmoid(direction_logits).unsqueeze(-1)  # [batch_size, n_assets, 1]

        return ratio, direction


class BasicLinearAgent(nn.Module):
    """
    Very basic linear trading agent for baseline comparison.

    Uses the formula: limit_price = price_open * (1 - param1 * vol_pred)
    - Only trades long positions (direction = 1.0)
    - Single learnable parameter param1
    - Uses vol_pred from EGARCH model directly
    """

    def __init__(self, n_assets, init_param1=0.5):
        """
        Initialize the basic linear agent.

        Args:
            n_assets (int): Number of assets/symbols
            init_param1 (float): Initial value for param1 parameter (default: 0.5)
        """
        super(BasicLinearAgent, self).__init__()

        self.n_assets = n_assets

        # Single learnable parameter for volatility multiplier
        # Initialize with reasonable value (0.5 means 5% vol_pred -> 2.5% discount)
        self.param1 = nn.Parameter(torch.full((n_assets,), init_param1))

        print(f"🎯 BasicLinearAgent initialized with param1={init_param1} for {n_assets} assets")

    def forward(self, vol_pred, vol_pred_history=None):
        """
        Forward pass of the basic linear agent.

        Args:
            vol_pred: Current volatility prediction [batch_size, n_assets, 1]
            vol_pred_history: Not used in basic model, kept for compatibility

        Returns:
            tuple: (ratio, direction)
                - ratio: [batch_size, n_assets, 1] calculated as 1 - param1 * vol_pred
                - direction: [batch_size, n_assets, 1] always 1.0 (long only)
        """
        batch_size = vol_pred.shape[0]

        # Clamp vol_pred to reasonable range
        vol_pred_clamped = torch.clamp(vol_pred, 1e-6, 0.5)  # Max 50% volatility

        # Clamp param1 to reasonable range (0.1 to 2.0)
        param1_clamped = torch.clamp(self.param1, 0.1, 2.0)

        # Calculate ratio: 1 - param1 * vol_pred
        # vol_pred: [batch_size, n_assets, 1]
        # param1: [n_assets]
        vol_pred_squeezed = vol_pred_clamped.squeeze(-1)  # [batch_size, n_assets]
        ratio_calc = 1.0 - param1_clamped * vol_pred_squeezed  # [batch_size, n_assets]

        # Clamp ratio to reasonable range (0.85 to 0.99 for realistic limit orders)
        ratio = torch.clamp(ratio_calc, 0.85, 0.99)
        ratio = ratio.unsqueeze(-1)  # [batch_size, n_assets, 1]

        # Direction: always long (1.0)
        direction = torch.ones_like(ratio)  # [batch_size, n_assets, 1]

        return ratio, direction

    def get_trading_signals(self, vol_pred, vol_pred_history, current_price):
        """
        Get trading signals for the basic linear agent.

        Args:
            vol_pred: Current volatility prediction [batch_size, n_assets, 1]
            vol_pred_history: Not used, kept for compatibility
            current_price: Current market prices [batch_size, n_assets]

        Returns:
            dict: Trading signals with keys:
                - 'ratio': Order placement ratio
                - 'direction': Trading direction (always 1.0 for long)
                - 'limit_price': Calculated limit order prices
                - 'is_long': Boolean mask (always True)
                - 'is_short': Boolean mask (always False)
                - 'param1': Current param1 values for monitoring
        """
        # Get ratio and direction
        ratio, direction = self.forward(vol_pred, vol_pred_history)

        # Calculate limit prices: current_price * ratio
        ratio_squeezed = ratio.squeeze(-1)  # [batch_size, n_assets]
        limit_price = current_price * ratio_squeezed

        # Create boolean masks
        is_long = torch.ones_like(ratio_squeezed, dtype=torch.bool)  # Always long
        is_short = torch.zeros_like(ratio_squeezed, dtype=torch.bool)  # Never short

        return {
            'ratio': ratio,
            'direction': direction,
            'limit_price': limit_price,
            'is_long': is_long,
            'is_short': is_short,
            'param1': self.param1.detach().clone()  # For monitoring parameter evolution
        }
