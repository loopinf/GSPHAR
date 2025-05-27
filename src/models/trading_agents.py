"""
Trading Agent Model Architectures

This module implements various trading agent models that use volatility predictions
and other market features to generate trading signals and position sizing decisions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class LinearTradingAgent(nn.Module):
    """
    Linear Trading Agent.
    
    Simple linear model that takes volatility predictions and market features
    to generate trading signals. Good baseline model for comparison.
    """
    
    def __init__(self,
                 input_features: int,
                 output_dim: int = 1,
                 use_bias: bool = True,
                 activation: str = "sigmoid"):
        """
        Initialize linear trading agent.
        
        Args:
            input_features: Number of input features
            output_dim: Output dimension (1 for single ratio, n_assets for multi-asset)
            use_bias: Whether to use bias term
            activation: Output activation function
        """
        super(LinearTradingAgent, self).__init__()
        
        self.input_features = input_features
        self.output_dim = output_dim
        self.activation = activation
        
        # Linear layer
        self.linear = nn.Linear(input_features, output_dim, bias=use_bias)
        
        # Output activation
        if activation == "sigmoid":
            self.output_activation = nn.Sigmoid()
        elif activation == "tanh":
            self.output_activation = nn.Tanh()
        elif activation == "softmax":
            self.output_activation = nn.Softmax(dim=-1)
        elif activation == "none":
            self.output_activation = nn.Identity()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of linear trading agent.
        
        Args:
            features: Input features [batch_size, input_features]
                     Expected features: [vol_pred, vol_confidence, returns, volume, ...]
            
        Returns:
            Dictionary containing trading signals and ratios
        """
        # Linear transformation
        logits = self.linear(features)
        
        # Apply activation
        if self.activation == "sigmoid":
            # Output in [0, 1] range - can be interpreted as position size
            ratios = self.output_activation(logits)
        elif self.activation == "tanh":
            # Output in [-1, 1] range - allows short positions
            ratios = self.output_activation(logits)
        elif self.activation == "softmax":
            # Output sums to 1 - portfolio weights
            ratios = self.output_activation(logits)
        else:
            # Raw output
            ratios = logits
        
        return {
            'ratios': ratios,
            'logits': logits,
            'features': features
        }


class NeuralTradingAgent(nn.Module):
    """
    Neural Network Trading Agent.
    
    Multi-layer neural network for more complex trading strategies.
    Can capture non-linear relationships between features and trading decisions.
    """
    
    def __init__(self,
                 input_features: int,
                 hidden_dims: List[int] = [64, 32],
                 output_dim: int = 1,
                 dropout: float = 0.1,
                 activation: str = "relu",
                 output_activation: str = "sigmoid",
                 batch_norm: bool = True):
        """
        Initialize neural trading agent.
        
        Args:
            input_features: Number of input features
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            dropout: Dropout rate
            activation: Hidden layer activation function
            output_activation: Output activation function
            batch_norm: Whether to use batch normalization
        """
        super(NeuralTradingAgent, self).__init__()
        
        self.input_features = input_features
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout = dropout
        
        # Build network layers
        layers = []
        prev_dim = input_features
        
        for hidden_dim in hidden_dims:
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU())
            elif activation == "elu":
                layers.append(nn.ELU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            
            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # Output activation
        if output_activation == "sigmoid":
            layers.append(nn.Sigmoid())
        elif output_activation == "tanh":
            layers.append(nn.Tanh())
        elif output_activation == "softmax":
            layers.append(nn.Softmax(dim=-1))
        
        self.network = nn.Sequential(*layers)
        
        # Additional layers for uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dims[-1] if hidden_dims else input_features, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()
        )
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of neural trading agent.
        
        Args:
            features: Input features [batch_size, input_features]
            
        Returns:
            Dictionary containing trading signals, ratios, and uncertainty
        """
        # Main network forward pass
        ratios = self.network(features)
        
        # Extract features from intermediate layer for uncertainty estimation
        x = features
        for layer in self.network[:-2]:  # Exclude last two layers
            x = layer(x)
        
        # Estimate uncertainty
        uncertainty = self.uncertainty_head(x)
        
        return {
            'ratios': ratios,
            'uncertainty': uncertainty,
            'features': features
        }


class TransformerTradingAgent(nn.Module):
    """
    Transformer-based Trading Agent.
    
    Uses transformer architecture to capture complex temporal dependencies
    and relationships between different market features.
    """
    
    def __init__(self,
                 input_features: int,
                 d_model: int = 128,
                 nhead: int = 8,
                 num_layers: int = 4,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 output_dim: int = 1,
                 max_seq_length: int = 100):
        """
        Initialize transformer trading agent.
        
        Args:
            input_features: Number of input features per time step
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward network dimension
            dropout: Dropout rate
            output_dim: Output dimension
            max_seq_length: Maximum sequence length
        """
        super(TransformerTradingAgent, self).__init__()
        
        self.input_features = input_features
        self.d_model = d_model
        self.output_dim = output_dim
        self.max_seq_length = max_seq_length
        
        # Input projection
        self.input_projection = nn.Linear(input_features, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_seq_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim),
            nn.Sigmoid()
        )
        
        # Attention weights for interpretability
        self.attention_weights = None
    
    def forward(self, features: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of transformer trading agent.
        
        Args:
            features: Input features [batch_size, seq_len, input_features]
            mask: Optional attention mask [batch_size, seq_len]
            
        Returns:
            Dictionary containing trading signals and attention weights
        """
        batch_size, seq_len, _ = features.shape
        
        # Input projection
        x = self.input_projection(features)  # [batch_size, seq_len, d_model]
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Transformer encoding
        if mask is not None:
            # Convert mask to attention mask format
            attention_mask = mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
            attention_mask = attention_mask.expand(batch_size, 1, seq_len, seq_len)
            attention_mask = attention_mask.masked_fill(attention_mask == 0, float('-inf'))
        else:
            attention_mask = None
        
        encoded = self.transformer(x, mask=attention_mask)  # [batch_size, seq_len, d_model]
        
        # Use last time step for prediction
        last_encoded = encoded[:, -1, :]  # [batch_size, d_model]
        
        # Generate trading ratios
        ratios = self.output_projection(last_encoded)  # [batch_size, output_dim]
        
        return {
            'ratios': ratios,
            'encoded_features': encoded,
            'attention_weights': self.attention_weights,
            'features': features
        }


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class EnsembleTradingAgent(nn.Module):
    """
    Ensemble Trading Agent.
    
    Combines multiple trading agents to make more robust trading decisions.
    """
    
    def __init__(self,
                 agents: List[nn.Module],
                 combination_method: str = "weighted_average",
                 weights: Optional[List[float]] = None):
        """
        Initialize ensemble trading agent.
        
        Args:
            agents: List of trading agent models
            combination_method: Method to combine predictions ("average", "weighted_average", "voting")
            weights: Weights for weighted combination
        """
        super(EnsembleTradingAgent, self).__init__()
        
        self.agents = nn.ModuleList(agents)
        self.combination_method = combination_method
        self.n_agents = len(agents)
        
        if weights is None:
            self.weights = torch.ones(self.n_agents) / self.n_agents
        else:
            self.weights = torch.tensor(weights)
            self.weights = self.weights / self.weights.sum()  # Normalize
        
        self.register_buffer('agent_weights', self.weights)
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of ensemble trading agent.
        
        Args:
            features: Input features
            
        Returns:
            Dictionary containing combined trading signals
        """
        # Get predictions from all agents
        agent_predictions = []
        agent_outputs = []
        
        for agent in self.agents:
            output = agent(features)
            agent_predictions.append(output['ratios'])
            agent_outputs.append(output)
        
        # Stack predictions
        all_predictions = torch.stack(agent_predictions, dim=0)  # [n_agents, batch_size, output_dim]
        
        # Combine predictions
        if self.combination_method == "average":
            combined_ratios = torch.mean(all_predictions, dim=0)
        elif self.combination_method == "weighted_average":
            weights = self.agent_weights.view(-1, 1, 1).to(features.device)
            combined_ratios = torch.sum(all_predictions * weights, dim=0)
        elif self.combination_method == "voting":
            # Simple majority voting (for binary decisions)
            binary_predictions = (all_predictions > 0.5).float()
            combined_ratios = torch.mean(binary_predictions, dim=0)
        else:
            raise ValueError(f"Unsupported combination method: {self.combination_method}")
        
        # Calculate prediction uncertainty (standard deviation across agents)
        prediction_std = torch.std(all_predictions, dim=0)
        
        return {
            'ratios': combined_ratios,
            'uncertainty': prediction_std,
            'individual_predictions': all_predictions,
            'agent_outputs': agent_outputs,
            'features': features
        }


# Utility functions for trading agents
def create_trading_features(volatility_predictions: Dict[str, torch.Tensor],
                          market_data: Dict[str, torch.Tensor],
                          additional_features: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
    """
    Create feature vector for trading agents from volatility predictions and market data.
    
    Args:
        volatility_predictions: Output from volatility model
        market_data: Market data (prices, volumes, etc.)
        additional_features: Additional features to include
        
    Returns:
        Feature tensor for trading agent input
    """
    features = []
    
    # Volatility features
    if 'volatility' in volatility_predictions:
        vol_pred = volatility_predictions['volatility']
        if vol_pred.dim() > 2:
            vol_pred = vol_pred.squeeze(-1)  # Remove horizon dimension if present
        features.append(vol_pred)
    
    if 'confidence_lower' in volatility_predictions and 'confidence_upper' in volatility_predictions:
        vol_uncertainty = volatility_predictions['confidence_upper'] - volatility_predictions['confidence_lower']
        if vol_uncertainty.dim() > 2:
            vol_uncertainty = vol_uncertainty.squeeze(-1)
        features.append(vol_uncertainty)
    
    # Market data features
    if 'returns' in market_data:
        features.append(market_data['returns'])
    
    if 'volume' in market_data:
        features.append(market_data['volume'])
    
    if 'price' in market_data:
        features.append(market_data['price'])
    
    # Additional features
    if additional_features:
        for key, value in additional_features.items():
            features.append(value)
    
    # Concatenate all features
    if features:
        return torch.cat(features, dim=-1)
    else:
        raise ValueError("No features provided for trading agent")
