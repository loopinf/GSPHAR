"""
Volatility-Specific Model Architectures

This module implements various volatility forecasting models including
EGARCH, GARCH, and neural network-based volatility models.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from arch import arch_model
import logging

logger = logging.getLogger(__name__)


class EGARCHVolatilityModel(nn.Module):
    """
    EGARCH (Exponential GARCH) Volatility Model.
    
    Implements EGARCH model for volatility forecasting with asymmetric effects.
    Can be used as a standalone model or as part of a multi-model pipeline.
    """
    
    def __init__(self, 
                 n_assets: int,
                 p: int = 1,
                 q: int = 1,
                 mean_model: str = "Zero",
                 distribution: str = "normal"):
        """
        Initialize EGARCH volatility model.
        
        Args:
            n_assets: Number of assets
            p: EGARCH lag order for conditional variance
            q: EGARCH lag order for innovation
            mean_model: Mean model specification
            distribution: Error distribution
        """
        super(EGARCHVolatilityModel, self).__init__()
        self.n_assets = n_assets
        self.p = p
        self.q = q
        self.mean_model = mean_model
        self.distribution = distribution
        
        # Store fitted models for each asset
        self.fitted_models = {}
        self.is_fitted = False
    
    def fit(self, returns_data: torch.Tensor, asset_names: Optional[List[str]] = None):
        """
        Fit EGARCH models for all assets.
        
        Args:
            returns_data: Return data [n_assets, time_steps] or [time_steps, n_assets]
            asset_names: Optional asset names
        """
        if returns_data.dim() == 3:
            returns_data = returns_data.squeeze(0)  # Remove batch dimension if present
        
        if returns_data.shape[0] > returns_data.shape[1]:
            returns_data = returns_data.T  # Ensure [n_assets, time_steps]
        
        if asset_names is None:
            asset_names = [f"Asset_{i}" for i in range(self.n_assets)]
        
        logger.info(f"Fitting EGARCH models for {self.n_assets} assets...")
        
        for i, asset_name in enumerate(asset_names[:self.n_assets]):
            try:
                # Get returns for this asset
                asset_returns = returns_data[i].detach().cpu().numpy()
                
                # Remove NaN values
                asset_returns = asset_returns[~np.isnan(asset_returns)]
                
                if len(asset_returns) < 100:
                    logger.warning(f"Insufficient data for {asset_name}: {len(asset_returns)} observations")
                    continue
                
                # Scale returns for numerical stability
                asset_returns_scaled = asset_returns * 100
                
                # Fit EGARCH model
                model = arch_model(
                    asset_returns_scaled,
                    vol='EGARCH',
                    p=self.p,
                    q=self.q,
                    mean=self.mean_model,
                    dist=self.distribution
                )
                
                fitted_model = model.fit(disp='off', show_warning=False)
                self.fitted_models[asset_name] = fitted_model
                
            except Exception as e:
                logger.warning(f"Failed to fit EGARCH for {asset_name}: {str(e)}")
                continue
        
        self.is_fitted = len(self.fitted_models) > 0
        logger.info(f"Successfully fitted {len(self.fitted_models)} EGARCH models")
    
    def forward(self, returns: torch.Tensor, horizon: int = 1) -> Dict[str, torch.Tensor]:
        """
        Generate volatility forecasts.
        
        Args:
            returns: Recent returns for conditioning [batch_size, n_assets, seq_len]
            horizon: Forecast horizon
            
        Returns:
            Dictionary containing volatility predictions and confidence intervals
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
        
        batch_size, n_assets, seq_len = returns.shape
        device = returns.device
        
        # Initialize output tensors
        vol_predictions = torch.zeros(batch_size, n_assets, horizon, device=device)
        vol_confidence_lower = torch.zeros(batch_size, n_assets, horizon, device=device)
        vol_confidence_upper = torch.zeros(batch_size, n_assets, horizon, device=device)
        
        for batch_idx in range(batch_size):
            for asset_idx, (asset_name, fitted_model) in enumerate(self.fitted_models.items()):
                if asset_idx >= n_assets:
                    break
                
                try:
                    # Get recent returns for this asset
                    recent_returns = returns[batch_idx, asset_idx].detach().cpu().numpy() * 100
                    
                    # Generate forecast
                    forecast = fitted_model.forecast(horizon=horizon, reindex=False)
                    
                    # Extract volatility forecast (convert back from scaled)
                    vol_forecast = np.sqrt(forecast.variance.values[-1]) / 100
                    
                    # Store predictions
                    for h in range(horizon):
                        vol_predictions[batch_idx, asset_idx, h] = torch.tensor(
                            vol_forecast[h] if h < len(vol_forecast) else vol_forecast[-1],
                            device=device
                        )
                        
                        # Simple confidence intervals (±2 std)
                        vol_confidence_lower[batch_idx, asset_idx, h] = vol_predictions[batch_idx, asset_idx, h] * 0.5
                        vol_confidence_upper[batch_idx, asset_idx, h] = vol_predictions[batch_idx, asset_idx, h] * 1.5
                
                except Exception as e:
                    logger.debug(f"Prediction failed for {asset_name}: {str(e)}")
                    # Use simple fallback prediction
                    recent_vol = torch.std(returns[batch_idx, asset_idx])
                    vol_predictions[batch_idx, asset_idx, :] = recent_vol
                    vol_confidence_lower[batch_idx, asset_idx, :] = recent_vol * 0.5
                    vol_confidence_upper[batch_idx, asset_idx, :] = recent_vol * 1.5
        
        return {
            'volatility': vol_predictions,
            'confidence_lower': vol_confidence_lower,
            'confidence_upper': vol_confidence_upper,
            'variance': vol_predictions ** 2
        }


class LSTMVolatilityModel(nn.Module):
    """
    LSTM-based Volatility Forecasting Model.
    
    Uses LSTM networks to capture temporal dependencies in volatility.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 output_dim: int = 1,
                 dropout: float = 0.1,
                 bidirectional: bool = False):
        """
        Initialize LSTM volatility model.
        
        Args:
            input_dim: Number of input features
            hidden_dim: Hidden dimension size
            num_layers: Number of LSTM layers
            output_dim: Output dimension (typically 1 for volatility)
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
        """
        super(LSTMVolatilityModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.bidirectional = bidirectional
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Output layers
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.output_layer = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Softplus()  # Ensure positive volatility
        )
        
        # Additional outputs for uncertainty quantification
        self.confidence_layer = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 2),  # Lower and upper bounds
            nn.Softplus()
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of LSTM volatility model.
        
        Args:
            x: Input features [batch_size, seq_len, input_dim]
            
        Returns:
            Dictionary containing volatility predictions and confidence intervals
        """
        batch_size, seq_len, _ = x.shape
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last time step output
        last_output = lstm_out[:, -1, :]  # [batch_size, hidden_dim]
        
        # Generate volatility prediction
        volatility = self.output_layer(last_output)  # [batch_size, output_dim]
        
        # Generate confidence intervals
        confidence_bounds = self.confidence_layer(last_output)  # [batch_size, 2]
        confidence_lower = volatility - confidence_bounds[:, 0:1]
        confidence_upper = volatility + confidence_bounds[:, 1:2]
        
        # Ensure positive bounds
        confidence_lower = torch.clamp(confidence_lower, min=1e-6)
        
        return {
            'volatility': volatility,
            'confidence_lower': confidence_lower,
            'confidence_upper': confidence_upper,
            'variance': volatility ** 2,
            'hidden_state': hidden,
            'cell_state': cell
        }


class GARCHVolatilityModel(nn.Module):
    """
    Standard GARCH Volatility Model.
    
    Implements GARCH(p,q) model for volatility forecasting.
    """
    
    def __init__(self,
                 n_assets: int,
                 p: int = 1,
                 q: int = 1,
                 mean_model: str = "Zero",
                 distribution: str = "normal"):
        """
        Initialize GARCH volatility model.
        
        Args:
            n_assets: Number of assets
            p: GARCH lag order for conditional variance
            q: GARCH lag order for innovation
            mean_model: Mean model specification
            distribution: Error distribution
        """
        super(GARCHVolatilityModel, self).__init__()
        self.n_assets = n_assets
        self.p = p
        self.q = q
        self.mean_model = mean_model
        self.distribution = distribution
        
        # Store fitted models for each asset
        self.fitted_models = {}
        self.is_fitted = False
    
    def fit(self, returns_data: torch.Tensor, asset_names: Optional[List[str]] = None):
        """
        Fit GARCH models for all assets.
        
        Args:
            returns_data: Return data [n_assets, time_steps] or [time_steps, n_assets]
            asset_names: Optional asset names
        """
        if returns_data.dim() == 3:
            returns_data = returns_data.squeeze(0)
        
        if returns_data.shape[0] > returns_data.shape[1]:
            returns_data = returns_data.T
        
        if asset_names is None:
            asset_names = [f"Asset_{i}" for i in range(self.n_assets)]
        
        logger.info(f"Fitting GARCH models for {self.n_assets} assets...")
        
        for i, asset_name in enumerate(asset_names[:self.n_assets]):
            try:
                # Get returns for this asset
                asset_returns = returns_data[i].detach().cpu().numpy()
                asset_returns = asset_returns[~np.isnan(asset_returns)]
                
                if len(asset_returns) < 100:
                    logger.warning(f"Insufficient data for {asset_name}")
                    continue
                
                # Scale returns
                asset_returns_scaled = asset_returns * 100
                
                # Fit GARCH model
                model = arch_model(
                    asset_returns_scaled,
                    vol='GARCH',
                    p=self.p,
                    q=self.q,
                    mean=self.mean_model,
                    dist=self.distribution
                )
                
                fitted_model = model.fit(disp='off', show_warning=False)
                self.fitted_models[asset_name] = fitted_model
                
            except Exception as e:
                logger.warning(f"Failed to fit GARCH for {asset_name}: {str(e)}")
                continue
        
        self.is_fitted = len(self.fitted_models) > 0
        logger.info(f"Successfully fitted {len(self.fitted_models)} GARCH models")
    
    def forward(self, returns: torch.Tensor, horizon: int = 1) -> Dict[str, torch.Tensor]:
        """
        Generate volatility forecasts using fitted GARCH models.
        
        Args:
            returns: Recent returns [batch_size, n_assets, seq_len]
            horizon: Forecast horizon
            
        Returns:
            Dictionary containing volatility predictions
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
        
        batch_size, n_assets, seq_len = returns.shape
        device = returns.device
        
        vol_predictions = torch.zeros(batch_size, n_assets, horizon, device=device)
        
        for batch_idx in range(batch_size):
            for asset_idx, (asset_name, fitted_model) in enumerate(self.fitted_models.items()):
                if asset_idx >= n_assets:
                    break
                
                try:
                    # Generate forecast
                    forecast = fitted_model.forecast(horizon=horizon, reindex=False)
                    vol_forecast = np.sqrt(forecast.variance.values[-1]) / 100
                    
                    for h in range(horizon):
                        vol_predictions[batch_idx, asset_idx, h] = torch.tensor(
                            vol_forecast[h] if h < len(vol_forecast) else vol_forecast[-1],
                            device=device
                        )
                
                except Exception as e:
                    logger.debug(f"Prediction failed for {asset_name}: {str(e)}")
                    recent_vol = torch.std(returns[batch_idx, asset_idx])
                    vol_predictions[batch_idx, asset_idx, :] = recent_vol
        
        return {
            'volatility': vol_predictions,
            'variance': vol_predictions ** 2
        }
