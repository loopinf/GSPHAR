"""
GARCH model implementation for volatility prediction.

This module implements a GARCH model that inherits from the base volatility model
interface, making it compatible with the existing GSPHAR pipeline.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple
import logging
import warnings
warnings.filterwarnings('ignore')

# Check arch availability
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    arch_model = None
    logger.warning("arch package not available. GARCH functionality will be limited.")

from .base_volatility_model import SingleAssetVolatilityModel

logger = logging.getLogger(__name__)


class GARCHModel(SingleAssetVolatilityModel):
    """
    GARCH model for single-asset volatility prediction.
    
    This implementation uses the arch library for GARCH fitting and integrates
    with the existing model pipeline through the base interface.
    """
    
    def __init__(self, 
                 model_type: str = 'GARCH',
                 p: int = 1, 
                 q: int = 1,
                 o: int = 0,
                 dist: str = 'normal',
                 scale_factor: float = 100.0,
                 min_periods: int = 50,
                 **kwargs):
        """
        Initialize GARCH model.
        
        Args:
            model_type: Type of model ('GARCH', 'EGARCH', 'GJR-GARCH')
            p: Number of lags for the squared residuals
            q: Number of lags for the conditional variance  
            o: Number of lags for asymmetric terms (for GJR-GARCH)
            dist: Distribution for residuals ('normal', 't', 'skewt')
            scale_factor: Factor to scale returns for numerical stability
            min_periods: Minimum number of observations required for fitting
            **kwargs: Additional parameters
        """
        if not ARCH_AVAILABLE:
            raise ImportError("arch package is required for GARCH functionality. Install with: pip install arch")
            
        super().__init__(model_name=f"{model_type}({p},{q})", **kwargs)
        
        self.model_type = model_type
        self.p = p
        self.q = q
        self.o = o
        self.dist = dist
        self.scale_factor = scale_factor
        self.min_periods = min_periods
        
        # GARCH model components
        self.arch_model = None
        self.fitted_model = None
        self.last_returns = None
        self.returns_history = []
        
        # Store model parameters
        self.model_params.update({
            'model_type': model_type,
            'p': p,
            'q': q, 
            'o': o,
            'dist': dist,
            'scale_factor': scale_factor,
            'min_periods': min_periods
        })
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for GARCH model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length) or (batch_size, n_features, seq_length)
            
        Returns:
            torch.Tensor: Predicted volatility
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
            
        # Handle different input shapes
        if x.dim() == 3:
            # Multi-asset format: (batch_size, n_features, seq_length)
            # Take the first feature for single-asset prediction
            x = x[:, 0, :]
        elif x.dim() == 2:
            # Single-asset format: (batch_size, seq_length)
            pass
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")
            
        batch_size = x.shape[0]
        predictions = []
        
        for i in range(batch_size):
            # Convert to returns if needed
            returns = self._convert_to_returns(x[i])
            
            # Update model with new data
            self._update_with_new_data(returns)
            
            # Make prediction
            vol_pred = self._predict_single()
            predictions.append(vol_pred)
            
        return torch.tensor(predictions, dtype=torch.float32, device=self._device).unsqueeze(-1)
    
    def fit(self, data: Union[torch.Tensor, np.ndarray], **kwargs) -> bool:
        """
        Fit GARCH model to training data.
        
        Args:
            data: Training data (prices or returns)
                 Shape: (n_samples,) or (n_samples, 1) for single asset
            **kwargs: Additional fitting parameters
            
        Returns:
            bool: True if fitting successful, False otherwise
        """
        try:
            # Convert to numpy if tensor
            if isinstance(data, torch.Tensor):
                data = data.detach().cpu().numpy()
                
            # Handle different shapes
            if data.ndim == 2:
                if data.shape[1] == 1:
                    data = data.squeeze()
                else:
                    # Multi-asset data, take first asset
                    data = data[:, 0]
                    
            # Convert to returns if data looks like prices
            returns = self._convert_to_returns(data)
            
            # Check minimum data requirement
            if len(returns) < self.min_periods:
                logger.warning(f"Insufficient data for GARCH fitting: {len(returns)} < {self.min_periods}")
                return False
                
            # Store returns history
            self.returns_history = returns.copy()
            
            # Scale returns for numerical stability
            scaled_returns = returns * self.scale_factor
            
            # Create GARCH model
            if self.model_type.upper() == 'EGARCH':
                self.arch_model = arch_model(
                    scaled_returns,
                    vol='EGARCH',
                    p=self.p,
                    q=self.q,
                    dist=self.dist
                )
            elif self.model_type.upper() == 'GJR-GARCH' or self.model_type.upper() == 'GJRGARCH':
                self.arch_model = arch_model(
                    scaled_returns,
                    vol='GARCH',
                    p=self.p,
                    o=self.o,
                    q=self.q,
                    dist=self.dist
                )
            else:  # Standard GARCH
                self.arch_model = arch_model(
                    scaled_returns,
                    vol='GARCH',
                    p=self.p,
                    q=self.q,
                    dist=self.dist
                )
                
            # Fit the model
            self.fitted_model = self.arch_model.fit(disp='off', show_warning=False)
            self.is_fitted = True
            
            logger.info(f"Successfully fitted {self.model_type} model")
            return True
            
        except Exception as e:
            logger.error(f"GARCH fitting failed: {str(e)}")
            return False
    
    def predict(self, data: Union[torch.Tensor, np.ndarray], 
                horizon: int = 1, **kwargs) -> torch.Tensor:
        """
        Generate volatility predictions.
        
        Args:
            data: Input data for prediction
            horizon: Number of periods to predict ahead
            **kwargs: Additional prediction parameters
            
        Returns:
            torch.Tensor: Predicted volatility values
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
            
        try:
            # Convert to numpy if tensor
            if isinstance(data, torch.Tensor):
                data = data.detach().cpu().numpy()
                
            # Handle different shapes
            if data.ndim == 2:
                if data.shape[1] == 1:
                    data = data.squeeze()
                else:
                    # Multi-asset data, take first asset
                    data = data[:, 0]
                    
            # Convert to returns
            returns = self._convert_to_returns(data)
            
            # Update model with new data if provided
            if len(returns) > 0:
                self._update_with_new_data(returns)
                
            # Make forecast
            forecast = self.fitted_model.forecast(horizon=horizon)
            
            # Extract volatility forecast and convert back to original scale
            vol_forecasts = []
            for h in range(horizon):
                vol_h = np.sqrt(forecast.variance.iloc[-1, h]) / self.scale_factor
                vol_forecasts.append(vol_h)
                
            return torch.tensor(vol_forecasts, dtype=torch.float32, device=self._device)
            
        except Exception as e:
            logger.error(f"GARCH prediction failed: {str(e)}")
            return torch.full((horizon,), float('nan'), device=self._device)
    
    def predict_rolling(self, data: Union[torch.Tensor, np.ndarray], 
                       window_size: int = 252, **kwargs) -> torch.Tensor:
        """
        Generate rolling volatility predictions.
        
        Args:
            data: Input data for prediction
            window_size: Size of rolling window for fitting
            **kwargs: Additional prediction parameters
            
        Returns:
            torch.Tensor: Rolling volatility predictions
        """
        # Convert to numpy if tensor
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
            
        # Handle different shapes
        if data.ndim == 2:
            if data.shape[1] == 1:
                data = data.squeeze()
            else:
                # Multi-asset data, take first asset
                data = data[:, 0]
                
        # Convert to returns
        returns = self._convert_to_returns(data)
        
        predictions = []
        
        # Start predictions after we have enough data
        start_idx = max(window_size, self.min_periods)
        
        for i in range(start_idx, len(returns)):
            # Get rolling window
            window_returns = returns[i-window_size:i]
            
            # Fit model on window
            temp_model = GARCHModel(
                model_type=self.model_type,
                p=self.p,
                q=self.q,
                o=self.o,
                dist=self.dist,
                scale_factor=self.scale_factor,
                min_periods=self.min_periods
            )
            
            if temp_model.fit(window_returns):
                vol_pred = temp_model.predict(np.array([]), horizon=1)
                predictions.append(vol_pred.item())
            else:
                predictions.append(float('nan'))
                
        # Pad with NaN for initial periods
        full_predictions = [float('nan')] * start_idx + predictions
        
        return torch.tensor(full_predictions, dtype=torch.float32, device=self._device)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information."""
        info = super().get_model_info()
        
        if self.fitted_model is not None:
            info.update({
                'aic': float(self.fitted_model.aic),
                'bic': float(self.fitted_model.bic),
                'log_likelihood': float(self.fitted_model.loglikelihood),
                'num_params': len(self.fitted_model.params),
                'convergence': self.fitted_model.convergence_flag == 0
            })
            
        return info
    
    def get_conditional_volatility(self) -> Optional[torch.Tensor]:
        """
        Get conditional volatility from fitted model.
        
        Returns:
            torch.Tensor: Conditional volatility series
        """
        if self.fitted_model is None:
            return None
            
        cond_vol = self.fitted_model.conditional_volatility / self.scale_factor
        return torch.tensor(cond_vol.values, dtype=torch.float32, device=self._device)
    
    def get_standardized_residuals(self) -> Optional[torch.Tensor]:
        """
        Get standardized residuals from fitted model.
        
        Returns:
            torch.Tensor: Standardized residuals
        """
        if self.fitted_model is None:
            return None
            
        std_resid = self.fitted_model.std_resid
        return torch.tensor(std_resid.values, dtype=torch.float32, device=self._device)
    
    def _convert_to_returns(self, data: np.ndarray) -> np.ndarray:
        """
        Convert data to returns if it looks like prices.
        
        Args:
            data: Input data
            
        Returns:
            np.ndarray: Returns data
        """
        # Simple heuristic: if all values are positive and range is large,
        # assume it's prices and convert to returns
        if np.all(data > 0) and (np.max(data) / np.min(data) > 2):
            returns = np.diff(np.log(data))
        else:
            # Assume it's already returns
            returns = data[1:] if len(data) > 1 else data
            
        # Remove any infinite or NaN values
        returns = returns[np.isfinite(returns)]
        
        return returns
    
    def _update_with_new_data(self, new_returns: np.ndarray) -> None:
        """
        Update the model with new return data.
        
        Args:
            new_returns: New returns to incorporate
        """
        if len(new_returns) > 0:
            self.returns_history = np.concatenate([self.returns_history, new_returns])
            # Keep only the most recent data to prevent memory issues
            if len(self.returns_history) > 2000:
                self.returns_history = self.returns_history[-2000:]
    
    def _predict_single(self) -> float:
        """
        Make a single-step volatility prediction.
        
        Returns:
            float: Predicted volatility
        """
        try:
            forecast = self.fitted_model.forecast(horizon=1)
            vol_pred = np.sqrt(forecast.variance.iloc[-1, 0]) / self.scale_factor
            return float(vol_pred)
        except Exception as e:
            logger.warning(f"Single prediction failed: {str(e)}")
            return float('nan')
