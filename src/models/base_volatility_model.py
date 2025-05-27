"""
Base interface for volatility prediction models.

This module defines the abstract base class that all volatility prediction models
should inherit from to ensure a consistent interface across GSPHAR, GARCH, and
future GNNHAR models.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)


class BaseVolatilityModel(nn.Module, ABC):
    """
    Abstract base class for volatility prediction models.
    
    This class defines the common interface that all volatility models
    (GSPHAR, GARCH, GNNHAR, etc.) should implement.
    """
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the base volatility model.
        
        Args:
            model_name: Name identifier for this model type
            **kwargs: Additional model-specific parameters
        """
        super().__init__()
        self.model_name = model_name
        self.model_params = kwargs
        self.is_fitted = False
        self._device = torch.device('cpu')
        
    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass for the model.
        
        Returns:
            torch.Tensor: Predicted volatility values
        """
        pass
    
    @abstractmethod
    def fit(self, data: Union[torch.Tensor, np.ndarray], **kwargs) -> bool:
        """
        Fit the model to training data.
        
        Args:
            data: Training data
            **kwargs: Additional fitting parameters
            
        Returns:
            bool: True if fitting successful, False otherwise
        """
        pass
    
    @abstractmethod
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
        pass
    
    def save_state(self, filepath: str) -> None:
        """
        Save model state to disk.
        
        Args:
            filepath: Path to save the model state
        """
        state = {
            'model_name': self.model_name,
            'model_params': self.model_params,
            'state_dict': self.state_dict(),
            'is_fitted': self.is_fitted,
            'device': str(self._device)
        }
        torch.save(state, filepath)
        logger.info(f"Model state saved to {filepath}")
    
    def load_state(self, filepath: str) -> bool:
        """
        Load model state from disk.
        
        Args:
            filepath: Path to load the model state from
            
        Returns:
            bool: True if loading successful, False otherwise
        """
        try:
            state = torch.load(filepath, map_location=self._device)
            self.model_name = state['model_name']
            self.model_params = state['model_params']
            self.load_state_dict(state['state_dict'])
            self.is_fitted = state['is_fitted']
            logger.info(f"Model state loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model state from {filepath}: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dict containing model information
        """
        return {
            'model_name': self.model_name,
            'model_params': self.model_params,
            'is_fitted': self.is_fitted,
            'device': str(self._device),
            'num_parameters': sum(p.numel() for p in self.parameters())
        }
    
    def to_device(self, device: Union[str, torch.device]) -> 'BaseVolatilityModel':
        """
        Move model to specified device.
        
        Args:
            device: Target device
            
        Returns:
            Self for method chaining
        """
        if isinstance(device, str):
            device = torch.device(device)
        self._device = device
        return self.to(device)
    
    def validate_input(self, data: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Validate and convert input data to appropriate tensor format.
        
        Args:
            data: Input data to validate
            
        Returns:
            torch.Tensor: Validated tensor data
            
        Raises:
            ValueError: If data format is invalid
        """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        elif not isinstance(data, torch.Tensor):
            raise ValueError(f"Data must be numpy array or torch tensor, got {type(data)}")
        
        if data.dtype != torch.float32:
            data = data.float()
            
        return data.to(self._device)
    
    def check_fitted(self) -> None:
        """
        Check if the model has been fitted.
        
        Raises:
            RuntimeError: If model has not been fitted
        """
        if not self.is_fitted:
            raise RuntimeError(f"{self.model_name} model has not been fitted yet. "
                             "Call fit() before making predictions.")


class MultiAssetVolatilityModel(BaseVolatilityModel):
    """
    Base class for multi-asset volatility models like GSPHAR.
    
    This extends the base interface with methods specific to models
    that handle multiple assets simultaneously.
    """
    
    def __init__(self, model_name: str, num_assets: int, **kwargs):
        """
        Initialize multi-asset volatility model.
        
        Args:
            model_name: Name identifier for this model type
            num_assets: Number of assets/symbols in the model
            **kwargs: Additional model-specific parameters
        """
        super().__init__(model_name, **kwargs)
        self.num_assets = num_assets
    
    @abstractmethod
    def get_cross_correlations(self) -> Optional[torch.Tensor]:
        """
        Get cross-correlation matrix between assets.
        
        Returns:
            torch.Tensor: Cross-correlation matrix or None if not applicable
        """
        pass
    
    @abstractmethod
    def predict_single_asset(self, data: Union[torch.Tensor, np.ndarray], 
                           asset_idx: int, horizon: int = 1) -> torch.Tensor:
        """
        Generate prediction for a single asset.
        
        Args:
            data: Input data
            asset_idx: Index of the asset to predict
            horizon: Prediction horizon
            
        Returns:
            torch.Tensor: Predicted volatility for the specified asset
        """
        pass


class SingleAssetVolatilityModel(BaseVolatilityModel):
    """
    Base class for single-asset volatility models like GARCH.
    
    This extends the base interface with methods specific to models
    that handle one asset at a time.
    """
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize single-asset volatility model.
        
        Args:
            model_name: Name identifier for this model type
            **kwargs: Additional model-specific parameters
        """
        super().__init__(model_name, **kwargs)
    
    @abstractmethod
    def get_model_parameters(self) -> Dict[str, float]:
        """
        Get fitted model parameters.
        
        Returns:
            Dict containing model parameters
        """
        pass
    
    @abstractmethod
    def compute_log_likelihood(self, data: Union[torch.Tensor, np.ndarray]) -> float:
        """
        Compute log-likelihood of the data given the fitted model.
        
        Args:
            data: Data to compute likelihood for
            
        Returns:
            float: Log-likelihood value
        """
        pass
