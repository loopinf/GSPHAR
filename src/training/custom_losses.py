"""
Custom loss functions for training GSPHAR models.

This module contains custom loss functions designed to emphasize
specific aspects of the prediction task, such as large jumps in
volatility or returns.
"""

import torch
import torch.nn as nn
import numpy as np


class WeightedMSELoss(nn.Module):
    """
    Weighted MSE loss that emphasizes large target values.
    
    This loss function applies higher weights to samples with target values
    above a specified threshold, making the model focus more on predicting
    large jumps correctly.
    """
    
    def __init__(self, threshold=0.5, weight_factor=5.0):
        """
        Initialize the weighted MSE loss.
        
        Args:
            threshold (float): Threshold above which to apply higher weights
            weight_factor (float): Factor to increase weights for values above threshold
        """
        super(WeightedMSELoss, self).__init__()
        self.threshold = threshold
        self.weight_factor = weight_factor
        
    def forward(self, predictions, targets):
        """
        Forward pass of the weighted loss function.
        
        Args:
            predictions (torch.Tensor): Model predictions
            targets (torch.Tensor): Ground truth values
            
        Returns:
            torch.Tensor: Weighted MSE loss
        """
        # Calculate squared errors
        squared_errors = (predictions - targets) ** 2
        
        # Create weights based on target magnitude
        weights = torch.ones_like(targets)
        large_value_mask = targets > self.threshold
        weights[large_value_mask] = self.weight_factor
        
        # Apply weights to squared errors
        weighted_squared_errors = weights * squared_errors
        
        # Return mean of weighted squared errors
        return weighted_squared_errors.mean()


class AsymmetricMSELoss(nn.Module):
    """
    Asymmetric MSE loss that penalizes underprediction more than overprediction.
    
    This loss function applies different weights depending on whether the model
    underpredicts or overpredicts the target value, making it particularly useful
    for risk-averse scenarios where underpredicting volatility is more costly.
    """
    
    def __init__(self, under_prediction_factor=2.0):
        """
        Initialize the asymmetric MSE loss.
        
        Args:
            under_prediction_factor (float): Factor to increase penalty for underprediction
        """
        super(AsymmetricMSELoss, self).__init__()
        self.under_prediction_factor = under_prediction_factor
        
    def forward(self, predictions, targets):
        """
        Forward pass of the asymmetric loss function.
        
        Args:
            predictions (torch.Tensor): Model predictions
            targets (torch.Tensor): Ground truth values
            
        Returns:
            torch.Tensor: Asymmetric MSE loss
        """
        # Calculate squared errors
        squared_errors = (predictions - targets) ** 2
        
        # Create weights based on prediction direction
        weights = torch.ones_like(targets)
        under_prediction_mask = predictions < targets
        weights[under_prediction_mask] = self.under_prediction_factor
        
        # Apply weights to squared errors
        weighted_squared_errors = weights * squared_errors
        
        # Return mean of weighted squared errors
        return weighted_squared_errors.mean()


class ThresholdMSELoss(nn.Module):
    """
    Threshold-based MSE loss that applies different penalties above and below thresholds.
    
    This loss function allows for fine-grained control over how the model treats
    different magnitudes of target values.
    """
    
    def __init__(self, thresholds=[0.2, 0.5, 1.0], weights=[1.0, 2.0, 5.0, 10.0]):
        """
        Initialize the threshold MSE loss.
        
        Args:
            thresholds (list): List of thresholds for different weight levels
            weights (list): List of weights to apply for each threshold range
                            (should be one more weight than thresholds)
        """
        super(ThresholdMSELoss, self).__init__()
        assert len(weights) == len(thresholds) + 1, "Number of weights should be number of thresholds + 1"
        self.thresholds = thresholds
        self.weights = weights
        
    def forward(self, predictions, targets):
        """
        Forward pass of the threshold loss function.
        
        Args:
            predictions (torch.Tensor): Model predictions
            targets (torch.Tensor): Ground truth values
            
        Returns:
            torch.Tensor: Threshold-based MSE loss
        """
        # Calculate squared errors
        squared_errors = (predictions - targets) ** 2
        
        # Create weights based on target magnitude
        weights = torch.ones_like(targets) * self.weights[0]
        
        # Apply different weights for each threshold range
        for i, threshold in enumerate(self.thresholds):
            mask = targets > threshold
            weights[mask] = self.weights[i + 1]
        
        # Apply weights to squared errors
        weighted_squared_errors = weights * squared_errors
        
        # Return mean of weighted squared errors
        return weighted_squared_errors.mean()


class HybridLoss(nn.Module):
    """
    Hybrid loss function that combines multiple loss components.
    
    This loss function allows for combining different loss functions with
    different weights, providing flexibility in how the model is trained.
    """
    
    def __init__(self, mse_weight=1.0, large_jump_weight=1.0, threshold=0.5, jump_factor=5.0):
        """
        Initialize the hybrid loss.
        
        Args:
            mse_weight (float): Weight for the standard MSE component
            large_jump_weight (float): Weight for the large jump component
            threshold (float): Threshold for defining large jumps
            jump_factor (float): Factor to increase weights for large jumps
        """
        super(HybridLoss, self).__init__()
        self.mse_weight = mse_weight
        self.large_jump_weight = large_jump_weight
        self.threshold = threshold
        self.jump_factor = jump_factor
        
    def forward(self, predictions, targets):
        """
        Forward pass of the hybrid loss function.
        
        Args:
            predictions (torch.Tensor): Model predictions
            targets (torch.Tensor): Ground truth values
            
        Returns:
            torch.Tensor: Hybrid loss
        """
        # Standard MSE component
        mse_loss = ((predictions - targets) ** 2).mean()
        
        # Large jump component
        large_jump_mask = targets > self.threshold
        if large_jump_mask.sum() > 0:
            large_jump_loss = (((predictions[large_jump_mask] - targets[large_jump_mask]) ** 2) * self.jump_factor).mean()
        else:
            large_jump_loss = torch.tensor(0.0, device=predictions.device)
        
        # Combine components
        total_loss = self.mse_weight * mse_loss + self.large_jump_weight * large_jump_loss
        
        return total_loss
