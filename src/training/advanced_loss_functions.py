"""
Advanced Loss Functions for Enhanced Trading Agent Training.

This module implements sophisticated loss functions for training trading agents:
- Multi-objective optimization
- Risk-adjusted returns
- Drawdown constraints
- Portfolio-level optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SharpeRatioLoss(nn.Module):
    """
    Loss function that maximizes Sharpe ratio (risk-adjusted returns).
    """
    
    def __init__(self, risk_free_rate: float = 0.0, min_std: float = 1e-6):
        super().__init__()
        self.risk_free_rate = risk_free_rate
        self.min_std = min_std
        
    def forward(self, returns: torch.Tensor) -> torch.Tensor:
        """
        Calculate negative Sharpe ratio as loss.
        
        Args:
            returns: Trading returns [batch_size, n_assets] or [batch_size * n_assets]
            
        Returns:
            Negative Sharpe ratio (to minimize)
        """
        # Flatten if needed
        if returns.dim() > 1:
            returns = returns.view(-1)
        
        # Remove zero returns (unfilled orders)
        non_zero_returns = returns[returns != 0]
        
        if len(non_zero_returns) < 2:
            return torch.tensor(0.0, device=returns.device)
        
        mean_return = non_zero_returns.mean()
        std_return = torch.clamp(non_zero_returns.std(), min=self.min_std)
        
        sharpe_ratio = (mean_return - self.risk_free_rate) / std_return
        return -sharpe_ratio  # Negative because we want to maximize


class DrawdownConstraintLoss(nn.Module):
    """
    Loss function that penalizes large drawdowns.
    """
    
    def __init__(self, max_drawdown: float = 0.1, penalty_weight: float = 10.0):
        super().__init__()
        self.max_drawdown = max_drawdown
        self.penalty_weight = penalty_weight
        
    def forward(self, returns: torch.Tensor) -> torch.Tensor:
        """
        Calculate drawdown penalty.
        
        Args:
            returns: Trading returns [batch_size, n_assets] or [batch_size * n_assets]
            
        Returns:
            Drawdown penalty
        """
        # Flatten if needed
        if returns.dim() > 1:
            returns = returns.view(-1)
        
        # Remove zero returns
        non_zero_returns = returns[returns != 0]
        
        if len(non_zero_returns) < 2:
            return torch.tensor(0.0, device=returns.device)
        
        # Calculate cumulative returns
        cumulative_returns = torch.cumsum(non_zero_returns, dim=0)
        
        # Calculate running maximum
        running_max = torch.cummax(cumulative_returns, dim=0)[0]
        
        # Calculate drawdown
        drawdown = (running_max - cumulative_returns) / (running_max + 1e-8)
        max_dd = drawdown.max()
        
        # Penalty if drawdown exceeds threshold
        penalty = torch.clamp(max_dd - self.max_drawdown, min=0.0)
        return self.penalty_weight * penalty


class FillRateOptimizationLoss(nn.Module):
    """
    Loss function that optimizes fill rate to target range.
    """
    
    def __init__(self, target_fill_rate: float = 0.15, tolerance: float = 0.05):
        super().__init__()
        self.target_fill_rate = target_fill_rate
        self.tolerance = tolerance
        
    def forward(self, returns: torch.Tensor) -> torch.Tensor:
        """
        Calculate fill rate optimization loss.
        
        Args:
            returns: Trading returns [batch_size, n_assets] or [batch_size * n_assets]
            
        Returns:
            Fill rate penalty
        """
        # Flatten if needed
        if returns.dim() > 1:
            returns = returns.view(-1)
        
        # Calculate fill rate
        total_orders = returns.numel()
        filled_orders = (returns != 0).sum().float()
        fill_rate = filled_orders / total_orders
        
        # Penalty if outside target range
        lower_bound = self.target_fill_rate - self.tolerance
        upper_bound = self.target_fill_rate + self.tolerance
        
        if fill_rate < lower_bound:
            penalty = (lower_bound - fill_rate) ** 2
        elif fill_rate > upper_bound:
            penalty = (fill_rate - upper_bound) ** 2
        else:
            penalty = torch.tensor(0.0, device=returns.device)
            
        return penalty


class ProfitConsistencyLoss(nn.Module):
    """
    Loss function that promotes consistent profits over time.
    """
    
    def __init__(self, consistency_weight: float = 1.0):
        super().__init__()
        self.consistency_weight = consistency_weight
        
    def forward(self, returns: torch.Tensor) -> torch.Tensor:
        """
        Calculate profit consistency loss.
        
        Args:
            returns: Trading returns [batch_size, n_assets] or [batch_size * n_assets]
            
        Returns:
            Consistency penalty
        """
        # Flatten if needed
        if returns.dim() > 1:
            returns = returns.view(-1)
        
        # Remove zero returns
        non_zero_returns = returns[returns != 0]
        
        if len(non_zero_returns) < 2:
            return torch.tensor(0.0, device=returns.device)
        
        # Calculate variance of returns (lower is better for consistency)
        return_variance = non_zero_returns.var()
        return self.consistency_weight * return_variance


class AdvancedTradingLoss(nn.Module):
    """
    Comprehensive multi-objective loss function for advanced trading agent training.
    """
    
    def __init__(
        self,
        profit_weight: float = 2.0,
        sharpe_weight: float = 1.5,
        drawdown_weight: float = 1.0,
        fill_rate_weight: float = 0.8,
        consistency_weight: float = 0.5,
        risk_penalty_weight: float = 0.3,
        target_fill_rate: float = 0.15,
        max_drawdown: float = 0.1
    ):
        super().__init__()
        
        self.profit_weight = profit_weight
        self.sharpe_weight = sharpe_weight
        self.drawdown_weight = drawdown_weight
        self.fill_rate_weight = fill_rate_weight
        self.consistency_weight = consistency_weight
        self.risk_penalty_weight = risk_penalty_weight
        
        # Initialize component loss functions
        self.sharpe_loss = SharpeRatioLoss()
        self.drawdown_loss = DrawdownConstraintLoss(max_drawdown=max_drawdown)
        self.fill_rate_loss = FillRateOptimizationLoss(target_fill_rate=target_fill_rate)
        self.consistency_loss = ProfitConsistencyLoss()
        
    def forward(
        self,
        returns: torch.Tensor,
        signals: Dict[str, torch.Tensor],
        market_data: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate comprehensive trading loss.
        
        Args:
            returns: Trading returns [batch_size, n_assets]
            signals: Trading signals from model
            market_data: Optional market data for additional constraints
            
        Returns:
            Tuple of (total_loss, loss_components)
        """
        device = returns.device
        
        # 1. Profit component (maximize returns)
        profit_loss = -returns.mean()
        
        # 2. Sharpe ratio component (risk-adjusted returns)
        sharpe_loss = self.sharpe_loss(returns)
        
        # 3. Drawdown constraint
        drawdown_loss = self.drawdown_loss(returns)
        
        # 4. Fill rate optimization
        fill_rate_loss = self.fill_rate_loss(returns)
        
        # 5. Profit consistency
        consistency_loss = self.consistency_loss(returns)
        
        # 6. Risk penalty (extreme positions)
        if 'ratio' in signals:
            ratio_penalty = torch.mean(torch.abs(signals['ratio'] - 0.95))
        else:
            ratio_penalty = torch.tensor(0.0, device=device)
        
        # Combine all loss components
        total_loss = (
            self.profit_weight * profit_loss +
            self.sharpe_weight * sharpe_loss +
            self.drawdown_weight * drawdown_loss +
            self.fill_rate_weight * fill_rate_loss +
            self.consistency_weight * consistency_loss +
            self.risk_penalty_weight * ratio_penalty
        )
        
        # Calculate metrics for monitoring
        fill_rate = (returns != 0).float().mean()
        avg_return = returns[returns != 0].mean() if (returns != 0).any() else torch.tensor(0.0)
        
        loss_components = {
            'total_loss': total_loss.item(),
            'profit_loss': profit_loss.item(),
            'sharpe_loss': sharpe_loss.item() if isinstance(sharpe_loss, torch.Tensor) else sharpe_loss,
            'drawdown_loss': drawdown_loss.item(),
            'fill_rate_loss': fill_rate_loss.item(),
            'consistency_loss': consistency_loss.item(),
            'risk_penalty': ratio_penalty.item(),
            'fill_rate': fill_rate.item(),
            'avg_return': avg_return.item() if isinstance(avg_return, torch.Tensor) else avg_return
        }
        
        return total_loss, loss_components


class CurriculumLearningScheduler:
    """
    Curriculum learning scheduler that adjusts loss weights during training.
    """
    
    def __init__(
        self,
        initial_weights: Dict[str, float],
        target_weights: Dict[str, float],
        transition_epochs: int = 20
    ):
        self.initial_weights = initial_weights
        self.target_weights = target_weights
        self.transition_epochs = transition_epochs
        
    def get_weights(self, epoch: int) -> Dict[str, float]:
        """
        Get loss weights for current epoch.
        
        Args:
            epoch: Current training epoch
            
        Returns:
            Dictionary of loss weights
        """
        if epoch >= self.transition_epochs:
            return self.target_weights
        
        # Linear interpolation between initial and target weights
        alpha = epoch / self.transition_epochs
        weights = {}
        
        for key in self.initial_weights:
            initial_val = self.initial_weights[key]
            target_val = self.target_weights[key]
            weights[key] = initial_val + alpha * (target_val - initial_val)
            
        return weights
    
    def update_loss_function(self, loss_fn: AdvancedTradingLoss, epoch: int):
        """
        Update loss function weights based on curriculum schedule.
        
        Args:
            loss_fn: Loss function to update
            epoch: Current training epoch
        """
        weights = self.get_weights(epoch)
        
        loss_fn.profit_weight = weights.get('profit_weight', loss_fn.profit_weight)
        loss_fn.sharpe_weight = weights.get('sharpe_weight', loss_fn.sharpe_weight)
        loss_fn.drawdown_weight = weights.get('drawdown_weight', loss_fn.drawdown_weight)
        loss_fn.fill_rate_weight = weights.get('fill_rate_weight', loss_fn.fill_rate_weight)
        loss_fn.consistency_weight = weights.get('consistency_weight', loss_fn.consistency_weight)
        loss_fn.risk_penalty_weight = weights.get('risk_penalty_weight', loss_fn.risk_penalty_weight)


def create_advanced_loss_function(
    stage: str = "initial",
    target_fill_rate: float = 0.15,
    max_drawdown: float = 0.1
) -> AdvancedTradingLoss:
    """
    Factory function to create advanced loss function for different training stages.
    
    Args:
        stage: Training stage ("initial", "intermediate", "advanced")
        target_fill_rate: Target fill rate for optimization
        max_drawdown: Maximum allowed drawdown
        
    Returns:
        Configured advanced loss function
    """
    if stage == "initial":
        # Focus on basic profitability and fill rates
        return AdvancedTradingLoss(
            profit_weight=3.0,
            sharpe_weight=0.5,
            drawdown_weight=0.5,
            fill_rate_weight=2.0,
            consistency_weight=0.2,
            risk_penalty_weight=1.0,
            target_fill_rate=target_fill_rate,
            max_drawdown=max_drawdown
        )
    elif stage == "intermediate":
        # Balance profitability with risk management
        return AdvancedTradingLoss(
            profit_weight=2.0,
            sharpe_weight=1.5,
            drawdown_weight=1.0,
            fill_rate_weight=1.0,
            consistency_weight=0.5,
            risk_penalty_weight=0.5,
            target_fill_rate=target_fill_rate,
            max_drawdown=max_drawdown
        )
    else:  # advanced
        # Focus on risk-adjusted returns and consistency
        return AdvancedTradingLoss(
            profit_weight=1.5,
            sharpe_weight=2.0,
            drawdown_weight=1.5,
            fill_rate_weight=0.8,
            consistency_weight=1.0,
            risk_penalty_weight=0.3,
            target_fill_rate=target_fill_rate,
            max_drawdown=max_drawdown
        )
