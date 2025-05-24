"""
Custom loss functions for trading strategies based on volatility predictions.
"""

import torch
import torch.nn as nn
import numpy as np


class TradingStrategyLoss(nn.Module):
    """
    Custom loss function for a volatility-based trading strategy.

    This loss function is designed for a strategy where:
    1. We predict volatility for the next period
    2. Place a buy order at price * (1 - predicted_volatility)
    3. If the order is filled (price drops below our threshold), we hold for a fixed period
    4. Exit the position after the holding period

    The loss function optimizes for:
    - Accurate prediction of price drops (to get orders filled)
    - Maximizing profit during the holding period
    - Avoiding losses on filled orders

    Args:
        alpha (float): Weight for the fill loss component. Default: 1.0
        beta (float): Weight for the profit component. Default: 1.0
        gamma (float): Weight for the loss avoidance component. Default: 2.0
        holding_period (int): Number of periods to hold the position. Default: 4
    """
    def __init__(self, alpha=1.0, beta=1.0, gamma=2.0, holding_period=4):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.holding_period = holding_period

    def forward(self, vol_pred, log_returns):
        """
        Calculate the trading strategy loss.

        Args:
            vol_pred (torch.Tensor): Predicted volatility, shape [batch_size]
            log_returns (torch.Tensor): Log returns for next periods,
                                        shape [batch_size, holding_period+1]
                                        where log_returns[:,0] is the next period return
                                        and log_returns[:,1:holding_period+1] are the returns
                                        during the holding period

        Returns:
            torch.Tensor: The calculated loss value
        """
        # Ensure vol_pred is positive (volatility should be positive)
        vol_pred = torch.abs(vol_pred)

        # Next period's log return (determines if order is filled)
        log_return_next = log_returns[:, 0]

        # Convert predicted volatility to log return threshold
        # If we predict 5% volatility, our threshold in log terms would be ln(1-0.05)
        # Ensure volatility is not greater than 1 to avoid negative values in log
        clipped_vol_pred = torch.clamp(vol_pred, 0.0, 0.99)
        log_entry_threshold = torch.log(1 - clipped_vol_pred)

        # Calculate holding period log return (sum the subsequent log returns)
        # Make sure we don't go out of bounds with the holding period
        actual_holding_period = min(self.holding_period, log_returns.shape[1] - 1)
        log_return_holding_period = torch.sum(log_returns[:, 1:1+actual_holding_period], dim=1)

        # Component 1: Fill Loss - Penalize when orders don't get filled
        # In log terms, we want log_return_next <= log_entry_threshold for the order to fill
        fill_loss = torch.max(torch.zeros_like(vol_pred), log_entry_threshold - log_return_next)**2

        # Create a mask for filled orders
        filled_orders = (log_return_next <= log_entry_threshold).float()

        # Component 2: Profit Loss - Reward when filled orders result in profit
        profit_loss = -filled_orders * log_return_holding_period

        # Component 3: Avoidance Loss - Penalize when filled orders result in losses
        avoidance_loss = torch.max(torch.zeros_like(vol_pred), -filled_orders * log_return_holding_period)**2

        # Combine components with weights
        total_loss = self.alpha * fill_loss + self.beta * profit_loss + self.gamma * avoidance_loss

        return total_loss.mean()


def convert_pct_change_to_log_returns(pct_changes):
    """
    Convert percentage changes to log returns.

    Args:
        pct_changes (numpy.ndarray or torch.Tensor): Percentage changes

    Returns:
        Same type as input: Log returns
    """
    # Add a small epsilon to avoid log(0) and handle negative values
    epsilon = 1e-8

    if isinstance(pct_changes, torch.Tensor):
        # Clip to avoid extreme values
        clipped_values = torch.clamp(1 + pct_changes, min=epsilon)
        return torch.log(clipped_values)
    else:
        # Replace NaN and inf values
        clean_values = np.nan_to_num(pct_changes, nan=0.0, posinf=0.0, neginf=0.0)
        # Clip to avoid extreme values
        clipped_values = np.clip(1 + clean_values, epsilon, None)
        return np.log(clipped_values)


def convert_log_returns_to_pct_change(log_returns):
    """
    Convert log returns to percentage changes.

    Args:
        log_returns (numpy.ndarray or torch.Tensor): Log returns

    Returns:
        Same type as input: Percentage changes
    """
    if isinstance(log_returns, torch.Tensor):
        return torch.exp(log_returns) - 1
    else:
        return np.exp(log_returns) - 1


def calculate_cumulative_return(log_returns):
    """
    Calculate cumulative return from a series of log returns.

    Args:
        log_returns (numpy.ndarray or torch.Tensor): Log returns

    Returns:
        Same type as input: Cumulative return as a percentage change
    """
    if isinstance(log_returns, torch.Tensor):
        cumulative_log_return = torch.sum(log_returns)
        return torch.exp(cumulative_log_return) - 1
    else:
        cumulative_log_return = np.sum(log_returns)
        return np.exp(cumulative_log_return) - 1


class MaximizeProfitLoss(nn.Module):
    """
    Custom loss function that directly maximizes trading profit.

    This loss function calculates the actual profit from the trading strategy
    and returns the negative profit as loss (so minimizing loss = maximizing profit).

    Strategy:
    1. Place limit buy order at current_price * (1 - predicted_volatility)
    2. Order fills if actual price drops to limit price or below
    3. Hold position for holding_period
    4. Calculate actual profit/loss
    5. Return negative profit as loss (to maximize profit)

    Args:
        holding_period (int): Number of periods to hold the position. Default: 24
        risk_penalty (float): Penalty factor for losses to encourage risk management. Default: 2.0
        no_fill_penalty (float): Penalty when orders don't fill. Default: 0.01
    """
    def __init__(self, holding_period=24, risk_penalty=2.0, no_fill_penalty=0.01):
        super().__init__()
        self.holding_period = holding_period
        self.risk_penalty = risk_penalty
        self.no_fill_penalty = no_fill_penalty

    def forward(self, vol_pred, log_returns):
        """
        Calculate the profit-maximizing loss.

        Args:
            vol_pred (torch.Tensor): Predicted volatility, shape [batch_size]
            log_returns (torch.Tensor): Log returns for next periods,
                                        shape [batch_size, holding_period+1]
                                        where log_returns[:,0] is the next period return
                                        and log_returns[:,1:holding_period+1] are the returns
                                        during the holding period

        Returns:
            torch.Tensor: The calculated loss value (negative profit)
        """
        # Ensure vol_pred is positive and reasonable
        vol_pred = torch.clamp(torch.abs(vol_pred), 0.001, 0.5)

        # Next period's log return (determines if order is filled)
        log_return_next = log_returns[:, 0]

        # Convert predicted volatility to log return threshold
        log_entry_threshold = torch.log(1 - vol_pred)

        # Calculate holding period log return
        actual_holding_period = min(self.holding_period, log_returns.shape[1] - 1)
        log_return_holding_period = torch.sum(log_returns[:, 1:1+actual_holding_period], dim=1)

        # Use smooth approximation for order filling to maintain gradients
        fill_probability = torch.sigmoid((log_entry_threshold - log_return_next) * 100)

        # Calculate actual profit for filled orders
        holding_period_profit = torch.exp(log_return_holding_period) - 1

        # Calculate expected strategy profit (weighted by fill probability)
        expected_strategy_profit = fill_probability * holding_period_profit

        # Apply risk penalty for losses (smooth version)
        loss_penalty = torch.clamp(-expected_strategy_profit, min=0)
        risk_adjusted_profit = expected_strategy_profit - self.risk_penalty * loss_penalty

        # Penalty for low fill probability (missed opportunities)
        no_fill_penalty = (1 - fill_probability) * self.no_fill_penalty

        # Total profit (including penalties)
        total_profit = risk_adjusted_profit - no_fill_penalty

        # Return negative profit as loss (minimize loss = maximize profit)
        return -total_profit.mean()


class SimpleProfitMaximizationLoss(nn.Module):
    """
    Simplified profit maximization loss function.

    This is the most direct approach: calculate actual trading profit
    and return negative profit as loss.

    Args:
        holding_period (int): Number of periods to hold the position. Default: 24
    """
    def __init__(self, holding_period=24):
        super().__init__()
        self.holding_period = holding_period

    def forward(self, vol_pred, log_returns):
        """
        Calculate simple profit maximization loss.

        Args:
            vol_pred (torch.Tensor): Predicted volatility, shape [batch_size]
            log_returns (torch.Tensor): Log returns, shape [batch_size, holding_period+1]

        Returns:
            torch.Tensor: Negative profit (loss to minimize)
        """
        # Clamp predictions to reasonable range (maintain gradients)
        vol_pred = torch.clamp(vol_pred, 0.001, 0.5)

        # Entry threshold and actual next return
        log_entry_threshold = torch.log(1 - vol_pred)
        log_return_next = log_returns[:, 0]

        # Use smooth approximation for order filling to maintain gradients
        # Instead of hard threshold, use sigmoid approximation
        fill_probability = torch.sigmoid((log_entry_threshold - log_return_next) * 100)

        # Calculate holding period return
        actual_holding_period = min(self.holding_period, log_returns.shape[1] - 1)
        log_return_holding = torch.sum(log_returns[:, 1:1+actual_holding_period], dim=1)

        # Calculate profit: convert log returns to actual percentage returns
        holding_period_profit = torch.exp(log_return_holding) - 1

        # Expected profit (weighted by fill probability)
        expected_profit = fill_probability * holding_period_profit

        # Return negative profit (minimizing this maximizes profit)
        return -expected_profit.mean()
