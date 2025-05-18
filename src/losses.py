"""
Custom loss functions for the GSPHAR model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AsymmetricMSELoss(nn.Module):
    """
    Asymmetric Mean Squared Error Loss.

    This loss function penalizes underpredictions more heavily than overpredictions.

    Args:
        alpha (float): Weight for underprediction penalty. Higher values penalize
                      underpredictions more heavily. Default: 1.5
    """
    def __init__(self, alpha=1.5):
        super(AsymmetricMSELoss, self).__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        """
        Forward pass.

        Args:
            pred (torch.Tensor): Predictions.
            target (torch.Tensor): Targets.

        Returns:
            torch.Tensor: Loss value.
        """
        # Calculate the error
        error = target - pred

        # Create a mask for underpredictions (error > 0)
        under_mask = (error > 0).float()

        # Create a mask for overpredictions (error <= 0)
        over_mask = (error <= 0).float()

        # Calculate the loss with asymmetric weighting
        loss = (under_mask * self.alpha * error**2 + over_mask * error**2).mean()

        return loss


class QLIKELoss(nn.Module):
    """
    QLIKE Loss (Quasi-Likelihood Loss).

    This loss is specifically designed for volatility forecasting and is based on
    the quasi-likelihood function. It naturally penalizes underpredictions more
    heavily than overpredictions.

    QLIKE = pred/target - log(pred/target) - 1

    A small epsilon is added to prevent division by zero and log of zero.

    Note: This implementation uses pred/target instead of target/pred to ensure
    numerical stability, as our target values can be very small.
    """
    def __init__(self, epsilon=1e-6):
        super(QLIKELoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        """
        Forward pass.

        Args:
            pred (torch.Tensor): Predictions.
            target (torch.Tensor): Targets.

        Returns:
            torch.Tensor: Loss value.
        """
        # Add epsilon to prevent division by zero and log of zero
        pred_safe = torch.clamp(pred, min=self.epsilon)
        target_safe = torch.clamp(target, min=self.epsilon)

        # Calculate the ratio (inverted from traditional QLIKE for numerical stability)
        ratio = pred_safe / target_safe

        # Calculate the QLIKE loss
        loss = ratio - torch.log(ratio) - 1

        # Return the mean loss
        return loss.mean()


class LogCoshLoss(nn.Module):
    """
    Log-Cosh Loss.

    A smooth approximation of the Huber loss that is differentiable everywhere.
    It behaves like MSE for small errors and like MAE for large errors, but
    transitions smoothly between them.

    log(cosh(x)) ≈ x^2/2 for small x
    log(cosh(x)) ≈ |x| - log(2) for large x
    """
    def __init__(self):
        super(LogCoshLoss, self).__init__()

    def forward(self, pred, target):
        """
        Forward pass.

        Args:
            pred (torch.Tensor): Predictions.
            target (torch.Tensor): Targets.

        Returns:
            torch.Tensor: Loss value.
        """
        # Calculate the error
        error = target - pred

        # Calculate the log-cosh loss
        loss = torch.log(torch.cosh(error))

        # Return the mean loss
        return loss.mean()


class AsymmetricLogCoshLoss(nn.Module):
    """
    Asymmetric Log-Cosh Loss.

    A combination of Log-Cosh loss with asymmetric weighting to penalize
    underpredictions more heavily than overpredictions.

    Args:
        alpha (float): Weight for underprediction penalty. Higher values penalize
                      underpredictions more heavily. Default: 1.5
    """
    def __init__(self, alpha=1.5):
        super(AsymmetricLogCoshLoss, self).__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        """
        Forward pass.

        Args:
            pred (torch.Tensor): Predictions.
            target (torch.Tensor): Targets.

        Returns:
            torch.Tensor: Loss value.
        """
        # Calculate the error
        error = target - pred

        # Create a mask for underpredictions (error > 0)
        under_mask = (error > 0).float()

        # Create a mask for overpredictions (error <= 0)
        over_mask = (error <= 0).float()

        # Calculate the log-cosh with asymmetric weighting
        loss = (under_mask * self.alpha * torch.log(torch.cosh(error)) +
                over_mask * torch.log(torch.cosh(error))).mean()

        return loss


class PinballLoss(nn.Module):
    """
    Pinball Loss (Quantile Loss).

    This loss is used for quantile regression and can be made asymmetric by
    adjusting the quantile parameter.

    Args:
        quantile (float): Quantile to target. Default: 0.5 (median)
                         Values > 0.5 penalize underpredictions more
                         Values < 0.5 penalize overpredictions more
    """
    def __init__(self, quantile=0.5):
        super(PinballLoss, self).__init__()
        self.quantile = quantile

    def forward(self, pred, target):
        """
        Forward pass.

        Args:
            pred (torch.Tensor): Predictions.
            target (torch.Tensor): Targets.

        Returns:
            torch.Tensor: Loss value.
        """
        # Calculate the error
        error = target - pred

        # Calculate the pinball loss
        loss = torch.max(self.quantile * error, (self.quantile - 1) * error)

        # Return the mean loss
        return loss.mean()


# Dictionary mapping loss function names to their classes
LOSS_FUNCTIONS = {
    'mse': nn.MSELoss,
    'mae': nn.L1Loss,
    'huber': nn.HuberLoss,
    'smooth_l1': nn.SmoothL1Loss,
    'asymmetric_mse': AsymmetricMSELoss,
    'qlike': QLIKELoss,
    'log_cosh': LogCoshLoss,
    'asymmetric_log_cosh': AsymmetricLogCoshLoss,
    'pinball': PinballLoss
}


def get_loss_function(loss_name, **kwargs):
    """
    Get a loss function by name.

    Args:
        loss_name (str): Name of the loss function.
        **kwargs: Additional arguments to pass to the loss function.

    Returns:
        nn.Module: Loss function.

    Raises:
        ValueError: If the loss function is not supported.
    """
    if loss_name not in LOSS_FUNCTIONS:
        raise ValueError(f"Unsupported loss function: {loss_name}. "
                         f"Supported loss functions: {list(LOSS_FUNCTIONS.keys())}")

    return LOSS_FUNCTIONS[loss_name](**kwargs)
