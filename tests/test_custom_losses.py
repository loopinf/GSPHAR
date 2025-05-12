#!/usr/bin/env python
"""
Test script for custom loss functions.

This script tests the behavior of custom loss functions to ensure they
work as expected, particularly for emphasizing large jumps in the target values.
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.training.custom_losses import (
    WeightedMSELoss,
    AsymmetricMSELoss,
    ThresholdMSELoss,
    HybridLoss
)


def test_weighted_mse_loss():
    """Test the WeightedMSELoss function."""
    print("\n=== Testing WeightedMSELoss ===")
    
    # Create test data
    targets = torch.tensor([0.1, 0.3, 0.6, 1.0, 1.5], dtype=torch.float32)
    
    # Test with perfect predictions
    predictions = targets.clone()
    loss_fn = WeightedMSELoss(threshold=0.5, weight_factor=5.0)
    loss = loss_fn(predictions, targets)
    print(f"Loss with perfect predictions: {loss.item()}")
    assert loss.item() == 0.0, "Loss should be 0 with perfect predictions"
    
    # Test with uniform error
    predictions = targets + 0.1
    loss = loss_fn(predictions, targets)
    standard_mse = ((predictions - targets) ** 2).mean().item()
    print(f"Standard MSE with uniform error: {standard_mse}")
    print(f"Weighted MSE with uniform error: {loss.item()}")
    assert loss.item() > standard_mse, "Weighted loss should be higher than standard MSE"
    
    # Test with errors only on large values
    predictions = targets.clone()
    predictions[targets > 0.5] += 0.1  # Add error only to values above threshold
    loss = loss_fn(predictions, targets)
    standard_mse = ((predictions - targets) ** 2).mean().item()
    print(f"Standard MSE with errors on large values: {standard_mse}")
    print(f"Weighted MSE with errors on large values: {loss.item()}")
    assert loss.item() > standard_mse, "Weighted loss should be higher than standard MSE"
    
    # Test with errors only on small values
    predictions = targets.clone()
    predictions[targets <= 0.5] += 0.1  # Add error only to values below threshold
    loss = loss_fn(predictions, targets)
    standard_mse = ((predictions - targets) ** 2).mean().item()
    print(f"Standard MSE with errors on small values: {standard_mse}")
    print(f"Weighted MSE with errors on small values: {loss.item()}")
    assert loss.item() == standard_mse, "Weighted loss should equal standard MSE when errors are only on small values"
    
    print("WeightedMSELoss tests passed!")


def test_asymmetric_mse_loss():
    """Test the AsymmetricMSELoss function."""
    print("\n=== Testing AsymmetricMSELoss ===")
    
    # Create test data
    targets = torch.tensor([0.1, 0.3, 0.6, 1.0, 1.5], dtype=torch.float32)
    
    # Test with perfect predictions
    predictions = targets.clone()
    loss_fn = AsymmetricMSELoss(under_prediction_factor=3.0)
    loss = loss_fn(predictions, targets)
    print(f"Loss with perfect predictions: {loss.item()}")
    assert loss.item() == 0.0, "Loss should be 0 with perfect predictions"
    
    # Test with underpredictions
    predictions = targets - 0.1
    loss = loss_fn(predictions, targets)
    standard_mse = ((predictions - targets) ** 2).mean().item()
    print(f"Standard MSE with underpredictions: {standard_mse}")
    print(f"Asymmetric MSE with underpredictions: {loss.item()}")
    assert loss.item() > standard_mse, "Asymmetric loss should be higher than standard MSE for underpredictions"
    
    # Test with overpredictions
    predictions = targets + 0.1
    loss = loss_fn(predictions, targets)
    standard_mse = ((predictions - targets) ** 2).mean().item()
    print(f"Standard MSE with overpredictions: {standard_mse}")
    print(f"Asymmetric MSE with overpredictions: {loss.item()}")
    assert loss.item() == standard_mse, "Asymmetric loss should equal standard MSE for overpredictions"
    
    print("AsymmetricMSELoss tests passed!")


def test_threshold_mse_loss():
    """Test the ThresholdMSELoss function."""
    print("\n=== Testing ThresholdMSELoss ===")
    
    # Create test data
    targets = torch.tensor([0.1, 0.3, 0.6, 1.0, 1.5], dtype=torch.float32)
    
    # Test with perfect predictions
    predictions = targets.clone()
    loss_fn = ThresholdMSELoss(thresholds=[0.2, 0.5, 1.0], weights=[1.0, 2.0, 5.0, 10.0])
    loss = loss_fn(predictions, targets)
    print(f"Loss with perfect predictions: {loss.item()}")
    assert loss.item() == 0.0, "Loss should be 0 with perfect predictions"
    
    # Test with uniform error
    predictions = targets + 0.1
    loss = loss_fn(predictions, targets)
    standard_mse = ((predictions - targets) ** 2).mean().item()
    print(f"Standard MSE with uniform error: {standard_mse}")
    print(f"Threshold MSE with uniform error: {loss.item()}")
    assert loss.item() > standard_mse, "Threshold loss should be higher than standard MSE"
    
    print("ThresholdMSELoss tests passed!")


def test_hybrid_loss():
    """Test the HybridLoss function."""
    print("\n=== Testing HybridLoss ===")
    
    # Create test data
    targets = torch.tensor([0.1, 0.3, 0.6, 1.0, 1.5], dtype=torch.float32)
    
    # Test with perfect predictions
    predictions = targets.clone()
    loss_fn = HybridLoss(mse_weight=1.0, large_jump_weight=2.0, threshold=0.5, jump_factor=5.0)
    loss = loss_fn(predictions, targets)
    print(f"Loss with perfect predictions: {loss.item()}")
    assert loss.item() == 0.0, "Loss should be 0 with perfect predictions"
    
    # Test with uniform error
    predictions = targets + 0.1
    loss = loss_fn(predictions, targets)
    standard_mse = ((predictions - targets) ** 2).mean().item()
    print(f"Standard MSE with uniform error: {standard_mse}")
    print(f"Hybrid loss with uniform error: {loss.item()}")
    assert loss.item() > standard_mse, "Hybrid loss should be higher than standard MSE"
    
    print("HybridLoss tests passed!")


def visualize_loss_behavior():
    """Visualize the behavior of different loss functions."""
    print("\n=== Visualizing Loss Functions ===")
    
    # Create a range of target values
    targets = torch.linspace(0.0, 2.0, 100)
    
    # Create predictions with constant error
    error = 0.1
    predictions_over = targets + error
    predictions_under = targets - error
    
    # Initialize loss functions
    mse_loss = torch.nn.MSELoss()
    weighted_loss = WeightedMSELoss(threshold=0.5, weight_factor=5.0)
    asymmetric_loss = AsymmetricMSELoss(under_prediction_factor=3.0)
    threshold_loss = ThresholdMSELoss(thresholds=[0.5, 1.0, 1.5], weights=[1.0, 2.0, 5.0, 10.0])
    hybrid_loss = HybridLoss(mse_weight=1.0, large_jump_weight=2.0, threshold=0.5, jump_factor=5.0)
    
    # Calculate losses for each target value
    mse_over = [mse_loss(predictions_over[i:i+1], targets[i:i+1]).item() for i in range(len(targets))]
    mse_under = [mse_loss(predictions_under[i:i+1], targets[i:i+1]).item() for i in range(len(targets))]
    
    weighted_over = [weighted_loss(predictions_over[i:i+1], targets[i:i+1]).item() for i in range(len(targets))]
    weighted_under = [weighted_loss(predictions_under[i:i+1], targets[i:i+1]).item() for i in range(len(targets))]
    
    asymmetric_over = [asymmetric_loss(predictions_over[i:i+1], targets[i:i+1]).item() for i in range(len(targets))]
    asymmetric_under = [asymmetric_loss(predictions_under[i:i+1], targets[i:i+1]).item() for i in range(len(targets))]
    
    threshold_over = [threshold_loss(predictions_over[i:i+1], targets[i:i+1]).item() for i in range(len(targets))]
    threshold_under = [threshold_loss(predictions_under[i:i+1], targets[i:i+1]).item() for i in range(len(targets))]
    
    hybrid_over = [hybrid_loss(predictions_over[i:i+1], targets[i:i+1]).item() for i in range(len(targets))]
    hybrid_under = [hybrid_loss(predictions_under[i:i+1], targets[i:i+1]).item() for i in range(len(targets))]
    
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(targets.numpy(), mse_over, label='MSE (Over)')
    plt.plot(targets.numpy(), weighted_over, label='Weighted MSE (Over)')
    plt.axvline(x=0.5, color='r', linestyle='--', label='Threshold')
    plt.title('MSE vs Weighted MSE (Overprediction)')
    plt.xlabel('Target Value')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(targets.numpy(), mse_under, label='MSE (Under)')
    plt.plot(targets.numpy(), asymmetric_under, label='Asymmetric MSE (Under)')
    plt.title('MSE vs Asymmetric MSE (Underprediction)')
    plt.xlabel('Target Value')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(targets.numpy(), mse_over, label='MSE (Over)')
    plt.plot(targets.numpy(), threshold_over, label='Threshold MSE (Over)')
    for threshold in [0.5, 1.0, 1.5]:
        plt.axvline(x=threshold, color='r', linestyle='--')
    plt.title('MSE vs Threshold MSE (Overprediction)')
    plt.xlabel('Target Value')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(targets.numpy(), mse_over, label='MSE (Over)')
    plt.plot(targets.numpy(), hybrid_over, label='Hybrid Loss (Over)')
    plt.axvline(x=0.5, color='r', linestyle='--', label='Threshold')
    plt.title('MSE vs Hybrid Loss (Overprediction)')
    plt.xlabel('Target Value')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save the plot with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f'plots/loss_functions_comparison_{timestamp}.png'
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    
    # Show the plot
    plt.close()


def main():
    """Run all tests."""
    print("Testing custom loss functions...")
    
    # Run tests
    test_weighted_mse_loss()
    test_asymmetric_mse_loss()
    test_threshold_mse_loss()
    test_hybrid_loss()
    
    # Visualize loss behavior
    visualize_loss_behavior()
    
    print("\nAll tests completed successfully!")


if __name__ == "__main__":
    main()
