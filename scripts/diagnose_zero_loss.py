#!/usr/bin/env python
"""
Diagnose why the trading loss went to zero during training.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.trading_loss import TradingStrategyLoss, convert_pct_change_to_log_returns

def analyze_zero_loss_problem():
    """Analyze why the trading loss goes to zero."""
    
    print("="*80)
    print("DIAGNOSING ZERO LOSS PROBLEM")
    print("="*80)
    
    # Create trading loss function
    trading_loss = TradingStrategyLoss(alpha=1.0, beta=0.1, gamma=2.0, holding_period=24)
    
    print("\n1. TESTING WITH REALISTIC VOLATILITY PREDICTIONS")
    print("-" * 50)
    
    # Test with realistic volatility predictions (small values)
    realistic_vol_preds = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    
    for vol_pred in realistic_vol_preds:
        print(f"\nTesting vol_pred = {vol_pred:.3f} ({vol_pred*100:.1f}%)")
        
        # Create test scenarios
        scenarios = [
            {"name": "Small drop", "next_return": -vol_pred * 0.8, "holding": [0.0001] * 24},
            {"name": "Exact drop", "next_return": -vol_pred, "holding": [0.0001] * 24},
            {"name": "Large drop", "next_return": -vol_pred * 1.5, "holding": [0.0001] * 24},
        ]
        
        for scenario in scenarios:
            vol_tensor = torch.tensor([vol_pred], dtype=torch.float32)
            returns_tensor = torch.tensor([[scenario["next_return"]] + scenario["holding"]], dtype=torch.float32)
            
            loss = trading_loss(vol_tensor, returns_tensor)
            
            # Check for problematic values
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  {scenario['name']}: NaN/Inf detected!")
            elif loss.item() == 0.0:
                print(f"  {scenario['name']}: Loss = 0.000000 (PROBLEM!)")
            else:
                print(f"  {scenario['name']}: Loss = {loss.item():.6f}")
    
    print("\n2. TESTING LOG OPERATIONS")
    print("-" * 50)
    
    # Test the log operations that might cause issues
    vol_preds = [0.001, 0.01, 0.1, 0.5, 0.99, 0.999]
    
    for vol_pred in vol_preds:
        try:
            log_threshold = torch.log(1 - vol_pred)
            print(f"vol_pred = {vol_pred:.3f} -> log(1 - vol_pred) = {log_threshold.item():.6f}")
            
            if torch.isnan(log_threshold) or torch.isinf(log_threshold):
                print(f"  WARNING: Invalid log value for vol_pred = {vol_pred}")
                
        except Exception as e:
            print(f"  ERROR with vol_pred = {vol_pred}: {e}")
    
    print("\n3. TESTING COMPONENT CALCULATION")
    print("-" * 50)
    
    # Test individual components
    vol_pred = 0.05
    next_return = -0.04  # Order should fill
    holding_returns = [-0.001] * 24  # Small losses
    
    vol_tensor = torch.tensor([vol_pred], dtype=torch.float32)
    returns_tensor = torch.tensor([[next_return] + holding_returns], dtype=torch.float32)
    
    # Manual calculation
    log_entry_threshold = torch.log(1 - vol_tensor)
    log_return_next = returns_tensor[:, 0]
    log_return_holding = torch.sum(returns_tensor[:, 1:], dim=1)
    
    print(f"vol_pred: {vol_pred}")
    print(f"log_entry_threshold: {log_entry_threshold.item():.6f}")
    print(f"log_return_next: {log_return_next.item():.6f}")
    print(f"log_return_holding: {log_return_holding.item():.6f}")
    
    # Fill loss
    fill_loss = torch.max(torch.zeros_like(vol_tensor), log_entry_threshold - log_return_next)**2
    print(f"fill_loss: {fill_loss.item():.6f}")
    
    # Order filled?
    filled_orders = (log_return_next <= log_entry_threshold).float()
    print(f"filled_orders: {filled_orders.item()}")
    
    # Profit loss
    profit_loss = -filled_orders * log_return_holding
    print(f"profit_loss: {profit_loss.item():.6f}")
    
    # Avoidance loss
    avoidance_loss = torch.max(torch.zeros_like(vol_tensor), -filled_orders * log_return_holding)**2
    print(f"avoidance_loss: {avoidance_loss.item():.6f}")
    
    # Total
    total_loss = fill_loss + profit_loss + avoidance_loss
    print(f"total_loss (manual): {total_loss.item():.6f}")
    
    # Using the function
    function_loss = trading_loss(vol_tensor, returns_tensor)
    print(f"total_loss (function): {function_loss.item():.6f}")
    
    print("\n4. TESTING GRADIENT FLOW")
    print("-" * 50)
    
    # Test if gradients are flowing properly
    vol_pred = torch.tensor([0.05], dtype=torch.float32, requires_grad=True)
    returns_tensor = torch.tensor([[next_return] + holding_returns], dtype=torch.float32)
    
    loss = trading_loss(vol_pred, returns_tensor)
    print(f"Loss with gradients: {loss.item():.6f}")
    
    if loss.item() > 0:
        loss.backward()
        print(f"Gradient: {vol_pred.grad.item():.6f}")
    else:
        print("Cannot compute gradient - loss is zero!")
    
    print("\n5. TESTING EXTREME CASES")
    print("-" * 50)
    
    # Test extreme cases that might cause numerical issues
    extreme_cases = [
        {"vol": 0.0, "next": 0.0, "name": "All zeros"},
        {"vol": 1e-8, "next": -1e-8, "name": "Very small values"},
        {"vol": 0.999, "next": -0.5, "name": "Very high volatility"},
        {"vol": 0.05, "next": 0.05, "name": "Positive return (no fill)"},
    ]
    
    for case in extreme_cases:
        vol_tensor = torch.tensor([case["vol"]], dtype=torch.float32)
        returns_tensor = torch.tensor([[case["next"]] + [0.0] * 24], dtype=torch.float32)
        
        try:
            loss = trading_loss(vol_tensor, returns_tensor)
            print(f"{case['name']}: Loss = {loss.item():.8f}")
        except Exception as e:
            print(f"{case['name']}: ERROR - {e}")

def test_model_predictions():
    """Test what happens when model predictions are very small."""
    
    print("\n" + "="*80)
    print("TESTING MODEL PREDICTION BEHAVIOR")
    print("="*80)
    
    # Simulate model predictions that might be causing zero loss
    model_predictions = [
        torch.tensor([[0.0] * 38], dtype=torch.float32),  # All zeros
        torch.tensor([[1e-8] * 38], dtype=torch.float32),  # Very small
        torch.tensor([[1e-4] * 38], dtype=torch.float32),  # Small but reasonable
        torch.tensor([[-0.01] * 38], dtype=torch.float32),  # Negative (problematic)
    ]
    
    # Create dummy log returns
    dummy_returns = torch.randn(1, 38, 25) * 0.01  # Small random returns
    
    trading_loss = TradingStrategyLoss(alpha=1.0, beta=0.1, gamma=2.0, holding_period=24)
    
    for i, pred in enumerate(model_predictions):
        print(f"\nTest {i+1}: Predictions = {pred[0, 0].item():.8f}")
        
        total_loss = 0.0
        valid_samples = 0
        
        for symbol_idx in range(38):
            symbol_pred = pred[0, symbol_idx].unsqueeze(0)
            symbol_returns = dummy_returns[0, symbol_idx, :].unsqueeze(0)
            
            try:
                sample_loss = trading_loss(symbol_pred, symbol_returns)
                if not torch.isnan(sample_loss) and not torch.isinf(sample_loss):
                    total_loss += sample_loss.item()
                    valid_samples += 1
            except Exception as e:
                print(f"  Error for symbol {symbol_idx}: {e}")
        
        if valid_samples > 0:
            avg_loss = total_loss / valid_samples
            print(f"  Average loss: {avg_loss:.8f}")
            print(f"  Valid samples: {valid_samples}/38")
        else:
            print(f"  No valid samples!")

def suggest_fixes():
    """Suggest fixes for the zero loss problem."""
    
    print("\n" + "="*80)
    print("SUGGESTED FIXES")
    print("="*80)
    
    print("""
PROBLEM DIAGNOSIS:
The loss is going to zero likely due to one or more of these issues:

1. MODEL PREDICTIONS TOO SMALL:
   - Model is predicting very small volatilities (near zero)
   - log(1 - vol_pred) ≈ -vol_pred for small vol_pred
   - This makes the loss components very small

2. NUMERICAL PRECISION ISSUES:
   - Very small numbers in loss calculation
   - Gradients become too small to update parameters
   - Model gets stuck in local minimum

3. LOSS FUNCTION DESIGN:
   - Loss can be negative (profit loss component)
   - Components may cancel each other out
   - Avoidance loss only activates for losses

SUGGESTED FIXES:

1. SCALE THE LOSS FUNCTION:
   ```python
   # Multiply final loss by a scaling factor
   total_loss = (fill_loss + profit_loss + avoidance_loss) * 1000
   ```

2. ADD REGULARIZATION:
   ```python
   # Add L2 regularization to prevent tiny predictions
   reg_loss = torch.mean(vol_pred ** 2) * 0.01
   total_loss = trading_loss + reg_loss
   ```

3. CLAMP PREDICTIONS:
   ```python
   # Ensure predictions are in reasonable range
   vol_pred = torch.clamp(vol_pred, min=0.001, max=0.5)
   ```

4. USE DIFFERENT LOSS FORMULATION:
   ```python
   # Use absolute values to prevent negative losses
   profit_loss = torch.abs(filled_orders * log_return_holding)
   ```

5. ADJUST LEARNING RATE:
   ```python
   # Use higher learning rate to overcome small gradients
   optimizer = optim.Adam(model.parameters(), lr=0.001)  # Instead of 0.0001
   ```

6. MODIFY TRAINING DATA:
   ```python
   # Ensure log returns have sufficient magnitude
   log_returns = log_returns * 10  # Scale up returns
   ```
    """)

def main():
    """Main function."""
    analyze_zero_loss_problem()
    test_model_predictions()
    suggest_fixes()

if __name__ == '__main__':
    main()
