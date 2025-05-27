#!/usr/bin/env python
"""
Test the profit maximization loss functions.
"""

import torch
import numpy as np
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.trading_loss import TradingStrategyLoss, MaximizeProfitLoss, SimpleProfitMaximizationLoss

def test_profit_maximization_losses():
    """Test the profit maximization loss functions."""

    print("="*80)
    print("TESTING PROFIT MAXIMIZATION LOSS FUNCTIONS")
    print("="*80)

    # Create loss functions
    simple_profit_loss = SimpleProfitMaximizationLoss(holding_period=24)
    advanced_profit_loss = MaximizeProfitLoss(holding_period=24, risk_penalty=2.0, no_fill_penalty=0.01)
    original_trading_loss = TradingStrategyLoss(alpha=1.0, beta=1.0, gamma=2.0, holding_period=24)

    # Test scenarios
    scenarios = [
        {
            "name": "High Profit Scenario",
            "vol_pred": 0.05,  # Predict 5% drop
            "actual_next": -0.06,  # Actual 6% drop (order fills)
            "holding_returns": [0.002] * 24,  # 0.2% per hour = 4.8% total
            "expected_profit": 0.048  # Should be around 4.8%
        },
        {
            "name": "Small Profit Scenario",
            "vol_pred": 0.03,  # Predict 3% drop
            "actual_next": -0.035,  # Actual 3.5% drop (order fills)
            "holding_returns": [0.0005] * 24,  # 0.05% per hour = 1.2% total
            "expected_profit": 0.012  # Should be around 1.2%
        },
        {
            "name": "Loss Scenario",
            "vol_pred": 0.04,  # Predict 4% drop
            "actual_next": -0.045,  # Actual 4.5% drop (order fills)
            "holding_returns": [-0.001] * 24,  # -0.1% per hour = -2.4% total
            "expected_profit": -0.024  # Should be around -2.4%
        },
        {
            "name": "No Fill Scenario",
            "vol_pred": 0.08,  # Predict 8% drop
            "actual_next": -0.03,  # Actual 3% drop (order doesn't fill)
            "holding_returns": [0.001] * 24,  # Irrelevant
            "expected_profit": 0.0  # No profit since no fill
        }
    ]

    print(f"Testing {len(scenarios)} scenarios with 3 loss functions...\n")

    for i, scenario in enumerate(scenarios, 1):
        print(f"SCENARIO {i}: {scenario['name']}")
        print("-" * 50)

        # Create tensors
        vol_pred = torch.tensor([scenario['vol_pred']], dtype=torch.float32)
        log_returns = torch.tensor([[scenario['actual_next']] + scenario['holding_returns']],
                                 dtype=torch.float32)

        # Calculate actual profit manually for verification
        vol_pred_clamped = torch.clamp(torch.abs(vol_pred), 0.001, 0.5)
        log_entry_threshold = torch.log(1 - vol_pred_clamped)
        log_return_next = log_returns[:, 0]
        filled_orders = (log_return_next <= log_entry_threshold).float()
        log_return_holding = torch.sum(log_returns[:, 1:], dim=1)
        actual_profit = filled_orders * (torch.exp(log_return_holding) - 1)

        print(f"Input:")
        print(f"  Predicted volatility: {scenario['vol_pred']:.3f} ({scenario['vol_pred']*100:.1f}%)")
        print(f"  Actual next return: {scenario['actual_next']:.3f} ({scenario['actual_next']*100:.1f}%)")
        print(f"  Holding period total: {sum(scenario['holding_returns']):.3f} ({sum(scenario['holding_returns'])*100:.1f}%)")
        print(f"  Order fills: {'Yes' if filled_orders.item() > 0 else 'No'}")
        print(f"  Actual profit: {actual_profit.item():.4f} ({actual_profit.item()*100:.2f}%)")
        print(f"  Expected profit: {scenario['expected_profit']:.4f} ({scenario['expected_profit']*100:.2f}%)")

        # Test loss functions
        print(f"\nLoss Function Results:")

        # Simple Profit Maximization Loss
        simple_loss = simple_profit_loss(vol_pred, log_returns)
        print(f"  Simple Profit Loss: {simple_loss.item():.6f} (negative profit)")
        print(f"    → Implied profit: {-simple_loss.item():.4f} ({-simple_loss.item()*100:.2f}%)")

        # Advanced Profit Maximization Loss
        advanced_loss = advanced_profit_loss(vol_pred, log_returns)
        print(f"  Advanced Profit Loss: {advanced_loss.item():.6f} (with risk penalties)")
        print(f"    → Implied profit: {-advanced_loss.item():.4f} ({-advanced_loss.item()*100:.2f}%)")

        # Original Trading Loss (for comparison)
        original_loss = original_trading_loss(vol_pred, log_returns)
        print(f"  Original Trading Loss: {original_loss.item():.6f} (multi-component)")

        # Verify consistency
        profit_diff = abs(actual_profit.item() - (-simple_loss.item()))
        if profit_diff < 0.0001:
            print(f"  ✅ Simple loss correctly represents negative profit")
        else:
            print(f"  ❌ Profit mismatch: {profit_diff:.6f}")

        print()

    print("="*80)
    print("GRADIENT TEST")
    print("="*80)

    # Test gradients to ensure they flow properly
    loss_functions = [
        ("Simple Profit Loss", simple_profit_loss),
        ("Advanced Profit Loss", advanced_profit_loss),
        ("Original Trading Loss", original_trading_loss)
    ]

    for name, loss_fn in loss_functions:
        vol_pred = torch.tensor([0.05], dtype=torch.float32, requires_grad=True)
        log_returns = torch.tensor([[-0.06] + [0.001] * 24], dtype=torch.float32)

        loss = loss_fn(vol_pred, log_returns)

        if loss.requires_grad:
            loss.backward()

            print(f"{name}:")
            print(f"  Loss: {loss.item():.6f}")
            print(f"  Gradient: {vol_pred.grad.item():.6f}")
            print(f"  Gradient direction: {'Increase vol_pred' if vol_pred.grad.item() > 0 else 'Decrease vol_pred'}")
        else:
            print(f"{name}:")
            print(f"  Loss: {loss.item():.6f}")
            print(f"  No gradient (loss doesn't require grad)")
        print()

def test_optimization_direction():
    """Test that the loss functions optimize in the right direction."""

    print("="*80)
    print("OPTIMIZATION DIRECTION TEST")
    print("="*80)

    simple_profit_loss = SimpleProfitMaximizationLoss(holding_period=24)

    # Create a scenario where we can test optimization
    # If we predict too high volatility, we miss fills
    # If we predict too low volatility, we get worse entry prices

    base_scenario = {
        "actual_next": -0.05,  # 5% drop actually happens
        "holding_returns": [0.001] * 24  # Positive holding period return
    }

    vol_predictions = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]

    print("Testing different volatility predictions for same market scenario:")
    print(f"Actual next return: {base_scenario['actual_next']:.1%}")
    print(f"Holding period return: {sum(base_scenario['holding_returns']):.1%}")
    print()

    losses = []
    profits = []

    for vol_pred in vol_predictions:
        vol_tensor = torch.tensor([vol_pred], dtype=torch.float32)
        returns_tensor = torch.tensor([[base_scenario['actual_next']] + base_scenario['holding_returns']],
                                    dtype=torch.float32)

        loss = simple_profit_loss(vol_tensor, returns_tensor)
        profit = -loss.item()  # Convert loss back to profit

        losses.append(loss.item())
        profits.append(profit)

        # Check if order fills
        log_entry_threshold = torch.log(1 - vol_tensor)
        log_return_next = returns_tensor[:, 0]
        fills = log_return_next <= log_entry_threshold

        print(f"Vol Pred: {vol_pred:.1%} → Loss: {loss.item():.6f}, Profit: {profit:.4f}, Fills: {'Yes' if fills else 'No'}")

    # Find optimal prediction
    best_idx = np.argmin(losses)
    best_vol_pred = vol_predictions[best_idx]
    best_profit = profits[best_idx]

    print(f"\nOptimal volatility prediction: {best_vol_pred:.1%}")
    print(f"Maximum profit: {best_profit:.4f} ({best_profit*100:.2f}%)")
    print(f"This should be close to the actual drop of {abs(base_scenario['actual_next']):.1%}")

def main():
    """Main function."""
    test_profit_maximization_losses()
    test_optimization_direction()

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("""
The profit maximization loss functions work by:

1. SIMPLE PROFIT MAXIMIZATION LOSS:
   - Calculates actual trading profit
   - Returns negative profit as loss
   - Minimizing loss = Maximizing profit
   - Direct and straightforward

2. ADVANCED PROFIT MAXIMIZATION LOSS:
   - Includes risk penalties for losses
   - Adds penalty for missed fills
   - More sophisticated risk management
   - Better for conservative strategies

3. OPTIMIZATION BEHAVIOR:
   - Model learns to predict volatility that maximizes profit
   - Balances between getting fills and good entry prices
   - Naturally learns optimal volatility predictions

NEXT STEPS:
- Run the full training script: python scripts/train_profit_maximization.py
- Compare results between different loss functions
- Analyze which approach gives best real-world profits
    """)

if __name__ == '__main__':
    main()
