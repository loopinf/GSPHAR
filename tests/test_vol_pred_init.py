#!/usr/bin/env python
"""
Test script for vol_pred initialization strategy.

This script tests that the agent model properly initializes with the
previous vol_pred strategy and can learn from there.
"""

import torch
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.agent_model import TradingAgentModel


def test_vol_pred_initialization():
    """Test that vol_pred initialization works correctly."""
    print("🧪 Testing vol_pred initialization...")
    
    batch_size = 4
    n_assets = 5
    history_length = 24
    
    # Create test inputs with realistic volatility values
    vol_pred = torch.tensor([
        [0.02],  # 2% volatility
        [0.05],  # 5% volatility  
        [0.10],  # 10% volatility
        [0.15],  # 15% volatility
        [0.03]   # 3% volatility
    ]).unsqueeze(0).repeat(batch_size, 1, 1)  # [batch_size, n_assets, 1]
    
    vol_pred_history = torch.randn(batch_size, n_assets, history_length) * 0.02 + 0.05
    
    # Test agent with vol_pred initialization
    agent_with_init = TradingAgentModel(
        n_assets=n_assets, 
        history_length=history_length,
        init_with_vol_pred=True
    )
    
    # Test agent without initialization (random)
    agent_without_init = TradingAgentModel(
        n_assets=n_assets, 
        history_length=history_length,
        init_with_vol_pred=False
    )
    
    print(f"  Input vol_pred: {vol_pred[0, :, 0].tolist()}")
    
    # Test forward pass with vol_pred initialization
    ratio_init, direction_init = agent_with_init.forward_with_vol_pred_init(vol_pred, vol_pred_history)
    
    # Test regular forward pass
    ratio_regular, direction_regular = agent_with_init.forward(vol_pred, vol_pred_history)
    ratio_random, direction_random = agent_without_init.forward(vol_pred, vol_pred_history)
    
    print(f"\n📊 Results for first sample:")
    print(f"  Expected ratios (1 - vol_pred): {(1.0 - vol_pred[0, :, 0]).tolist()}")
    print(f"  Vol_pred init ratios: {ratio_init[0, :, 0].tolist()}")
    print(f"  Regular ratios: {ratio_regular[0, :, 0].tolist()}")
    print(f"  Random ratios: {ratio_random[0, :, 0].tolist()}")
    
    print(f"\n  Vol_pred init directions: {direction_init[0, :, 0].tolist()}")
    print(f"  Regular directions: {direction_regular[0, :, 0].tolist()}")
    print(f"  Random directions: {direction_random[0, :, 0].tolist()}")
    
    # Check that vol_pred initialization produces ratios closer to (1 - vol_pred)
    expected_ratios = torch.clamp(1.0 - vol_pred[0, :, 0], 0.85, 0.99)
    
    init_error = torch.abs(ratio_init[0, :, 0] - expected_ratios).mean()
    regular_error = torch.abs(ratio_regular[0, :, 0] - expected_ratios).mean()
    random_error = torch.abs(ratio_random[0, :, 0] - expected_ratios).mean()
    
    print(f"\n📏 Distance from expected (1 - vol_pred):")
    print(f"  Vol_pred init error: {init_error:.4f}")
    print(f"  Regular error: {regular_error:.4f}")
    print(f"  Random error: {random_error:.4f}")
    
    # Check that initialized agent favors long positions
    init_long_ratio = (direction_init > 0.5).float().mean()
    random_long_ratio = (direction_random > 0.5).float().mean()
    
    print(f"\n📈 Long position preference:")
    print(f"  Vol_pred init long ratio: {init_long_ratio:.3f}")
    print(f"  Random long ratio: {random_long_ratio:.3f}")
    
    # Assertions
    assert init_error < regular_error, "Vol_pred init should be closer to expected ratios"
    assert init_error < random_error, "Vol_pred init should be closer to expected ratios than random"
    assert init_long_ratio > 0.7, "Initialized agent should favor long positions"
    
    print("✅ Vol_pred initialization test passed!")
    return True


def test_trading_signals_with_init():
    """Test trading signals with vol_pred initialization."""
    print("\n🧪 Testing trading signals with vol_pred initialization...")
    
    batch_size = 2
    n_assets = 3
    
    # Create test data
    vol_pred = torch.tensor([[[0.05], [0.10], [0.02]]]).repeat(batch_size, 1, 1)
    vol_pred_history = torch.randn(batch_size, n_assets, 24) * 0.02 + 0.05
    current_price = torch.tensor([[100.0, 200.0, 50.0]]).repeat(batch_size, 1)
    
    # Create agent
    agent = TradingAgentModel(n_assets=n_assets, init_with_vol_pred=True)
    
    # Get trading signals
    signals = agent.get_trading_signals(vol_pred, vol_pred_history, current_price)
    
    print(f"  Current prices: {current_price[0].tolist()}")
    print(f"  Vol predictions: {vol_pred[0, :, 0].tolist()}")
    print(f"  Limit prices: {signals['limit_price'][0].tolist()}")
    print(f"  Ratios: {signals['ratio'][0, :, 0].tolist()}")
    print(f"  Directions: {signals['direction'][0, :, 0].tolist()}")
    print(f"  Is long: {signals['is_long'][0].tolist()}")
    print(f"  Is short: {signals['is_short'][0].tolist()}")
    
    # Check that limit prices are reasonable
    ratios = signals['ratio'][0, :, 0]
    expected_limit_prices = current_price[0] * ratios
    actual_limit_prices = signals['limit_price'][0]
    
    # For mostly long positions, limit prices should be below current prices
    long_mask = signals['is_long'][0]
    if long_mask.any():
        long_limits = actual_limit_prices[long_mask]
        long_current = current_price[0][long_mask]
        assert (long_limits <= long_current).all(), "Long limit prices should be below current prices"
    
    print("✅ Trading signals test passed!")
    return True


def compare_strategies():
    """Compare vol_pred initialization with previous strategy."""
    print("\n🔄 Comparing with previous vol_pred strategy...")
    
    # Simulate previous strategy
    vol_pred_value = 0.05  # 5% volatility
    current_price = 100.0
    
    # Previous strategy: limit_price = current_price * (1 - vol_pred)
    previous_limit_price = current_price * (1 - vol_pred_value)
    
    # New agent strategy
    vol_pred_tensor = torch.tensor([[[vol_pred_value]]])
    vol_pred_history = torch.randn(1, 1, 24) * 0.02 + 0.05
    current_price_tensor = torch.tensor([[current_price]])
    
    agent = TradingAgentModel(n_assets=1, init_with_vol_pred=True)
    signals = agent.get_trading_signals(vol_pred_tensor, vol_pred_history, current_price_tensor)
    
    agent_limit_price = signals['limit_price'][0, 0].item()
    agent_ratio = signals['ratio'][0, 0, 0].item()
    
    print(f"  Vol prediction: {vol_pred_value}")
    print(f"  Current price: ${current_price}")
    print(f"  Previous strategy limit: ${previous_limit_price:.2f}")
    print(f"  Agent limit price: ${agent_limit_price:.2f}")
    print(f"  Agent ratio: {agent_ratio:.3f}")
    print(f"  Expected ratio (1-vol): {1-vol_pred_value:.3f}")
    
    # They should be similar (within initialization tolerance)
    price_diff = abs(agent_limit_price - previous_limit_price)
    print(f"  Price difference: ${price_diff:.2f}")
    
    print("✅ Strategy comparison completed!")
    return True


def main():
    """Run all vol_pred initialization tests."""
    print("🚀 VOL_PRED INITIALIZATION TESTING")
    print("=" * 50)
    
    tests = [
        test_vol_pred_initialization,
        test_trading_signals_with_init,
        compare_strategies
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{i+1}. {test.__name__}: {status}")
    
    total_passed = sum(results)
    total_tests = len(results)
    print(f"\nOverall: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("🎉 All vol_pred initialization tests passed!")
    else:
        print("⚠️  Some tests failed. Check implementation.")


if __name__ == "__main__":
    main()
