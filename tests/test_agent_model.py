#!/usr/bin/env python
"""
Test script for agent model implementation.

This script tests the agent model, trading loss functions, and dataset
to ensure everything works correctly before full training.
"""

import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.flexible_gsphar import FlexibleGSPHAR
from src.models.agent_model import TradingAgentModel, SimpleTradingAgent
from src.agent_trading_loss import AgentTradingLoss, AgentSharpeRatioLoss, AgentAdvancedTradingLoss
from src.data.agent_trading_dataset import load_agent_trading_data


def test_agent_model():
    """Test agent model forward pass."""
    print("🧪 Testing Agent Model...")
    
    batch_size = 4
    n_assets = 5
    history_length = 24
    
    # Create test inputs
    vol_pred = torch.randn(batch_size, n_assets, 1) * 0.1 + 0.05  # Around 5% volatility
    vol_pred_history = torch.randn(batch_size, n_assets, history_length) * 0.1 + 0.05
    
    # Test TradingAgentModel
    agent = TradingAgentModel(n_assets=n_assets, history_length=history_length)
    ratio, direction = agent(vol_pred, vol_pred_history)
    
    print(f"  Input shapes: vol_pred {vol_pred.shape}, vol_pred_history {vol_pred_history.shape}")
    print(f"  Output shapes: ratio {ratio.shape}, direction {direction.shape}")
    print(f"  Ratio range: [{ratio.min().item():.3f}, {ratio.max().item():.3f}]")
    print(f"  Direction range: [{direction.min().item():.3f}, {direction.max().item():.3f}]")
    
    # Test trading signals
    current_price = torch.ones(batch_size, n_assets) * 100.0  # $100 base price
    signals = agent.get_trading_signals(vol_pred, vol_pred_history, current_price)
    
    print(f"  Limit price range: [{signals['limit_price'].min().item():.2f}, {signals['limit_price'].max().item():.2f}]")
    print(f"  Long positions: {signals['is_long'].sum().item()}/{signals['is_long'].numel()}")
    print(f"  Short positions: {signals['is_short'].sum().item()}/{signals['is_short'].numel()}")
    
    # Test SimpleTradingAgent
    simple_agent = SimpleTradingAgent(n_assets=n_assets, history_length=history_length)
    simple_ratio, simple_direction = simple_agent(vol_pred, vol_pred_history)
    
    print(f"  Simple agent output shapes: ratio {simple_ratio.shape}, direction {simple_direction.shape}")
    
    print("✅ Agent model tests passed!")
    return True


def test_trading_loss():
    """Test trading loss functions."""
    print("\n🧪 Testing Trading Loss Functions...")
    
    batch_size = 4
    n_assets = 5
    holding_period = 4
    
    # Create test inputs
    ratio = torch.rand(batch_size, n_assets, 1) * 0.5 + 0.5  # 0.5 to 1.0
    direction = torch.rand(batch_size, n_assets, 1)  # 0 to 1
    
    # Create synthetic OHLCV data
    time_periods = holding_period + 1
    ohlcv_data = torch.zeros(batch_size, n_assets, time_periods, 5)
    
    # Fill with realistic OHLCV data
    base_price = 100.0
    for b in range(batch_size):
        for a in range(n_assets):
            for t in range(time_periods):
                # Generate price with some volatility
                price_change = np.random.normal(0, 0.02)  # 2% volatility
                if t == 0:
                    close_price = base_price
                else:
                    close_price = ohlcv_data[b, a, t-1, 3] * (1 + price_change)
                
                open_price = close_price * np.random.uniform(0.99, 1.01)
                high_price = max(open_price, close_price) * np.random.uniform(1.0, 1.02)
                low_price = min(open_price, close_price) * np.random.uniform(0.98, 1.0)
                volume = np.random.uniform(1000, 10000)
                
                ohlcv_data[b, a, t, 0] = open_price
                ohlcv_data[b, a, t, 1] = high_price
                ohlcv_data[b, a, t, 2] = low_price
                ohlcv_data[b, a, t, 3] = close_price
                ohlcv_data[b, a, t, 4] = volume
    
    # Test AgentTradingLoss
    basic_loss = AgentTradingLoss(holding_period=holding_period)
    loss_value = basic_loss(ratio, direction, ohlcv_data)
    metrics = basic_loss.calculate_metrics(ratio, direction, ohlcv_data)
    
    print(f"  Basic loss value: {loss_value.item():.6f}")
    print(f"  Fill rate: {metrics['fill_rate']:.3f}")
    print(f"  Avg profit when filled: {metrics['avg_profit_when_filled']:.4f}")
    print(f"  Long ratio: {metrics['long_ratio']:.3f}")
    print(f"  Total trades: {metrics['total_trades']}")
    
    # Test AgentSharpeRatioLoss
    sharpe_loss = AgentSharpeRatioLoss(holding_period=holding_period)
    sharpe_value = sharpe_loss(ratio, direction, ohlcv_data)
    print(f"  Sharpe loss value: {sharpe_value.item():.6f}")
    
    # Test AgentAdvancedTradingLoss
    advanced_loss = AgentAdvancedTradingLoss(holding_period=holding_period)
    advanced_value = advanced_loss(ratio, direction, ohlcv_data)
    print(f"  Advanced loss value: {advanced_value.item():.6f}")
    
    print("✅ Trading loss tests passed!")
    return True


def test_dataset():
    """Test agent trading dataset."""
    print("\n🧪 Testing Agent Trading Dataset...")
    
    try:
        # Load dataset with debug mode
        dataset, metadata = load_agent_trading_data(
            volatility_file="data/crypto_rv1h_38_20200822_20250116.csv",
            lags=[1, 4, 24],
            holding_period=4,
            history_length=24,
            debug=True
        )
        
        print(f"  Dataset length: {len(dataset)}")
        print(f"  Assets: {metadata['n_assets']}")
        print(f"  Date range: {metadata['date_range']}")
        
        # Test getting a sample
        sample = dataset[0]
        sample_info = dataset.get_sample_info(0)
        
        print(f"  Sample keys: {list(sample.keys())}")
        print(f"  x_lags shapes: {[x.shape for x in sample['x_lags']]}")
        print(f"  vol_targets shape: {sample['vol_targets'].shape}")
        print(f"  vol_pred_history shape: {sample['vol_pred_history'].shape}")
        print(f"  ohlcv_data shape: {sample['ohlcv_data'].shape}")
        print(f"  Prediction time: {sample_info['prediction_time']}")
        
        # Test data loader
        train_loader = data.DataLoader(dataset, batch_size=4, shuffle=True)
        batch = next(iter(train_loader))
        
        print(f"  Batch x_lags shapes: {[x.shape for x in batch['x_lags']]}")
        print(f"  Batch vol_targets shape: {batch['vol_targets'].shape}")
        print(f"  Batch vol_pred_history shape: {batch['vol_pred_history'].shape}")
        print(f"  Batch ohlcv_data shape: {batch['ohlcv_data'].shape}")
        
        print("✅ Dataset tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        return False


def test_integration():
    """Test full integration of all components."""
    print("\n🧪 Testing Full Integration...")
    
    try:
        # Load small dataset
        dataset, metadata = load_agent_trading_data(
            volatility_file="data/crypto_rv1h_38_20200822_20250116.csv",
            lags=[1, 4, 24],
            holding_period=4,
            history_length=24,
            debug=True
        )
        
        # Create data loader
        train_loader = data.DataLoader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(train_loader))
        
        # Create models
        n_assets = metadata['n_assets']
        
        # Create correlation matrix for volatility model
        vol_df = pd.read_csv("data/crypto_rv1h_38_20200822_20250116.csv", index_col=0, parse_dates=True)
        corr_sample = vol_df.iloc[-100:] if len(vol_df) > 100 else vol_df
        corr_matrix = corr_sample.corr().values
        A = (corr_matrix + corr_matrix.T) / 2
        
        volatility_model = FlexibleGSPHAR(
            lags=[1, 4, 24],
            output_dim=1,
            filter_size=n_assets,
            A=A
        )
        
        agent_model = TradingAgentModel(
            n_assets=n_assets,
            history_length=24
        )
        
        # Test forward pass
        x_lags = batch['x_lags']
        vol_pred_history = batch['vol_pred_history']
        ohlcv_data = batch['ohlcv_data']
        
        # Volatility prediction
        vol_pred = volatility_model(*x_lags)
        print(f"  Volatility prediction shape: {vol_pred.shape}")
        
        # Agent prediction
        ratio, direction = agent_model(vol_pred, vol_pred_history)
        print(f"  Agent outputs: ratio {ratio.shape}, direction {direction.shape}")
        
        # Loss calculation
        loss_fn = AgentTradingLoss(holding_period=4)
        loss = loss_fn(ratio, direction, ohlcv_data)
        print(f"  Loss value: {loss.item():.6f}")
        
        # Test backward pass
        loss.backward()
        print(f"  Backward pass successful")
        
        # Check gradients
        agent_grad_norm = torch.norm(torch.stack([p.grad.norm() for p in agent_model.parameters() if p.grad is not None]))
        print(f"  Agent gradient norm: {agent_grad_norm.item():.6f}")
        
        print("✅ Integration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🚀 AGENT MODEL TESTING")
    print("=" * 50)
    
    tests = [
        test_agent_model,
        test_trading_loss,
        test_dataset,
        test_integration
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
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
        print("🎉 All tests passed! Ready for training.")
    else:
        print("⚠️  Some tests failed. Please fix issues before training.")


if __name__ == "__main__":
    main()
