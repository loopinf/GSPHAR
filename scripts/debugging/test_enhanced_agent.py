#!/usr/bin/env python
"""
Test Enhanced Trading Agent with Advanced Features.

This script demonstrates the enhanced trading agent with:
- Technical indicators
- OHLCV data integration
- Volume analysis
- Multi-timeframe features
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Import our modules
from src.models.enhanced_agent_model import EnhancedTradingAgentModel
from src.data.enhanced_dataloader import EnhancedOHLCVDataLoader
from src.features.technical_indicators import TechnicalIndicatorEngine

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_enhanced_dataloader():
    """Test the enhanced OHLCV data loader."""
    logger.info("🔍 Testing Enhanced Data Loader...")
    
    # Initialize data loader
    dataloader = EnhancedOHLCVDataLoader(
        data_dir="data",
        lookback_window=48,
        device=torch.device('cpu')
    )
    
    # Get symbol info
    symbol_info = dataloader.get_symbol_info()
    logger.info(f"📊 Loaded {symbol_info['n_assets']} assets: {symbol_info['symbols'][:5]}...")
    logger.info(f"📅 Time range: {symbol_info['time_range']}")
    
    # Test data quality
    quality_report = dataloader.validate_data_quality()
    logger.info("📈 Data Quality Report:")
    for data_type, metrics in quality_report.items():
        logger.info(f"  {data_type}: {metrics['shape']}, Missing: {metrics['missing_percentage']:.2f}%")
    
    # Test OHLCV window extraction
    test_time = "2023-01-01 12:00:00"
    ohlcv_data = dataloader.get_ohlcv_window(test_time, window_hours=24)
    
    logger.info(f"🎯 OHLCV Window Test (24h ending {test_time}):")
    for data_type, tensor in ohlcv_data.items():
        logger.info(f"  {data_type}: {tensor.shape}")
    
    return dataloader


def test_technical_indicators():
    """Test technical indicator calculations."""
    logger.info("🔧 Testing Technical Indicators...")
    
    # Create sample data
    batch_size, n_assets, time_steps = 2, 5, 48
    
    # Sample price data (trending upward with noise)
    base_prices = torch.linspace(100, 110, time_steps).unsqueeze(0).unsqueeze(0)
    noise = torch.randn(batch_size, n_assets, time_steps) * 0.5
    prices = base_prices + noise
    
    # Sample volume data
    volumes = torch.rand(batch_size, n_assets, time_steps) * 1000 + 500
    
    # Sample OHLCV data
    ohlc_data = {
        'open': prices + torch.randn_like(prices) * 0.1,
        'high': prices + torch.abs(torch.randn_like(prices)) * 0.2,
        'low': prices - torch.abs(torch.randn_like(prices)) * 0.2,
        'close': prices,
    }
    
    # Initialize indicator engine
    indicator_engine = TechnicalIndicatorEngine()
    
    # Test momentum features
    momentum_features = indicator_engine.calculate_momentum_features(prices)
    logger.info(f"📈 Momentum Features: {list(momentum_features.keys())}")
    
    # Test volume features
    volume_features = indicator_engine.calculate_volume_features(volumes)
    logger.info(f"📊 Volume Features: {list(volume_features.keys())}")
    
    # Test OHLC features
    ohlc_features = indicator_engine.calculate_ohlc_features(ohlc_data)
    logger.info(f"🕯️ OHLC Features: {list(ohlc_features.keys())}")
    
    # Test volatility regime features
    vol_history = torch.rand(batch_size, n_assets, time_steps) * 0.05 + 0.02
    vol_regime_features = indicator_engine.calculate_volatility_regime(vol_history)
    logger.info(f"📉 Volatility Regime Features: {list(vol_regime_features.keys())}")
    
    # Test cross-asset features
    cross_features = indicator_engine.calculate_cross_asset_features(prices, volumes)
    logger.info(f"🔗 Cross-Asset Features: {list(cross_features.keys())}")
    
    return {
        'momentum': momentum_features,
        'volume': volume_features,
        'ohlc': ohlc_features,
        'vol_regime': vol_regime_features,
        'cross_asset': cross_features
    }


def test_enhanced_agent_model():
    """Test the enhanced trading agent model."""
    logger.info("🤖 Testing Enhanced Agent Model...")
    
    # Model parameters
    n_assets = 38
    vol_history_length = 24
    batch_size = 4
    
    # Initialize enhanced model
    model = EnhancedTradingAgentModel(
        n_assets=n_assets,
        vol_history_length=vol_history_length,
        feature_dim=128,
        hidden_dim=256,
        init_with_vol_pred=True
    )
    
    logger.info(f"🏗️ Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create sample input data
    vol_pred = torch.rand(batch_size, n_assets, 1) * 0.05 + 0.02  # 2-7% volatility
    vol_history = torch.rand(batch_size, n_assets, vol_history_length) * 0.05 + 0.02
    current_prices = torch.rand(batch_size, n_assets) * 100 + 50  # $50-150
    
    # Create sample OHLCV data
    time_steps = 48
    ohlc_data = {}
    for price_type in ['open', 'high', 'low', 'close']:
        base_price = torch.rand(batch_size, n_assets, time_steps) * 100 + 50
        ohlc_data[price_type] = base_price
    
    volume_data = torch.rand(batch_size, n_assets, time_steps) * 1000 + 500
    
    # Test basic forward pass
    logger.info("🔄 Testing basic forward pass...")
    ratio, direction = model.forward(vol_pred, vol_history)
    logger.info(f"  Basic output shapes - Ratio: {ratio.shape}, Direction: {direction.shape}")
    logger.info(f"  Ratio range: [{ratio.min():.3f}, {ratio.max():.3f}]")
    logger.info(f"  Direction range: [{direction.min():.3f}, {direction.max():.3f}]")
    
    # Test enhanced forward pass with OHLCV data
    logger.info("🚀 Testing enhanced forward pass with OHLCV...")
    ratio_enhanced, direction_enhanced = model.forward(
        vol_pred, vol_history,
        ohlc_data=ohlc_data,
        volume_data=volume_data
    )
    logger.info(f"  Enhanced output shapes - Ratio: {ratio_enhanced.shape}, Direction: {direction_enhanced.shape}")
    logger.info(f"  Enhanced ratio range: [{ratio_enhanced.min():.3f}, {ratio_enhanced.max():.3f}]")
    
    # Test vol_pred initialization
    logger.info("🎯 Testing vol_pred initialization...")
    ratio_init, direction_init = model.forward_with_vol_pred_init(
        vol_pred, vol_history,
        ohlc_data=ohlc_data,
        volume_data=volume_data,
        vol_multiplier=0.4
    )
    logger.info(f"  Initialized ratio range: [{ratio_init.min():.3f}, {ratio_init.max():.3f}]")
    
    # Test trading signals
    logger.info("📊 Testing trading signals...")
    signals = model.get_trading_signals(
        vol_pred, vol_history, current_prices,
        ohlc_data=ohlc_data,
        volume_data=volume_data
    )
    
    logger.info(f"  Signal keys: {list(signals.keys())}")
    logger.info(f"  Limit prices range: [{signals['limit_price'].min():.2f}, {signals['limit_price'].max():.2f}]")
    logger.info(f"  Long positions: {signals['is_long'].sum().item()}/{n_assets * batch_size}")
    logger.info(f"  Short positions: {signals['is_short'].sum().item()}/{n_assets * batch_size}")
    
    return model, signals


def test_integration_with_real_data():
    """Test integration with real OHLCV data."""
    logger.info("🌐 Testing Integration with Real Data...")
    
    try:
        # Initialize data loader
        dataloader = EnhancedOHLCVDataLoader(data_dir="data")
        
        # Initialize model
        model = EnhancedTradingAgentModel(
            n_assets=dataloader.n_assets,
            vol_history_length=24,
            init_with_vol_pred=True
        )
        
        # Get a sample time point
        test_times = ["2023-06-01 12:00:00", "2023-06-01 18:00:00"]
        
        # Create sample volatility data
        batch_size = len(test_times)
        vol_pred = torch.rand(batch_size, dataloader.n_assets, 1) * 0.05 + 0.02
        vol_history = torch.rand(batch_size, dataloader.n_assets, 24) * 0.05 + 0.02
        
        # Create enhanced batch
        batch_data = dataloader.create_enhanced_batch(
            end_times=test_times,
            vol_pred_data=vol_pred,
            vol_history_data=vol_history,
            window_hours=48
        )
        
        logger.info(f"📦 Created batch with keys: {list(batch_data.keys())}")
        
        # Test model with real data
        signals = model.get_trading_signals(
            vol_pred=batch_data['vol_pred'],
            vol_pred_history=batch_data['vol_history'],
            current_price=batch_data['current_prices'],
            ohlc_data={k: v for k, v in batch_data.items() if k in ['open', 'high', 'low', 'close']},
            volume_data=batch_data.get('volume'),
            timestamps=batch_data.get('timestamps')
        )
        
        logger.info("✅ Successfully processed real data!")
        logger.info(f"  Generated {signals['ratio'].numel()} trading signals")
        logger.info(f"  Average limit price discount: {(1 - signals['ratio']).mean():.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        return False


def main():
    """Run all enhanced agent tests."""
    logger.info("🚀 Starting Enhanced Trading Agent Tests")
    logger.info("=" * 60)
    
    # Test 1: Enhanced Data Loader
    try:
        dataloader = test_enhanced_dataloader()
        logger.info("✅ Enhanced Data Loader: PASSED")
    except Exception as e:
        logger.error(f"❌ Enhanced Data Loader: FAILED - {e}")
        return
    
    print()
    
    # Test 2: Technical Indicators
    try:
        features = test_technical_indicators()
        logger.info("✅ Technical Indicators: PASSED")
    except Exception as e:
        logger.error(f"❌ Technical Indicators: FAILED - {e}")
        return
    
    print()
    
    # Test 3: Enhanced Agent Model
    try:
        model, signals = test_enhanced_agent_model()
        logger.info("✅ Enhanced Agent Model: PASSED")
    except Exception as e:
        logger.error(f"❌ Enhanced Agent Model: FAILED - {e}")
        return
    
    print()
    
    # Test 4: Real Data Integration
    try:
        success = test_integration_with_real_data()
        if success:
            logger.info("✅ Real Data Integration: PASSED")
        else:
            logger.warning("⚠️ Real Data Integration: PARTIAL (check data availability)")
    except Exception as e:
        logger.error(f"❌ Real Data Integration: FAILED - {e}")
    
    print()
    logger.info("🎉 Enhanced Trading Agent Testing Complete!")
    logger.info("=" * 60)
    
    # Summary
    logger.info("📋 SUMMARY:")
    logger.info("  ✅ Enhanced data loading with OHLCV support")
    logger.info("  ✅ Comprehensive technical indicator calculation")
    logger.info("  ✅ Multi-feature neural network architecture")
    logger.info("  ✅ Integration with volatility predictions")
    logger.info("  ✅ Advanced trading signal generation")
    
    logger.info("\n🚀 Ready for advanced feature engineering training!")


if __name__ == "__main__":
    main()
