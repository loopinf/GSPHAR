#!/usr/bin/env python
"""
Comprehensive reproducibility testing for the recommended strategy.
Tests across different time periods, market conditions, and sample sizes.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime, timedelta

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.ohlcv_trading_dataset import load_ohlcv_trading_data
from src.models.flexible_gsphar import FlexibleGSPHAR


def execute_strategy(model, dataset, test_indices, symbols, strategy_name="0.9% + Top 5"):
    """Execute the recommended strategy on given test indices."""
    
    total_pnl = 0.0
    total_trades = 0
    total_filled = 0
    all_trades = []
    total_fees = 0.0
    periods_with_trades = 0
    
    # Strategy parameters
    abs_threshold = 0.009  # 0.9%
    max_assets = 5
    position_size = 600
    
    with torch.no_grad():
        for period_idx, idx in enumerate(test_indices):
            sample = dataset[idx]
            sample_info = dataset.get_sample_info(idx)
            
            x_lags = [x.unsqueeze(0) for x in sample['x_lags']]
            vol_pred = model(*x_lags)
            vol_pred_np = vol_pred.squeeze().cpu().numpy()
            
            ohlcv_data = sample['ohlcv_data'].numpy()
            
            # Dual filter: absolute threshold + ranking
            qualifying_assets = []
            for i, vol in enumerate(vol_pred_np):
                if vol >= abs_threshold:
                    qualifying_assets.append((i, vol))
            
            # Rank qualifying assets and select top N
            qualifying_assets.sort(key=lambda x: x[1], reverse=True)
            selected_assets = [asset[0] for asset in qualifying_assets[:max_assets]]
            
            # Execute trades on selected assets
            period_pnl = 0.0
            period_trades = 0
            period_filled = 0
            period_fees = 0.0
            
            if len(selected_assets) > 0:
                periods_with_trades += 1
            
            for asset_idx in selected_assets:
                symbol = symbols[asset_idx]
                vol_pred_asset = vol_pred_np[asset_idx]
                
                # Get prices
                open_price = ohlcv_data[asset_idx, 0, 0]
                current_low = ohlcv_data[asset_idx, 0, 2]
                exit_price = ohlcv_data[asset_idx, 4, 3]
                
                # Calculate limit price
                limit_price = open_price * (1 - vol_pred_asset)
                
                # Check if order fills
                order_fills = current_low <= limit_price
                
                period_trades += 1
                total_trades += 1
                
                if order_fills:
                    # Calculate profit
                    gross_profit_pct = (exit_price - limit_price) / limit_price
                    total_fee_rate = 2 * 0.0002  # 0.04% total fees
                    net_profit_pct = gross_profit_pct - total_fee_rate
                    
                    trade_pnl = net_profit_pct * position_size
                    trade_fees = total_fee_rate * position_size
                    
                    period_pnl += trade_pnl
                    period_filled += 1
                    period_fees += trade_fees
                    
                    all_trades.append({
                        'period': period_idx + 1,
                        'symbol': symbol,
                        'vol_pred': vol_pred_asset,
                        'position_size': position_size,
                        'trade_pnl': trade_pnl,
                        'trade_fees': trade_fees,
                        'gross_profit_pct': gross_profit_pct * 100,
                        'net_profit_pct': net_profit_pct * 100,
                        'timestamp': sample_info['prediction_time']
                    })
            
            total_pnl += period_pnl
            total_filled += period_filled
            total_fees += period_fees
    
    # Calculate metrics
    fill_rate = total_filled / total_trades if total_trades > 0 else 0
    avg_pnl_per_trade = total_pnl / total_filled if total_filled > 0 else 0
    avg_pnl_per_period = total_pnl / len(test_indices)
    trades_per_period = total_trades / len(test_indices)
    active_period_rate = periods_with_trades / len(test_indices)
    
    return {
        'strategy_name': strategy_name,
        'total_pnl': total_pnl,
        'total_trades': total_trades,
        'total_filled': total_filled,
        'total_fees': total_fees,
        'fill_rate': fill_rate,
        'avg_pnl_per_trade': avg_pnl_per_trade,
        'avg_pnl_per_period': avg_pnl_per_period,
        'trades_per_period': trades_per_period,
        'active_period_rate': active_period_rate,
        'periods_with_trades': periods_with_trades,
        'all_trades': all_trades,
        'test_periods': len(test_indices)
    }


def test_different_time_periods():
    """Test strategy across different time periods."""
    print("🕐 TESTING ACROSS DIFFERENT TIME PERIODS")
    print("=" * 70)
    
    # Load model and data
    model_path = "models/fixed_stage1_model_20250524_202400.pt"
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    test_indices = checkpoint['test_indices']
    
    # Load volatility data for correlation matrix
    vol_df = pd.read_csv("data/crypto_rv1h_38_20200822_20250116.csv", index_col=0, parse_dates=True)
    symbols = vol_df.columns.tolist()
    corr_sample = vol_df.iloc[-1000:] if len(vol_df) > 1000 else vol_df
    corr_matrix = corr_sample.corr().values
    A = (corr_matrix + corr_matrix.T) / 2
    
    # Create model
    model = FlexibleGSPHAR(
        lags=[1, 4, 24],
        output_dim=1,
        filter_size=38,
        A=A
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load dataset
    dataset, _ = load_ohlcv_trading_data(
        volatility_file="data/crypto_rv1h_38_20200822_20250116.csv",
        lags=[1, 4, 24],
        holding_period=4,
        debug=False
    )
    
    # Test different time periods
    time_period_tests = [
        {
            'name': 'First 100 periods',
            'indices': test_indices[:100]
        },
        {
            'name': 'Second 100 periods', 
            'indices': test_indices[100:200]
        },
        {
            'name': 'Third 100 periods',
            'indices': test_indices[200:300]
        },
        {
            'name': 'Fourth 100 periods',
            'indices': test_indices[300:400]
        },
        {
            'name': 'Random 100 periods (1)',
            'indices': np.random.choice(test_indices, 100, replace=False).tolist()
        },
        {
            'name': 'Random 100 periods (2)',
            'indices': np.random.choice(test_indices, 100, replace=False).tolist()
        },
        {
            'name': 'Large sample (500 periods)',
            'indices': test_indices[:500]
        },
        {
            'name': 'Very large sample (1000 periods)',
            'indices': test_indices[:1000]
        }
    ]
    
    results = []
    
    for test_config in time_period_tests:
        print(f"\n🧪 Testing: {test_config['name']}")
        print("-" * 50)
        
        result = execute_strategy(model, dataset, test_config['indices'], symbols, test_config['name'])
        results.append(result)
        
        print(f"  Periods: {result['test_periods']}")
        print(f"  Total PnL: ${result['total_pnl']:.2f}")
        print(f"  PnL per Trade: ${result['avg_pnl_per_trade']:.2f}")
        print(f"  Trades per Period: {result['trades_per_period']:.1f}")
        print(f"  Fill Rate: {result['fill_rate']:.1%}")
        print(f"  Active Periods: {result['active_period_rate']:.1%}")
    
    return results


def test_different_sample_sizes():
    """Test strategy with different sample sizes to check consistency."""
    print(f"\n🔢 TESTING DIFFERENT SAMPLE SIZES")
    print("=" * 70)
    
    # Load model and data (same setup as above)
    model_path = "models/fixed_stage1_model_20250524_202400.pt"
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    test_indices = checkpoint['test_indices']
    
    vol_df = pd.read_csv("data/crypto_rv1h_38_20200822_20250116.csv", index_col=0, parse_dates=True)
    symbols = vol_df.columns.tolist()
    corr_sample = vol_df.iloc[-1000:] if len(vol_df) > 1000 else vol_df
    corr_matrix = corr_sample.corr().values
    A = (corr_matrix + corr_matrix.T) / 2
    
    model = FlexibleGSPHAR(lags=[1, 4, 24], output_dim=1, filter_size=38, A=A)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    dataset, _ = load_ohlcv_trading_data(
        volatility_file="data/crypto_rv1h_38_20200822_20250116.csv",
        lags=[1, 4, 24], holding_period=4, debug=False
    )
    
    # Test different sample sizes
    sample_sizes = [20, 50, 100, 200, 500, 1000, 2000]
    results = []
    
    for sample_size in sample_sizes:
        if sample_size > len(test_indices):
            continue
            
        print(f"\n🧪 Testing sample size: {sample_size}")
        print("-" * 30)
        
        # Test multiple random samples of this size
        sample_results = []
        for i in range(3):  # 3 random samples per size
            random_indices = np.random.choice(test_indices, sample_size, replace=False).tolist()
            result = execute_strategy(model, dataset, random_indices, symbols, f"Sample {sample_size}-{i+1}")
            sample_results.append(result)
        
        # Calculate statistics across samples
        pnl_per_trades = [r['avg_pnl_per_trade'] for r in sample_results]
        trades_per_periods = [r['trades_per_period'] for r in sample_results]
        fill_rates = [r['fill_rate'] for r in sample_results]
        
        avg_pnl_per_trade = np.mean(pnl_per_trades)
        std_pnl_per_trade = np.std(pnl_per_trades)
        avg_trades_per_period = np.mean(trades_per_periods)
        avg_fill_rate = np.mean(fill_rates)
        
        print(f"  PnL per Trade: ${avg_pnl_per_trade:.2f} ± ${std_pnl_per_trade:.2f}")
        print(f"  Trades per Period: {avg_trades_per_period:.1f}")
        print(f"  Fill Rate: {avg_fill_rate:.1%}")
        
        results.append({
            'sample_size': sample_size,
            'avg_pnl_per_trade': avg_pnl_per_trade,
            'std_pnl_per_trade': std_pnl_per_trade,
            'avg_trades_per_period': avg_trades_per_period,
            'avg_fill_rate': avg_fill_rate,
            'individual_results': sample_results
        })
    
    return results


def analyze_consistency(time_results, sample_results):
    """Analyze consistency across different tests."""
    print(f"\n📊 CONSISTENCY ANALYSIS")
    print("=" * 70)
    
    # Analyze time period consistency
    time_pnl_per_trades = [r['avg_pnl_per_trade'] for r in time_results if r['avg_pnl_per_trade'] > 0]
    time_trades_per_periods = [r['trades_per_period'] for r in time_results]
    time_fill_rates = [r['fill_rate'] for r in time_results if r['fill_rate'] > 0]
    
    print(f"📈 TIME PERIOD CONSISTENCY:")
    print(f"  PnL per Trade: ${np.mean(time_pnl_per_trades):.2f} ± ${np.std(time_pnl_per_trades):.2f}")
    print(f"  Range: ${np.min(time_pnl_per_trades):.2f} to ${np.max(time_pnl_per_trades):.2f}")
    print(f"  Coefficient of Variation: {np.std(time_pnl_per_trades)/np.mean(time_pnl_per_trades):.2%}")
    
    print(f"\n  Trades per Period: {np.mean(time_trades_per_periods):.1f} ± {np.std(time_trades_per_periods):.1f}")
    print(f"  Fill Rate: {np.mean(time_fill_rates):.1%} ± {np.std(time_fill_rates):.1%}")
    
    # Analyze sample size consistency
    print(f"\n📊 SAMPLE SIZE CONSISTENCY:")
    for result in sample_results:
        cv = result['std_pnl_per_trade'] / result['avg_pnl_per_trade'] if result['avg_pnl_per_trade'] > 0 else 0
        print(f"  Size {result['sample_size']:4d}: ${result['avg_pnl_per_trade']:.2f} ± ${result['std_pnl_per_trade']:.2f} (CV: {cv:.1%})")
    
    # Overall assessment
    print(f"\n🎯 REPRODUCIBILITY ASSESSMENT:")
    
    # Check if results are consistent
    cv_threshold = 0.3  # 30% coefficient of variation
    cv_time = np.std(time_pnl_per_trades) / np.mean(time_pnl_per_trades)
    
    if cv_time < cv_threshold:
        print(f"✅ HIGHLY REPRODUCIBLE: CV = {cv_time:.1%} (< {cv_threshold:.0%})")
        reproducibility = "HIGH"
    elif cv_time < 0.5:
        print(f"⚠️  MODERATELY REPRODUCIBLE: CV = {cv_time:.1%}")
        reproducibility = "MODERATE"
    else:
        print(f"❌ LOW REPRODUCIBILITY: CV = {cv_time:.1%} (> 50%)")
        reproducibility = "LOW"
    
    # Check for outliers
    q1 = np.percentile(time_pnl_per_trades, 25)
    q3 = np.percentile(time_pnl_per_trades, 75)
    iqr = q3 - q1
    outlier_threshold = 1.5 * iqr
    
    outliers = [x for x in time_pnl_per_trades if x < q1 - outlier_threshold or x > q3 + outlier_threshold]
    
    if len(outliers) == 0:
        print(f"✅ NO OUTLIERS DETECTED: Results are stable")
    else:
        print(f"⚠️  {len(outliers)} OUTLIERS DETECTED: Some periods show unusual performance")
    
    # Minimum performance check
    min_pnl_per_trade = np.min(time_pnl_per_trades)
    expected_min = 10.0  # $10 minimum expected
    
    if min_pnl_per_trade > expected_min:
        print(f"✅ CONSISTENT PROFITABILITY: Minimum ${min_pnl_per_trade:.2f} > ${expected_min}")
    else:
        print(f"⚠️  VARIABLE PROFITABILITY: Minimum ${min_pnl_per_trade:.2f} < ${expected_min}")
    
    return {
        'reproducibility': reproducibility,
        'cv_time': cv_time,
        'outliers': len(outliers),
        'min_performance': min_pnl_per_trade,
        'avg_performance': np.mean(time_pnl_per_trades),
        'std_performance': np.std(time_pnl_per_trades)
    }


def main():
    """Main reproducibility testing function."""
    print("🧪 COMPREHENSIVE STRATEGY REPRODUCIBILITY TESTING")
    print("=" * 80)
    print("Testing the recommended '0.9% Threshold + Top 5' strategy")
    print("across different time periods and sample sizes...")
    
    # Set random seed for reproducible random sampling
    np.random.seed(42)
    
    # Test 1: Different time periods
    time_results = test_different_time_periods()
    
    # Test 2: Different sample sizes
    sample_results = test_different_sample_sizes()
    
    # Test 3: Consistency analysis
    consistency = analyze_consistency(time_results, sample_results)
    
    # Final recommendation
    print(f"\n🎯 FINAL REPRODUCIBILITY VERDICT:")
    print("=" * 50)
    
    if consistency['reproducibility'] == 'HIGH' and consistency['min_performance'] > 10:
        print(f"✅ STRATEGY IS HIGHLY REPRODUCIBLE AND READY FOR DEPLOYMENT")
        print(f"   - Consistent performance across time periods")
        print(f"   - Low variability (CV: {consistency['cv_time']:.1%})")
        print(f"   - No significant outliers")
        print(f"   - Minimum performance: ${consistency['min_performance']:.2f}")
        recommendation = "DEPLOY"
    elif consistency['reproducibility'] in ['HIGH', 'MODERATE'] and consistency['min_performance'] > 5:
        print(f"⚠️  STRATEGY IS MODERATELY REPRODUCIBLE - DEPLOY WITH CAUTION")
        print(f"   - Generally consistent but some variability")
        print(f"   - Monitor performance closely")
        print(f"   - Consider additional risk management")
        recommendation = "DEPLOY_WITH_CAUTION"
    else:
        print(f"❌ STRATEGY SHOWS LOW REPRODUCIBILITY - NEEDS MORE WORK")
        print(f"   - High variability across periods")
        print(f"   - Inconsistent performance")
        print(f"   - Requires further optimization")
        recommendation = "DO_NOT_DEPLOY"
    
    print(f"\n📋 RECOMMENDATION: {recommendation}")
    
    return {
        'time_results': time_results,
        'sample_results': sample_results,
        'consistency': consistency,
        'recommendation': recommendation
    }


if __name__ == "__main__":
    results = main()
