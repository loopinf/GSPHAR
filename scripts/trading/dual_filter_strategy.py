#!/usr/bin/env python
"""
Enhanced selective strategy with dual filtering:
1. Absolute threshold: Only trade if vol_pred > threshold
2. Relative ranking: Select top N from qualifying assets
"""

import torch
import numpy as np
import pandas as pd
import os
import sys

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.ohlcv_trading_dataset import load_ohlcv_trading_data
from src.models.flexible_gsphar import FlexibleGSPHAR


def test_dual_filter_strategies():
    """Test dual filter strategies: absolute threshold + relative ranking."""
    print("🎯 DUAL FILTER STRATEGY: ABSOLUTE THRESHOLD + RANKING")
    print("=" * 70)
    
    # Load model
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
    
    # Test different dual filter strategies
    strategies = [
        {
            'name': 'Original (All Trades)',
            'method': 'all_trades',
            'position_size': 100
        },
        {
            'name': 'Threshold 0.8% + Top 10',
            'method': 'dual_filter',
            'abs_threshold': 0.008,  # 0.8%
            'max_assets': 10,
            'position_size': 300
        },
        {
            'name': 'Threshold 0.9% + Top 10',
            'method': 'dual_filter',
            'abs_threshold': 0.009,  # 0.9%
            'max_assets': 10,
            'position_size': 300
        },
        {
            'name': 'Threshold 1.0% + Top 10',
            'method': 'dual_filter',
            'abs_threshold': 0.010,  # 1.0%
            'max_assets': 10,
            'position_size': 400
        },
        {
            'name': 'Threshold 0.8% + Top 5',
            'method': 'dual_filter',
            'abs_threshold': 0.008,  # 0.8%
            'max_assets': 5,
            'position_size': 500
        },
        {
            'name': 'Threshold 0.9% + Top 5',
            'method': 'dual_filter',
            'abs_threshold': 0.009,  # 0.9%
            'max_assets': 5,
            'position_size': 600
        },
        {
            'name': 'Threshold 1.0% + Top 5',
            'method': 'dual_filter',
            'abs_threshold': 0.010,  # 1.0%
            'max_assets': 5,
            'position_size': 800
        },
        {
            'name': 'Dynamic: 75th Percentile + Top 8',
            'method': 'dynamic_threshold',
            'percentile': 75,
            'max_assets': 8,
            'position_size': 400
        }
    ]
    
    # Test on 20 periods
    test_periods = test_indices[:20]
    results = {}
    
    for strategy_config in strategies:
        print(f"\n🧪 TESTING: {strategy_config['name']}")
        print("-" * 50)
        
        total_pnl = 0.0
        total_trades = 0
        total_filled = 0
        all_trades = []
        total_fees = 0.0
        periods_with_trades = 0
        
        with torch.no_grad():
            for period_idx, idx in enumerate(test_periods):
                sample = dataset[idx]
                sample_info = dataset.get_sample_info(idx)
                
                x_lags = [x.unsqueeze(0) for x in sample['x_lags']]
                vol_pred = model(*x_lags)
                vol_pred_np = vol_pred.squeeze().cpu().numpy()
                
                ohlcv_data = sample['ohlcv_data'].numpy()
                
                # Select trades based on strategy
                if strategy_config['method'] == 'all_trades':
                    # Trade all assets
                    selected_assets = list(range(len(symbols)))
                    
                elif strategy_config['method'] == 'dual_filter':
                    # Dual filter: absolute threshold + ranking
                    abs_threshold = strategy_config['abs_threshold']
                    max_assets = strategy_config['max_assets']
                    
                    # Step 1: Filter by absolute threshold
                    qualifying_assets = []
                    for i, vol in enumerate(vol_pred_np):
                        if vol >= abs_threshold:
                            qualifying_assets.append((i, vol))
                    
                    # Step 2: Rank qualifying assets and select top N
                    qualifying_assets.sort(key=lambda x: x[1], reverse=True)
                    selected_assets = [asset[0] for asset in qualifying_assets[:max_assets]]
                    
                elif strategy_config['method'] == 'dynamic_threshold':
                    # Dynamic threshold based on percentile
                    percentile = strategy_config['percentile']
                    max_assets = strategy_config['max_assets']
                    
                    # Calculate dynamic threshold
                    threshold = np.percentile(vol_pred_np, percentile)
                    
                    # Filter and rank
                    qualifying_assets = []
                    for i, vol in enumerate(vol_pred_np):
                        if vol >= threshold:
                            qualifying_assets.append((i, vol))
                    
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
                        
                        position_size = strategy_config['position_size']
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
                            'net_profit_pct': net_profit_pct * 100
                        })
                
                total_pnl += period_pnl
                total_filled += period_filled
                total_fees += period_fees
        
        # Calculate metrics
        fill_rate = total_filled / total_trades if total_trades > 0 else 0
        avg_pnl_per_trade = total_pnl / total_filled if total_filled > 0 else 0
        avg_pnl_per_period = total_pnl / len(test_periods)
        trades_per_period = total_trades / len(test_periods)
        active_period_rate = periods_with_trades / len(test_periods)
        
        # Calculate efficiency metrics
        pnl_per_trade_attempt = total_pnl / total_trades if total_trades > 0 else 0
        fee_efficiency = total_pnl / total_fees if total_fees > 0 else 0
        
        results[strategy_config['name']] = {
            'total_pnl': total_pnl,
            'total_trades': total_trades,
            'total_filled': total_filled,
            'total_fees': total_fees,
            'fill_rate': fill_rate,
            'avg_pnl_per_trade': avg_pnl_per_trade,
            'avg_pnl_per_period': avg_pnl_per_period,
            'trades_per_period': trades_per_period,
            'pnl_per_trade_attempt': pnl_per_trade_attempt,
            'fee_efficiency': fee_efficiency,
            'active_period_rate': active_period_rate,
            'periods_with_trades': periods_with_trades,
            'all_trades': all_trades
        }
        
        print(f"  📊 RESULTS:")
        print(f"    Total PnL: ${total_pnl:.2f}")
        print(f"    Total Trades: {total_trades}")
        print(f"    Fill Rate: {fill_rate:.1%}")
        print(f"    Trades per Period: {trades_per_period:.1f}")
        print(f"    Active Periods: {periods_with_trades}/{len(test_periods)} ({active_period_rate:.1%})")
        print(f"    PnL per Filled Trade: ${avg_pnl_per_trade:.2f}")
        print(f"    PnL per Trade Attempt: ${pnl_per_trade_attempt:.2f}")
        print(f"    Total Fees: ${total_fees:.2f}")
        print(f"    Fee Efficiency: {fee_efficiency:.1f}x")
    
    # Compare strategies
    print(f"\n" + "="*70)
    print("📊 DUAL FILTER STRATEGY COMPARISON")
    print("="*70)
    
    comparison_data = {}
    for name, result in results.items():
        comparison_data[name] = {
            'Total PnL': f"${result['total_pnl']:.0f}",
            'Fill Rate': f"{result['fill_rate']:.1%}",
            'Trades/Period': f"{result['trades_per_period']:.1f}",
            'Active Periods': f"{result['active_period_rate']:.1%}",
            'PnL/Trade': f"${result['avg_pnl_per_trade']:.2f}",
            'PnL/Attempt': f"${result['pnl_per_trade_attempt']:.2f}",
            'Fee Efficiency': f"{result['fee_efficiency']:.1f}x"
        }
    
    comparison_df = pd.DataFrame(comparison_data).T
    print(comparison_df)
    
    # Analysis and recommendations
    print(f"\n🎯 DUAL FILTER ANALYSIS:")
    print("=" * 50)
    
    # Find best strategies by different metrics
    best_total_pnl = max(results.items(), key=lambda x: x[1]['total_pnl'])
    best_per_trade = max(results.items(), key=lambda x: x[1]['avg_pnl_per_trade'])
    best_efficiency = max(results.items(), key=lambda x: x[1]['pnl_per_trade_attempt'])
    most_selective = min([r for r in results.items() if r[1]['total_trades'] > 0], 
                        key=lambda x: x[1]['trades_per_period'])
    
    print(f"✅ Highest Total PnL: {best_total_pnl[0]} (${best_total_pnl[1]['total_pnl']:.0f})")
    print(f"✅ Best PnL per Trade: {best_per_trade[0]} (${best_per_trade[1]['avg_pnl_per_trade']:.2f})")
    print(f"✅ Best Efficiency: {best_efficiency[0]} (${best_efficiency[1]['pnl_per_trade_attempt']:.2f})")
    print(f"✅ Most Selective: {most_selective[0]} ({most_selective[1]['trades_per_period']:.1f} trades/period)")
    
    # Threshold analysis
    print(f"\n💡 THRESHOLD INSIGHTS:")
    threshold_strategies = [r for r in results.items() if 'Threshold' in r[0]]
    
    for name, result in threshold_strategies:
        threshold = name.split('%')[0].split()[-1]
        print(f"  {threshold}% threshold: {result['trades_per_period']:.1f} trades/period, "
              f"${result['avg_pnl_per_trade']:.2f}/trade, "
              f"{result['active_period_rate']:.1%} active periods")
    
    # Recommendations
    print(f"\n🚀 RECOMMENDATIONS:")
    print(f"🎯 For Maximum Profit: {best_total_pnl[0]}")
    print(f"🎯 For Highest Quality: {best_per_trade[0]}")
    print(f"🎯 For Best Efficiency: {best_efficiency[0]}")
    print(f"🎯 For Ultra Selective: {most_selective[0]}")
    
    print(f"\n💡 DUAL FILTER BENEFITS:")
    print(f"✅ Quality Control: Only trade meaningful predictions")
    print(f"✅ Dynamic Adaptation: Fewer trades in calm periods")
    print(f"✅ Risk Management: Avoid low-conviction trades")
    print(f"✅ Fee Optimization: Reduce unnecessary transactions")
    
    return results


if __name__ == "__main__":
    results = test_dual_filter_strategies()
