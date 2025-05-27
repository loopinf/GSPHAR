#!/usr/bin/env python
"""
Realistic selective trading strategy using data-driven thresholds.
Reduces trading frequency while maintaining profitability.
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


def test_realistic_selective_strategies():
    """Test realistic selective strategies with data-driven thresholds."""
    print("🎯 REALISTIC SELECTIVE TRADING STRATEGIES")
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
    
    # Test different realistic strategies
    strategies = [
        {
            'name': 'Original (All Trades)',
            'method': 'all_trades',
            'position_size': 100
        },
        {
            'name': 'Top 50% Vol Predictions',
            'method': 'top_percentile',
            'vol_percentile': 50,
            'position_size': 150
        },
        {
            'name': 'Top 25% Vol Predictions', 
            'method': 'top_percentile',
            'vol_percentile': 75,
            'position_size': 200
        },
        {
            'name': 'Top 10% Vol Predictions',
            'method': 'top_percentile', 
            'vol_percentile': 90,
            'position_size': 300
        },
        {
            'name': 'Best 15 Assets per Period',
            'method': 'top_n_assets',
            'n_assets': 15,
            'position_size': 200
        },
        {
            'name': 'Best 10 Assets per Period',
            'method': 'top_n_assets',
            'n_assets': 10,
            'position_size': 300
        },
        {
            'name': 'Best 5 Assets per Period',
            'method': 'top_n_assets',
            'n_assets': 5,
            'position_size': 500
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
                    
                elif strategy_config['method'] == 'top_percentile':
                    # Trade assets with vol predictions above percentile
                    vol_threshold = np.percentile(vol_pred_np, strategy_config['vol_percentile'])
                    selected_assets = [i for i, vol in enumerate(vol_pred_np) if vol >= vol_threshold]
                    
                elif strategy_config['method'] == 'top_n_assets':
                    # Trade top N assets by vol prediction
                    n_assets = strategy_config['n_assets']
                    asset_scores = [(i, vol) for i, vol in enumerate(vol_pred_np)]
                    asset_scores.sort(key=lambda x: x[1], reverse=True)
                    selected_assets = [asset_scores[i][0] for i in range(min(n_assets, len(asset_scores)))]
                
                # Execute trades on selected assets
                period_pnl = 0.0
                period_trades = 0
                period_filled = 0
                period_fees = 0.0
                
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
            'all_trades': all_trades
        }
        
        print(f"  📊 RESULTS:")
        print(f"    Total PnL: ${total_pnl:.2f}")
        print(f"    Total Trades: {total_trades}")
        print(f"    Fill Rate: {fill_rate:.1%}")
        print(f"    Trades per Period: {trades_per_period:.1f}")
        print(f"    PnL per Filled Trade: ${avg_pnl_per_trade:.2f}")
        print(f"    PnL per Trade Attempt: ${pnl_per_trade_attempt:.2f}")
        print(f"    Total Fees: ${total_fees:.2f}")
        print(f"    Fee Efficiency: {fee_efficiency:.1f}x")
    
    # Compare strategies
    print(f"\n" + "="*70)
    print("📊 STRATEGY COMPARISON")
    print("="*70)
    
    comparison_data = {}
    for name, result in results.items():
        comparison_data[name] = {
            'Total PnL': f"${result['total_pnl']:.0f}",
            'Fill Rate': f"{result['fill_rate']:.1%}",
            'Trades/Period': f"{result['trades_per_period']:.1f}",
            'PnL/Trade': f"${result['avg_pnl_per_trade']:.2f}",
            'PnL/Attempt': f"${result['pnl_per_trade_attempt']:.2f}",
            'Total Fees': f"${result['total_fees']:.0f}",
            'Fee Efficiency': f"{result['fee_efficiency']:.1f}x"
        }
    
    comparison_df = pd.DataFrame(comparison_data).T
    print(comparison_df)
    
    # Analysis and recommendations
    print(f"\n🎯 ANALYSIS & RECOMMENDATIONS:")
    print("=" * 50)
    
    # Find best strategies by different metrics
    best_total_pnl = max(results.items(), key=lambda x: x[1]['total_pnl'])
    best_per_trade = max(results.items(), key=lambda x: x[1]['avg_pnl_per_trade'])
    best_per_attempt = max(results.items(), key=lambda x: x[1]['pnl_per_trade_attempt'])
    lowest_fill_rate = min(results.items(), key=lambda x: x[1]['fill_rate'])
    best_fee_efficiency = max(results.items(), key=lambda x: x[1]['fee_efficiency'])
    
    print(f"✅ Highest Total PnL: {best_total_pnl[0]} (${best_total_pnl[1]['total_pnl']:.0f})")
    print(f"✅ Best PnL per Filled Trade: {best_per_trade[0]} (${best_per_trade[1]['avg_pnl_per_trade']:.2f})")
    print(f"✅ Best PnL per Attempt: {best_per_attempt[0]} (${best_per_attempt[1]['pnl_per_trade_attempt']:.2f})")
    print(f"✅ Most Selective: {lowest_fill_rate[0]} ({lowest_fill_rate[1]['fill_rate']:.1%} fill rate)")
    print(f"✅ Best Fee Efficiency: {best_fee_efficiency[0]} ({best_fee_efficiency[1]['fee_efficiency']:.1f}x)")
    
    # Recommendations based on goals
    print(f"\n💡 RECOMMENDATIONS BY GOAL:")
    print(f"🎯 For Maximum Profit: {best_total_pnl[0]}")
    print(f"🎯 For Selectivity: {lowest_fill_rate[0]}")
    print(f"🎯 For Efficiency: {best_per_attempt[0]}")
    print(f"🎯 For Low Fees: {best_fee_efficiency[0]}")
    
    # Calculate improvement over original
    original_result = results['Original (All Trades)']
    
    print(f"\n📈 IMPROVEMENTS OVER ORIGINAL:")
    for name, result in results.items():
        if name == 'Original (All Trades)':
            continue
        
        pnl_improvement = (result['total_pnl'] - original_result['total_pnl']) / original_result['total_pnl'] * 100
        trade_reduction = (1 - result['total_trades'] / original_result['total_trades']) * 100
        fee_reduction = (1 - result['total_fees'] / original_result['total_fees']) * 100
        
        print(f"  {name}:")
        print(f"    PnL change: {pnl_improvement:+.1f}%")
        print(f"    Trade reduction: {trade_reduction:.1f}%")
        print(f"    Fee reduction: {fee_reduction:.1f}%")
    
    return results


if __name__ == "__main__":
    results = test_realistic_selective_strategies()
