#!/usr/bin/env python
"""
Adaptive strategy that adjusts to market conditions for better reproducibility.
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


def execute_adaptive_strategy(model, dataset, test_indices, symbols):
    """Execute adaptive strategy that adjusts to market conditions."""
    
    total_pnl = 0.0
    total_trades = 0
    total_filled = 0
    all_trades = []
    total_fees = 0.0
    periods_with_trades = 0
    
    with torch.no_grad():
        for period_idx, idx in enumerate(test_indices):
            sample = dataset[idx]
            sample_info = dataset.get_sample_info(idx)
            
            x_lags = [x.unsqueeze(0) for x in sample['x_lags']]
            vol_pred = model(*x_lags)
            vol_pred_np = vol_pred.squeeze().cpu().numpy()
            
            ohlcv_data = sample['ohlcv_data'].numpy()
            
            # ADAPTIVE THRESHOLDS based on current predictions
            vol_75th = np.percentile(vol_pred_np, 75)  # Top 25%
            vol_50th = np.percentile(vol_pred_np, 50)  # Top 50%
            
            # Determine market regime
            avg_vol_pred = np.mean(vol_pred_np)
            vol_std = np.std(vol_pred_np)
            
            if avg_vol_pred > 0.010:  # High volatility regime
                threshold = vol_75th
                max_assets = 5
                position_size = 600
                regime = "HIGH_VOL"
            elif avg_vol_pred > 0.008:  # Medium volatility regime
                threshold = vol_50th
                max_assets = 8
                position_size = 400
                regime = "MED_VOL"
            else:  # Low volatility regime
                threshold = vol_50th
                max_assets = 10
                position_size = 300
                regime = "LOW_VOL"
            
            # Select assets based on adaptive criteria
            qualifying_assets = []
            for i, vol in enumerate(vol_pred_np):
                if vol >= threshold:
                    qualifying_assets.append((i, vol))
            
            # Rank and select
            qualifying_assets.sort(key=lambda x: x[1], reverse=True)
            selected_assets = [asset[0] for asset in qualifying_assets[:max_assets]]
            
            # Execute trades
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
                    total_fee_rate = 2 * 0.0002
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
                        'regime': regime,
                        'threshold': threshold,
                        'avg_vol_pred': avg_vol_pred,
                        'timestamp': sample_info['prediction_time']
                    })
            
            total_pnl += period_pnl
            total_filled += period_filled
            total_fees += period_fees
    
    # Calculate metrics
    fill_rate = total_filled / total_trades if total_trades > 0 else 0
    avg_pnl_per_trade = total_pnl / total_filled if total_filled > 0 else 0
    trades_per_period = total_trades / len(test_indices)
    active_period_rate = periods_with_trades / len(test_indices)
    
    return {
        'total_pnl': total_pnl,
        'avg_pnl_per_trade': avg_pnl_per_trade,
        'trades_per_period': trades_per_period,
        'fill_rate': fill_rate,
        'active_period_rate': active_period_rate,
        'all_trades': all_trades
    }


def test_adaptive_vs_fixed():
    """Compare adaptive strategy vs fixed strategy."""
    print("🔄 TESTING ADAPTIVE vs FIXED STRATEGY")
    print("=" * 70)
    
    # Load model and data
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
    
    # Test on different time periods
    time_periods = [
        ("First 100", test_indices[:100]),
        ("Second 100", test_indices[100:200]),
        ("Third 100", test_indices[200:300]),
        ("Fourth 100", test_indices[300:400]),
        ("Large Sample", test_indices[:500])
    ]
    
    results = []
    
    for period_name, indices in time_periods:
        print(f"\n🧪 Testing: {period_name}")
        print("-" * 40)
        
        # Test adaptive strategy
        adaptive_result = execute_adaptive_strategy(model, dataset, indices, symbols)
        
        print(f"📊 ADAPTIVE STRATEGY:")
        print(f"  PnL per Trade: ${adaptive_result['avg_pnl_per_trade']:.2f}")
        print(f"  Trades per Period: {adaptive_result['trades_per_period']:.1f}")
        print(f"  Active Periods: {adaptive_result['active_period_rate']:.1%}")
        print(f"  Fill Rate: {adaptive_result['fill_rate']:.1%}")
        
        results.append({
            'period': period_name,
            'adaptive_pnl_per_trade': adaptive_result['avg_pnl_per_trade'],
            'adaptive_trades_per_period': adaptive_result['trades_per_period'],
            'adaptive_active_rate': adaptive_result['active_period_rate']
        })
    
    # Analyze consistency
    print(f"\n📊 ADAPTIVE STRATEGY CONSISTENCY:")
    print("=" * 50)
    
    adaptive_pnls = [r['adaptive_pnl_per_trade'] for r in results]
    adaptive_trades = [r['adaptive_trades_per_period'] for r in results]
    adaptive_active = [r['adaptive_active_rate'] for r in results]
    
    print(f"PnL per Trade: ${np.mean(adaptive_pnls):.2f} ± ${np.std(adaptive_pnls):.2f}")
    print(f"Range: ${np.min(adaptive_pnls):.2f} to ${np.max(adaptive_pnls):.2f}")
    print(f"Coefficient of Variation: {np.std(adaptive_pnls)/np.mean(adaptive_pnls):.1%}")
    
    print(f"\nTrades per Period: {np.mean(adaptive_trades):.1f} ± {np.std(adaptive_trades):.1f}")
    print(f"Active Period Rate: {np.mean(adaptive_active):.1%} ± {np.std(adaptive_active):.1%}")
    
    # Assessment
    cv_adaptive = np.std(adaptive_pnls) / np.mean(adaptive_pnls)
    min_pnl_adaptive = np.min(adaptive_pnls)
    
    print(f"\n🎯 ADAPTIVE STRATEGY ASSESSMENT:")
    
    if cv_adaptive < 0.15 and min_pnl_adaptive > 8:
        print(f"✅ HIGHLY CONSISTENT AND PROFITABLE")
        assessment = "EXCELLENT"
    elif cv_adaptive < 0.25 and min_pnl_adaptive > 5:
        print(f"✅ GOOD CONSISTENCY AND PROFITABILITY")
        assessment = "GOOD"
    elif cv_adaptive < 0.35:
        print(f"⚠️  MODERATE CONSISTENCY")
        assessment = "MODERATE"
    else:
        print(f"❌ POOR CONSISTENCY")
        assessment = "POOR"
    
    print(f"CV: {cv_adaptive:.1%}, Min PnL: ${min_pnl_adaptive:.2f}")
    
    return assessment, results


if __name__ == "__main__":
    assessment, results = test_adaptive_vs_fixed()
    
    print(f"\n🎯 FINAL RECOMMENDATION:")
    if assessment in ["EXCELLENT", "GOOD"]:
        print(f"✅ DEPLOY ADAPTIVE STRATEGY")
        print(f"The adaptive approach shows much better consistency")
    else:
        print(f"⚠️  CONTINUE DEVELOPMENT")
        print(f"Need further improvements before deployment")
