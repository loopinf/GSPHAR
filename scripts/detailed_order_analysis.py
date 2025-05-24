#!/usr/bin/env python
"""
Detailed analysis showing actual orders and results for each symbol.
This provides complete transparency into the trading strategy execution.
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


def get_symbol_names():
    """Get the actual cryptocurrency symbol names."""
    # Load the volatility data to get symbol names
    vol_df = pd.read_csv("data/crypto_rv1h_38_20200822_20250116.csv", index_col=0, parse_dates=True)
    symbols = vol_df.columns.tolist()
    return symbols


def analyze_detailed_orders():
    """Analyze detailed orders and results for each symbol."""
    print("📊 DETAILED ORDER ANALYSIS BY SYMBOL")
    print("=" * 80)
    
    # Load the improved model
    model_path = "models/improved_model_20250524_172018.pt"
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    metadata = checkpoint['metadata']
    parameters = checkpoint['parameters']
    test_indices = checkpoint['test_indices']
    
    # Get symbol names
    symbols = get_symbol_names()
    print(f"✅ Found {len(symbols)} symbols: {symbols[:5]}... (showing first 5)")
    
    # Load volatility data for correlation matrix
    vol_df = pd.read_csv("data/crypto_rv1h_38_20200822_20250116.csv", index_col=0, parse_dates=True)
    corr_sample = vol_df.iloc[-1000:] if len(vol_df) > 1000 else vol_df
    corr_matrix = corr_sample.corr().values
    A = (corr_matrix + corr_matrix.T) / 2
    
    # Create model
    model = FlexibleGSPHAR(
        lags=parameters['lags'],
        output_dim=1,
        filter_size=len(metadata['assets']),
        A=A
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load dataset
    dataset, _ = load_ohlcv_trading_data(
        volatility_file="data/crypto_rv1h_38_20200822_20250116.csv",
        lags=parameters['lags'],
        holding_period=parameters['holding_period'],
        debug=False
    )
    
    # Analyze first 10 periods for detailed view
    test_sample_indices = test_indices[:10]
    print(f"\n🔍 ANALYZING FIRST {len(test_sample_indices)} PERIODS")
    print("=" * 80)
    
    all_orders = []
    period_summaries = []
    
    with torch.no_grad():
        for period_idx, idx in enumerate(test_sample_indices):
            sample = dataset[idx]
            sample_info = dataset.get_sample_info(idx)
            
            x_lags = [x.unsqueeze(0) for x in sample['x_lags']]
            vol_pred = model(*x_lags)
            vol_pred_np = vol_pred.squeeze().cpu().numpy()
            
            ohlcv_data = sample['ohlcv_data'].numpy()
            timestamp = sample_info['prediction_time']
            
            print(f"\n📅 PERIOD {period_idx + 1}: {timestamp}")
            print("-" * 60)
            
            period_pnl = 0.0
            period_orders = []
            filled_orders = 0
            total_orders = 0
            
            # Analyze each symbol
            for asset_idx, symbol in enumerate(symbols):
                # Get prices
                open_price = ohlcv_data[asset_idx, 0, 0]      # T+0 OPEN
                current_low = ohlcv_data[asset_idx, 0, 2]     # T+0 LOW
                current_high = ohlcv_data[asset_idx, 0, 1]    # T+0 HIGH
                current_close = ohlcv_data[asset_idx, 0, 3]   # T+0 CLOSE
                exit_price = ohlcv_data[asset_idx, 4, 3]      # T+4 CLOSE
                
                # Get volatility prediction
                asset_vol_pred = vol_pred_np[asset_idx]
                
                # Calculate limit order price
                limit_price = open_price * (1 - asset_vol_pred)
                
                # Check if order fills
                order_fills = current_low <= limit_price
                
                # Calculate potential profit
                if order_fills:
                    gross_profit_pct = (exit_price - limit_price) / limit_price
                    total_fee_rate = 2 * 0.0002  # 0.04% total fees
                    net_profit_pct = gross_profit_pct - total_fee_rate
                    asset_pnl = net_profit_pct * 100  # $100 position
                    period_pnl += asset_pnl
                    filled_orders += 1
                else:
                    asset_pnl = 0.0
                
                total_orders += 1
                
                # Store order details
                order_detail = {
                    'period': period_idx + 1,
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'asset_idx': asset_idx,
                    'vol_pred': asset_vol_pred,
                    'open_price': open_price,
                    'limit_price': limit_price,
                    'current_low': current_low,
                    'current_high': current_high,
                    'current_close': current_close,
                    'exit_price': exit_price,
                    'order_fills': order_fills,
                    'asset_pnl': asset_pnl,
                    'discount_pct': (open_price - limit_price) / open_price * 100,
                    'price_drop_pct': (open_price - current_low) / open_price * 100
                }
                
                all_orders.append(order_detail)
                period_orders.append(order_detail)
            
            # Show top 5 filled orders and top 5 unfilled orders for this period
            filled_orders_period = [o for o in period_orders if o['order_fills']]
            unfilled_orders_period = [o for o in period_orders if not o['order_fills']]
            
            # Sort by PnL for filled orders
            filled_orders_period.sort(key=lambda x: x['asset_pnl'], reverse=True)
            unfilled_orders_period.sort(key=lambda x: x['discount_pct'], reverse=True)
            
            print(f"📈 TOP 5 FILLED ORDERS (Best PnL):")
            for i, order in enumerate(filled_orders_period[:5]):
                print(f"  {i+1}. {order['symbol']}: "
                      f"Vol={order['vol_pred']:.3f} ({order['vol_pred']*100:.1f}%), "
                      f"Limit=${order['limit_price']:.4f}, "
                      f"Low=${order['current_low']:.4f}, "
                      f"PnL=${order['asset_pnl']:.2f}")
            
            print(f"\n❌ TOP 5 UNFILLED ORDERS (Highest Discount):")
            for i, order in enumerate(unfilled_orders_period[:5]):
                print(f"  {i+1}. {order['symbol']}: "
                      f"Vol={order['vol_pred']:.3f} ({order['vol_pred']*100:.1f}%), "
                      f"Limit=${order['limit_price']:.4f}, "
                      f"Low=${order['current_low']:.4f}, "
                      f"Discount={order['discount_pct']:.2f}%")
            
            # Period summary
            fill_rate = filled_orders / total_orders
            avg_vol_pred = np.mean([o['vol_pred'] for o in period_orders])
            
            print(f"\n📊 PERIOD SUMMARY:")
            print(f"  Total Orders: {total_orders}")
            print(f"  Filled Orders: {filled_orders} ({fill_rate:.1%})")
            print(f"  Period PnL: ${period_pnl:.2f}")
            print(f"  Avg Vol Pred: {avg_vol_pred:.3f} ({avg_vol_pred*100:.1f}%)")
            
            period_summaries.append({
                'period': period_idx + 1,
                'timestamp': timestamp,
                'total_orders': total_orders,
                'filled_orders': filled_orders,
                'fill_rate': fill_rate,
                'period_pnl': period_pnl,
                'avg_vol_pred': avg_vol_pred
            })
    
    # Overall analysis
    print(f"\n" + "="*80)
    print("📊 OVERALL ANALYSIS ACROSS ALL PERIODS")
    print("="*80)
    
    # Convert to DataFrame for analysis
    orders_df = pd.DataFrame(all_orders)
    
    # Symbol-level analysis
    symbol_stats = orders_df.groupby('symbol').agg({
        'order_fills': ['count', 'sum', 'mean'],
        'asset_pnl': ['sum', 'mean'],
        'vol_pred': 'mean',
        'discount_pct': 'mean'
    }).round(3)
    
    symbol_stats.columns = ['Total_Orders', 'Filled_Orders', 'Fill_Rate', 
                           'Total_PnL', 'Avg_PnL', 'Avg_Vol_Pred', 'Avg_Discount']
    
    # Sort by total PnL
    symbol_stats = symbol_stats.sort_values('Total_PnL', ascending=False)
    
    print(f"\n🏆 TOP 10 PERFORMING SYMBOLS:")
    print(symbol_stats.head(10).to_string())
    
    print(f"\n📉 BOTTOM 10 PERFORMING SYMBOLS:")
    print(symbol_stats.tail(10).to_string())
    
    # Fill rate analysis
    print(f"\n📊 FILL RATE ANALYSIS:")
    print(f"Overall Fill Rate: {orders_df['order_fills'].mean():.1%}")
    print(f"Symbols with >50% Fill Rate: {(symbol_stats['Fill_Rate'] > 0.5).sum()}")
    print(f"Symbols with <20% Fill Rate: {(symbol_stats['Fill_Rate'] < 0.2).sum()}")
    
    # Volatility prediction analysis
    print(f"\n📈 VOLATILITY PREDICTION ANALYSIS:")
    print(f"Average Vol Prediction: {orders_df['vol_pred'].mean():.3f} ({orders_df['vol_pred'].mean()*100:.1f}%)")
    print(f"Vol Prediction Range: {orders_df['vol_pred'].min():.3f} to {orders_df['vol_pred'].max():.3f}")
    print(f"Vol Prediction Std: {orders_df['vol_pred'].std():.3f}")
    
    # Save detailed results
    orders_df.to_csv('detailed_orders_analysis.csv', index=False)
    symbol_stats.to_csv('symbol_performance_analysis.csv')
    
    print(f"\n✅ Detailed analysis saved to:")
    print(f"  - detailed_orders_analysis.csv (all orders)")
    print(f"  - symbol_performance_analysis.csv (symbol stats)")
    
    return orders_df, symbol_stats


if __name__ == "__main__":
    orders_df, symbol_stats = analyze_detailed_orders()
