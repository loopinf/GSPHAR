#!/usr/bin/env python
"""
Implement selective trading strategy with higher alpha, lower frequency.
Focus on quality over quantity - fewer trades but higher confidence.
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


class SelectiveTradingStrategy:
    """
    Selective trading strategy that focuses on high-confidence, high-alpha opportunities.
    Reduces trading frequency while increasing profit per trade.
    """
    
    def __init__(self, model, confidence_threshold=0.7, vol_threshold=0.02, 
                 max_trades_per_period=10, position_size=200):
        """
        Initialize selective trading strategy.
        
        Args:
            model: Trained prediction model
            confidence_threshold: Minimum confidence to trade (0-1)
            vol_threshold: Minimum volatility prediction to consider (e.g., 2%)
            max_trades_per_period: Maximum trades per time period
            position_size: Position size per trade (higher since fewer trades)
        """
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.vol_threshold = vol_threshold
        self.max_trades_per_period = max_trades_per_period
        self.position_size = position_size
    
    def calculate_prediction_confidence(self, vol_predictions, historical_accuracy=None):
        """
        Calculate confidence score for each prediction.
        Higher confidence = more reliable prediction.
        """
        # Method 1: Based on prediction magnitude
        # Higher volatility predictions tend to be more reliable
        magnitude_confidence = np.clip(vol_predictions / 0.05, 0, 1)  # Normalize to 5%
        
        # Method 2: Based on prediction consistency across assets
        # If many assets have similar predictions, higher confidence
        pred_std = np.std(vol_predictions)
        consistency_confidence = np.exp(-pred_std * 10)  # Lower std = higher confidence
        
        # Method 3: Based on historical model accuracy (if available)
        if historical_accuracy is not None:
            accuracy_confidence = historical_accuracy
        else:
            accuracy_confidence = 0.5  # Default moderate confidence
        
        # Combine confidence measures
        combined_confidence = (
            0.4 * magnitude_confidence + 
            0.3 * consistency_confidence + 
            0.3 * accuracy_confidence
        )
        
        return combined_confidence
    
    def select_best_opportunities(self, vol_predictions, confidence_scores, symbols, ohlcv_data):
        """
        Select the best trading opportunities from all available assets.
        """
        opportunities = []
        
        for asset_idx, symbol in enumerate(symbols):
            vol_pred = vol_predictions[asset_idx]
            confidence = confidence_scores[asset_idx]
            
            # Basic filters
            if vol_pred < self.vol_threshold:
                continue
            if confidence < self.confidence_threshold:
                continue
            
            # Calculate opportunity score
            open_price = ohlcv_data[asset_idx, 0, 0]
            current_low = ohlcv_data[asset_idx, 0, 2]
            intraday_drop = (open_price - current_low) / open_price
            
            # Opportunity score combines prediction, confidence, and current market state
            opportunity_score = (
                vol_pred * 0.4 +           # Predicted volatility
                confidence * 0.4 +         # Prediction confidence  
                intraday_drop * 0.2        # Current market drop
            )
            
            opportunities.append({
                'asset_idx': asset_idx,
                'symbol': symbol,
                'vol_pred': vol_pred,
                'confidence': confidence,
                'opportunity_score': opportunity_score,
                'open_price': open_price,
                'current_low': current_low,
                'intraday_drop': intraday_drop
            })
        
        # Sort by opportunity score and select top N
        opportunities.sort(key=lambda x: x['opportunity_score'], reverse=True)
        selected = opportunities[:self.max_trades_per_period]
        
        return selected
    
    def execute_selective_trades(self, selected_opportunities, ohlcv_data):
        """
        Execute trades only on selected high-confidence opportunities.
        """
        trades = []
        total_pnl = 0.0
        total_fee_rate = 2 * 0.0002  # 0.04% total fees
        
        for opp in selected_opportunities:
            asset_idx = opp['asset_idx']
            symbol = opp['symbol']
            vol_pred = opp['vol_pred']
            
            # Get prices
            open_price = ohlcv_data[asset_idx, 0, 0]
            current_low = ohlcv_data[asset_idx, 0, 2]
            exit_price = ohlcv_data[asset_idx, 4, 3]  # T+4 close
            
            # Calculate limit price with higher discount for selectivity
            limit_price = open_price * (1 - vol_pred)
            
            # Check if order fills
            order_fills = current_low <= limit_price
            
            if order_fills:
                # Calculate profit with larger position size
                gross_profit_pct = (exit_price - limit_price) / limit_price
                net_profit_pct = gross_profit_pct - total_fee_rate
                trade_pnl = net_profit_pct * self.position_size
                total_pnl += trade_pnl
                
                trade_result = {
                    'symbol': symbol,
                    'vol_pred': vol_pred,
                    'confidence': opp['confidence'],
                    'opportunity_score': opp['opportunity_score'],
                    'open_price': open_price,
                    'limit_price': limit_price,
                    'current_low': current_low,
                    'exit_price': exit_price,
                    'order_fills': True,
                    'gross_profit_pct': gross_profit_pct * 100,
                    'net_profit_pct': net_profit_pct * 100,
                    'trade_pnl': trade_pnl,
                    'position_size': self.position_size
                }
            else:
                trade_result = {
                    'symbol': symbol,
                    'vol_pred': vol_pred,
                    'confidence': opp['confidence'],
                    'opportunity_score': opp['opportunity_score'],
                    'open_price': open_price,
                    'limit_price': limit_price,
                    'current_low': current_low,
                    'exit_price': exit_price,
                    'order_fills': False,
                    'gross_profit_pct': 0,
                    'net_profit_pct': 0,
                    'trade_pnl': 0,
                    'position_size': self.position_size
                }
            
            trades.append(trade_result)
        
        return trades, total_pnl


def test_selective_strategy():
    """Test the selective trading strategy."""
    print("🎯 TESTING SELECTIVE TRADING STRATEGY")
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
    
    # Test different strategy configurations
    strategies = [
        {
            'name': 'Original (All Trades)',
            'confidence_threshold': 0.0,
            'vol_threshold': 0.0,
            'max_trades_per_period': 38,
            'position_size': 100
        },
        {
            'name': 'Selective (High Confidence)',
            'confidence_threshold': 0.7,
            'vol_threshold': 0.015,  # 1.5%
            'max_trades_per_period': 15,
            'position_size': 200
        },
        {
            'name': 'Ultra Selective (Very High Confidence)',
            'confidence_threshold': 0.8,
            'vol_threshold': 0.02,   # 2.0%
            'max_trades_per_period': 10,
            'position_size': 300
        },
        {
            'name': 'Hyper Selective (Extreme Confidence)',
            'confidence_threshold': 0.9,
            'vol_threshold': 0.025,  # 2.5%
            'max_trades_per_period': 5,
            'position_size': 500
        }
    ]
    
    # Test each strategy
    test_periods = test_indices[:20]  # Test on 20 periods
    
    results = {}
    
    for strategy_config in strategies:
        print(f"\n🧪 TESTING: {strategy_config['name']}")
        print("-" * 50)
        
        strategy = SelectiveTradingStrategy(
            model=model,
            confidence_threshold=strategy_config['confidence_threshold'],
            vol_threshold=strategy_config['vol_threshold'],
            max_trades_per_period=strategy_config['max_trades_per_period'],
            position_size=strategy_config['position_size']
        )
        
        total_pnl = 0.0
        total_trades = 0
        total_filled = 0
        all_trades = []
        
        with torch.no_grad():
            for period_idx, idx in enumerate(test_periods):
                sample = dataset[idx]
                sample_info = dataset.get_sample_info(idx)
                
                x_lags = [x.unsqueeze(0) for x in sample['x_lags']]
                vol_pred = model(*x_lags)
                vol_pred_np = vol_pred.squeeze().cpu().numpy()
                
                ohlcv_data = sample['ohlcv_data'].numpy()
                
                # Calculate confidence scores
                confidence_scores = strategy.calculate_prediction_confidence(vol_pred_np)
                
                # Select opportunities
                selected_opportunities = strategy.select_best_opportunities(
                    vol_pred_np, confidence_scores, symbols, ohlcv_data
                )
                
                # Execute trades
                period_trades, period_pnl = strategy.execute_selective_trades(
                    selected_opportunities, ohlcv_data
                )
                
                total_pnl += period_pnl
                total_trades += len(period_trades)
                total_filled += sum(1 for t in period_trades if t['order_fills'])
                all_trades.extend(period_trades)
        
        # Calculate metrics
        fill_rate = total_filled / total_trades if total_trades > 0 else 0
        avg_pnl_per_trade = total_pnl / total_filled if total_filled > 0 else 0
        trades_per_period = total_trades / len(test_periods)
        
        results[strategy_config['name']] = {
            'total_pnl': total_pnl,
            'total_trades': total_trades,
            'total_filled': total_filled,
            'fill_rate': fill_rate,
            'avg_pnl_per_trade': avg_pnl_per_trade,
            'trades_per_period': trades_per_period,
            'all_trades': all_trades
        }
        
        print(f"  Total PnL: ${total_pnl:.2f}")
        print(f"  Total Trades: {total_trades}")
        print(f"  Fill Rate: {fill_rate:.1%}")
        print(f"  Avg PnL per Filled Trade: ${avg_pnl_per_trade:.2f}")
        print(f"  Trades per Period: {trades_per_period:.1f}")
    
    # Compare strategies
    print(f"\n" + "="*70)
    print("📊 STRATEGY COMPARISON")
    print("="*70)
    
    comparison_df = pd.DataFrame({
        name: {
            'Total PnL': f"${result['total_pnl']:.2f}",
            'Fill Rate': f"{result['fill_rate']:.1%}",
            'Trades/Period': f"{result['trades_per_period']:.1f}",
            'PnL/Trade': f"${result['avg_pnl_per_trade']:.2f}",
            'Total Trades': result['total_trades']
        }
        for name, result in results.items()
    }).T
    
    print(comparison_df)
    
    # Recommendations
    print(f"\n🎯 RECOMMENDATIONS:")
    
    best_total_pnl = max(results.values(), key=lambda x: x['total_pnl'])
    best_per_trade = max(results.values(), key=lambda x: x['avg_pnl_per_trade'])
    lowest_fill_rate = min(results.values(), key=lambda x: x['fill_rate'])
    
    print(f"✅ Highest Total PnL: {[k for k, v in results.items() if v == best_total_pnl][0]}")
    print(f"✅ Highest PnL per Trade: {[k for k, v in results.items() if v == best_per_trade][0]}")
    print(f"✅ Most Selective (Lowest Fill Rate): {[k for k, v in results.items() if v == lowest_fill_rate][0]}")
    
    return results


if __name__ == "__main__":
    results = test_selective_strategy()
