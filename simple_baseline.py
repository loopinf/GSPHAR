#!/usr/bin/env python3
"""
Simple Baseline Model - Clean and Reliable

This is a straightforward baseline for comparison with complex models.
Uses simple rolling volatility + basic mean reversion strategy.

Strategy:
1. Calculate rolling volatility (24-hour window)
2. Buy when price drops > 2 * volatility (oversold)
3. Sell after fixed holding period (24 hours)
4. Track performance metrics

This serves as the performance baseline for all future model improvements.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class SimpleBaseline:
    """Simple baseline trading model"""
    
    def __init__(self, vol_window=24, vol_threshold=2.0, holding_period=24, transaction_cost=0.001):
        self.vol_window = vol_window
        self.vol_threshold = vol_threshold  # Buy when price drops > threshold * volatility
        self.holding_period = holding_period
        self.transaction_cost = transaction_cost
        self.results = {}
        
    def load_data(self, data_path='data/simple/crypto_close_simple.csv'):
        """Load simple price data"""
        print(f"📊 Loading data from {data_path}")
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        
        # Keep only first 10 assets (the ones with data)
        df = df.iloc[:, :10].dropna()
        
        print(f"✅ Loaded {len(df)} rows, {len(df.columns)} assets")
        print(f"Assets: {list(df.columns)}")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
        
        return df
    
    def calculate_signals(self, prices):
        """Calculate trading signals based on price drops and volatility"""
        # Calculate returns and rolling volatility
        returns = prices.pct_change()
        volatility = returns.rolling(self.vol_window).std()
        
        # Calculate price changes
        price_changes = prices.pct_change()
        
        # Buy signal: price drops more than threshold * volatility (oversold condition)
        buy_signals = (price_changes < -self.vol_threshold * volatility) & (~volatility.isna())
        
        return buy_signals, volatility
    
    def backtest_asset(self, prices, buy_signals):
        """Backtest strategy for single asset"""
        trade_returns = []
        trade_dates = []
        
        i = 0
        while i < len(prices) - self.holding_period:
            if buy_signals.iloc[i]:
                # Buy at current price
                entry_price = prices.iloc[i]
                
                # Sell after holding period
                exit_price = prices.iloc[i + self.holding_period]
                
                # Calculate return
                trade_return = (exit_price - entry_price) / entry_price
                trade_return = trade_return - (2 * self.transaction_cost)  # Buy + sell costs
                
                trade_returns.append(trade_return)
                trade_dates.append(prices.index[i])
                
                # Skip ahead to avoid overlapping trades
                i += self.holding_period
            else:
                i += 1
        
        return trade_returns, trade_dates
    
    def calculate_metrics(self, trade_returns):
        """Calculate performance metrics"""
        if len(trade_returns) == 0:
            return {
                'total_return': 0,
                'avg_return': 0,
                'sharpe_ratio': 0,
                'win_rate': 0,
                'max_drawdown': 0,
                'total_trades': 0
            }
        
        trade_returns = np.array(trade_returns)
        
        total_return = trade_returns.sum() * 100
        avg_return = trade_returns.mean() * 100
        sharpe_ratio = trade_returns.mean() / trade_returns.std() if trade_returns.std() > 0 else 0
        win_rate = (trade_returns > 0).sum() / len(trade_returns) * 100
        
        # Calculate max drawdown
        cumulative = np.cumprod(1 + trade_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        return {
            'total_return': total_return,
            'avg_return': avg_return,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'total_trades': len(trade_returns)
        }
    
    def run_backtest(self, data_path='data/simple/crypto_close_simple.csv', train_split=0.7):
        """Run complete backtest"""
        print("🚀 Running Simple Baseline Model")
        print("="*50)
        
        # Load data
        prices = self.load_data(data_path)
        
        # Split data
        split_idx = int(len(prices) * train_split)
        test_prices = prices.iloc[split_idx:]
        
        print(f"📊 Test period: {test_prices.index[0]} to {test_prices.index[-1]} ({len(test_prices)} periods)")
        
        # Calculate signals
        buy_signals, volatility = self.calculate_signals(test_prices)
        
        print(f"🔧 Strategy: Buy when price drops > {self.vol_threshold} * volatility, hold for {self.holding_period}h")
        
        # Backtest each asset
        results = {}
        
        for asset in test_prices.columns:
            asset_prices = test_prices[asset]
            asset_signals = buy_signals[asset]
            
            # Run backtest
            trade_returns, trade_dates = self.backtest_asset(asset_prices, asset_signals)
            
            # Calculate metrics
            metrics = self.calculate_metrics(trade_returns)
            metrics['asset'] = asset
            metrics['signal_rate'] = asset_signals.sum() / len(asset_signals) * 100
            
            results[asset] = metrics
        
        self.results = results
        return results
    
    def print_results(self):
        """Print formatted results"""
        print("\n" + "="*80)
        print("SIMPLE BASELINE RESULTS")
        print("="*80)
        print(f"Vol Window: {self.vol_window}h | Vol Threshold: {self.vol_threshold} | Holding: {self.holding_period}h | Cost: {self.transaction_cost*100:.1f}%")
        print("-"*80)
        
        # Print individual asset results
        for asset, metrics in self.results.items():
            print(f"{asset:10} | Return: {metrics['total_return']:6.2f}% | Sharpe: {metrics['sharpe_ratio']:5.2f} | "
                  f"Win: {metrics['win_rate']:5.1f}% | Drawdown: {metrics['max_drawdown']:6.2f}% | "
                  f"Trades: {metrics['total_trades']:3.0f} | Signals: {metrics['signal_rate']:4.1f}%")
        
        print("-"*80)
        
        # Calculate summary
        total_return = sum(m['total_return'] for m in self.results.values())
        avg_sharpe = np.mean([m['sharpe_ratio'] for m in self.results.values()])
        avg_win_rate = np.mean([m['win_rate'] for m in self.results.values()])
        avg_drawdown = np.mean([m['max_drawdown'] for m in self.results.values()])
        total_trades = sum(m['total_trades'] for m in self.results.values())
        avg_signal_rate = np.mean([m['signal_rate'] for m in self.results.values()])
        
        print(f"{'SUMMARY':10} | Return: {total_return:6.2f}% | Sharpe: {avg_sharpe:5.2f} | "
              f"Win: {avg_win_rate:5.1f}% | Drawdown: {avg_drawdown:6.2f}% | "
              f"Trades: {total_trades:3.0f} | Signals: {avg_signal_rate:4.1f}%")
        
        print("="*80)
        
        return pd.DataFrame(self.results).T

def optimize_baseline():
    """Test different parameter combinations"""
    print("🔍 Optimizing Simple Baseline")
    print("="*50)
    
    # Parameter grid
    vol_thresholds = [1.5, 2.0, 2.5, 3.0]
    holding_periods = [12, 24, 48]
    
    best_return = -np.inf
    best_params = None
    all_results = []
    
    for vol_thresh in vol_thresholds:
        for hold_period in holding_periods:
            print(f"\n--- Testing vol_threshold={vol_thresh}, holding_period={hold_period}h ---")
            
            model = SimpleBaseline(
                vol_threshold=vol_thresh,
                holding_period=hold_period
            )
            
            results = model.run_backtest()
            results_df = model.print_results()
            
            total_return = results_df['total_return'].sum()
            avg_sharpe = results_df['sharpe_ratio'].mean()
            
            all_results.append({
                'vol_threshold': vol_thresh,
                'holding_period': hold_period,
                'total_return': total_return,
                'avg_sharpe': avg_sharpe
            })
            
            if total_return > best_return:
                best_return = total_return
                best_params = (vol_thresh, hold_period)
    
    print(f"\n🏆 BEST CONFIGURATION")
    print(f"Vol Threshold: {best_params[0]}")
    print(f"Holding Period: {best_params[1]}h")
    print(f"Total Return: {best_return:.2f}%")
    
    return pd.DataFrame(all_results), best_params

if __name__ == "__main__":
    # Run simple baseline
    print("Running Simple Baseline Model...")
    
    model = SimpleBaseline(vol_threshold=2.0, holding_period=24)
    results = model.run_backtest()
    results_df = model.print_results()
    
    print("\n" + "="*50)
    print("✅ SIMPLE BASELINE COMPLETE")
    print("="*50)
    print("This is your performance baseline for comparison.")
    print("Strategy: Buy on oversold conditions (price drops > 2*volatility)")
    print("Use this benchmark for all future model improvements.")
