#!/usr/bin/env python
"""
Analyze the specific market period (Aug-Nov 2020) to understand why results are so good.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def analyze_crypto_performance():
    """Analyze crypto performance during Aug-Nov 2020."""
    print("📈 CRYPTO MARKET ANALYSIS: AUG-NOV 2020")
    print("=" * 60)
    
    # Get all OHLCV files
    ohlcv_files = glob.glob("data/ohlcv_1h/*.csv")
    
    if not ohlcv_files:
        print("❌ No OHLCV files found")
        return
    
    # Test period
    start_date = "2020-08-23"
    end_date = "2020-11-14"
    
    total_returns = []
    volatilities = []
    asset_names = []
    
    for file_path in ohlcv_files[:10]:  # Analyze first 10 assets
        try:
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            
            # Filter to test period
            mask = (df.index >= start_date) & (df.index <= end_date)
            period_df = df[mask]
            
            if len(period_df) < 100:  # Need sufficient data
                continue
                
            asset_name = os.path.basename(file_path).split('_')[0]
            
            # Calculate returns
            start_price = period_df['close'].iloc[0]
            end_price = period_df['close'].iloc[-1]
            total_return = (end_price - start_price) / start_price * 100
            
            # Calculate volatility
            hourly_returns = period_df['close'].pct_change().dropna()
            volatility = hourly_returns.std() * np.sqrt(24) * 100  # Daily volatility %
            
            # Calculate max drawdown
            cumulative = (1 + hourly_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min() * 100
            
            print(f"\n{asset_name}:")
            print(f"  Total Return: {total_return:.1f}%")
            print(f"  Daily Volatility: {volatility:.1f}%")
            print(f"  Max Drawdown: {max_drawdown:.1f}%")
            print(f"  Data Points: {len(period_df)}")
            
            # Check for extreme movements
            max_hourly_gain = hourly_returns.max() * 100
            max_hourly_loss = hourly_returns.min() * 100
            
            print(f"  Max Hourly Gain: {max_hourly_gain:.1f}%")
            print(f"  Max Hourly Loss: {max_hourly_loss:.1f}%")
            
            if total_return > 50:
                print(f"  🚀 MASSIVE BULL RUN!")
            elif total_return > 20:
                print(f"  📈 Strong Bull Market")
            elif total_return < -20:
                print(f"  📉 Bear Market")
            else:
                print(f"  📊 Normal Market")
                
            total_returns.append(total_return)
            volatilities.append(volatility)
            asset_names.append(asset_name)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Summary statistics
    if total_returns:
        print(f"\n📊 PORTFOLIO SUMMARY:")
        print(f"  Average Return: {np.mean(total_returns):.1f}%")
        print(f"  Median Return: {np.median(total_returns):.1f}%")
        print(f"  Best Performer: {np.max(total_returns):.1f}%")
        print(f"  Worst Performer: {np.min(total_returns):.1f}%")
        print(f"  Average Volatility: {np.mean(volatilities):.1f}%")
        
        # Check if this was an exceptional period
        positive_returns = sum(1 for r in total_returns if r > 0)
        print(f"  Positive Returns: {positive_returns}/{len(total_returns)} assets")
        
        if np.mean(total_returns) > 30:
            print(f"  🚨 EXCEPTIONAL BULL MARKET PERIOD!")
            print(f"  This explains the high profitability!")


def analyze_specific_trades():
    """Analyze specific profitable trades to understand the pattern."""
    print(f"\n🔍 ANALYZING SPECIFIC TRADE PATTERNS")
    print("=" * 60)
    
    # Load Bitcoin data as example
    btc_file = "data/ohlcv_1h/BTCUSDT_1h.csv"
    if not os.path.exists(btc_file):
        print("❌ Bitcoin data not found")
        return
    
    df = pd.read_csv(btc_file, index_col=0, parse_dates=True)
    
    # Filter to test period
    start_date = "2020-08-23"
    end_date = "2020-11-14"
    mask = (df.index >= start_date) & (df.index <= end_date)
    period_df = df[mask].copy()
    
    print(f"Analyzing {len(period_df)} hours of Bitcoin data")
    
    # Simulate the trading strategy
    vol_prediction = 0.022  # 2.2% average from our model
    profitable_trades = 0
    total_trades = 0
    total_profit = 0
    
    for i in range(len(period_df) - 4):
        # Current prices
        current_price = period_df['close'].iloc[i]
        next_low = period_df['low'].iloc[i + 1]
        exit_price = period_df['close'].iloc[i + 4]
        
        # Calculate limit order
        limit_price = current_price * (1 - vol_prediction)
        
        # Check if order fills
        if next_low <= limit_price:
            # Calculate profit
            gross_profit_pct = (exit_price - limit_price) / limit_price
            net_profit_pct = gross_profit_pct - 0.0004  # 0.04% fees
            
            if net_profit_pct > 0:
                profitable_trades += 1
            
            total_profit += net_profit_pct * 100  # $100 position
            total_trades += 1
    
    if total_trades > 0:
        win_rate = profitable_trades / total_trades
        avg_profit = total_profit / total_trades
        
        print(f"\nBitcoin Trading Simulation:")
        print(f"  Total Trades: {total_trades}")
        print(f"  Profitable Trades: {profitable_trades}")
        print(f"  Win Rate: {win_rate:.1%}")
        print(f"  Total Profit: ${total_profit:.2f}")
        print(f"  Average Profit per Trade: ${avg_profit:.2f}")
        
        if win_rate > 0.8:
            print(f"  🚨 EXTREMELY HIGH WIN RATE!")
            print(f"  This suggests favorable market conditions")


def check_volatility_predictions():
    """Check if 2.2% volatility predictions are reasonable."""
    print(f"\n📊 VOLATILITY PREDICTION ANALYSIS")
    print("=" * 60)
    
    # Load realized volatility data
    vol_file = "data/crypto_rv1h_38_20200822_20250116.csv"
    if not os.path.exists(vol_file):
        print("❌ Volatility data not found")
        return
    
    vol_df = pd.read_csv(vol_file, index_col=0, parse_dates=True)
    
    # Filter to test period
    start_date = "2020-08-23"
    end_date = "2020-11-14"
    mask = (vol_df.index >= start_date) & (vol_df.index <= end_date)
    period_vol = vol_df[mask]
    
    print(f"Analyzing {len(period_vol)} hours of volatility data")
    
    # Calculate statistics
    avg_vol = period_vol.mean().mean()
    median_vol = period_vol.median().median()
    std_vol = period_vol.std().mean()
    
    print(f"Realized Volatility Statistics:")
    print(f"  Average: {avg_vol:.4f} ({avg_vol*100:.2f}%)")
    print(f"  Median: {median_vol:.4f} ({median_vol*100:.2f}%)")
    print(f"  Std Dev: {std_vol:.4f} ({std_vol*100:.2f}%)")
    
    # Compare with model predictions
    model_pred = 0.022
    print(f"\nModel Prediction: {model_pred:.4f} ({model_pred*100:.2f}%)")
    
    if model_pred > avg_vol * 1.5:
        print(f"  ⚠️  Model over-predicting volatility")
    elif model_pred < avg_vol * 0.5:
        print(f"  ⚠️  Model under-predicting volatility")
    else:
        print(f"  ✅ Model predictions reasonable")
    
    # Check if low volatility period
    if avg_vol < 0.015:  # < 1.5%
        print(f"  📉 LOW VOLATILITY PERIOD")
        print(f"  This makes limit orders more likely to fill!")


def main():
    """Main analysis function."""
    print("🔍 MARKET PERIOD ANALYSIS: WHY RESULTS ARE SO GOOD?")
    print("=" * 80)
    
    analyze_crypto_performance()
    analyze_specific_trades()
    check_volatility_predictions()
    
    print(f"\n🎯 CONCLUSION:")
    print("=" * 40)
    print("The exceptional results may be due to:")
    print("1. 🚀 Bull market period (Aug-Nov 2020)")
    print("2. 📈 Strong crypto performance during this time")
    print("3. 📉 Relatively low volatility making fills easier")
    print("4. 🎯 Model predictions well-calibrated for this period")
    print("\nTo validate robustness, test on:")
    print("- Bear market periods (2022)")
    print("- High volatility periods")
    print("- Different market regimes")


if __name__ == "__main__":
    main()
