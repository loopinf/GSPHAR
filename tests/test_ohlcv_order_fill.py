#!/usr/bin/env python
"""
Test script to demonstrate the correct way to check order fills using OHLCV data.

This shows why we need the LOW price to properly determine if a limit order fills,
not just the closing price.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def create_ohlcv_scenario():
    """
    Create a realistic OHLCV scenario that demonstrates the order fill issue.
    """
    print("🎯 OHLCV ORDER FILL DEMONSTRATION")
    print("="*80)
    
    # Create realistic OHLCV data
    dates = pd.date_range('2024-01-01', periods=20, freq='h')
    
    # Scenario: Price opens high, drops low (fills order), then recovers
    ohlcv_data = [
        # [Open, High, Low, Close, Volume]
        [100.00, 102.00, 99.00, 101.50, 1000],   # T+0: Prediction time
        [101.50, 103.00, 95.00, 102.00, 1200],   # T+1: Drops to 95, recovers to 102
        [102.00, 104.00, 101.00, 103.50, 1100],  # T+2: Normal movement
        [103.50, 105.00, 102.50, 104.00, 1050],  # T+3: Continues up
        [104.00, 106.00, 103.00, 105.50, 1150],  # T+4: More gains
        [105.50, 107.00, 104.50, 106.00, 1080],  # T+5: Steady rise
        [106.00, 108.00, 105.00, 107.50, 1200],  # T+6: Continued growth
        [107.50, 109.00, 106.50, 108.00, 1100],  # T+7: More gains
        [108.00, 110.00, 107.00, 109.50, 1250],  # T+8: Strong performance
        [109.50, 111.00, 108.50, 110.00, 1180],  # T+9: Final period
    ]
    
    # Extend with more periods
    for i in range(10):
        last_close = ohlcv_data[-1][3]
        change = np.random.normal(0, 0.01)
        new_open = last_close
        new_high = new_open * (1 + abs(change) + 0.005)
        new_low = new_open * (1 - abs(change) - 0.005)
        new_close = new_open * (1 + change)
        new_volume = np.random.randint(1000, 1300)
        
        ohlcv_data.append([new_open, new_high, new_low, new_close, new_volume])
    
    # Create DataFrame
    df = pd.DataFrame(ohlcv_data[:len(dates)], 
                     columns=['open', 'high', 'low', 'close', 'volume'],
                     index=dates)
    
    return df


def demonstrate_order_fill_difference(df):
    """
    Demonstrate the difference between checking close vs low for order fills.
    """
    print("\n📊 ORDER FILL CHECK COMPARISON")
    print("="*60)
    
    # Trading scenario
    prediction_idx = 0
    vol_prediction = 0.04  # 4% volatility prediction
    holding_period = 8
    
    prediction_time = df.index[prediction_idx]
    current_price = df.iloc[prediction_idx]['close']
    limit_price = current_price * (1 - vol_prediction)
    
    print(f"📅 PREDICTION TIME: {prediction_time}")
    print(f"💰 CURRENT PRICE (Close): ${current_price:.2f}")
    print(f"📊 PREDICTED VOLATILITY: {vol_prediction:.1%}")
    print(f"🎯 LIMIT ORDER PRICE: ${limit_price:.2f}")
    print()
    
    # Check next period
    next_time = df.index[prediction_idx + 1]
    next_ohlc = df.iloc[prediction_idx + 1]
    
    print(f"⏰ NEXT PERIOD ({next_time}):")
    print(f"   Open:  ${next_ohlc['open']:.2f}")
    print(f"   High:  ${next_ohlc['high']:.2f}")
    print(f"   Low:   ${next_ohlc['low']:.2f}")
    print(f"   Close: ${next_ohlc['close']:.2f}")
    print()
    
    # Method 1: Check only closing price (WRONG)
    order_filled_close = next_ohlc['close'] <= limit_price
    print(f"❌ WRONG METHOD (Close price only):")
    print(f"   Close ${next_ohlc['close']:.2f} <= Limit ${limit_price:.2f}? {order_filled_close}")
    print(f"   Order filled: {'YES' if order_filled_close else 'NO'}")
    print()
    
    # Method 2: Check low price (CORRECT)
    order_filled_low = next_ohlc['low'] <= limit_price
    print(f"✅ CORRECT METHOD (Low price check):")
    print(f"   Low ${next_ohlc['low']:.2f} <= Limit ${limit_price:.2f}? {order_filled_low}")
    print(f"   Order filled: {'YES' if order_filled_low else 'NO'}")
    print()
    
    # Show the impact
    if order_filled_close != order_filled_low:
        print(f"🚨 CRITICAL DIFFERENCE:")
        print(f"   Close-based check: {'FILL' if order_filled_close else 'NO FILL'}")
        print(f"   Low-based check:   {'FILL' if order_filled_low else 'NO FILL'}")
        print(f"   This affects profit calculation and model training!")
    else:
        print(f"✅ Both methods agree: {'FILL' if order_filled_low else 'NO FILL'}")
    
    return order_filled_close, order_filled_low, limit_price


def calculate_correct_profit_with_ohlcv(df, prediction_idx, vol_prediction, holding_period):
    """
    Calculate profit using correct OHLCV-based order fill logic.
    """
    print("\n" + "="*80)
    print("CORRECT PROFIT CALCULATION WITH OHLCV")
    print("="*80)
    
    # Setup
    current_price = df.iloc[prediction_idx]['close']
    limit_price = current_price * (1 - vol_prediction)
    
    # Check if order fills using LOW price
    next_period = df.iloc[prediction_idx + 1]
    order_fills = next_period['low'] <= limit_price
    
    print(f"📊 TRADING SETUP:")
    print(f"   Current price: ${current_price:.2f}")
    print(f"   Limit price: ${limit_price:.2f}")
    print(f"   Next period low: ${next_period['low']:.2f}")
    print(f"   Order fills: {'YES' if order_fills else 'NO'}")
    print()
    
    if order_fills:
        # Determine actual entry price
        # If low <= limit_price, we get filled at the limit price
        entry_price = limit_price
        
        # Calculate profit over holding period
        exit_price = df.iloc[prediction_idx + holding_period]['close']
        holding_profit = (exit_price - entry_price) / entry_price
        
        print(f"📈 POSITION TRACKING:")
        print(f"   Entry price: ${entry_price:.2f}")
        print(f"   Exit price: ${exit_price:.2f}")
        print(f"   Holding profit: {holding_profit:.4f} ({holding_profit*100:.2f}%)")
        print()
        
        # Show period-by-period evolution
        print(f"📊 PRICE EVOLUTION:")
        print(f"{'Period':<8} {'OHLC':<25} {'Profit from Entry':<15}")
        print("-" * 60)
        
        for i in range(min(holding_period + 1, len(df) - prediction_idx)):
            period_data = df.iloc[prediction_idx + i]
            period_time = df.index[prediction_idx + i]
            
            if i == 0:
                print(f"T+{i:<6} O:{period_data['open']:6.2f} H:{period_data['high']:6.2f} "
                      f"L:{period_data['low']:6.2f} C:{period_data['close']:6.2f}   (Prediction)")
            elif i == 1:
                profit_at_close = ((period_data['close'] - entry_price) / entry_price) * 100
                print(f"T+{i:<6} O:{period_data['open']:6.2f} H:{period_data['high']:6.2f} "
                      f"L:{period_data['low']:6.2f} C:{period_data['close']:6.2f}   {profit_at_close:+6.2f}% (Entry)")
            else:
                profit_at_close = ((period_data['close'] - entry_price) / entry_price) * 100
                print(f"T+{i:<6} O:{period_data['open']:6.2f} H:{period_data['high']:6.2f} "
                      f"L:{period_data['low']:6.2f} C:{period_data['close']:6.2f}   {profit_at_close:+6.2f}%")
        
        return holding_profit, True
    else:
        print(f"❌ Order does not fill - no profit calculation")
        return 0.0, False


def create_corrected_ohlcv_loss_function():
    """
    Create a corrected loss function that uses OHLCV data properly.
    """
    print("\n" + "="*80)
    print("CORRECTED OHLCV-BASED LOSS FUNCTION")
    print("="*80)
    
    class OHLCVProfitMaximizationLoss(torch.nn.Module):
        """
        Corrected loss function that uses OHLCV data for proper order fill detection.
        """
        
        def __init__(self, holding_period=24):
            super().__init__()
            self.holding_period = holding_period
        
        def forward(self, vol_pred, ohlcv_data, prediction_idx):
            """
            Calculate loss using OHLCV data for proper order fill detection.
            
            Args:
                vol_pred: Volatility prediction [batch_size]
                ohlcv_data: OHLCV tensor [batch_size, sequence_length, 5] (O,H,L,C,V)
                prediction_idx: Index where prediction is made
            """
            batch_size = vol_pred.shape[0]
            
            # Clamp volatility prediction
            vol_pred = torch.clamp(vol_pred, 0.001, 0.5)
            
            # Get current price (close at prediction time)
            current_price = ohlcv_data[:, prediction_idx, 3]  # Close price
            
            # Calculate limit order price
            limit_price = current_price * (1 - vol_pred)
            
            # Check if order fills using LOW price of next period
            next_low = ohlcv_data[:, prediction_idx + 1, 2]  # Low price
            
            # Order fills if low price touches or goes below limit price
            order_fills = (next_low <= limit_price).float()
            
            # Use smooth approximation for gradients
            price_diff = (limit_price - next_low) / limit_price
            fill_probability = torch.sigmoid(price_diff * 100)
            
            # Calculate holding period profit from entry price to exit price
            exit_price = ohlcv_data[:, prediction_idx + self.holding_period, 3]  # Close at exit
            holding_profit = (exit_price - limit_price) / limit_price
            
            # Expected profit (weighted by fill probability)
            expected_profit = fill_probability * holding_profit
            
            # Return negative profit (minimizing this maximizes profit)
            return -expected_profit.mean()
    
    print("✅ OHLCV-based loss function created!")
    print()
    print("🎯 KEY IMPROVEMENTS:")
    print("1. ✅ Uses LOW price to check order fills")
    print("2. ✅ Calculates profit from actual entry price (limit price)")
    print("3. ✅ Accounts for intraday price movements")
    print("4. ✅ Maintains gradient flow with smooth approximations")
    print("5. ✅ Realistic trading simulation")
    
    return OHLCVProfitMaximizationLoss


def visualize_ohlcv_order_fill(df, prediction_idx, vol_prediction, holding_period):
    """
    Create visualization showing OHLCV data and order fill logic.
    """
    print("\n" + "="*80)
    print("CREATING OHLCV VISUALIZATION")
    print("="*80)
    
    # Setup
    current_price = df.iloc[prediction_idx]['close']
    limit_price = current_price * (1 - vol_prediction)
    
    # Create candlestick-style plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle('OHLCV-Based Order Fill Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: OHLCV Candlestick Chart
    plot_range = slice(max(0, prediction_idx - 2), min(len(df), prediction_idx + holding_period + 2))
    plot_df = df.iloc[plot_range]
    
    # Create candlestick representation
    for i, (idx, row) in enumerate(plot_df.iterrows()):
        x = i
        
        # Candlestick body
        body_color = 'green' if row['close'] >= row['open'] else 'red'
        body_height = abs(row['close'] - row['open'])
        body_bottom = min(row['open'], row['close'])
        
        # Draw high-low line
        ax1.plot([x, x], [row['low'], row['high']], 'k-', linewidth=1)
        
        # Draw body
        ax1.bar(x, body_height, bottom=body_bottom, width=0.6, 
               color=body_color, alpha=0.7, edgecolor='black')
        
        # Mark special periods
        if i == prediction_idx - plot_range.start:
            ax1.text(x, row['high'] + 1, 'PRED', ha='center', fontweight='bold', color='blue')
        elif i == prediction_idx + 1 - plot_range.start:
            ax1.text(x, row['high'] + 1, 'FILL?', ha='center', fontweight='bold', color='orange')
    
    # Add limit price line
    ax1.axhline(y=limit_price, color='red', linestyle='--', linewidth=2, 
               label=f'Limit Price (${limit_price:.2f})', alpha=0.8)
    
    # Mark order fill point
    next_period_idx = prediction_idx + 1 - plot_range.start
    if next_period_idx < len(plot_df):
        next_row = plot_df.iloc[next_period_idx]
        if next_row['low'] <= limit_price:
            ax1.scatter([next_period_idx], [next_row['low']], 
                       color='green', s=100, zorder=5, 
                       label='Order Fills Here!', marker='v')
    
    ax1.set_title('OHLCV Chart: Order Fill Detection', fontsize=14)
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Set x-axis labels
    ax1.set_xticks(range(len(plot_df)))
    ax1.set_xticklabels([f'T+{i-prediction_idx+plot_range.start}' for i in range(len(plot_df))], 
                       rotation=45)
    
    # Plot 2: Profit Evolution
    if df.iloc[prediction_idx + 1]['low'] <= limit_price:
        profits = []
        times = []
        
        for i in range(holding_period + 1):
            if prediction_idx + i < len(df):
                period_price = df.iloc[prediction_idx + i]['close']
                if i == 0:
                    profit = 0  # Prediction time
                else:
                    profit = ((period_price - limit_price) / limit_price) * 100
                
                profits.append(profit)
                times.append(i)
        
        ax2.plot(times, profits, 'b-', linewidth=2, marker='o', label='Profit Evolution')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.fill_between(times, profits, alpha=0.3, 
                        color='green' if profits[-1] > 0 else 'red')
        
        # Mark final profit
        if profits:
            ax2.scatter([times[-1]], [profits[-1]], color='red', s=100, zorder=5,
                       label=f'Final: {profits[-1]:+.2f}%')
        
        ax2.set_title(f'Profit Evolution (Entry at ${limit_price:.2f})')
    else:
        ax2.text(0.5, 0.5, 'Order Did Not Fill\nNo Profit to Track', 
                ha='center', va='center', transform=ax2.transAxes, 
                fontsize=14, bbox=dict(boxstyle='round', facecolor='lightcoral'))
        ax2.set_title('No Position Taken')
    
    ax2.set_xlabel('Time Periods')
    ax2.set_ylabel('Profit (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs('plots/ohlcv_analysis', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f'plots/ohlcv_analysis/ohlcv_order_fill_{timestamp}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 OHLCV visualization saved to: {plot_path}")
    
    return plot_path


def main():
    """
    Main function to demonstrate OHLCV-based order fill logic.
    """
    print("🎯 OHLCV-BASED ORDER FILL ANALYSIS")
    print("="*80)
    
    # Create OHLCV data
    df = create_ohlcv_scenario()
    
    # Demonstrate the difference
    order_filled_close, order_filled_low, limit_price = demonstrate_order_fill_difference(df)
    
    # Calculate correct profit
    profit, filled = calculate_correct_profit_with_ohlcv(df, 0, 0.04, 8)
    
    # Create corrected loss function
    corrected_loss_fn = create_corrected_ohlcv_loss_function()
    
    # Create visualization
    plot_path = visualize_ohlcv_order_fill(df, 0, 0.04, 8)
    
    # Summary
    print("\n" + "="*80)
    print("📋 OHLCV ORDER FILL ANALYSIS SUMMARY")
    print("="*80)
    print("🚨 CRITICAL FINDINGS:")
    print(f"   Close-based fill check: {'FILL' if order_filled_close else 'NO FILL'}")
    print(f"   Low-based fill check:   {'FILL' if order_filled_low else 'NO FILL'}")
    
    if order_filled_close != order_filled_low:
        print(f"   ⚠️  METHODS DISAGREE - This affects model training!")
    else:
        print(f"   ✅ Methods agree")
    
    print()
    print("✅ IMPROVEMENTS IMPLEMENTED:")
    print("   1. Use LOW price for order fill detection")
    print("   2. Proper OHLCV-based profit calculation")
    print("   3. Realistic intraday price movement handling")
    print("   4. Corrected loss function for model training")
    print()
    print(f"📊 Visualization: {plot_path}")


if __name__ == '__main__':
    main()
