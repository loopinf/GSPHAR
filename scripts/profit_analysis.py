#!/usr/bin/env python
"""
Detailed profit analysis for the volatility-based trading strategy.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.trading_loss import TradingStrategyLoss, convert_log_returns_to_pct_change, calculate_cumulative_return

def calculate_strategy_profit(current_price, vol_prediction, actual_next_return, holding_period_returns):
    """
    Calculate the exact profit from the trading strategy.
    
    Args:
        current_price (float): Current asset price
        vol_prediction (float): Predicted volatility (e.g., 0.05 for 5%)
        actual_next_return (float): Actual return in next period (e.g., -0.04 for -4%)
        holding_period_returns (list): Returns during holding period
    
    Returns:
        dict: Detailed profit analysis
    """
    
    # Step 1: Calculate limit order price
    limit_price = current_price * (1 - vol_prediction)
    
    # Step 2: Calculate actual next price
    actual_next_price = current_price * (1 + actual_next_return)
    
    # Step 3: Check if order fills
    order_fills = actual_next_price <= limit_price
    
    if not order_fills:
        return {
            'order_fills': False,
            'limit_price': limit_price,
            'actual_next_price': actual_next_price,
            'entry_price': None,
            'exit_price': None,
            'profit_loss': 0.0,
            'profit_percentage': 0.0,
            'total_return': 0.0
        }
    
    # Step 4: Calculate entry price (we get filled at the actual price, not limit price)
    entry_price = actual_next_price
    
    # Step 5: Calculate exit price after holding period
    cumulative_holding_return = sum(holding_period_returns)
    exit_price = entry_price * (1 + cumulative_holding_return)
    
    # Step 6: Calculate profit/loss
    profit_loss = exit_price - entry_price
    profit_percentage = (exit_price - entry_price) / entry_price
    total_return = (exit_price - current_price) / current_price
    
    return {
        'order_fills': True,
        'limit_price': limit_price,
        'actual_next_price': actual_next_price,
        'entry_price': entry_price,
        'exit_price': exit_price,
        'profit_loss': profit_loss,
        'profit_percentage': profit_percentage,
        'total_return': total_return
    }

def analyze_profit_scenarios():
    """Analyze profit under various scenarios."""
    
    print("="*80)
    print("TRADING STRATEGY PROFIT ANALYSIS")
    print("="*80)
    
    # Base parameters
    current_price = 50000  # e.g., Bitcoin at $50,000
    holding_period = 24    # 24 hours
    
    # Test scenarios
    scenarios = [
        {
            "name": "Perfect Prediction - Bull Market",
            "vol_pred": 0.05,
            "actual_next": -0.05,  # Exactly as predicted
            "holding_returns": [0.001] * 24,  # 0.1% per hour = 2.4% total
            "description": "Order fills exactly as predicted, market goes up during holding"
        },
        {
            "name": "Under-prediction - Bull Market", 
            "vol_pred": 0.03,
            "actual_next": -0.05,  # Bigger drop than predicted
            "holding_returns": [0.002] * 24,  # 0.2% per hour = 4.8% total
            "description": "Bigger drop than expected, strong recovery"
        },
        {
            "name": "Over-prediction - No Fill",
            "vol_pred": 0.08,
            "actual_next": -0.03,  # Smaller drop than predicted
            "holding_returns": [0.001] * 24,  # Irrelevant
            "description": "Predicted too big a drop, order doesn't fill"
        },
        {
            "name": "Good Prediction - Bear Market",
            "vol_pred": 0.04,
            "actual_next": -0.045,  # Slightly bigger drop
            "holding_returns": [-0.001] * 24,  # -0.1% per hour = -2.4% total
            "description": "Order fills, but market continues down"
        },
        {
            "name": "Extreme Loss Scenario",
            "vol_pred": 0.03,
            "actual_next": -0.04,  # 4% drop
            "holding_returns": [-0.005] * 24,  # -0.5% per hour = -12% total
            "description": "Order fills into a major crash"
        },
        {
            "name": "Small Volatility - Sideways Market",
            "vol_pred": 0.01,
            "actual_next": -0.012,  # Small drop
            "holding_returns": [0.0002] * 24,  # 0.02% per hour = 0.48% total
            "description": "Small movements, minimal profit"
        },
        {
            "name": "High Volatility - Recovery",
            "vol_pred": 0.10,
            "actual_next": -0.12,  # Big drop
            "holding_returns": [0.008] * 24,  # 0.8% per hour = 19.2% total
            "description": "Major drop followed by strong recovery"
        }
    ]
    
    results = []
    
    print(f"Base Parameters:")
    print(f"  Current Price: ${current_price:,.2f}")
    print(f"  Holding Period: {holding_period} hours")
    print(f"  Capital: $10,000 (example)")
    print()
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"SCENARIO {i}: {scenario['name']}")
        print("-" * 60)
        print(f"Description: {scenario['description']}")
        print()
        
        # Calculate profit
        profit_analysis = calculate_strategy_profit(
            current_price=current_price,
            vol_prediction=scenario['vol_pred'],
            actual_next_return=scenario['actual_next'],
            holding_period_returns=scenario['holding_returns']
        )
        
        # Store results
        result = {
            'scenario': scenario['name'],
            'vol_pred': scenario['vol_pred'],
            'actual_next': scenario['actual_next'],
            'holding_total': sum(scenario['holding_returns']),
            **profit_analysis
        }
        results.append(result)
        
        # Print detailed analysis
        print(f"Strategy Details:")
        print(f"  Predicted Volatility: {scenario['vol_pred']:.1%}")
        print(f"  Limit Order Price: ${profit_analysis['limit_price']:,.2f}")
        print(f"  Actual Next Price: ${profit_analysis['actual_next_price']:,.2f}")
        print(f"  Order Fills: {'YES' if profit_analysis['order_fills'] else 'NO'}")
        
        if profit_analysis['order_fills']:
            print(f"  Entry Price: ${profit_analysis['entry_price']:,.2f}")
            print(f"  Exit Price: ${profit_analysis['exit_price']:,.2f}")
            print(f"  Holding Period Return: {sum(scenario['holding_returns']):.2%}")
            print()
            print(f"Profit Analysis:")
            print(f"  Profit/Loss per Share: ${profit_analysis['profit_loss']:,.2f}")
            print(f"  Profit Percentage: {profit_analysis['profit_percentage']:.2%}")
            print(f"  Total Return: {profit_analysis['total_return']:.2%}")
            
            # Calculate with example capital
            capital = 10000
            shares = capital / profit_analysis['entry_price']
            total_profit = shares * profit_analysis['profit_loss']
            final_value = capital + total_profit
            
            print()
            print(f"With $10,000 Capital:")
            print(f"  Shares Purchased: {shares:.4f}")
            print(f"  Total Profit/Loss: ${total_profit:,.2f}")
            print(f"  Final Portfolio Value: ${final_value:,.2f}")
            print(f"  ROI: {(final_value - capital) / capital:.2%}")
        else:
            print(f"  No position taken (order didn't fill)")
            print(f"  Capital remains: $10,000")
            print(f"  ROI: 0.00%")
        
        print()
    
    return results

def create_profit_visualization(results):
    """Create visualizations of profit scenarios."""
    
    # Filter only scenarios where orders filled
    filled_results = [r for r in results if r['order_fills']]
    
    if not filled_results:
        print("No scenarios with filled orders to visualize.")
        return
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Trading Strategy Profit Analysis', fontsize=16)
    
    # Plot 1: Profit vs Volatility Prediction
    ax1 = axes[0, 0]
    vol_preds = [r['vol_pred'] for r in filled_results]
    profits = [r['profit_percentage'] for r in filled_results]
    colors = ['green' if p > 0 else 'red' for p in profits]
    
    bars1 = ax1.bar(range(len(filled_results)), profits, color=colors, alpha=0.7)
    ax1.set_xticks(range(len(filled_results)))
    ax1.set_xticklabels([f"{v:.1%}" for v in vol_preds], rotation=45)
    ax1.set_xlabel('Predicted Volatility')
    ax1.set_ylabel('Profit Percentage')
    ax1.set_title('Profit vs Volatility Prediction')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Add value labels on bars
    for bar, profit in zip(bars1, profits):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + (0.005 if height >= 0 else -0.015),
                f'{profit:.1%}', ha='center', va='bottom' if height >= 0 else 'top')
    
    # Plot 2: Entry vs Exit Prices
    ax2 = axes[0, 1]
    entry_prices = [r['entry_price'] for r in filled_results]
    exit_prices = [r['exit_price'] for r in filled_results]
    scenario_names = [r['scenario'][:20] + '...' if len(r['scenario']) > 20 else r['scenario'] 
                     for r in filled_results]
    
    x = np.arange(len(filled_results))
    width = 0.35
    
    bars2 = ax2.bar(x - width/2, entry_prices, width, label='Entry Price', alpha=0.7)
    bars3 = ax2.bar(x + width/2, exit_prices, width, label='Exit Price', alpha=0.7)
    
    ax2.set_xlabel('Scenarios')
    ax2.set_ylabel('Price ($)')
    ax2.set_title('Entry vs Exit Prices')
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenario_names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Total Return Distribution
    ax3 = axes[1, 0]
    total_returns = [r['total_return'] for r in filled_results]
    colors = ['green' if r > 0 else 'red' for r in total_returns]
    
    bars4 = ax3.bar(range(len(filled_results)), total_returns, color=colors, alpha=0.7)
    ax3.set_xticks(range(len(filled_results)))
    ax3.set_xticklabels([f"S{i+1}" for i in range(len(filled_results))])
    ax3.set_xlabel('Scenario')
    ax3.set_ylabel('Total Return')
    ax3.set_title('Total Return by Scenario')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Add value labels
    for bar, ret in zip(bars4, total_returns):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + (0.005 if height >= 0 else -0.015),
                f'{ret:.1%}', ha='center', va='bottom' if height >= 0 else 'top')
    
    # Plot 4: Risk-Return Scatter
    ax4 = axes[1, 1]
    holding_returns = [r['holding_total'] for r in filled_results]
    
    scatter = ax4.scatter(holding_returns, total_returns, 
                         c=[r['vol_pred'] for r in filled_results], 
                         cmap='viridis', s=100, alpha=0.7)
    
    ax4.set_xlabel('Holding Period Return')
    ax4.set_ylabel('Total Strategy Return')
    ax4.set_title('Risk-Return Analysis')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax4.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Predicted Volatility')
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs('plots/profit_analysis', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f'plots/profit_analysis/trading_strategy_profits_{timestamp}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved profit analysis plot to: {plot_path}")
    
    return plot_path

def calculate_expected_profit():
    """Calculate expected profit based on historical data patterns."""
    
    print("\n" + "="*80)
    print("EXPECTED PROFIT CALCULATION")
    print("="*80)
    
    print("""
STRATEGY SUMMARY:
1. Place limit buy order at: Current_Price × (1 - Predicted_Volatility)
2. Order fills if actual price drops to or below limit price
3. Hold position for 24 hours
4. Exit at market price

PROFIT FORMULA:
- Entry Price = Actual price when order fills
- Exit Price = Entry Price × (1 + Holding_Period_Return)
- Profit = Exit Price - Entry Price
- ROI = (Exit Price - Entry Price) / Entry Price

KEY INSIGHTS FROM ANALYSIS:
1. Strategy only profits when:
   - Order fills (actual drop ≥ predicted drop)
   - Market recovers during holding period

2. Risk factors:
   - Over-prediction → No fills → No profit
   - Under-prediction → Good fills but may miss bigger opportunities
   - Market continues down during holding period → Losses

3. Optimal conditions:
   - Accurate volatility predictions
   - Mean-reverting market behavior
   - Positive market trend during holding period
    """)

def main():
    """Main function."""
    print("Analyzing trading strategy profits...")
    
    # Run profit analysis
    results = analyze_profit_scenarios()
    
    # Create visualizations
    plot_path = create_profit_visualization(results)
    
    # Calculate expected profits
    calculate_expected_profit()
    
    # Summary statistics
    filled_results = [r for r in results if r['order_fills']]
    if filled_results:
        avg_profit = np.mean([r['profit_percentage'] for r in filled_results])
        win_rate = len([r for r in filled_results if r['profit_percentage'] > 0]) / len(filled_results)
        
        print(f"\nSTRATEGY PERFORMANCE SUMMARY:")
        print(f"  Scenarios with fills: {len(filled_results)}/{len(results)}")
        print(f"  Average profit per trade: {avg_profit:.2%}")
        print(f"  Win rate: {win_rate:.1%}")
        print(f"  Best trade: {max(r['profit_percentage'] for r in filled_results):.2%}")
        print(f"  Worst trade: {min(r['profit_percentage'] for r in filled_results):.2%}")

if __name__ == '__main__':
    main()
