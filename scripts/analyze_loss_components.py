#!/usr/bin/env python
"""
Analyze the balance between Fill Loss, Profit Loss, and Avoidance Loss components.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.trading_loss import TradingStrategyLoss, convert_pct_change_to_log_returns

def analyze_loss_components():
    """Analyze the individual components of TradingStrategyLoss."""
    
    print("="*80)
    print("TRADING LOSS COMPONENTS ANALYSIS")
    print("="*80)
    
    # Test different parameter combinations
    parameter_sets = [
        {"alpha": 1.0, "beta": 1.0, "gamma": 2.0, "name": "Original (α=1, β=1, γ=2)"},
        {"alpha": 1.0, "beta": 0.1, "gamma": 2.0, "name": "Reduced Profit (α=1, β=0.1, γ=2)"},
        {"alpha": 0.5, "beta": 1.0, "gamma": 2.0, "name": "Reduced Fill (α=0.5, β=1, γ=2)"},
        {"alpha": 1.0, "beta": 1.0, "gamma": 1.0, "name": "Reduced Avoidance (α=1, β=1, γ=1)"},
        {"alpha": 2.0, "beta": 0.5, "gamma": 1.0, "name": "Balanced (α=2, β=0.5, γ=1)"},
    ]
    
    # Create test scenarios
    scenarios = [
        {
            "name": "Perfect Prediction - Profitable",
            "vol_pred": 0.05,  # Predict 5% drop
            "actual_next": -0.05,  # Actual 5% drop (order fills)
            "holding_returns": [0.001] * 24,  # Small positive returns
        },
        {
            "name": "Under-prediction - Missed Opportunity", 
            "vol_pred": 0.03,  # Predict 3% drop
            "actual_next": -0.05,  # Actual 5% drop (order fills easily)
            "holding_returns": [0.002] * 24,  # Good returns
        },
        {
            "name": "Over-prediction - No Fill",
            "vol_pred": 0.08,  # Predict 8% drop
            "actual_next": -0.03,  # Actual 3% drop (order doesn't fill)
            "holding_returns": [0.001] * 24,  # Irrelevant (no position)
        },
        {
            "name": "Good Prediction - But Loss",
            "vol_pred": 0.04,  # Predict 4% drop
            "actual_next": -0.045,  # Actual 4.5% drop (order fills)
            "holding_returns": [-0.001] * 24,  # Negative returns
        },
        {
            "name": "Extreme Loss Scenario",
            "vol_pred": 0.03,  # Predict 3% drop
            "actual_next": -0.04,  # Actual 4% drop (order fills)
            "holding_returns": [-0.005] * 24,  # Large negative returns
        }
    ]
    
    # Analyze each parameter set
    results = []
    
    for param_set in parameter_sets:
        print(f"\n{'-'*60}")
        print(f"PARAMETER SET: {param_set['name']}")
        print(f"{'-'*60}")
        
        # Create loss function
        trading_loss = TradingStrategyLoss(
            alpha=param_set['alpha'],
            beta=param_set['beta'],
            gamma=param_set['gamma'],
            holding_period=24
        )
        
        param_results = []
        
        for scenario in scenarios:
            print(f"\nScenario: {scenario['name']}")
            print("-" * 40)
            
            # Convert to tensors
            vol_pred = torch.tensor([scenario['vol_pred']], dtype=torch.float32)
            log_returns = torch.tensor([[scenario['actual_next']] + scenario['holding_returns']], 
                                     dtype=torch.float32)
            
            # Calculate individual components
            log_entry_threshold = torch.log(1 - vol_pred)
            log_return_next = log_returns[:, 0]
            log_return_holding = torch.sum(log_returns[:, 1:], dim=1)
            
            # Fill Loss
            fill_loss = torch.max(torch.zeros_like(vol_pred), 
                                 log_entry_threshold - log_return_next)**2
            fill_loss_weighted = param_set['alpha'] * fill_loss
            
            # Determine if order filled
            filled_orders = (log_return_next <= log_entry_threshold).float()
            
            # Profit Loss
            profit_loss = -filled_orders * log_return_holding
            profit_loss_weighted = param_set['beta'] * profit_loss
            
            # Avoidance Loss
            avoidance_loss = torch.max(torch.zeros_like(vol_pred), 
                                      -filled_orders * log_return_holding)**2
            avoidance_loss_weighted = param_set['gamma'] * avoidance_loss
            
            # Total Loss
            total_loss = fill_loss_weighted + profit_loss_weighted + avoidance_loss_weighted
            
            # Store results
            scenario_result = {
                'scenario': scenario['name'],
                'vol_pred': scenario['vol_pred'],
                'actual_next': scenario['actual_next'],
                'order_filled': filled_orders.item() > 0,
                'holding_return': log_return_holding.item(),
                'fill_loss': fill_loss.item(),
                'fill_loss_weighted': fill_loss_weighted.item(),
                'profit_loss': profit_loss.item(),
                'profit_loss_weighted': profit_loss_weighted.item(),
                'avoidance_loss': avoidance_loss.item(),
                'avoidance_loss_weighted': avoidance_loss_weighted.item(),
                'total_loss': total_loss.item(),
                'param_set': param_set['name']
            }
            param_results.append(scenario_result)
            
            # Print detailed breakdown
            print(f"Predicted volatility: {scenario['vol_pred']:.3f} ({scenario['vol_pred']*100:.1f}%)")
            print(f"Actual next return: {scenario['actual_next']:.3f} ({scenario['actual_next']*100:.1f}%)")
            print(f"Order filled: {'Yes' if filled_orders.item() > 0 else 'No'}")
            print(f"Holding period return: {log_return_holding.item():.4f} ({(torch.exp(log_return_holding).item()-1)*100:.2f}%)")
            print(f"")
            print(f"Loss Components (unweighted):")
            print(f"  Fill Loss: {fill_loss.item():.6f}")
            print(f"  Profit Loss: {profit_loss.item():.6f}")
            print(f"  Avoidance Loss: {avoidance_loss.item():.6f}")
            print(f"")
            print(f"Loss Components (weighted):")
            print(f"  Fill Loss (α={param_set['alpha']}): {fill_loss_weighted.item():.6f}")
            print(f"  Profit Loss (β={param_set['beta']}): {profit_loss_weighted.item():.6f}")
            print(f"  Avoidance Loss (γ={param_set['gamma']}): {avoidance_loss_weighted.item():.6f}")
            print(f"  TOTAL LOSS: {total_loss.item():.6f}")
            
            # Show component percentages
            if total_loss.item() > 0:
                fill_pct = (fill_loss_weighted.item() / total_loss.item()) * 100
                profit_pct = (profit_loss_weighted.item() / total_loss.item()) * 100
                avoidance_pct = (avoidance_loss_weighted.item() / total_loss.item()) * 100
                print(f"")
                print(f"Component Percentages:")
                print(f"  Fill: {fill_pct:.1f}%")
                print(f"  Profit: {profit_pct:.1f}%")
                print(f"  Avoidance: {avoidance_pct:.1f}%")
        
        results.extend(param_results)
    
    return results

def create_comparison_plots(results):
    """Create comparison plots for different parameter sets."""
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Create plots directory
    os.makedirs('plots/loss_analysis', exist_ok=True)
    
    # Plot 1: Total loss comparison across scenarios
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Trading Loss Components Analysis', fontsize=16)
    
    # Get unique parameter sets and scenarios
    param_sets = df['param_set'].unique()
    scenarios = df['scenario'].unique()
    
    # Plot 1: Total Loss by Scenario
    ax1 = axes[0, 0]
    for param_set in param_sets:
        subset = df[df['param_set'] == param_set]
        ax1.plot(range(len(scenarios)), subset['total_loss'], 'o-', label=param_set, linewidth=2)
    ax1.set_xticks(range(len(scenarios)))
    ax1.set_xticklabels([s.replace(' - ', '\n') for s in scenarios], rotation=45, ha='right')
    ax1.set_ylabel('Total Loss')
    ax1.set_title('Total Loss by Scenario')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Fill Loss Component
    ax2 = axes[0, 1]
    for param_set in param_sets:
        subset = df[df['param_set'] == param_set]
        ax2.plot(range(len(scenarios)), subset['fill_loss_weighted'], 'o-', label=param_set, linewidth=2)
    ax2.set_xticks(range(len(scenarios)))
    ax2.set_xticklabels([s.replace(' - ', '\n') for s in scenarios], rotation=45, ha='right')
    ax2.set_ylabel('Fill Loss (Weighted)')
    ax2.set_title('Fill Loss Component')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Profit Loss Component
    ax3 = axes[0, 2]
    for param_set in param_sets:
        subset = df[df['param_set'] == param_set]
        ax3.plot(range(len(scenarios)), subset['profit_loss_weighted'], 'o-', label=param_set, linewidth=2)
    ax3.set_xticks(range(len(scenarios)))
    ax3.set_xticklabels([s.replace(' - ', '\n') for s in scenarios], rotation=45, ha='right')
    ax3.set_ylabel('Profit Loss (Weighted)')
    ax3.set_title('Profit Loss Component')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Avoidance Loss Component
    ax4 = axes[1, 0]
    for param_set in param_sets:
        subset = df[df['param_set'] == param_set]
        ax4.plot(range(len(scenarios)), subset['avoidance_loss_weighted'], 'o-', label=param_set, linewidth=2)
    ax4.set_xticks(range(len(scenarios)))
    ax4.set_xticklabels([s.replace(' - ', '\n') for s in scenarios], rotation=45, ha='right')
    ax4.set_ylabel('Avoidance Loss (Weighted)')
    ax4.set_title('Avoidance Loss Component')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Component Balance for Original Parameters
    ax5 = axes[1, 1]
    original_data = df[df['param_set'] == 'Original (α=1, β=1, γ=2)']
    components = ['fill_loss_weighted', 'profit_loss_weighted', 'avoidance_loss_weighted']
    component_names = ['Fill Loss', 'Profit Loss', 'Avoidance Loss']
    
    x = np.arange(len(scenarios))
    width = 0.25
    
    for i, (comp, name) in enumerate(zip(components, component_names)):
        ax5.bar(x + i*width, original_data[comp], width, label=name, alpha=0.8)
    
    ax5.set_xticks(x + width)
    ax5.set_xticklabels([s.replace(' - ', '\n') for s in scenarios], rotation=45, ha='right')
    ax5.set_ylabel('Loss Value')
    ax5.set_title('Component Balance (Original Parameters)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Loss Ratios
    ax6 = axes[1, 2]
    for param_set in param_sets:
        subset = df[df['param_set'] == param_set]
        # Calculate ratio of max component to total loss
        max_component = np.maximum.reduce([
            subset['fill_loss_weighted'].values,
            subset['profit_loss_weighted'].values,
            subset['avoidance_loss_weighted'].values
        ])
        ratios = max_component / (subset['total_loss'].values + 1e-8)  # Avoid division by zero
        ax6.plot(range(len(scenarios)), ratios, 'o-', label=param_set, linewidth=2)
    
    ax6.set_xticks(range(len(scenarios)))
    ax6.set_xticklabels([s.replace(' - ', '\n') for s in scenarios], rotation=45, ha='right')
    ax6.set_ylabel('Max Component / Total Loss')
    ax6.set_title('Component Dominance Ratio')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f'plots/loss_analysis/loss_components_analysis_{timestamp}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved analysis plot to: {plot_path}")
    
    return plot_path

def recommend_parameters(results):
    """Recommend optimal parameter settings based on analysis."""
    
    print("\n" + "="*80)
    print("PARAMETER RECOMMENDATIONS")
    print("="*80)
    
    df = pd.DataFrame(results)
    
    # Analyze each parameter set
    param_sets = df['param_set'].unique()
    
    print("\nParameter Set Performance Summary:")
    print("-" * 50)
    
    for param_set in param_sets:
        subset = df[df['param_set'] == param_set]
        
        # Calculate statistics
        avg_total_loss = subset['total_loss'].mean()
        max_total_loss = subset['total_loss'].max()
        min_total_loss = subset['total_loss'].min()
        
        # Calculate component balance (standard deviation of component percentages)
        total_losses = subset['total_loss'].values
        fill_pcts = (subset['fill_loss_weighted'].values / (total_losses + 1e-8)) * 100
        profit_pcts = (subset['profit_loss_weighted'].values / (total_losses + 1e-8)) * 100
        avoidance_pcts = (subset['avoidance_loss_weighted'].values / (total_losses + 1e-8)) * 100
        
        balance_score = np.std([fill_pcts.mean(), profit_pcts.mean(), avoidance_pcts.mean()])
        
        print(f"\n{param_set}:")
        print(f"  Average Total Loss: {avg_total_loss:.4f}")
        print(f"  Loss Range: {min_total_loss:.4f} - {max_total_loss:.4f}")
        print(f"  Component Balance Score: {balance_score:.2f} (lower = more balanced)")
        print(f"  Avg Component %: Fill={fill_pcts.mean():.1f}%, Profit={profit_pcts.mean():.1f}%, Avoidance={avoidance_pcts.mean():.1f}%")
    
    print(f"\n" + "="*60)
    print("RECOMMENDATIONS:")
    print("="*60)
    
    print("""
1. ORIGINAL PARAMETERS (α=1, β=1, γ=2):
   - Avoidance loss dominates due to γ=2
   - Good for risk-averse strategies
   - May over-penalize profitable but risky trades
   
2. REDUCED PROFIT (α=1, β=0.1, γ=2):
   - Reduces profit-seeking behavior
   - Good for conservative strategies
   - May miss profitable opportunities
   
3. BALANCED APPROACH (α=2, β=0.5, γ=1):
   - More emphasis on order filling
   - Moderate profit seeking
   - Reasonable loss avoidance
   - RECOMMENDED for most trading strategies
   
4. AGGRESSIVE PROFIT (α=0.5, β=2, γ=0.5):
   - High profit seeking
   - Lower loss avoidance
   - Good for high-risk/high-reward strategies
   
5. CONSERVATIVE (α=1, β=0.1, γ=3):
   - Very high loss avoidance
   - Low profit seeking
   - Good for capital preservation
    """)

def main():
    """Main function."""
    print("Analyzing TradingStrategyLoss component balance...")
    
    # Run analysis
    results = analyze_loss_components()
    
    # Create plots
    plot_path = create_comparison_plots(results)
    
    # Provide recommendations
    recommend_parameters(results)
    
    print(f"\nAnalysis complete! Check the plot at: {plot_path}")

if __name__ == '__main__':
    main()
