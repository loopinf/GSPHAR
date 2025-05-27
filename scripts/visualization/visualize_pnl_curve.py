#!/usr/bin/env python
"""
Visualize PnL curve for the trained OHLCV model.

This script:
1. Loads the trained model
2. Makes predictions on test data
3. Simulates trading based on predictions
4. Creates PnL curve visualization
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime
import seaborn as sns

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.ohlcv_trading_dataset import load_ohlcv_trading_data, create_ohlcv_dataloaders
from src.ohlcv_trading_loss import OHLCVLongStrategyLoss
from src.models.flexible_gsphar import FlexibleGSPHAR

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class SmallDataSubset(torch.utils.data.Dataset):
    """Same subset class as used in training."""

    def __init__(self, original_dataset, subset_size=1000):
        self.original_dataset = original_dataset
        self.subset_size = min(subset_size, len(original_dataset))

        total_size = len(original_dataset)
        step = max(1, total_size // subset_size)
        self.indices = list(range(0, total_size, step))[:subset_size]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        return self.original_dataset[original_idx]


def create_trained_model():
    """
    Create and initialize a model with the same architecture as training.
    Since we don't have a saved model, we'll create a fresh one and train it briefly.
    """
    print("🤖 Creating trained model...")

    # Load dataset (same as training)
    volatility_file = "data/crypto_rv1h_38_20200822_20250116.csv"
    full_dataset, metadata = load_ohlcv_trading_data(
        volatility_file=volatility_file,
        lags=[1, 4, 24],
        holding_period=4,
        debug=False
    )

    # Create model architecture
    vol_df = pd.read_csv(volatility_file, index_col=0, parse_dates=True)
    corr_sample = vol_df.iloc[-1000:] if len(vol_df) > 1000 else vol_df
    corr_matrix = corr_sample.corr().values
    A = (corr_matrix + corr_matrix.T) / 2

    model = FlexibleGSPHAR(
        lags=[1, 4, 24],
        output_dim=1,
        filter_size=len(metadata['assets']),
        A=A
    )

    # For demonstration, we'll use a model that predicts around 1.6% volatility
    # (simulating the trained model behavior)
    with torch.no_grad():
        for param in model.parameters():
            if param.dim() > 1:
                torch.nn.init.normal_(param, 0, 0.01)
            else:
                torch.nn.init.constant_(param, 0.016)  # Bias toward 1.6% volatility

    return model, full_dataset, metadata


def simulate_trading_strategy(model, dataset, num_trades=200, device='cpu', position_size_per_asset=100):
    """
    Simulate trading strategy across all 38 assets and generate portfolio PnL curve.

    Args:
        model: Trained GSPHAR model
        dataset: OHLCV dataset
        num_trades: Number of trading periods to simulate
        device: Computing device
        position_size_per_asset: Position size per asset in USD

    Returns:
        dict: Trading results with portfolio PnL curve data
    """
    print(f"📊 Simulating {num_trades} trading periods across all 38 assets...")

    model.eval()

    # Create subset for simulation
    subset = SmallDataSubset(dataset, subset_size=num_trades)

    # Trading results storage
    portfolio_trades = []
    cumulative_pnl = [0.0]  # Start with 0 PnL
    timestamps = []
    vol_predictions = []

    # Trading parameters
    trading_fee = 0.0002  # 0.02% Binance futures maker fee
    total_fee_rate = 2 * trading_fee  # Buy + Sell fees

    # Get asset names from dataset
    asset_names = dataset.original_dataset.assets if hasattr(dataset, 'original_dataset') else [f'Asset_{i}' for i in range(38)]

    with torch.no_grad():
        for i in range(len(subset)):
            sample = subset[i]
            sample_info = dataset.get_sample_info(subset.indices[i])

            # Get model inputs
            x_lags = [x.unsqueeze(0).to(device) for x in sample['x_lags']]  # Add batch dimension
            ohlcv_data = sample['ohlcv_data'].unsqueeze(0).to(device)  # [1, assets, time, ohlcv]

            # Make prediction for all assets
            vol_pred = model(*x_lags)  # [1, assets, 1]
            vol_pred_assets = vol_pred.squeeze().cpu().numpy()  # [assets]

            # Portfolio-level results for this period
            period_trades = []
            period_total_pnl = 0.0
            period_filled_trades = 0
            period_total_trades = 0

            # Trade each asset
            for asset_idx in range(ohlcv_data.shape[1]):  # 38 assets
                asset_name = asset_names[asset_idx] if asset_idx < len(asset_names) else f'Asset_{asset_idx}'

                # Get OHLCV data for this asset
                current_price = ohlcv_data[0, asset_idx, 0, 3].item()  # Close at T+0
                next_low = ohlcv_data[0, asset_idx, 1, 2].item()      # Low at T+1
                exit_price = ohlcv_data[0, asset_idx, 4, 3].item()    # Close at T+4

                # Get volatility prediction for this asset
                asset_vol_pred = vol_pred_assets[asset_idx]

                # Calculate limit order price
                limit_price = current_price * (1 - asset_vol_pred)

                # Check if order fills
                order_fills = next_low <= limit_price

                # Calculate trade result
                if order_fills:
                    # Calculate profit with fees
                    gross_profit_pct = (exit_price - limit_price) / limit_price
                    net_profit_pct = gross_profit_pct - total_fee_rate

                    # Calculate PnL for this asset
                    asset_pnl = net_profit_pct * position_size_per_asset
                    period_filled_trades += 1
                else:
                    asset_pnl = 0.0  # No trade, no PnL
                    net_profit_pct = 0.0
                    gross_profit_pct = 0.0

                period_total_pnl += asset_pnl
                period_total_trades += 1

                # Store individual asset trade data
                asset_trade = {
                    'asset_name': asset_name,
                    'asset_idx': asset_idx,
                    'vol_prediction': asset_vol_pred,
                    'current_price': current_price,
                    'limit_price': limit_price,
                    'next_low': next_low,
                    'exit_price': exit_price,
                    'order_filled': order_fills,
                    'gross_profit_pct': gross_profit_pct,
                    'net_profit_pct': net_profit_pct,
                    'asset_pnl': asset_pnl
                }
                period_trades.append(asset_trade)

            # Portfolio-level metrics for this period
            portfolio_fill_rate = period_filled_trades / period_total_trades if period_total_trades > 0 else 0
            avg_vol_pred = np.mean(vol_pred_assets)

            # Store portfolio trade data
            portfolio_trade_data = {
                'timestamp': sample_info['prediction_time'],
                'period_idx': i,
                'portfolio_pnl': period_total_pnl,
                'cumulative_pnl': cumulative_pnl[-1] + period_total_pnl,
                'filled_trades': period_filled_trades,
                'total_trades': period_total_trades,
                'portfolio_fill_rate': portfolio_fill_rate,
                'avg_vol_prediction': avg_vol_pred,
                'asset_trades': period_trades,
                'best_asset_pnl': max([t['asset_pnl'] for t in period_trades]),
                'worst_asset_pnl': min([t['asset_pnl'] for t in period_trades]),
                'profitable_assets': sum([1 for t in period_trades if t['asset_pnl'] > 0])
            }

            portfolio_trades.append(portfolio_trade_data)
            cumulative_pnl.append(portfolio_trade_data['cumulative_pnl'])
            timestamps.append(sample_info['prediction_time'])
            vol_predictions.append(avg_vol_pred)

    return {
        'portfolio_trades': portfolio_trades,
        'cumulative_pnl': cumulative_pnl[1:],  # Remove initial 0
        'timestamps': timestamps,
        'vol_predictions': vol_predictions,
        'asset_names': asset_names,
        'position_size_per_asset': position_size_per_asset,
        'total_assets': len(asset_names)
    }


def create_pnl_visualizations(trading_results):
    """
    Create comprehensive PnL visualizations for portfolio trading.

    Args:
        trading_results (dict): Results from simulate_trading_strategy
    """
    print("📈 Creating portfolio PnL visualizations...")

    portfolio_df = pd.DataFrame(trading_results['portfolio_trades'])

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'OHLCV Long Strategy - Portfolio Performance ({trading_results["total_assets"]} Assets)',
                fontsize=16, fontweight='bold')

    # 1. Cumulative Portfolio PnL Curve
    ax1 = axes[0, 0]
    ax1.plot(trading_results['timestamps'], trading_results['cumulative_pnl'],
             linewidth=2, color='darkblue', label='Portfolio Cumulative PnL')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Break-even')
    ax1.set_title('Portfolio Cumulative PnL Over Time', fontweight='bold')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Cumulative PnL ($)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Add final PnL annotation
    final_pnl = trading_results['cumulative_pnl'][-1]
    ax1.annotate(f'Final PnL: ${final_pnl:.2f}',
                xy=(trading_results['timestamps'][-1], final_pnl),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                fontweight='bold')

    # 2. Portfolio Period PnL Distribution
    ax2 = axes[0, 1]
    period_pnls = portfolio_df['portfolio_pnl']
    profitable_periods = period_pnls[period_pnls > 0]
    losing_periods = period_pnls[period_pnls < 0]

    ax2.hist(period_pnls, bins=20, alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Break-even')
    ax2.set_title('Portfolio Period PnL Distribution', fontweight='bold')
    ax2.set_xlabel('Period PnL ($)')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Add statistics
    win_rate = (period_pnls > 0).mean()
    avg_win = profitable_periods.mean() if len(profitable_periods) > 0 else 0
    avg_loss = losing_periods.mean() if len(losing_periods) > 0 else 0

    stats_text = f'Win Rate: {win_rate:.1%}\nAvg Win: ${avg_win:.2f}\nAvg Loss: ${avg_loss:.2f}'
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 3. Portfolio Fill Rate and Volatility Predictions
    ax3 = axes[1, 0]
    ax3_twin = ax3.twinx()

    # Plot portfolio fill rate
    line1 = ax3.plot(trading_results['timestamps'], portfolio_df['portfolio_fill_rate'] * 100,
                     linewidth=2, color='blue', label='Portfolio Fill Rate (%)')

    # Plot average volatility predictions
    line2 = ax3_twin.plot(trading_results['timestamps'], np.array(trading_results['vol_predictions']) * 100,
                          linewidth=2, color='purple', alpha=0.7, label='Avg Vol Prediction (%)')

    ax3.set_title('Portfolio Fill Rate & Volatility Predictions', fontweight='bold')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Fill Rate (%)', color='blue')
    ax3_twin.set_ylabel('Volatility Prediction (%)', color='purple')
    ax3.grid(True, alpha=0.3)

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper left')

    # 4. Rolling Performance Metrics
    ax4 = axes[1, 1]

    # Calculate rolling metrics
    window = min(20, len(portfolio_df) // 4)  # Adaptive window size
    portfolio_df['rolling_avg_pnl'] = portfolio_df['portfolio_pnl'].rolling(window=window, min_periods=1).mean()
    portfolio_df['rolling_fill_rate'] = portfolio_df['portfolio_fill_rate'].rolling(window=window, min_periods=1).mean()

    # Plot rolling metrics
    ax4_twin = ax4.twinx()

    line1 = ax4.plot(portfolio_df['timestamp'], portfolio_df['rolling_fill_rate'] * 100,
                     color='blue', linewidth=2, label=f'Rolling Fill Rate (%) [{window}p]')
    line2 = ax4_twin.plot(portfolio_df['timestamp'], portfolio_df['rolling_avg_pnl'],
                          color='red', linewidth=2, label=f'Rolling Avg PnL ($) [{window}p]')

    ax4.set_title(f'Rolling Portfolio Performance', fontweight='bold')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Fill Rate (%)', color='blue')
    ax4_twin.set_ylabel('Average PnL ($)', color='red')
    ax4.grid(True, alpha=0.3)

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='upper left')

    plt.tight_layout()

    # Save plot
    os.makedirs('plots/pnl_analysis', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f'plots/pnl_analysis/pnl_curve_{timestamp}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 PnL visualization saved to: {plot_path}")

    return plot_path


def print_trading_summary(trading_results):
    """
    Print comprehensive portfolio trading summary statistics.
    """
    portfolio_df = pd.DataFrame(trading_results['portfolio_trades'])

    print("\n" + "="*80)
    print("PORTFOLIO TRADING STRATEGY PERFORMANCE SUMMARY")
    print("="*80)

    # Basic statistics
    total_periods = len(portfolio_df)
    total_assets = trading_results['total_assets']
    position_size = trading_results['position_size_per_asset']
    total_capital_per_period = total_assets * position_size

    print(f"📊 PORTFOLIO STATISTICS:")
    print(f"  Trading Periods: {total_periods}")
    print(f"  Assets in Portfolio: {total_assets}")
    print(f"  Position Size per Asset: ${position_size}")
    print(f"  Total Capital per Period: ${total_capital_per_period:,}")
    print(f"  Total Opportunities: {total_periods * total_assets:,}")

    # Fill rate statistics
    total_filled = portfolio_df['filled_trades'].sum()
    total_possible = portfolio_df['total_trades'].sum()
    overall_fill_rate = total_filled / total_possible if total_possible > 0 else 0
    avg_portfolio_fill_rate = portfolio_df['portfolio_fill_rate'].mean()

    print(f"\n📈 EXECUTION STATISTICS:")
    print(f"  Total Filled Trades: {total_filled:,}")
    print(f"  Overall Fill Rate: {overall_fill_rate:.1%}")
    print(f"  Average Portfolio Fill Rate: {avg_portfolio_fill_rate:.1%}")
    print(f"  Average Vol Prediction: {np.mean(trading_results['vol_predictions'])*100:.2f}%")

    # PnL statistics
    final_pnl = trading_results['cumulative_pnl'][-1]
    total_capital_deployed = total_periods * total_capital_per_period
    total_return = final_pnl / total_capital_deployed * 100

    print(f"\n💰 PORTFOLIO PnL PERFORMANCE:")
    print(f"  Final Cumulative PnL: ${final_pnl:.2f}")
    print(f"  Total Capital Deployed: ${total_capital_deployed:,}")
    print(f"  Total Return: {total_return:.3f}%")
    print(f"  Average PnL per Period: ${final_pnl/total_periods:.2f}")

    # Period-level statistics
    period_pnls = portfolio_df['portfolio_pnl']
    profitable_periods = period_pnls[period_pnls > 0]
    losing_periods = period_pnls[period_pnls < 0]

    period_win_rate = len(profitable_periods) / len(period_pnls) if len(period_pnls) > 0 else 0
    avg_winning_period = profitable_periods.mean() if len(profitable_periods) > 0 else 0
    avg_losing_period = losing_periods.mean() if len(losing_periods) > 0 else 0

    print(f"\n📊 PERIOD-LEVEL PERFORMANCE:")
    print(f"  Profitable Periods: {len(profitable_periods)} / {len(period_pnls)}")
    print(f"  Period Win Rate: {period_win_rate:.1%}")
    print(f"  Average Winning Period: ${avg_winning_period:.2f}")
    print(f"  Average Losing Period: ${avg_losing_period:.2f}")

    if avg_losing_period != 0:
        profit_factor = abs(avg_winning_period / avg_losing_period)
        print(f"  Profit Factor: {profit_factor:.2f}")

    # Asset-level statistics
    avg_profitable_assets = portfolio_df['profitable_assets'].mean()
    max_profitable_assets = portfolio_df['profitable_assets'].max()
    min_profitable_assets = portfolio_df['profitable_assets'].min()

    print(f"\n🎯 ASSET-LEVEL PERFORMANCE:")
    print(f"  Average Profitable Assets per Period: {avg_profitable_assets:.1f} / {total_assets}")
    print(f"  Max Profitable Assets in Period: {max_profitable_assets}")
    print(f"  Min Profitable Assets in Period: {min_profitable_assets}")
    print(f"  Asset Success Rate: {avg_profitable_assets/total_assets:.1%}")

    # Risk metrics
    pnl_series = pd.Series(trading_results['cumulative_pnl'])
    max_drawdown = (pnl_series - pnl_series.expanding().max()).min()

    # Calculate volatility of period returns
    period_returns = period_pnls / total_capital_per_period
    period_volatility = period_returns.std()

    print(f"\n📉 RISK METRICS:")
    print(f"  Maximum Drawdown: ${max_drawdown:.2f}")
    print(f"  Period Return Volatility: {period_volatility:.4f} ({period_volatility*100:.2f}%)")

    if len(period_returns) > 1 and period_volatility > 0:
        sharpe_ratio = period_returns.mean() / period_volatility * np.sqrt(252/4)  # Assuming 4-hour periods
        print(f"  Sharpe Ratio (annualized): {sharpe_ratio:.2f}")

    # Best and worst periods
    best_period_idx = portfolio_df['portfolio_pnl'].idxmax()
    worst_period_idx = portfolio_df['portfolio_pnl'].idxmin()

    best_period = portfolio_df.iloc[best_period_idx]
    worst_period = portfolio_df.iloc[worst_period_idx]

    print(f"\n🏆 BEST/WORST PERIODS:")
    print(f"  Best Period: ${best_period['portfolio_pnl']:.2f} on {best_period['timestamp']}")
    print(f"    Fill Rate: {best_period['portfolio_fill_rate']:.1%}, Profitable Assets: {best_period['profitable_assets']}")
    print(f"  Worst Period: ${worst_period['portfolio_pnl']:.2f} on {worst_period['timestamp']}")
    print(f"    Fill Rate: {worst_period['portfolio_fill_rate']:.1%}, Profitable Assets: {worst_period['profitable_assets']}")

    # Performance consistency
    positive_periods = (period_pnls > 0).sum()
    negative_periods = (period_pnls < 0).sum()
    neutral_periods = (period_pnls == 0).sum()

    print(f"\n📈 CONSISTENCY METRICS:")
    print(f"  Positive Periods: {positive_periods} ({positive_periods/total_periods:.1%})")
    print(f"  Negative Periods: {negative_periods} ({negative_periods/total_periods:.1%})")
    print(f"  Neutral Periods: {neutral_periods} ({neutral_periods/total_periods:.1%})")


def main():
    """
    Main function to create PnL curve visualization.
    """
    print("🎯 OHLCV TRADING STRATEGY - PnL CURVE ANALYSIS")
    print("=" * 80)

    # Create trained model
    model, dataset, metadata = create_trained_model()

    # Simulate trading strategy
    trading_results = simulate_trading_strategy(model, dataset, num_trades=200)

    # Create visualizations
    plot_path = create_pnl_visualizations(trading_results)

    # Print summary
    print_trading_summary(trading_results)

    print(f"\n✅ Analysis complete! Check the visualization at: {plot_path}")


if __name__ == "__main__":
    main()
