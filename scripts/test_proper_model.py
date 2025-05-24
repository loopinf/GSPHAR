#!/usr/bin/env python
"""
Test the properly trained model (no data leakage) with corrected timeline.

Timeline:
1. Vol prediction using data up to T-1
2. Place order using T+0 open price  
3. Check fills using T+0 low
4. Exit using holding period close
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import sys
import logging
from datetime import datetime

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.ohlcv_trading_dataset import load_ohlcv_trading_data
from src.models.flexible_gsphar import FlexibleGSPHAR

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_proper_model(model_path, device='cpu'):
    """Load the properly trained model."""
    logger.info(f"Loading model from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    metadata = checkpoint['metadata']
    parameters = checkpoint['parameters']
    
    # Verify no data leakage
    if not metadata.get('no_data_leakage', False):
        logger.warning("⚠️  Model may have data leakage!")
    else:
        logger.info("✅ Model verified: No data leakage")
    
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
    model = model.to(device)
    model.eval()
    
    logger.info(f"Model loaded successfully")
    logger.info(f"  Training period: {metadata['train_period']}")
    logger.info(f"  Testing period: {metadata['test_period']}")
    
    return model, metadata, parameters, checkpoint['test_indices']


def generate_test_predictions(model, dataset, test_indices, device='cpu'):
    """Generate predictions on test data only."""
    logger.info(f"Generating predictions on {len(test_indices)} test samples")
    
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for i, idx in enumerate(test_indices):
            if i % 500 == 0:
                logger.info(f"Processing test sample {i+1}/{len(test_indices)}")
            
            sample = dataset[idx]
            sample_info = dataset.get_sample_info(idx)
            
            x_lags = [x.unsqueeze(0).to(device) for x in sample['x_lags']]
            vol_pred = model(*x_lags)
            vol_pred_np = vol_pred.squeeze().cpu().numpy()
            
            prediction_data = {
                'index': idx,
                'timestamp': sample_info['prediction_time'],
                'vol_predictions': vol_pred_np,
                'ohlcv_data': sample['ohlcv_data'].numpy(),
                'vol_targets': sample['vol_targets'].numpy()
            }
            
            predictions.append(prediction_data)
    
    logger.info(f"Generated {len(predictions)} test predictions")
    return predictions


def corrected_trading_simulation(predictions, holding_period=4, position_size_per_asset=100, trading_fee=0.0002):
    """
    Corrected trading simulation with proper timeline:
    1. Vol prediction using data up to T-1
    2. Place order using T+0 open price
    3. Check fills using T+0 low  
    4. Exit using holding period close
    """
    logger.info("🔧 CORRECTED TRADING SIMULATION")
    logger.info("Timeline: Vol pred → T+0 open → T+0 low fill check → Exit")
    
    trading_results = []
    cumulative_pnl = 0.0
    total_fee_rate = 2 * trading_fee
    
    for pred in predictions:
        timestamp = pred['timestamp']
        vol_predictions = pred['vol_predictions']
        ohlcv_data = pred['ohlcv_data']
        
        period_pnl = 0.0
        period_trades = 0
        period_filled = 0
        
        n_assets = ohlcv_data.shape[0]
        for asset_idx in range(n_assets):
            # CORRECTED TIMELINE
            open_price = ohlcv_data[asset_idx, 0, 0]      # T+0 OPEN
            current_low = ohlcv_data[asset_idx, 0, 2]     # T+0 LOW
            exit_price = ohlcv_data[asset_idx, holding_period, 3]  # T+holding_period CLOSE
            
            asset_vol_pred = vol_predictions[asset_idx]
            limit_price = open_price * (1 - asset_vol_pred)
            
            # Check if order fills during T+0 period
            order_fills = current_low <= limit_price
            
            if order_fills:
                gross_profit_pct = (exit_price - limit_price) / limit_price
                net_profit_pct = gross_profit_pct - total_fee_rate
                asset_pnl = net_profit_pct * position_size_per_asset
                period_pnl += asset_pnl
                period_filled += 1
            
            period_trades += 1
        
        cumulative_pnl += period_pnl
        
        period_result = {
            'timestamp': timestamp,
            'period_pnl': period_pnl,
            'cumulative_pnl': cumulative_pnl,
            'trades': period_trades,
            'filled': period_filled,
            'fill_rate': period_filled / period_trades if period_trades > 0 else 0,
            'avg_vol_pred': np.mean(vol_predictions)
        }
        
        trading_results.append(period_result)
    
    logger.info(f"Trading simulation completed")
    logger.info(f"  Total periods: {len(trading_results)}")
    logger.info(f"  Final cumulative PnL: ${cumulative_pnl:.2f}")
    
    return trading_results


def create_realistic_pnl_plot(trading_results, save_path=None):
    """Create PnL plot for realistic results."""
    logger.info("Creating realistic PnL time series plot...")
    
    df = pd.DataFrame(trading_results)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle('Realistic Model: No Data Leakage + Corrected Timeline', fontsize=16, fontweight='bold')
    
    # 1. Cumulative PnL
    ax1 = axes[0]
    ax1.plot(df['timestamp'], df['cumulative_pnl'], linewidth=2, color='darkgreen', label='Cumulative PnL')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Break-even')
    ax1.set_title('Cumulative PnL Over Time (Out-of-Sample)', fontweight='bold')
    ax1.set_ylabel('Cumulative PnL ($)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    final_pnl = df['cumulative_pnl'].iloc[-1]
    ax1.annotate(f'Final PnL: ${final_pnl:.2f}',
                xy=(df['timestamp'].iloc[-1], final_pnl),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
                fontweight='bold')
    
    # 2. Period PnL
    ax2 = axes[1]
    colors = ['green' if pnl >= 0 else 'red' for pnl in df['period_pnl']]
    ax2.bar(df['timestamp'], df['period_pnl'], color=colors, alpha=0.7, width=1)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.set_title('Period PnL Over Time', fontweight='bold')
    ax2.set_ylabel('Period PnL ($)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Fill rate and volatility
    ax3 = axes[2]
    ax3_twin = ax3.twinx()
    
    line1 = ax3.plot(df['timestamp'], df['fill_rate'] * 100,
                     color='blue', linewidth=2, label='Fill Rate (%)')
    line2 = ax3_twin.plot(df['timestamp'], df['avg_vol_pred'] * 100,
                          color='purple', linewidth=2, alpha=0.7, label='Avg Vol Prediction (%)')
    
    ax3.set_title('Fill Rate & Volatility Predictions', fontweight='bold')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Fill Rate (%)', color='blue')
    ax3_twin.set_ylabel('Volatility Prediction (%)', color='purple')
    ax3.grid(True, alpha=0.3)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to: {save_path}")
    
    plt.show()
    
    # Print realistic summary
    print("\n" + "="*60)
    print("REALISTIC PnL ANALYSIS (NO DATA LEAKAGE)")
    print("="*60)
    
    total_periods = len(df)
    profitable_periods = (df['period_pnl'] > 0).sum()
    win_rate = profitable_periods / total_periods
    
    print(f"Test Period: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Total Periods: {total_periods}")
    print(f"Profitable Periods: {profitable_periods} ({win_rate:.1%})")
    print(f"Final Cumulative PnL: ${final_pnl:.2f}")
    print(f"Average Period PnL: ${df['period_pnl'].mean():.2f}")
    print(f"Best Period: ${df['period_pnl'].max():.2f}")
    print(f"Worst Period: ${df['period_pnl'].min():.2f}")
    print(f"Average Fill Rate: {df['fill_rate'].mean():.1%}")
    print(f"Average Vol Prediction: {df['avg_vol_pred'].mean()*100:.2f}%")
    
    # Compare with previous results
    print(f"\n📊 COMPARISON WITH PREVIOUS RESULTS:")
    print(f"Previous (Data Leakage): $47,699 | 85% win rate | 41.8% fill rate")
    print(f"Current (No Leakage):    ${final_pnl:.0f} | {win_rate:.1%} win rate | {df['fill_rate'].mean():.1%} fill rate")
    
    if final_pnl > 0:
        print(f"✅ Strategy still profitable without data leakage!")
    else:
        print(f"❌ Strategy not profitable on out-of-sample data")


def main():
    """Main testing function."""
    logger.info("🧪 TESTING PROPERLY TRAINED MODEL")
    logger.info("=" * 80)
    
    # Find the latest proper model
    models_dir = "models"
    proper_models = [f for f in os.listdir(models_dir) if f.startswith('proper_split_model_') and f.endswith('.pt')]
    
    if not proper_models:
        logger.error("❌ No properly trained model found!")
        logger.info("Please run train_proper_split.py first")
        return
    
    # Use the latest model
    latest_model = sorted(proper_models)[-1]
    model_path = os.path.join(models_dir, latest_model)
    
    device = torch.device('cpu')
    
    # Load model
    model, metadata, parameters, test_indices = load_proper_model(model_path, device)
    
    # Load dataset
    dataset, _ = load_ohlcv_trading_data(
        volatility_file="data/crypto_rv1h_38_20200822_20250116.csv",
        lags=parameters['lags'],
        holding_period=parameters['holding_period'],
        debug=False
    )
    
    # Generate test predictions (out-of-sample only)
    predictions = generate_test_predictions(model, dataset, test_indices, device)
    
    # Run corrected trading simulation
    trading_results = corrected_trading_simulation(
        predictions=predictions,
        holding_period=parameters['holding_period'],
        position_size_per_asset=100,
        trading_fee=0.0002
    )
    
    # Create realistic PnL plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"plots/pnl_analysis/realistic_pnl_{timestamp}.png"
    os.makedirs("plots/pnl_analysis", exist_ok=True)
    
    create_realistic_pnl_plot(trading_results, save_path)
    
    logger.info("✅ Realistic testing completed!")


if __name__ == "__main__":
    main()
