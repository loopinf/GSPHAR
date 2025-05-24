#!/usr/bin/env python
"""
Generate PnL curve on time axis using the trained two-stage model.

This script:
1. Loads the trained two-stage model
2. Generates predictions on sequential time data
3. Simulates trading strategy with proper timestamps
4. Creates PnL curve with time on x-axis
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import sys
from datetime import datetime
import logging

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.ohlcv_trading_dataset import load_ohlcv_trading_data
from src.ohlcv_trading_loss import OHLCVLongStrategyLoss
from src.models.flexible_gsphar import FlexibleGSPHAR

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_trained_model(model_path, device='cpu'):
    """
    Load the trained two-stage model.

    Args:
        model_path: Path to the saved model
        device: Device to load model on

    Returns:
        tuple: (model, metadata, parameters)
    """
    logger.info(f"Loading trained model from: {model_path}")

    # Load checkpoint (with weights_only=False for compatibility)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Extract metadata and parameters
    metadata = checkpoint['metadata']
    parameters = checkpoint['parameters']

    # Load volatility data for correlation matrix
    vol_df = pd.read_csv("data/crypto_rv1h_38_20200822_20250116.csv", index_col=0, parse_dates=True)
    corr_sample = vol_df.iloc[-1000:] if len(vol_df) > 1000 else vol_df
    corr_matrix = corr_sample.corr().values
    A = (corr_matrix + corr_matrix.T) / 2

    # Create model with same architecture
    model = FlexibleGSPHAR(
        lags=parameters['lags'],
        output_dim=1,
        filter_size=len(metadata['assets']),
        A=A
    )

    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    logger.info(f"Model loaded successfully")
    logger.info(f"  Assets: {len(metadata['assets'])}")
    logger.info(f"  Lags: {parameters['lags']}")
    logger.info(f"  Holding period: {parameters['holding_period']}")

    return model, metadata, parameters


def generate_sequential_predictions(model, dataset, start_idx=0, num_periods=1000, device='cpu'):
    """
    Generate predictions on sequential time periods.

    Args:
        model: Trained GSPHAR model
        dataset: OHLCV dataset
        start_idx: Starting index in dataset
        num_periods: Number of sequential periods to predict
        device: Computing device

    Returns:
        dict: Sequential predictions with timestamps
    """
    logger.info(f"Generating {num_periods} sequential predictions starting from index {start_idx}")

    model.eval()
    predictions = []

    # Ensure we don't exceed dataset bounds
    end_idx = min(start_idx + num_periods, len(dataset))
    actual_periods = end_idx - start_idx

    logger.info(f"Actual periods to process: {actual_periods}")

    with torch.no_grad():
        for i in range(start_idx, end_idx):
            if i % 100 == 0:
                logger.info(f"Processing period {i - start_idx + 1}/{actual_periods}")

            # Get sample
            sample = dataset[i]
            sample_info = dataset.get_sample_info(i)

            # Get model inputs
            x_lags = [x.unsqueeze(0).to(device) for x in sample['x_lags']]  # Add batch dimension

            # Make prediction
            vol_pred = model(*x_lags)  # [1, assets, 1]
            vol_pred_np = vol_pred.squeeze().cpu().numpy()  # [assets]

            # Store prediction with metadata
            prediction_data = {
                'index': i,
                'timestamp': sample_info['prediction_time'],
                'vol_predictions': vol_pred_np,
                'avg_vol_prediction': np.mean(vol_pred_np),
                'ohlcv_data': sample['ohlcv_data'].numpy(),  # [assets, time_periods, 5]
                'vol_targets': sample['vol_targets'].numpy()  # [assets, 1]
            }

            predictions.append(prediction_data)

    logger.info(f"Generated {len(predictions)} predictions")
    return predictions


def simulate_trading_pnl(predictions, holding_period=4, position_size_per_asset=100, trading_fee=0.0002):
    """
    Simulate trading strategy and calculate PnL over time.

    Args:
        predictions: List of prediction dictionaries
        holding_period: Holding period in hours
        position_size_per_asset: Position size per asset in USD
        trading_fee: Trading fee rate (0.02% = 0.0002)

    Returns:
        dict: Trading results with PnL time series
    """
    logger.info(f"Simulating trading strategy...")
    logger.info(f"  Holding period: {holding_period} hours")
    logger.info(f"  Position size per asset: ${position_size_per_asset}")
    logger.info(f"  Trading fee: {trading_fee*100:.2f}%")

    # Trading results storage
    trading_results = []
    cumulative_pnl = 0.0
    total_fee_rate = 2 * trading_fee  # Buy + Sell fees

    for pred in predictions:
        timestamp = pred['timestamp']
        vol_predictions = pred['vol_predictions']
        ohlcv_data = pred['ohlcv_data']  # [assets, time_periods, 5]

        # Portfolio-level results for this period
        period_pnl = 0.0
        period_trades = 0
        period_filled = 0

        # Trade each asset
        n_assets = ohlcv_data.shape[0]
        for asset_idx in range(n_assets):
            # Get OHLCV data for this asset
            current_price = ohlcv_data[asset_idx, 0, 3]  # Close at T+0
            next_low = ohlcv_data[asset_idx, 1, 2]       # Low at T+1
            exit_price = ohlcv_data[asset_idx, holding_period, 3]  # Close at T+holding_period

            # Get volatility prediction for this asset
            asset_vol_pred = vol_predictions[asset_idx]

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
                period_pnl += asset_pnl
                period_filled += 1

            period_trades += 1

        # Update cumulative PnL
        cumulative_pnl += period_pnl

        # Store period results
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


def create_pnl_time_series_plot(trading_results, save_path=None):
    """
    Create PnL curve plot with time on x-axis.

    Args:
        trading_results: List of trading result dictionaries
        save_path: Optional path to save the plot
    """
    logger.info("Creating PnL time series plot...")

    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(trading_results)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle('Two-Stage Model: Trading Strategy PnL Analysis Over Time', fontsize=16, fontweight='bold')

    # 1. Cumulative PnL over time
    ax1 = axes[0]
    ax1.plot(df['timestamp'], df['cumulative_pnl'], linewidth=2, color='darkblue', label='Cumulative PnL')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Break-even')
    ax1.set_title('Cumulative PnL Over Time', fontweight='bold')
    ax1.set_ylabel('Cumulative PnL ($)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Format x-axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # Add final PnL annotation
    final_pnl = df['cumulative_pnl'].iloc[-1]
    ax1.annotate(f'Final PnL: ${final_pnl:.2f}',
                xy=(df['timestamp'].iloc[-1], final_pnl),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                fontweight='bold')

    # 2. Period PnL over time
    ax2 = axes[1]
    colors = ['green' if pnl >= 0 else 'red' for pnl in df['period_pnl']]
    ax2.bar(df['timestamp'], df['period_pnl'], color=colors, alpha=0.7, width=1)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.set_title('Period PnL Over Time', fontweight='bold')
    ax2.set_ylabel('Period PnL ($)')
    ax2.grid(True, alpha=0.3)

    # Format x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    # 3. Fill rate and volatility predictions over time
    ax3 = axes[2]
    ax3_twin = ax3.twinx()

    # Plot fill rate
    line1 = ax3.plot(df['timestamp'], df['fill_rate'] * 100,
                     color='blue', linewidth=2, label='Fill Rate (%)')

    # Plot average volatility predictions
    line2 = ax3_twin.plot(df['timestamp'], df['avg_vol_pred'] * 100,
                          color='purple', linewidth=2, alpha=0.7, label='Avg Vol Prediction (%)')

    ax3.set_title('Fill Rate & Volatility Predictions Over Time', fontweight='bold')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Fill Rate (%)', color='blue')
    ax3_twin.set_ylabel('Volatility Prediction (%)', color='purple')
    ax3.grid(True, alpha=0.3)

    # Format x-axis
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper left')

    plt.tight_layout()

    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to: {save_path}")

    plt.show()

    # Print summary statistics
    print("\n" + "="*60)
    print("PnL TIME SERIES SUMMARY")
    print("="*60)

    total_periods = len(df)
    profitable_periods = (df['period_pnl'] > 0).sum()
    win_rate = profitable_periods / total_periods

    print(f"Time Period: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Total Periods: {total_periods}")
    print(f"Profitable Periods: {profitable_periods} ({win_rate:.1%})")
    print(f"Final Cumulative PnL: ${final_pnl:.2f}")
    print(f"Average Period PnL: ${df['period_pnl'].mean():.2f}")
    print(f"Best Period: ${df['period_pnl'].max():.2f}")
    print(f"Worst Period: ${df['period_pnl'].min():.2f}")
    print(f"Average Fill Rate: {df['fill_rate'].mean():.1%}")
    print(f"Average Vol Prediction: {df['avg_vol_pred'].mean()*100:.2f}%")


def main():
    """
    Main function to generate PnL time series.
    """
    logger.info("🎯 GENERATING PnL TIME SERIES WITH TWO-STAGE MODEL")
    logger.info("=" * 80)

    # Parameters
    model_path = "models/two_stage_model_20250524_132116.pt"  # Update with actual path
    start_idx = 0  # Start from beginning of dataset
    num_periods = 2000  # Number of periods to analyze
    device = torch.device('cpu')

    # Check if model exists
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        logger.info("Available models:")
        models_dir = "models"
        if os.path.exists(models_dir):
            for f in os.listdir(models_dir):
                if f.endswith('.pt') and 'two_stage' in f:
                    logger.info(f"  {f}")
        return

    # Load trained model
    model, metadata, parameters = load_trained_model(model_path, device)

    # Load dataset
    logger.info("Loading OHLCV dataset...")
    dataset, _ = load_ohlcv_trading_data(
        volatility_file="data/crypto_rv1h_38_20200822_20250116.csv",
        lags=parameters['lags'],
        holding_period=parameters['holding_period'],
        debug=False
    )

    logger.info(f"Dataset loaded: {len(dataset)} total samples")

    # Generate sequential predictions
    predictions = generate_sequential_predictions(
        model=model,
        dataset=dataset,
        start_idx=start_idx,
        num_periods=num_periods,
        device=device
    )

    # Simulate trading and calculate PnL
    trading_results = simulate_trading_pnl(
        predictions=predictions,
        holding_period=parameters['holding_period'],
        position_size_per_asset=100,  # $100 per asset
        trading_fee=0.0002  # 0.02% Binance futures fee
    )

    # Create PnL time series plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"plots/pnl_analysis/two_stage_pnl_time_series_{timestamp}.png"
    os.makedirs("plots/pnl_analysis", exist_ok=True)

    create_pnl_time_series_plot(trading_results, save_path)

    logger.info("PnL time series analysis completed!")


if __name__ == "__main__":
    main()
