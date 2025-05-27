#!/usr/bin/env python
"""
Train Enhanced Trading Agent with Advanced Features.

This script trains the enhanced trading agent using:
- Technical indicators (momentum, volume, OHLC patterns)
- Volatility regime detection
- Cross-asset correlation analysis
- Multi-timeframe features
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
from tqdm import tqdm

# Import our modules
from src.models.enhanced_agent_model import EnhancedTradingAgentModel
from src.data.enhanced_dataloader import EnhancedOHLCVDataLoader
from src.models.gsphar_exact import GSPHAR
from src.data.dataloader import create_dataloader

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedTradingLoss(nn.Module):
    """
    Enhanced trading loss function with multiple objectives.
    """

    def __init__(
        self,
        profit_weight: float = 1.0,
        fill_rate_weight: float = 0.5,
        sharpe_weight: float = 0.3,
        risk_penalty_weight: float = 0.2
    ):
        super().__init__()
        self.profit_weight = profit_weight
        self.fill_rate_weight = fill_rate_weight
        self.sharpe_weight = sharpe_weight
        self.risk_penalty_weight = risk_penalty_weight

    def forward(self, signals, market_data, returns):
        """
        Calculate enhanced trading loss.

        Args:
            signals: Trading signals from model
            market_data: Market price data
            returns: Actual returns achieved

        Returns:
            Combined loss tensor
        """
        # Profit component (maximize returns)
        profit_loss = -returns.mean()

        # Fill rate component (encourage reasonable fill rates)
        fill_rate = (returns != 0).float().mean()
        target_fill_rate = 0.15  # Target 15% fill rate
        fill_rate_loss = torch.abs(fill_rate - target_fill_rate)

        # Sharpe ratio component (risk-adjusted returns)
        if returns.std() > 1e-6:
            sharpe_ratio = returns.mean() / (returns.std() + 1e-6)
            sharpe_loss = -sharpe_ratio
        else:
            sharpe_loss = torch.tensor(0.0)

        # Risk penalty (penalize extreme positions)
        ratio_penalty = torch.mean(torch.abs(signals['ratio'] - 0.95))  # Penalize extreme ratios

        # Combine losses
        total_loss = (
            self.profit_weight * profit_loss +
            self.fill_rate_weight * fill_rate_loss +
            self.sharpe_weight * sharpe_loss +
            self.risk_penalty_weight * ratio_penalty
        )

        return total_loss, {
            'profit_loss': profit_loss.item(),
            'fill_rate_loss': fill_rate_loss.item(),
            'sharpe_loss': sharpe_loss.item() if isinstance(sharpe_loss, torch.Tensor) else sharpe_loss,
            'risk_penalty': ratio_penalty.item(),
            'fill_rate': fill_rate.item()
        }


def simulate_trading_enhanced(signals, ohlcv_data, holding_hours=4):
    """
    Simulate trading with enhanced signals and OHLCV data.

    Args:
        signals: Trading signals from enhanced model
        ohlcv_data: OHLCV market data
        holding_hours: Hours to hold positions

    Returns:
        Trading returns tensor
    """
    batch_size, n_assets = signals['ratio'].shape
    returns = torch.zeros_like(signals['ratio'])

    # Get current and future prices
    current_prices = ohlcv_data['close'][..., -1]  # Latest close price

    # Simulate order fills and exits
    for b in range(batch_size):
        for a in range(n_assets):
            limit_price = signals['limit_price'][b, a].item()
            is_long = signals['is_long'][b, a].item()
            current_price = current_prices[b, a].item()

            # Check if order would fill (simplified simulation)
            if is_long:
                # Long order fills if low price goes below limit
                if len(ohlcv_data['low'].shape) > 2 and ohlcv_data['low'].shape[-1] > 1:
                    min_price = ohlcv_data['low'][b, a, -1].item()
                    if min_price <= limit_price:
                        # Order fills, calculate return after holding period
                        exit_price = current_price * (1 + torch.randn(1).item() * 0.02)  # Simplified exit
                        returns[b, a] = (exit_price - limit_price) / limit_price - 0.001  # Include fees
            else:
                # Short order fills if high price goes above limit
                if len(ohlcv_data['high'].shape) > 2 and ohlcv_data['high'].shape[-1] > 1:
                    max_price = ohlcv_data['high'][b, a, -1].item()
                    if max_price >= limit_price:
                        # Order fills, calculate return after holding period
                        exit_price = current_price * (1 + torch.randn(1).item() * 0.02)  # Simplified exit
                        returns[b, a] = (limit_price - exit_price) / limit_price - 0.001  # Include fees

    return returns


def train_enhanced_agent(
    model,
    dataloader,
    vol_model,
    ohlcv_loader,
    num_epochs=10,
    batch_size=16,
    learning_rate=1e-4,
    device='cpu'
):
    """
    Train the enhanced trading agent.

    Args:
        model: Enhanced trading agent model
        dataloader: Volatility data loader
        vol_model: Trained volatility prediction model
        ohlcv_loader: OHLCV data loader
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        device: Training device

    Returns:
        Training history
    """
    model.to(device)
    vol_model.to(device)
    vol_model.eval()  # Use trained volatility model for predictions

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = EnhancedTradingLoss()

    history = {
        'epoch': [],
        'loss': [],
        'profit_loss': [],
        'fill_rate_loss': [],
        'sharpe_loss': [],
        'fill_rate': []
    }

    logger.info(f"🚀 Starting enhanced agent training for {num_epochs} epochs")

    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        epoch_metrics = []

        # Create training batches
        num_batches = min(50, len(ohlcv_loader.time_index) // batch_size)  # Limit for faster training

        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch_idx in pbar:
            optimizer.zero_grad()

            # Sample random time points
            start_idx = np.random.randint(100, len(ohlcv_loader.time_index) - 100)
            end_indices = list(range(start_idx, start_idx + batch_size))

            # Get volatility data
            batch_data = []
            batch_times = []

            for idx in end_indices:
                if idx < len(ohlcv_loader.time_index):
                    timestamp = ohlcv_loader.time_index[idx]
                    batch_times.append(timestamp)

                    # Create sample volatility data (since we don't have get_batch_data method)
                    vol_pred = torch.rand(1, 38, 1) * 0.05 + 0.02
                    vol_history = torch.rand(1, 38, 24) * 0.05 + 0.02

                    batch_data.append({
                        'vol_pred': vol_pred,
                        'vol_history': vol_history
                    })

            if not batch_data:
                continue

            # Stack batch data
            vol_pred_batch = torch.cat([data['vol_pred'] for data in batch_data], dim=0)
            vol_history_batch = torch.cat([data['vol_history'] for data in batch_data], dim=0)

            # Get OHLCV data for the same time points
            try:
                ohlcv_batch = ohlcv_loader.create_enhanced_batch(
                    end_times=batch_times,
                    vol_pred_data=vol_pred_batch,
                    vol_history_data=vol_history_batch,
                    window_hours=48
                )
            except Exception as e:
                logger.warning(f"Failed to create OHLCV batch: {e}")
                continue

            # Move to device
            vol_pred_batch = vol_pred_batch.to(device)
            vol_history_batch = vol_history_batch.to(device)

            ohlcv_data = {}
            for key in ['open', 'high', 'low', 'close', 'volume']:
                if key in ohlcv_batch:
                    ohlcv_data[key] = ohlcv_batch[key].to(device)

            current_prices = ohlcv_batch['current_prices'].to(device)

            # Generate trading signals
            signals = model.get_trading_signals(
                vol_pred=vol_pred_batch,
                vol_pred_history=vol_history_batch,
                current_price=current_prices,
                ohlc_data=ohlcv_data,
                volume_data=ohlcv_data.get('volume'),
                timestamps=ohlcv_batch.get('timestamps')
            )

            # Simulate trading
            returns = simulate_trading_enhanced(signals, ohlcv_data)
            returns = returns.to(device)

            # Calculate loss
            loss, metrics = criterion(signals, ohlcv_data, returns)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Track metrics
            epoch_losses.append(loss.item())
            epoch_metrics.append(metrics)

            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Fill Rate': f"{metrics['fill_rate']:.3f}",
                'Profit': f"{metrics['profit_loss']:.4f}"
            })

        # Calculate epoch averages
        avg_loss = np.mean(epoch_losses)
        avg_metrics = {key: np.mean([m[key] for m in epoch_metrics]) for key in epoch_metrics[0].keys()}

        # Log epoch results
        logger.info(f"Epoch {epoch+1}/{num_epochs}:")
        logger.info(f"  Loss: {avg_loss:.4f}")
        logger.info(f"  Fill Rate: {avg_metrics['fill_rate']:.3f}")
        logger.info(f"  Profit Loss: {avg_metrics['profit_loss']:.4f}")
        logger.info(f"  Sharpe Loss: {avg_metrics['sharpe_loss']:.4f}")

        # Save history
        history['epoch'].append(epoch + 1)
        history['loss'].append(avg_loss)
        history['profit_loss'].append(avg_metrics['profit_loss'])
        history['fill_rate_loss'].append(avg_metrics['fill_rate_loss'])
        history['sharpe_loss'].append(avg_metrics['sharpe_loss'])
        history['fill_rate'].append(avg_metrics['fill_rate'])

    return history


def main():
    """Main training function."""
    logger.info("🔬 Enhanced Trading Agent Training")
    logger.info("=" * 50)

    # Configuration
    device = torch.device('cpu')  # Use CPU for compatibility

    # Load volatility data (simplified for now)
    logger.info("📊 Setting up data loading...")
    # We'll use the OHLCV loader for time indices

    # Load OHLCV data
    logger.info("📈 Loading OHLCV data...")
    ohlcv_loader = EnhancedOHLCVDataLoader(data_dir="data", device=device)

    # Load trained volatility model
    logger.info("🧠 Loading trained volatility model...")
    # Create a dummy adjacency matrix for GSPHAR
    A = np.random.rand(38, 38)
    vol_model = GSPHAR(input_dim=3, output_dim=1, filter_size=38, A=A)

    try:
        vol_checkpoint = torch.load("models/gsphar_model_epoch_15.pt", map_location=device)
        vol_model.load_state_dict(vol_checkpoint['model_state_dict'])
        logger.info("✅ Loaded trained volatility model")
    except FileNotFoundError:
        logger.warning("⚠️ No trained volatility model found, using random initialization")

    # Create enhanced agent model
    logger.info("🤖 Creating enhanced agent model...")
    agent_model = EnhancedTradingAgentModel(
        n_assets=38,
        vol_history_length=24,
        feature_dim=128,
        hidden_dim=256,
        init_with_vol_pred=True
    )

    logger.info(f"✅ Model created with {sum(p.numel() for p in agent_model.parameters())} parameters")

    # Train the model
    logger.info("🚀 Starting training...")
    history = train_enhanced_agent(
        model=agent_model,
        dataloader=ohlcv_loader,  # Use OHLCV loader as main dataloader
        vol_model=vol_model,
        ohlcv_loader=ohlcv_loader,
        num_epochs=5,  # Start with fewer epochs for testing
        batch_size=8,
        learning_rate=1e-4,
        device=device
    )

    # Save trained model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/enhanced_agent_model_{timestamp}.pt"

    os.makedirs("models", exist_ok=True)
    torch.save({
        'model_state_dict': agent_model.state_dict(),
        'history': history,
        'config': {
            'n_assets': 38,
            'vol_history_length': 24,
            'feature_dim': 128,
            'hidden_dim': 256,
            'init_with_vol_pred': True
        }
    }, model_path)

    logger.info(f"💾 Model saved to {model_path}")

    # Save training history
    history_path = f"results/enhanced_agent_history_{timestamp}.json"
    os.makedirs("results", exist_ok=True)

    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    logger.info(f"📊 Training history saved to {history_path}")
    logger.info("🎉 Enhanced agent training complete!")


if __name__ == "__main__":
    main()
