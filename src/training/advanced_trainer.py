"""
ption
Advanced Training Pipeline for Enhanced Trading Agent.

This module implements sophisticated training strategies:
- Curriculum learning
- Dynamic loss weighting
- Early stopping with validation
- Model checkpointing
- Performance monitoring
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.training.advanced_loss_functions import AdvancedTradingLoss, CurriculumLearningScheduler
from src.models.enhanced_agent_model import EnhancedTradingAgentModel
from src.data.enhanced_dataloader import EnhancedOHLCVDataLoader

logger = logging.getLogger(__name__)


class AdvancedTradingTrainer:
    """
    Advanced trainer for enhanced trading agent with sophisticated optimization strategies.
    """

    def __init__(
        self,
        model: EnhancedTradingAgentModel,
        dataloader: EnhancedOHLCVDataLoader,
        device: torch.device = torch.device('cpu'),
        save_dir: str = "models/advanced_training"
    ):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.current_epoch = 0
        self.best_score = float('-inf')
        self.patience_counter = 0
        self.training_history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'train_sharpe': [],
            'val_sharpe': [],
            'train_fill_rate': [],
            'val_fill_rate': [],
            'train_avg_return': [],
            'val_avg_return': [],
            'learning_rate': []
        }

    def create_train_val_split(
        self,
        train_ratio: float = 0.8,
        validation_months: int = 3
    ) -> Tuple[List[pd.Timestamp], List[pd.Timestamp]]:
        """
        Create time-based train/validation split.

        Args:
            train_ratio: Ratio of data for training
            validation_months: Months of data for validation

        Returns:
            Tuple of (train_timestamps, val_timestamps)
        """
        all_timestamps = self.dataloader.time_index

        # Use last validation_months for validation
        val_start = all_timestamps[-validation_months * 30 * 24:]  # Approximate
        val_timestamps = val_start.tolist()

        # Use earlier data for training
        train_end_idx = len(all_timestamps) - len(val_timestamps)
        train_timestamps = all_timestamps[:train_end_idx].tolist()

        logger.info(f"Train samples: {len(train_timestamps)}, Val samples: {len(val_timestamps)}")
        return train_timestamps, val_timestamps

    def simulate_enhanced_trading_differentiable(
        self,
        signals: Dict[str, torch.Tensor],
        ohlcv_data: Dict[str, torch.Tensor],
        holding_hours: int = 4,
        transaction_cost: float = 0.001,
        temperature: float = 10.0
    ) -> torch.Tensor:
        """
        Differentiable trading simulation that maintains gradient flow.

        Args:
            signals: Trading signals from model
            ohlcv_data: OHLCV market data
            holding_hours: Hours to hold positions
            transaction_cost: Transaction cost per trade
            temperature: Temperature for soft approximations

        Returns:
            Trading returns tensor (differentiable)
        """
        batch_size, n_assets = signals['ratio'].shape
        device = signals['ratio'].device

        # Extract signals (these maintain gradients)
        limit_ratios = signals['ratio']  # [batch_size, n_assets]
        is_long_probs = signals['is_long']  # [batch_size, n_assets] (0-1 probabilities)

        # Get current prices from OHLCV data
        if 'close' in ohlcv_data and ohlcv_data['close'].shape[-1] > 0:
            current_prices = ohlcv_data['close'][:, :, -1]  # [batch_size, n_assets]
        else:
            # Fallback to random prices if no OHLCV data
            current_prices = torch.rand(batch_size, n_assets, device=device) * 100 + 50

        # Calculate limit prices from ratios
        limit_prices = current_prices * limit_ratios  # [batch_size, n_assets]

        # Soft fill probability based on how aggressive the order is
        # More aggressive orders (further from current price) have lower fill probability
        price_distance = torch.abs(limit_prices - current_prices) / current_prices
        base_fill_prob = 0.2  # Base 20% fill rate
        fill_prob = base_fill_prob * torch.exp(-price_distance * temperature)
        fill_prob = torch.clamp(fill_prob, 0.01, 0.5)  # 1-50% fill rate range

        # Simulate market returns (differentiable random-like behavior)
        # Use deterministic function of current prices for reproducibility
        market_seed = torch.sin(current_prices * 1000) * torch.cos(current_prices * 500)
        market_returns = torch.tanh(market_seed) * 0.02  # ±2% market movement

        # Calculate exit prices after holding period
        exit_prices = current_prices * (1 + market_returns)

        # Calculate raw returns for long and short positions
        long_returns = (exit_prices - limit_prices) / limit_prices
        short_returns = (limit_prices - exit_prices) / limit_prices

        # Blend long and short returns based on position direction
        position_returns = is_long_probs * long_returns + (1 - is_long_probs) * short_returns

        # Apply transaction costs
        net_returns = position_returns - transaction_cost

        # Apply fill probability (soft gating)
        filled_returns = fill_prob * net_returns

        return filled_returns

    def _calculate_fill_probability(
        self,
        limit_price: float,
        market_high: float,
        market_low: float,
        is_long: bool
    ) -> float:
        """Calculate probability of order fill based on market movement."""
        if is_long:
            # Long order fills if market goes below limit price
            if market_low <= limit_price:
                return 1.0
            else:
                # Partial probability based on how close market got
                distance = (limit_price - market_low) / market_low
                return max(0.0, 1.0 - distance * 10)  # Adjust sensitivity
        else:
            # Short order fills if market goes above limit price
            if market_high >= limit_price:
                return 1.0
            else:
                distance = (market_high - limit_price) / limit_price
                return max(0.0, 1.0 - distance * 10)

    def _simulate_exit_price(
        self,
        entry_close: float,
        holding_hours: int,
        is_long: bool
    ) -> float:
        """Simulate exit price after holding period."""
        # Simple random walk simulation
        volatility = 0.02  # 2% hourly volatility
        price_change = np.random.normal(0, volatility * np.sqrt(holding_hours))
        exit_price = entry_close * (1 + price_change)
        return max(exit_price, entry_close * 0.5)  # Prevent extreme losses

    def train_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        loss_function: AdvancedTradingLoss,
        train_timestamps: List[pd.Timestamp],
        batch_size: int = 16,
        max_batches: int = 100
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            optimizer: Optimizer for training
            loss_function: Loss function to use
            train_timestamps: Training timestamps
            batch_size: Batch size for training
            max_batches: Maximum batches per epoch

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        epoch_losses = []
        epoch_metrics = []

        # Sample random batches from training data
        num_batches = min(max_batches, len(train_timestamps) // batch_size)

        pbar = tqdm(range(num_batches), desc=f"Epoch {self.current_epoch + 1}")

        for batch_idx in pbar:
            optimizer.zero_grad()

            # Sample random timestamps for this batch
            batch_timestamps = np.random.choice(
                train_timestamps, size=batch_size, replace=False
            ).tolist()

            try:
                # Create batch data
                vol_pred = torch.rand(batch_size, 38, 1, dtype=torch.float32, device=self.device) * 0.05 + 0.02
                vol_history = torch.rand(batch_size, 38, 24, dtype=torch.float32, device=self.device) * 0.05 + 0.02

                # Get OHLCV data
                ohlcv_batch = self.dataloader.create_enhanced_batch(
                    end_times=batch_timestamps,
                    vol_pred_data=vol_pred,
                    vol_history_data=vol_history,
                    window_hours=48
                )

                # Move to device and ensure float32
                ohlcv_data = {}
                for key in ['open', 'high', 'low', 'close', 'volume']:
                    if key in ohlcv_batch:
                        ohlcv_data[key] = ohlcv_batch[key].to(self.device, dtype=torch.float32)

                current_prices = ohlcv_batch['current_prices'].to(self.device, dtype=torch.float32)

                # Generate trading signals
                signals = self.model.get_trading_signals(
                    vol_pred=vol_pred,
                    vol_pred_history=vol_history,
                    current_price=current_prices,
                    ohlc_data=ohlcv_data,
                    volume_data=ohlcv_data.get('volume'),
                    timestamps=ohlcv_batch.get('timestamps')
                )

                # Simulate trading (differentiable)
                returns = self.simulate_enhanced_trading_differentiable(signals, ohlcv_data)

                # Calculate loss
                loss, metrics = loss_function(returns, signals, ohlcv_data)

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                # Track metrics
                epoch_losses.append(loss.item())
                epoch_metrics.append(metrics)

                # Update progress bar
                pbar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Fill Rate': f"{metrics['fill_rate']:.3f}",
                    'Avg Return': f"{metrics['avg_return']:.4f}"
                })

            except Exception as e:
                logger.warning(f"Batch {batch_idx} failed: {e}")
                continue

        # Calculate epoch averages
        if epoch_metrics:
            avg_metrics = {
                key: np.mean([m[key] for m in epoch_metrics])
                for key in epoch_metrics[0].keys()
            }
            avg_metrics['epoch_loss'] = np.mean(epoch_losses)
        else:
            avg_metrics = {'epoch_loss': float('inf')}

        return avg_metrics

    def validate_epoch(
        self,
        loss_function: AdvancedTradingLoss,
        val_timestamps: List[pd.Timestamp],
        batch_size: int = 16,
        max_batches: int = 50
    ) -> Dict[str, float]:
        """
        Validate for one epoch.

        Args:
            loss_function: Loss function to use
            val_timestamps: Validation timestamps
            batch_size: Batch size for validation
            max_batches: Maximum batches for validation

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        val_losses = []
        val_metrics = []

        num_batches = min(max_batches, len(val_timestamps) // batch_size)

        with torch.no_grad():
            for batch_idx in range(num_batches):
                # Sample validation batch
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(val_timestamps))
                batch_timestamps = val_timestamps[start_idx:end_idx]

                if len(batch_timestamps) < batch_size:
                    continue

                try:
                    # Create validation batch (similar to training)
                    vol_pred = torch.rand(len(batch_timestamps), 38, 1, dtype=torch.float32, device=self.device) * 0.05 + 0.02
                    vol_history = torch.rand(len(batch_timestamps), 38, 24, dtype=torch.float32, device=self.device) * 0.05 + 0.02

                    ohlcv_batch = self.dataloader.create_enhanced_batch(
                        end_times=batch_timestamps,
                        vol_pred_data=vol_pred,
                        vol_history_data=vol_history,
                        window_hours=48
                    )

                    ohlcv_data = {}
                    for key in ['open', 'high', 'low', 'close', 'volume']:
                        if key in ohlcv_batch:
                            ohlcv_data[key] = ohlcv_batch[key].to(self.device, dtype=torch.float32)

                    current_prices = ohlcv_batch['current_prices'].to(self.device, dtype=torch.float32)

                    signals = self.model.get_trading_signals(
                        vol_pred=vol_pred,
                        vol_pred_history=vol_history,
                        current_price=current_prices,
                        ohlc_data=ohlcv_data,
                        volume_data=ohlcv_data.get('volume'),
                        timestamps=ohlcv_batch.get('timestamps')
                    )

                    returns = self.simulate_enhanced_trading_differentiable(signals, ohlcv_data)
                    loss, metrics = loss_function(returns, signals, ohlcv_data)

                    val_losses.append(loss.item())
                    val_metrics.append(metrics)

                except Exception as e:
                    logger.warning(f"Validation batch {batch_idx} failed: {e}")
                    continue

        # Calculate validation averages
        if val_metrics:
            avg_val_metrics = {
                key: np.mean([m[key] for m in val_metrics])
                for key in val_metrics[0].keys()
            }
            avg_val_metrics['val_loss'] = np.mean(val_losses)
        else:
            avg_val_metrics = {'val_loss': float('inf')}

        return avg_val_metrics

    def train(
        self,
        num_epochs: int = 50,
        learning_rate: float = 1e-4,
        batch_size: int = 16,
        patience: int = 10,
        min_delta: float = 1e-4,
        use_curriculum: bool = True,
        save_checkpoints: bool = True
    ) -> Dict[str, Any]:
        """
        Main training loop with advanced optimization strategies.

        Args:
            num_epochs: Number of training epochs
            learning_rate: Initial learning rate
            batch_size: Batch size for training
            patience: Early stopping patience
            min_delta: Minimum improvement for early stopping
            use_curriculum: Whether to use curriculum learning
            save_checkpoints: Whether to save model checkpoints

        Returns:
            Training history and final metrics
        """
        logger.info(f"🚀 Starting advanced training for {num_epochs} epochs")

        # Create train/validation split
        train_timestamps, val_timestamps = self.create_train_val_split()

        # Initialize optimizer and scheduler
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5,
            betas=(0.9, 0.999)
        )

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )

        # Initialize loss function and curriculum scheduler
        loss_function = AdvancedTradingLoss(
            profit_weight=3.0,
            sharpe_weight=0.5,
            fill_rate_weight=2.0,
            target_fill_rate=0.15
        )

        if use_curriculum:
            curriculum_scheduler = CurriculumLearningScheduler(
                initial_weights={
                    'profit_weight': 3.0,
                    'sharpe_weight': 0.5,
                    'fill_rate_weight': 2.0,
                    'consistency_weight': 0.2
                },
                target_weights={
                    'profit_weight': 1.5,
                    'sharpe_weight': 2.0,
                    'fill_rate_weight': 0.8,
                    'consistency_weight': 1.0
                },
                transition_epochs=num_epochs // 2
            )

        # Training loop
        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Update curriculum if enabled
            if use_curriculum:
                curriculum_scheduler.update_loss_function(loss_function, epoch)

            # Training phase
            train_metrics = self.train_epoch(
                optimizer, loss_function, train_timestamps, batch_size
            )

            # Validation phase
            val_metrics = self.validate_epoch(
                loss_function, val_timestamps, batch_size
            )

            # Update learning rate scheduler
            val_score = -val_metrics.get('val_loss', float('inf'))
            scheduler.step(val_score)

            # Log epoch results
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Epoch {epoch + 1}/{num_epochs}:")
            logger.info(f"  Train Loss: {train_metrics.get('epoch_loss', 0):.4f}")
            logger.info(f"  Val Loss: {val_metrics.get('val_loss', 0):.4f}")
            logger.info(f"  Train Fill Rate: {train_metrics.get('fill_rate', 0):.3f}")
            logger.info(f"  Val Fill Rate: {val_metrics.get('fill_rate', 0):.3f}")
            logger.info(f"  Learning Rate: {current_lr:.2e}")

            # Update training history
            self.training_history['epoch'].append(epoch + 1)
            self.training_history['train_loss'].append(train_metrics.get('epoch_loss', 0))
            self.training_history['val_loss'].append(val_metrics.get('val_loss', 0))
            self.training_history['train_fill_rate'].append(train_metrics.get('fill_rate', 0))
            self.training_history['val_fill_rate'].append(val_metrics.get('fill_rate', 0))
            self.training_history['train_avg_return'].append(train_metrics.get('avg_return', 0))
            self.training_history['val_avg_return'].append(val_metrics.get('avg_return', 0))
            self.training_history['learning_rate'].append(current_lr)

            # Calculate Sharpe ratios for history
            train_sharpe = -train_metrics.get('sharpe_loss', 0) if 'sharpe_loss' in train_metrics else 0
            val_sharpe = -val_metrics.get('sharpe_loss', 0) if 'sharpe_loss' in val_metrics else 0
            self.training_history['train_sharpe'].append(train_sharpe)
            self.training_history['val_sharpe'].append(val_sharpe)

            # Early stopping and checkpointing
            if val_score > self.best_score + min_delta:
                self.best_score = val_score
                self.patience_counter = 0

                if save_checkpoints:
                    self.save_checkpoint(epoch, val_score, "best")

            else:
                self.patience_counter += 1

            # Save regular checkpoint
            if save_checkpoints and (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, val_score, f"epoch_{epoch + 1}")

            # Early stopping
            if self.patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

        # Save final model and create visualizations
        if save_checkpoints:
            self.save_checkpoint(epoch, val_score, "final")

        self.create_training_visualizations()

        return {
            'training_history': self.training_history,
            'best_score': self.best_score,
            'final_epoch': epoch + 1,
            'model_path': str(self.save_dir)
        }

    def save_checkpoint(self, epoch: int, score: float, checkpoint_type: str):
        """Save model checkpoint with training state."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = self.save_dir / f"enhanced_agent_{checkpoint_type}_{timestamp}.pt"

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'score': score,
            'training_history': self.training_history,
            'model_config': {
                'n_assets': self.model.n_assets,
                'vol_history_length': self.model.vol_history_length,
                'feature_dim': self.model.feature_dim,
                'hidden_dim': self.model.hidden_dim,
                'init_with_vol_pred': self.model.init_with_vol_pred
            }
        }, checkpoint_path)

        logger.info(f"💾 Saved checkpoint: {checkpoint_path}")

    def create_training_visualizations(self):
        """Create comprehensive training visualizations."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Training & Validation Loss',
                'Fill Rates',
                'Average Returns',
                'Sharpe Ratios',
                'Learning Rate',
                'Performance Summary'
            ],
            vertical_spacing=0.08
        )

        epochs = self.training_history['epoch']

        # Loss curves
        fig.add_trace(
            go.Scatter(x=epochs, y=self.training_history['train_loss'],
                      name='Train Loss', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=self.training_history['val_loss'],
                      name='Val Loss', line=dict(color='red')),
            row=1, col=1
        )

        # Fill rates
        fig.add_trace(
            go.Scatter(x=epochs, y=self.training_history['train_fill_rate'],
                      name='Train Fill Rate', line=dict(color='green')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=self.training_history['val_fill_rate'],
                      name='Val Fill Rate', line=dict(color='orange')),
            row=1, col=2
        )

        # Average returns
        fig.add_trace(
            go.Scatter(x=epochs, y=self.training_history['train_avg_return'],
                      name='Train Avg Return', line=dict(color='purple')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=self.training_history['val_avg_return'],
                      name='Val Avg Return', line=dict(color='brown')),
            row=2, col=1
        )

        # Sharpe ratios
        fig.add_trace(
            go.Scatter(x=epochs, y=self.training_history['train_sharpe'],
                      name='Train Sharpe', line=dict(color='cyan')),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=self.training_history['val_sharpe'],
                      name='Val Sharpe', line=dict(color='magenta')),
            row=2, col=2
        )

        # Learning rate
        fig.add_trace(
            go.Scatter(x=epochs, y=self.training_history['learning_rate'],
                      name='Learning Rate', line=dict(color='black')),
            row=3, col=1
        )

        # Performance summary (final metrics)
        final_metrics = {
            'Final Train Loss': self.training_history['train_loss'][-1],
            'Final Val Loss': self.training_history['val_loss'][-1],
            'Final Fill Rate': self.training_history['val_fill_rate'][-1],
            'Best Score': self.best_score
        }

        fig.add_trace(
            go.Bar(x=list(final_metrics.keys()), y=list(final_metrics.values()),
                   name='Final Metrics'),
            row=3, col=2
        )

        fig.update_layout(
            title=f"Enhanced Trading Agent Training Progress - {timestamp}",
            height=1000,
            showlegend=True
        )

        # Save visualization
        plot_path = self.save_dir / f"training_progress_{timestamp}.html"
        fig.write_html(plot_path)
        logger.info(f"📊 Training visualization saved: {plot_path}")

        # Save training history as CSV
        history_df = pd.DataFrame(self.training_history)
        csv_path = self.save_dir / f"training_history_{timestamp}.csv"
        history_df.to_csv(csv_path, index=False)
        logger.info(f"📈 Training history saved: {csv_path}")
