#!/usr/bin/env python
"""
Training script for agent-based trading model.

This script implements a three-stage training approach:
1. Stage 1: Train volatility prediction model (supervised learning)
2. Stage 2: Generate volatility predictions for history
3. Stage 3: Train agent model with frozen volatility model
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import pandas as pd
import os
import sys
import logging
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# EGARCH modeling
from arch import arch_model

from src.models.flexible_gsphar import FlexibleGSPHAR
from src.models.agent_model import TradingAgentModel, SimpleTradingAgent, BasicLinearAgent
from src.agent_trading_loss import AgentTradingLoss, AgentSharpeRatioLoss, AgentAdvancedTradingLoss
from src.data.agent_trading_dataset import load_agent_trading_data
from src.data.ohlcv_trading_dataset import load_ohlcv_trading_data

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EGARCHVolatilityModel:
    """EGARCH model for volatility prediction - compatible with agent training pipeline."""

    def __init__(self, n_assets, p=1, q=1):
        """
        Initialize EGARCH model for multiple assets.

        Args:
            n_assets: Number of cryptocurrency assets
            p: Number of lags for the squared residuals
            q: Number of lags for the conditional variance
        """
        self.n_assets = n_assets
        self.p = p
        self.q = q
        self.fitted_models = {}  # Store fitted models for each asset
        self.asset_names = []

    def fit(self, price_data, asset_names):
        """
        Fit EGARCH models for all assets.

        Args:
            price_data: DataFrame with price data (columns = asset names)
            asset_names: List of asset names
        """
        self.asset_names = asset_names
        logger.info(f"🔄 Fitting EGARCH models for {len(asset_names)} assets...")

        fitted_count = 0
        for asset in tqdm(asset_names, desc="Fitting EGARCH models"):
            try:
                if asset in price_data.columns:
                    # Calculate returns
                    prices = price_data[asset].dropna()
                    returns = prices.pct_change().dropna()

                    # Remove extreme outliers (beyond 3 standard deviations)
                    returns_clean = returns[np.abs(returns - returns.mean()) <= 3 * returns.std()]

                    if len(returns_clean) < 100:  # Need minimum data
                        logger.warning(f"Insufficient data for {asset}: {len(returns_clean)} observations")
                        continue

                    # Fit EGARCH model
                    model = arch_model(
                        returns_clean * 100,  # Scale for numerical stability
                        vol='EGARCH',
                        p=self.p,
                        q=self.q,
                        dist='normal'
                    )

                    fitted_model = model.fit(disp='off', show_warning=False)
                    self.fitted_models[asset] = fitted_model
                    fitted_count += 1

                else:
                    logger.warning(f"Asset {asset} not found in price data")

            except Exception as e:
                logger.warning(f"Failed to fit EGARCH for {asset}: {str(e)}")
                continue

        logger.info(f"✅ Successfully fitted {fitted_count}/{len(asset_names)} EGARCH models")
        return fitted_count > 0

    def predict_volatility(self, batch_size=1):
        """
        Predict volatility for all assets (compatible with agent training).

        Args:
            batch_size: Batch size for predictions

        Returns:
            torch.Tensor: Volatility predictions [batch_size, n_assets, 1]
        """
        vol_predictions = []

        for asset in self.asset_names:
            if asset in self.fitted_models:
                try:
                    # Get volatility forecast
                    forecast = self.fitted_models[asset].forecast(horizon=1)
                    vol_forecast = np.sqrt(forecast.variance.iloc[-1, 0]) / 100

                    # Clamp to reasonable range
                    vol_forecast = np.clip(vol_forecast, 0.001, 0.5)
                    vol_predictions.append(vol_forecast)

                except Exception as e:
                    logger.warning(f"Prediction failed for {asset}: {str(e)}")
                    vol_predictions.append(0.02)  # Default 2% volatility
            else:
                vol_predictions.append(0.02)  # Default for unfitted models

        # Convert to tensor format expected by agent model
        vol_array = np.array(vol_predictions)
        vol_tensor = torch.tensor(vol_array, dtype=torch.float32)

        # Expand to batch format: [batch_size, n_assets, 1]
        vol_tensor = vol_tensor.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1)

        return vol_tensor

    def to(self, device):
        """Compatibility method for device placement."""
        return self

    def eval(self):
        """Compatibility method for evaluation mode."""
        return self


def train_stage1_volatility(model, train_loader, val_loader, device, n_epochs=15, lr=0.001):
    """
    Stage 1: Train volatility prediction model with supervised learning.
    """
    logger.info(f"🎯 STAGE 1: VOLATILITY PREDICTION TRAINING")
    logger.info(f"  Epochs: {n_epochs}")
    logger.info(f"  Learning rate: {lr}")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=3)

    train_history = {'train_loss': [], 'val_loss': [], 'lr': []}
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 7

    for epoch in range(n_epochs):
        # Training phase
        model.train()
        train_losses = []

        train_pbar = tqdm(train_loader, desc=f"Stage 1 Train {epoch+1}", leave=False)
        for batch in train_pbar:
            x_lags = batch['x_lags']
            vol_targets = batch['vol_targets']

            # Move to device
            x_lags = [x.to(device) for x in x_lags]
            vol_targets = vol_targets.to(device)

            # Forward pass
            optimizer.zero_grad()
            vol_pred = model(*x_lags)
            loss = criterion(vol_pred, vol_targets)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())
            train_pbar.set_postfix({'Loss': f"{loss.item():.6f}"})

        # Validation phase
        model.eval()
        val_losses = []

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Stage 1 Val {epoch+1}", leave=False)
            for batch in val_pbar:
                x_lags = batch['x_lags']
                vol_targets = batch['vol_targets']

                x_lags = [x.to(device) for x in x_lags]
                vol_targets = vol_targets.to(device)

                vol_pred = model(*x_lags)
                loss = criterion(vol_pred, vol_targets)
                val_losses.append(loss.item())
                val_pbar.set_postfix({'Loss': f"{loss.item():.6f}"})

        # Calculate epoch metrics
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        current_lr = optimizer.param_groups[0]['lr']

        train_history['train_loss'].append(avg_train_loss)
        train_history['val_loss'].append(avg_val_loss)
        train_history['lr'].append(current_lr)

        logger.info(f"Epoch {epoch+1}/{n_epochs}: Train Loss: {avg_train_loss:.6f}, "
                   f"Val Loss: {avg_val_loss:.6f}, LR: {current_lr:.6f}")

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'models/best_volatility_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    logger.info(f"✅ Stage 1 completed. Best validation loss: {best_val_loss:.6f}")
    return train_history


def generate_volatility_predictions(model, dataset, device):
    """
    Stage 2: Generate volatility predictions for the entire dataset.
    """
    logger.info(f"🔮 STAGE 2: GENERATING VOLATILITY PREDICTIONS")

    model.eval()
    predictions = {}

    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Generating predictions"):
            sample = dataset[i]
            x_lags = [x.unsqueeze(0).to(device) for x in sample['x_lags']]

            vol_pred = model(*x_lags)
            vol_pred_np = vol_pred.squeeze().cpu().numpy()

            predictions[i] = vol_pred_np

    logger.info(f"✅ Generated {len(predictions)} volatility predictions")
    return predictions


def train_stage3_agent(volatility_model, agent_model, train_loader, val_loader, device,
                      n_epochs=20, lr=0.0005, loss_type='basic', agent_type='full'):
    """
    Stage 3: Train agent model with frozen volatility model.
    """
    logger.info(f"🤖 STAGE 3: AGENT MODEL TRAINING")
    logger.info(f"  Epochs: {n_epochs}")
    logger.info(f"  Learning rate: {lr}")
    logger.info(f"  Loss type: {loss_type}")

    # EGARCH model doesn't have parameters to freeze (it's already fitted)
    volatility_model.eval()

    # Setup agent training with more conservative parameters
    optimizer = optim.Adam(agent_model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # Choose loss function
    if loss_type == 'basic':
        criterion = AgentTradingLoss(holding_period=4, trading_fee=0.0002)
    elif loss_type == 'sharpe':
        criterion = AgentSharpeRatioLoss(holding_period=4, trading_fee=0.0002)
    elif loss_type == 'advanced':
        criterion = AgentAdvancedTradingLoss(holding_period=4)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    train_history = {'train_loss': [], 'val_loss': [], 'train_metrics': [], 'val_metrics': []}
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 8

    for epoch in range(n_epochs):
        # Training phase
        agent_model.train()
        train_losses = []
        train_metrics_list = []

        train_pbar = tqdm(train_loader, desc=f"Stage 3 Train {epoch+1}", leave=False)
        for batch in train_pbar:
            x_lags = batch['x_lags']
            vol_pred_history = batch['vol_pred_history']
            ohlcv_data = batch['ohlcv_data']

            # Move to device
            x_lags = [x.to(device) for x in x_lags]
            vol_pred_history = vol_pred_history.to(device)
            ohlcv_data = ohlcv_data.to(device)

            # Get volatility predictions from EGARCH
            batch_size = ohlcv_data.shape[0]
            vol_pred = volatility_model.predict_volatility(batch_size=batch_size).to(device)

            # Agent forward pass
            optimizer.zero_grad()
            if hasattr(agent_model, 'forward_with_vol_pred_init'):
                ratio, direction = agent_model.forward_with_vol_pred_init(vol_pred, vol_pred_history)
            else:
                ratio, direction = agent_model(vol_pred, vol_pred_history)

            # Calculate loss
            loss = criterion(ratio, direction, ohlcv_data)

            # Backward pass with more aggressive gradient clipping
            if not torch.isnan(loss) and not torch.isinf(loss) and loss.item() < 1000:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent_model.parameters(), max_norm=0.5)
                optimizer.step()

                train_losses.append(loss.item())

                # Calculate metrics
                if hasattr(criterion, 'calculate_metrics'):
                    metrics = criterion.calculate_metrics(ratio, direction, ohlcv_data)
                    train_metrics_list.append(metrics)

                    train_pbar.set_postfix({
                        'Loss': f"{loss.item():.6f}",
                        'Fill': f"{metrics.get('fill_rate', 0):.3f}",
                        'Profit': f"{metrics.get('avg_profit_when_filled', 0):.4f}",
                        'Long%': f"{metrics.get('long_ratio', 0.5):.2f}"
                    })
                else:
                    train_pbar.set_postfix({'Loss': f"{loss.item():.6f}"})

        # Validation phase
        agent_model.eval()
        val_losses = []
        val_metrics_list = []

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Stage 3 Val {epoch+1}", leave=False)
            for batch in val_pbar:
                x_lags = batch['x_lags']
                vol_pred_history = batch['vol_pred_history']
                ohlcv_data = batch['ohlcv_data']

                x_lags = [x.to(device) for x in x_lags]
                vol_pred_history = vol_pred_history.to(device)
                ohlcv_data = ohlcv_data.to(device)

                # Get volatility predictions from EGARCH
                batch_size = ohlcv_data.shape[0]
                vol_pred = volatility_model.predict_volatility(batch_size=batch_size).to(device)

                # Agent forward pass
                if hasattr(agent_model, 'forward_with_vol_pred_init'):
                    ratio, direction = agent_model.forward_with_vol_pred_init(vol_pred, vol_pred_history)
                else:
                    ratio, direction = agent_model(vol_pred, vol_pred_history)

                # Calculate loss
                loss = criterion(ratio, direction, ohlcv_data)
                val_losses.append(loss.item())

                # Calculate metrics
                if hasattr(criterion, 'calculate_metrics'):
                    metrics = criterion.calculate_metrics(ratio, direction, ohlcv_data)
                    val_metrics_list.append(metrics)

                    val_pbar.set_postfix({
                        'Loss': f"{loss.item():.6f}",
                        'Fill': f"{metrics.get('fill_rate', 0):.3f}",
                        'Profit': f"{metrics.get('avg_profit_when_filled', 0):.4f}",
                        'Long%': f"{metrics.get('long_ratio', 0.5):.2f}"
                    })
                else:
                    val_pbar.set_postfix({'Loss': f"{loss.item():.6f}"})

        # Calculate epoch metrics
        avg_train_loss = np.mean(train_losses) if train_losses else float('inf')
        avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
        current_lr = optimizer.param_groups[0]['lr']

        train_history['train_loss'].append(avg_train_loss)
        train_history['val_loss'].append(avg_val_loss)

        if train_metrics_list:
            avg_train_metrics = {k: np.mean([m[k] for m in train_metrics_list])
                               for k in train_metrics_list[0].keys()}
            train_history['train_metrics'].append(avg_train_metrics)

        if val_metrics_list:
            avg_val_metrics = {k: np.mean([m[k] for m in val_metrics_list])
                             for k in val_metrics_list[0].keys()}
            train_history['val_metrics'].append(avg_val_metrics)

        logger.info(f"Epoch {epoch+1}/{n_epochs}: Train Loss: {avg_train_loss:.6f}, "
                   f"Val Loss: {avg_val_loss:.6f}, LR: {current_lr:.6f}")

        if train_metrics_list and val_metrics_list:
            logger.info(f"  Train Fill Rate: {avg_train_metrics['fill_rate']:.3f}, "
                       f"Val Fill Rate: {avg_val_metrics['fill_rate']:.3f}")
            logger.info(f"  Train Long Ratio: {avg_train_metrics['long_ratio']:.3f}, "
                       f"Val Long Ratio: {avg_val_metrics['long_ratio']:.3f}")

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best agent model
            torch.save({
                'model_state_dict': agent_model.state_dict(),
                'agent_type': agent_type,
                'n_assets': agent_model.n_assets,
                'loss_type': loss_type
            }, f'models/best_agent_model_{agent_type}_{loss_type}.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    logger.info(f"✅ Stage 3 completed. Best validation loss: {best_val_loss:.6f}")
    return train_history


def main():
    """Main training function."""
    logger.info("🚀 AGENT-BASED TRADING MODEL TRAINING")
    logger.info("=" * 80)

    # Parameters
    device = torch.device('cpu')
    batch_size = 16
    stage1_epochs = 10  # Volatility model training (reduced)
    stage3_epochs = 15  # Agent model training (reduced)
    stage1_lr = 0.001
    stage3_lr = 0.0001  # Much lower learning rate for agent
    history_length = 24
    holding_period = 4

    # Model selection: 'basic_linear', 'simple', or 'full'
    agent_type = 'basic_linear'  # Use basic linear agent for baseline

    print(f"\n🎯 EGARCH-BASED BASICLINEAR AGENT")
    print(f"   Formula: limit_price = price * (1 - param1 * EGARCH_vol_pred)")
    print(f"   Agent type: {agent_type}")
    print(f"   Using EGARCH models instead of GSPHAR neural networks")

    # Load dataset
    logger.info("📊 Loading dataset...")

    # First, create a volatility proxy from 1-hour price data for the dataset
    logger.info("🔄 Creating volatility proxy from 1-hour price data...")
    price_df_for_dataset = pd.read_pickle("data/df_cl_h1.pickle")

    # Calculate rolling volatility from returns as proxy for the dataset
    # This is different from the EGARCH training - here we need historical volatility for the dataset
    returns_df = price_df_for_dataset.pct_change().dropna()

    # Calculate rolling 24-hour volatility as proxy for realized volatility
    rolling_vol_df = returns_df.rolling(window=24, min_periods=12).std().dropna()

    # Save temporary volatility file for dataset loading
    temp_vol_file = "data/temp_1h_volatility_proxy.csv"
    rolling_vol_df.to_csv(temp_vol_file)
    logger.info(f"   Created volatility proxy: {rolling_vol_df.shape}")
    logger.info(f"   Saved to: {temp_vol_file}")

    dataset, metadata = load_agent_trading_data(
        volatility_file=temp_vol_file,
        lags=[1, 4, 24],
        holding_period=holding_period,
        history_length=history_length,
        debug=True  # Use subset for initial testing
    )

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = data.random_split(dataset, [train_size, val_size])

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    logger.info(f"  Train samples: {len(train_dataset)}")
    logger.info(f"  Validation samples: {len(val_dataset)}")

    # Create models
    logger.info("🏗️ Creating models...")

    # Load 1-hour price data for EGARCH model fitting
    logger.info("📊 Loading 1-hour price data for EGARCH training...")
    price_df = pd.read_pickle("data/df_cl_h1.pickle")
    logger.info(f"   Price data shape: {price_df.shape}")
    logger.info(f"   Available symbols: {list(price_df.columns)}")
    logger.info(f"   Date range: {price_df.index[0]} to {price_df.index[-1]}")

    # Create EGARCH volatility model (replaces GSPHAR)
    volatility_model = EGARCHVolatilityModel(
        n_assets=len(metadata['assets']),
        p=1,  # EGARCH parameters
        q=1
    ).to(device)

    # Fit EGARCH models to historical price data (using returns calculated from prices)
    logger.info("🔄 Fitting EGARCH models to 1-hour return data...")
    volatility_model.fit(price_df, metadata['assets'])

    # Agent model selection
    if agent_type == 'basic_linear':
        agent_model = BasicLinearAgent(
            n_assets=len(metadata['assets']),
            init_param1=0.5  # Start with 0.5 multiplier
        ).to(device)
        logger.info(f"🎯 Using BasicLinearAgent with formula: limit_price = price * (1 - param1 * vol_pred)")
    elif agent_type == 'simple':
        agent_model = SimpleTradingAgent(
            n_assets=len(metadata['assets']),
            history_length=history_length,
            hidden_dim=32
        ).to(device)
        logger.info(f"🎯 Using SimpleTradingAgent")
    else:  # 'full'
        agent_model = TradingAgentModel(
            n_assets=len(metadata['assets']),
            history_length=history_length,
            hidden_dim=64,
            dropout=0.1,
            init_with_vol_pred=True  # Initialize with previous strategy
        ).to(device)
        logger.info(f"🎯 Using full TradingAgentModel")

    logger.info(f"EGARCH models fitted: {len(volatility_model.fitted_models)}/{volatility_model.n_assets}")
    logger.info(f"Agent model parameters: {sum(p.numel() for p in agent_model.parameters())}")

    # Create models directory
    os.makedirs('models', exist_ok=True)

    # Stage 1: EGARCH models are already fitted (no neural network training needed)
    logger.info("✅ EGARCH models fitted and ready for volatility predictions")
    stage1_history = {'train_loss': [], 'val_loss': [], 'lr': []}  # Empty history for compatibility

    # Stage 2: Generate predictions (for future use)
    # For now, we'll use the proxy historical volatility in the dataset

    # Stage 3: Train agent model
    for loss_type in ['basic', 'advanced']:
        logger.info(f"\n🎯 Training agent with {loss_type} loss...")

        stage3_history = train_stage3_agent(
            volatility_model=volatility_model,
            agent_model=agent_model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            n_epochs=stage3_epochs,
            lr=stage3_lr,
            loss_type=loss_type,
            agent_type=agent_type
        )

        # Save training history
        torch.save({
            'stage1_history': stage1_history,
            'stage3_history': stage3_history,
            'metadata': metadata,
            'loss_type': loss_type,
            'agent_type': agent_type
        }, f'models/agent_training_history_{agent_type}_{loss_type}.pt')

    logger.info("🎉 Training completed successfully!")

    # Quick test of EGARCH predictions
    print(f"\n📊 EGARCH VOLATILITY PREDICTIONS SAMPLE:")
    sample_vol_pred = volatility_model.predict_volatility(batch_size=1)
    print(f"   Sample vol_pred shape: {sample_vol_pred.shape}")
    print(f"   Sample vol_pred range: {sample_vol_pred.min():.4f} - {sample_vol_pred.max():.4f}")
    print(f"   Sample vol_pred mean: {sample_vol_pred.mean():.4f}")

    # Test BasicLinearAgent with EGARCH predictions
    print(f"\n🧪 TESTING BASICLINEAR AGENT WITH EGARCH:")
    test_price = torch.tensor([[100.0] * len(metadata['assets'])])  # $100 for all assets
    test_signals = agent_model.get_trading_signals(sample_vol_pred, None, test_price)

    print(f"   Current price: $100.00")
    print(f"   Limit price range: ${test_signals['limit_price'].min():.2f} - ${test_signals['limit_price'].max():.2f}")
    print(f"   Average discount: {((test_price.mean() - test_signals['limit_price'].mean()) / test_price.mean() * 100).item():.2f}%")
    print(f"   Learned param1 range: {test_signals['param1'].min():.4f} - {test_signals['param1'].max():.4f}")

    print(f"\n✅ EGARCH-based BasicLinearAgent is ready for comparison!")
    print(f"\n� Run PnL analysis with: python scripts/egarch_pnl_analysis.py")

    # Cleanup temporary files
    temp_vol_file = "data/temp_1h_volatility_proxy.csv"
    if os.path.exists(temp_vol_file):
        os.remove(temp_vol_file)
        logger.info(f"🧹 Cleaned up temporary file: {temp_vol_file}")


if __name__ == "__main__":
    main()
