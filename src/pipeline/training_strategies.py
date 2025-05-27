"""
Training Strategy Classes for Flexible Training Pipeline.

This module implements different training strategies that can be selected
through the experiment configuration system.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional
import logging
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm

from .experiment_config import TrainingConfig, TrainingApproach

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """Container for training results."""
    train_losses: List[float]
    val_losses: List[float]
    train_metrics: Dict[str, List[float]]
    val_metrics: Dict[str, List[float]]
    best_epoch: int
    best_val_loss: float
    final_model_state: Dict[str, Any]


class BaseTrainingStrategy(ABC):
    """Base class for training strategies."""

    def __init__(self, config: TrainingConfig, device: str = "cpu"):
        self.config = config
        self.device = device
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def train(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ) -> TrainingResult:
        """Execute the training strategy."""
        pass

    def _calculate_metrics(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        loss_val: float
    ) -> Dict[str, float]:
        """Calculate additional metrics beyond loss."""
        metrics = {"loss": loss_val}

        # Mean Absolute Error
        mae = torch.mean(torch.abs(outputs - targets)).item()
        metrics["mae"] = mae

        # Root Mean Square Error
        rmse = torch.sqrt(torch.mean((outputs - targets) ** 2)).item()
        metrics["rmse"] = rmse

        # Mean Absolute Percentage Error
        mape = torch.mean(torch.abs((outputs - targets) / targets)) * 100
        metrics["mape"] = mape.item() if torch.isfinite(mape) else float('inf')

        return metrics


class SingleStageStrategy(BaseTrainingStrategy):
    """Standard single-stage training approach."""

    def train(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ) -> TrainingResult:
        """Execute single-stage training."""
        self.logger.info("Starting single-stage training")

        train_losses = []
        val_losses = []
        train_metrics = {"mae": [], "rmse": [], "mape": []}
        val_metrics = {"mae": [], "rmse": [], "mape": []}

        best_val_loss = float('inf')
        best_epoch = 0
        best_model_state = None

        for epoch in range(self.config.n_epochs):
            # Training phase
            model.train()
            epoch_train_loss = 0.0
            epoch_train_metrics = {"mae": 0.0, "rmse": 0.0, "mape": 0.0}

            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.n_epochs} [Train]")
            for batch_idx, batch in enumerate(train_pbar):
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    inputs, targets = batch
                else:
                    inputs, targets = batch['features'], batch['targets']

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()

                # Gradient clipping if specified
                if self.config.grad_clip_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip_norm)

                optimizer.step()

                batch_loss = loss.item()
                epoch_train_loss += batch_loss

                # Calculate metrics
                batch_metrics = self._calculate_metrics(outputs, targets, batch_loss)
                for key in epoch_train_metrics:
                    epoch_train_metrics[key] += batch_metrics[key]

                train_pbar.set_postfix({"loss": f"{batch_loss:.4f}"})

            # Average training metrics
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            for key in train_metrics:
                avg_metric = epoch_train_metrics[key] / len(train_loader)
                train_metrics[key].append(avg_metric)

            # Validation phase
            model.eval()
            epoch_val_loss = 0.0
            epoch_val_metrics = {"mae": 0.0, "rmse": 0.0, "mape": 0.0}

            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{self.config.n_epochs} [Val]")
                for batch in val_pbar:
                    if isinstance(batch, (list, tuple)) and len(batch) == 2:
                        inputs, targets = batch
                    else:
                        inputs, targets = batch['features'], batch['targets']

                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    outputs = model(inputs)
                    loss = loss_fn(outputs, targets)

                    batch_loss = loss.item()
                    epoch_val_loss += batch_loss

                    # Calculate metrics
                    batch_metrics = self._calculate_metrics(outputs, targets, batch_loss)
                    for key in epoch_val_metrics:
                        epoch_val_metrics[key] += batch_metrics[key]

                    val_pbar.set_postfix({"loss": f"{batch_loss:.4f}"})

            # Average validation metrics
            avg_val_loss = epoch_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            for key in val_metrics:
                avg_metric = epoch_val_metrics[key] / len(val_loader)
                val_metrics[key].append(avg_metric)

            # Learning rate scheduling
            if scheduler:
                scheduler.step()

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                best_model_state = model.state_dict().copy()

            # Early stopping
            if self.config.early_stopping_patience:
                if epoch - best_epoch >= self.config.early_stopping_patience:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break

            self.logger.info(
                f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}"
            )

        return TrainingResult(
            train_losses=train_losses,
            val_losses=val_losses,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            final_model_state=best_model_state
        )


class TwoStageStrategy(BaseTrainingStrategy):
    """Two-stage training: pre-training + fine-tuning."""

    def train(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ) -> TrainingResult:
        """Execute two-stage training."""
        self.logger.info("Starting two-stage training")

        # Stage 1: Pre-training with MSE loss
        self.logger.info("Stage 1: Pre-training with MSE loss")
        stage1_epochs = self.config.n_epochs // 2

        # Create MSE loss for pre-training
        mse_loss = nn.MSELoss()

        # Stage 1 training
        stage1_config = TrainingConfig(
            approach=TrainingApproach.SINGLE_STAGE,
            n_epochs=stage1_epochs,
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            early_stopping_patience=self.config.early_stopping_patience,
            grad_clip_norm=self.config.grad_clip_norm
        )

        single_stage = SingleStageStrategy(stage1_config, self.device)
        stage1_result = single_stage.train(
            model, train_loader, val_loader, mse_loss, optimizer, scheduler
        )

        # Load best model from stage 1
        model.load_state_dict(stage1_result.final_model_state)

        # Stage 2: Fine-tuning with original loss
        self.logger.info("Stage 2: Fine-tuning with original loss")
        stage2_epochs = self.config.n_epochs - stage1_epochs

        # Reduce learning rate for fine-tuning
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1

        stage2_config = TrainingConfig(
            approach=TrainingApproach.SINGLE_STAGE,
            n_epochs=stage2_epochs,
            learning_rate=self.config.learning_rate * 0.1,
            batch_size=self.config.batch_size,
            early_stopping_patience=self.config.early_stopping_patience,
            grad_clip_norm=self.config.grad_clip_norm
        )

        single_stage = SingleStageStrategy(stage2_config, self.device)
        stage2_result = single_stage.train(
            model, train_loader, val_loader, loss_fn, optimizer, scheduler
        )

        # Combine results
        combined_losses = stage1_result.train_losses + stage2_result.train_losses
        combined_val_losses = stage1_result.val_losses + stage2_result.val_losses

        # Combine metrics
        combined_train_metrics = {}
        combined_val_metrics = {}
        for key in stage1_result.train_metrics:
            combined_train_metrics[key] = (
                stage1_result.train_metrics[key] + stage2_result.train_metrics[key]
            )
            combined_val_metrics[key] = (
                stage1_result.val_metrics[key] + stage2_result.val_metrics[key]
            )

        return TrainingResult(
            train_losses=combined_losses,
            val_losses=combined_val_losses,
            train_metrics=combined_train_metrics,
            val_metrics=combined_val_metrics,
            best_epoch=stage1_epochs + stage2_result.best_epoch,
            best_val_loss=stage2_result.best_val_loss,
            final_model_state=stage2_result.final_model_state
        )


class ProfitMaximizationStrategy(BaseTrainingStrategy):
    """Profit maximization training with trading-specific objectives."""

    def train(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ) -> TrainingResult:
        """Execute profit maximization training."""
        self.logger.info("Starting profit maximization training")

        # Use the existing single-stage strategy but with enhanced metrics
        # that focus on trading performance
        strategy = SingleStageStrategy(self.config, self.device)
        result = strategy.train(model, train_loader, val_loader, loss_fn, optimizer, scheduler)

        # TODO: Add trading-specific metrics calculation
        # - Sharpe ratio calculation
        # - Maximum drawdown
        # - Trading signals analysis

        return result


class OHLCVBasedStrategy(BaseTrainingStrategy):
    """OHLCV-based training strategy."""

    def train(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ) -> TrainingResult:
        """Execute OHLCV-based training."""
        self.logger.info("Starting OHLCV-based training")

        # For now, use single-stage strategy
        # TODO: Implement OHLCV-specific training logic
        strategy = SingleStageStrategy(self.config, self.device)
        return strategy.train(model, train_loader, val_loader, loss_fn, optimizer, scheduler)


class GARCHPipelineStrategy(BaseTrainingStrategy):
    """GARCH pipeline training strategy."""

    def train(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ) -> TrainingResult:
        """Execute GARCH pipeline training."""
        self.logger.info("Starting GARCH pipeline training")

        # GARCH models typically don't use PyTorch training loops
        # This is a placeholder for GARCH-specific training logic

        # TODO: Implement GARCH-specific training
        # - Use arch package for GARCH model fitting
        # - Return appropriate training results

        # For now, return empty results
        return TrainingResult(
            train_losses=[],
            val_losses=[],
            train_metrics={"mae": [], "rmse": [], "mape": []},
            val_metrics={"mae": [], "rmse": [], "mape": []},
            best_epoch=0,
            best_val_loss=0.0,
            final_model_state={}
        )


class MultiModelSequentialStrategy(BaseTrainingStrategy):
    """
    Multi-Model Sequential Training Strategy.

    Trains volatility model first, then uses its predictions to train trading agent.
    This is the recommended approach for separate volatility and trading models.
    """

    def __init__(self, config: TrainingConfig, device: str = "cpu"):
        super().__init__(config, device)
        self.volatility_model = None
        self.trading_model = None
        self.volatility_loss_fn = None
        self.trading_loss_fn = None

    def train(
        self,
        models: dict,  # {'volatility': vol_model, 'trading': trading_model}
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        loss_fns: dict,  # {'volatility': vol_loss, 'trading': trading_loss}
        optimizers: dict,  # {'volatility': vol_opt, 'trading': trading_opt}
        schedulers: Optional[dict] = None
    ) -> dict:
        """
        Execute multi-model sequential training.

        Args:
            models: Dictionary containing volatility and trading models
            train_loader: Training data loader
            val_loader: Validation data loader
            loss_fns: Dictionary containing loss functions for each model
            optimizers: Dictionary containing optimizers for each model
            schedulers: Optional dictionary containing schedulers

        Returns:
            Dictionary containing training results for both models
        """
        self.logger.info("Starting multi-model sequential training")

        # Extract models and components
        self.volatility_model = models['volatility']
        self.trading_model = models['trading']
        self.volatility_loss_fn = loss_fns['volatility']
        self.trading_loss_fn = loss_fns['trading']

        vol_optimizer = optimizers['volatility']
        trading_optimizer = optimizers['trading']

        vol_scheduler = schedulers.get('volatility') if schedulers else None
        trading_scheduler = schedulers.get('trading') if schedulers else None

        # Stage 1: Train volatility model
        self.logger.info("Stage 1: Training volatility model")
        vol_result = self._train_volatility_model(
            train_loader, val_loader, vol_optimizer, vol_scheduler
        )

        # Stage 2: Train trading agent using volatility predictions
        self.logger.info("Stage 2: Training trading agent with volatility predictions")
        trading_result = self._train_trading_model(
            train_loader, val_loader, trading_optimizer, trading_scheduler
        )

        return {
            'volatility': vol_result,
            'trading': trading_result,
            'combined_loss': vol_result.best_val_loss + trading_result.best_val_loss
        }

    def _train_volatility_model(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ) -> TrainingResult:
        """Train the volatility model."""
        train_losses = []
        val_losses = []
        train_metrics = {"mae": [], "rmse": [], "mape": []}
        val_metrics = {"mae": [], "rmse": [], "mape": []}

        best_val_loss = float('inf')
        best_epoch = 0
        best_model_state = None

        for epoch in range(self.config.stage1_epochs):
            # Training phase
            self.volatility_model.train()
            epoch_train_loss = 0.0
            epoch_train_metrics = {"mae": 0.0, "rmse": 0.0, "mape": 0.0}

            train_pbar = tqdm(train_loader, desc=f"Vol Epoch {epoch+1}/{self.config.stage1_epochs} [Train]")
            for batch_idx, batch in enumerate(train_pbar):
                # Extract volatility-relevant data from batch
                vol_data = self._extract_volatility_data(batch)

                optimizer.zero_grad()

                # Forward pass through volatility model
                vol_outputs = self.volatility_model(vol_data['features'])

                # Calculate volatility loss
                vol_loss = self.volatility_loss_fn(vol_outputs, vol_data['targets'])
                vol_loss.backward()

                # Gradient clipping
                if hasattr(self.config, 'grad_clip_norm') and self.config.grad_clip_norm:
                    torch.nn.utils.clip_grad_norm_(self.volatility_model.parameters(), self.config.grad_clip_norm)

                optimizer.step()

                batch_loss = vol_loss.item()
                epoch_train_loss += batch_loss

                # Calculate metrics
                if 'volatility' in vol_outputs and 'realized_volatility' in vol_data['targets']:
                    batch_metrics = self._calculate_metrics(
                        vol_outputs['volatility'],
                        vol_data['targets']['realized_volatility'],
                        batch_loss
                    )
                    for key in epoch_train_metrics:
                        epoch_train_metrics[key] += batch_metrics[key]

                train_pbar.set_postfix({"vol_loss": f"{batch_loss:.4f}"})

            # Average training metrics
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            for key in train_metrics:
                avg_metric = epoch_train_metrics[key] / len(train_loader)
                train_metrics[key].append(avg_metric)

            # Validation phase
            self.volatility_model.eval()
            epoch_val_loss = 0.0
            epoch_val_metrics = {"mae": 0.0, "rmse": 0.0, "mape": 0.0}

            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Vol Epoch {epoch+1} [Val]")
                for batch in val_pbar:
                    vol_data = self._extract_volatility_data(batch)

                    vol_outputs = self.volatility_model(vol_data['features'])
                    vol_loss = self.volatility_loss_fn(vol_outputs, vol_data['targets'])

                    batch_loss = vol_loss.item()
                    epoch_val_loss += batch_loss

                    # Calculate metrics
                    if 'volatility' in vol_outputs and 'realized_volatility' in vol_data['targets']:
                        batch_metrics = self._calculate_metrics(
                            vol_outputs['volatility'],
                            vol_data['targets']['realized_volatility'],
                            batch_loss
                        )
                        for key in epoch_val_metrics:
                            epoch_val_metrics[key] += batch_metrics[key]

                    val_pbar.set_postfix({"vol_loss": f"{batch_loss:.4f}"})

            # Average validation metrics
            avg_val_loss = epoch_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            for key in val_metrics:
                avg_metric = epoch_val_metrics[key] / len(val_loader)
                val_metrics[key].append(avg_metric)

            # Learning rate scheduling
            if scheduler:
                scheduler.step(avg_val_loss)

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                best_model_state = self.volatility_model.state_dict().copy()

            # Early stopping
            if self.config.early_stopping and hasattr(self.config, 'patience'):
                if epoch - best_epoch >= self.config.patience:
                    self.logger.info(f"Early stopping volatility training at epoch {epoch}")
                    break

            self.logger.info(
                f"Vol Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}"
            )

        # Load best model
        if best_model_state:
            self.volatility_model.load_state_dict(best_model_state)

        return TrainingResult(
            train_losses=train_losses,
            val_losses=val_losses,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            final_model_state=best_model_state
        )

    def _train_trading_model(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ) -> TrainingResult:
        """Train the trading model using volatility predictions."""
        train_losses = []
        val_losses = []
        train_metrics = {"mae": [], "rmse": [], "mape": []}
        val_metrics = {"mae": [], "rmse": [], "mape": []}

        best_val_loss = float('inf')
        best_epoch = 0
        best_model_state = None

        # Freeze volatility model
        self.volatility_model.eval()
        for param in self.volatility_model.parameters():
            param.requires_grad = False

        for epoch in range(self.config.stage2_epochs):
            # Training phase
            self.trading_model.train()
            epoch_train_loss = 0.0
            epoch_train_metrics = {"mae": 0.0, "rmse": 0.0, "mape": 0.0}

            train_pbar = tqdm(train_loader, desc=f"Trading Epoch {epoch+1}/{self.config.stage2_epochs} [Train]")
            for batch_idx, batch in enumerate(train_pbar):
                # Extract data for both models
                vol_data = self._extract_volatility_data(batch)
                trading_data = self._extract_trading_data(batch)

                optimizer.zero_grad()

                # Get volatility predictions (no gradients)
                with torch.no_grad():
                    vol_outputs = self.volatility_model(vol_data['features'])

                # Create trading features using volatility predictions
                trading_features = self._create_trading_features(vol_outputs, trading_data)

                # Forward pass through trading model
                trading_outputs = self.trading_model(trading_features)

                # Calculate trading loss
                trading_loss = self.trading_loss_fn(trading_outputs, trading_data['targets'])
                trading_loss.backward()

                # Gradient clipping
                if hasattr(self.config, 'grad_clip_norm') and self.config.grad_clip_norm:
                    torch.nn.utils.clip_grad_norm_(self.trading_model.parameters(), self.config.grad_clip_norm)

                optimizer.step()

                batch_loss = trading_loss.item()
                epoch_train_loss += batch_loss

                # Calculate metrics (if applicable)
                if 'ratios' in trading_outputs and 'target_ratios' in trading_data['targets']:
                    batch_metrics = self._calculate_metrics(
                        trading_outputs['ratios'],
                        trading_data['targets']['target_ratios'],
                        batch_loss
                    )
                    for key in epoch_train_metrics:
                        epoch_train_metrics[key] += batch_metrics[key]

                train_pbar.set_postfix({"trading_loss": f"{batch_loss:.4f}"})

            # Average training metrics
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            for key in train_metrics:
                avg_metric = epoch_train_metrics[key] / len(train_loader)
                train_metrics[key].append(avg_metric)

            # Validation phase
            self.trading_model.eval()
            epoch_val_loss = 0.0
            epoch_val_metrics = {"mae": 0.0, "rmse": 0.0, "mape": 0.0}

            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Trading Epoch {epoch+1} [Val]")
                for batch in val_pbar:
                    vol_data = self._extract_volatility_data(batch)
                    trading_data = self._extract_trading_data(batch)

                    # Get volatility predictions
                    vol_outputs = self.volatility_model(vol_data['features'])

                    # Create trading features
                    trading_features = self._create_trading_features(vol_outputs, trading_data)

                    # Forward pass through trading model
                    trading_outputs = self.trading_model(trading_features)

                    # Calculate trading loss
                    trading_loss = self.trading_loss_fn(trading_outputs, trading_data['targets'])

                    batch_loss = trading_loss.item()
                    epoch_val_loss += batch_loss

                    # Calculate metrics
                    if 'ratios' in trading_outputs and 'target_ratios' in trading_data['targets']:
                        batch_metrics = self._calculate_metrics(
                            trading_outputs['ratios'],
                            trading_data['targets']['target_ratios'],
                            batch_loss
                        )
                        for key in epoch_val_metrics:
                            epoch_val_metrics[key] += batch_metrics[key]

                    val_pbar.set_postfix({"trading_loss": f"{batch_loss:.4f}"})

            # Average validation metrics
            avg_val_loss = epoch_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            for key in val_metrics:
                avg_metric = epoch_val_metrics[key] / len(val_loader)
                val_metrics[key].append(avg_metric)

            # Learning rate scheduling
            if scheduler:
                scheduler.step(avg_val_loss)

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                best_model_state = self.trading_model.state_dict().copy()

            # Early stopping
            if self.config.early_stopping and hasattr(self.config, 'patience'):
                if epoch - best_epoch >= self.config.patience:
                    self.logger.info(f"Early stopping trading training at epoch {epoch}")
                    break

            self.logger.info(
                f"Trading Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}"
            )

        # Load best model
        if best_model_state:
            self.trading_model.load_state_dict(best_model_state)

        return TrainingResult(
            train_losses=train_losses,
            val_losses=val_losses,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            final_model_state=best_model_state
        )

    def _extract_volatility_data(self, batch) -> dict:
        """Extract volatility-relevant data from batch."""
        # This is a placeholder - actual implementation depends on data loader format
        if isinstance(batch, dict):
            return {
                'features': batch.get('returns', batch.get('features')),
                'targets': {
                    'realized_volatility': batch.get('realized_volatility', batch.get('targets'))
                }
            }
        else:
            # Assume tuple format (features, targets)
            features, targets = batch
            return {
                'features': features,
                'targets': {'realized_volatility': targets}
            }

    def _extract_trading_data(self, batch) -> dict:
        """Extract trading-relevant data from batch."""
        # This is a placeholder - actual implementation depends on data loader format
        if isinstance(batch, dict):
            return {
                'market_data': {
                    'returns': batch.get('returns'),
                    'volume': batch.get('volume'),
                    'price': batch.get('price')
                },
                'targets': {
                    'target_ratios': batch.get('target_ratios', batch.get('targets'))
                }
            }
        else:
            # Assume tuple format
            features, targets = batch
            return {
                'market_data': {'returns': features},
                'targets': {'target_ratios': targets}
            }

    def _create_trading_features(self, vol_outputs: dict, trading_data: dict) -> torch.Tensor:
        """Create feature vector for trading model from volatility predictions and market data."""
        features = []

        # Add volatility predictions
        if 'volatility' in vol_outputs:
            vol_pred = vol_outputs['volatility']
            if vol_pred.dim() > 2:
                vol_pred = vol_pred.squeeze(-1)  # Remove horizon dimension
            features.append(vol_pred)

        # Add volatility confidence/uncertainty
        if 'confidence_lower' in vol_outputs and 'confidence_upper' in vol_outputs:
            vol_uncertainty = vol_outputs['confidence_upper'] - vol_outputs['confidence_lower']
            if vol_uncertainty.dim() > 2:
                vol_uncertainty = vol_uncertainty.squeeze(-1)
            features.append(vol_uncertainty)

        # Add market data features
        market_data = trading_data.get('market_data', {})
        for key in ['returns', 'volume', 'price']:
            if key in market_data and market_data[key] is not None:
                features.append(market_data[key])

        # Concatenate all features
        if features:
            return torch.cat(features, dim=-1)
        else:
            # Fallback: use volatility prediction only
            vol_pred = vol_outputs['volatility']
            if vol_pred.dim() > 2:
                vol_pred = vol_pred.squeeze(-1)
            return vol_pred