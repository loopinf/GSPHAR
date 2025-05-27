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
