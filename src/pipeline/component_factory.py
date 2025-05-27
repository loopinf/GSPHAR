"""
Factory classes for creating models, loss functions, and training strategies.

This module implements the factory pattern to create different components
based on configuration specifications.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Tuple, Optional
import logging

from .experiment_config import (
    ModelConfig, LossConfig, TrainingConfig, DataConfig,
    ModelType, LossType, TrainingApproach, DataLoaderType
)

# Import existing components
try:
    from ..models.flexible_gsphar import FlexibleGSPHAR
    from ..training.custom_losses import (
        WeightedMSE, AsymmetricMSE, ThresholdMSE, HybridLoss
    )
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from models.flexible_gsphar import FlexibleGSPHAR
    from training.custom_losses import (
        WeightedMSE, AsymmetricMSE, ThresholdMSE, HybridLoss
    )
try:
    from ..trading_loss import TradingStrategyLoss
    from ..ohlcv_trading_loss import OHLCVLongStrategyLoss, OHLCVSharpeRatioLoss
    from ..data.ohlcv_trading_dataset import load_ohlcv_trading_data, create_ohlcv_dataloaders
except ImportError:
    from trading_loss import TradingStrategyLoss
    from ohlcv_trading_loss import OHLCVLongStrategyLoss, OHLCVSharpeRatioLoss
    from data.ohlcv_trading_dataset import load_ohlcv_trading_data, create_ohlcv_dataloaders

logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory for creating model instances."""

    @staticmethod
    def create_model(config: ModelConfig, device: str = "cpu") -> nn.Module:
        """Create a model based on configuration."""
        logger.info(f"Creating model: {config.model_type.value}")

        if config.model_type == ModelType.FLEXIBLE_GSPHAR:
            return ModelFactory._create_flexible_gsphar(config, device)
        elif config.model_type == ModelType.GARCH:
            return ModelFactory._create_garch_model(config)
        elif config.model_type == ModelType.EGARCH:
            return ModelFactory._create_egarch_model(config)
        # Volatility-specific models
        elif config.model_type == ModelType.EGARCH_VOLATILITY:
            return ModelFactory._create_egarch_volatility_model(config, device)
        elif config.model_type == ModelType.GARCH_VOLATILITY:
            return ModelFactory._create_garch_volatility_model(config, device)
        elif config.model_type == ModelType.LSTM_VOLATILITY:
            return ModelFactory._create_lstm_volatility_model(config, device)
        # Trading-specific models
        elif config.model_type == ModelType.LINEAR_TRADING_AGENT:
            return ModelFactory._create_linear_trading_agent(config, device)
        elif config.model_type == ModelType.NEURAL_TRADING_AGENT:
            return ModelFactory._create_neural_trading_agent(config, device)
        elif config.model_type == ModelType.TRANSFORMER_TRADING_AGENT:
            return ModelFactory._create_transformer_trading_agent(config, device)
        else:
            raise ValueError(f"Unsupported model type: {config.model_type}")

    @staticmethod
    def _create_flexible_gsphar(config: ModelConfig, device: str) -> FlexibleGSPHAR:
        """Create FlexibleGSPHAR model."""
        # Set default parameters
        params = {
            "n_assets": config.n_assets or 38,
            "A": config.A or 10,
            "dropout_rate": config.dropout_rate,
            **config.params
        }

        model = FlexibleGSPHAR(**params)
        return model.to(device)

    @staticmethod
    def _create_garch_model(config: ModelConfig):
        """Create GARCH model wrapper."""
        from ..scripts.garch_pipeline_strategy import GARCHModel

        params = {
            "model_type": "GARCH",
            "param1": config.param1,
            "n_hold": config.n_hold,
            **config.params
        }

        return GARCHModel(**params)

    @staticmethod
    def _create_egarch_volatility_model(config: ModelConfig, device: str):
        """Create EGARCH volatility model."""
        try:
            from ..models.volatility_models import EGARCHVolatilityModel
        except ImportError:
            from models.volatility_models import EGARCHVolatilityModel

        params = {
            "n_assets": config.n_assets or 20,
            "p": config.params.get("p", 1),
            "q": config.params.get("q", 1),
            "mean_model": config.params.get("mean_model", "Zero"),
            "distribution": config.params.get("distribution", "normal")
        }

        model = EGARCHVolatilityModel(**params)
        return model.to(device)

    @staticmethod
    def _create_garch_volatility_model(config: ModelConfig, device: str):
        """Create GARCH volatility model."""
        from ..models.volatility_models import GARCHVolatilityModel

        params = {
            "n_assets": config.n_assets or 20,
            "p": config.params.get("p", 1),
            "q": config.params.get("q", 1),
            "mean_model": config.params.get("mean_model", "Zero"),
            "distribution": config.params.get("distribution", "normal")
        }

        model = GARCHVolatilityModel(**params)
        return model.to(device)

    @staticmethod
    def _create_lstm_volatility_model(config: ModelConfig, device: str):
        """Create LSTM volatility model."""
        from ..models.volatility_models import LSTMVolatilityModel

        params = {
            "input_dim": config.input_features or 5,
            "hidden_dim": config.hidden_dim or 64,
            "num_layers": config.params.get("num_layers", 2),
            "output_dim": config.output_dim or 1,
            "dropout": config.dropout_rate,
            "bidirectional": config.params.get("bidirectional", False)
        }

        model = LSTMVolatilityModel(**params)
        return model.to(device)

    @staticmethod
    def _create_linear_trading_agent(config: ModelConfig, device: str):
        """Create linear trading agent."""
        from ..models.trading_agents import LinearTradingAgent

        params = {
            "input_features": config.input_features or 10,
            "output_dim": config.output_dim or 1,
            "use_bias": config.params.get("use_bias", True),
            "activation": config.params.get("activation", "sigmoid")
        }

        model = LinearTradingAgent(**params)
        return model.to(device)

    @staticmethod
    def _create_neural_trading_agent(config: ModelConfig, device: str):
        """Create neural trading agent."""
        from ..models.trading_agents import NeuralTradingAgent

        params = {
            "input_features": config.input_features or 10,
            "hidden_dims": config.params.get("hidden_dims", [64, 32]),
            "output_dim": config.output_dim or 1,
            "dropout": config.dropout_rate,
            "activation": config.params.get("activation", "relu"),
            "output_activation": config.params.get("output_activation", "sigmoid"),
            "batch_norm": config.params.get("batch_norm", True)
        }

        model = NeuralTradingAgent(**params)
        return model.to(device)

    @staticmethod
    def _create_transformer_trading_agent(config: ModelConfig, device: str):
        """Create transformer trading agent."""
        from ..models.trading_agents import TransformerTradingAgent

        params = {
            "input_features": config.input_features or 10,
            "d_model": config.params.get("d_model", 128),
            "nhead": config.params.get("nhead", 8),
            "num_layers": config.params.get("num_layers", 4),
            "dim_feedforward": config.params.get("dim_feedforward", 512),
            "dropout": config.dropout_rate,
            "output_dim": config.output_dim or 1,
            "max_seq_length": config.params.get("max_seq_length", 100)
        }

        model = TransformerTradingAgent(**params)
        return model.to(device)

    @staticmethod
    def _create_egarch_model(config: ModelConfig):
        """Create EGARCH model wrapper."""
        from ..scripts.garch_pipeline_strategy import GARCHModel

        params = {
            "model_type": "EGARCH",
            "param1": config.param1,
            "n_hold": config.n_hold,
            **config.params
        }

        return GARCHModel(**params)


class LossFactory:
    """Factory for creating loss functions."""

    @staticmethod
    def create_loss(config: LossConfig) -> nn.Module:
        """Create a loss function based on configuration."""
        logger.info(f"Creating loss function: {config.loss_type.value}")

        if config.loss_type == LossType.MSE:
            return nn.MSELoss()
        elif config.loss_type == LossType.WEIGHTED_MSE:
            return LossFactory._create_weighted_mse(config)
        elif config.loss_type == LossType.ASYMMETRIC_MSE:
            return LossFactory._create_asymmetric_mse(config)
        elif config.loss_type == LossType.THRESHOLD_MSE:
            return LossFactory._create_threshold_mse(config)
        elif config.loss_type == LossType.HYBRID_LOSS:
            return LossFactory._create_hybrid_loss(config)
        elif config.loss_type == LossType.TRADING_STRATEGY_LOSS:
            return LossFactory._create_trading_strategy_loss(config)
        elif config.loss_type == LossType.OHLCV_LONG_STRATEGY_LOSS:
            return LossFactory._create_ohlcv_long_strategy_loss(config)
        elif config.loss_type == LossType.OHLCV_SHARPE_RATIO_LOSS:
            return LossFactory._create_ohlcv_sharpe_ratio_loss(config)
        # Volatility-specific losses
        elif config.loss_type == LossType.QLIKE_LOSS:
            return LossFactory._create_qlike_loss(config)
        elif config.loss_type == LossType.LOG_LIKELIHOOD_LOSS:
            return LossFactory._create_log_likelihood_loss(config)
        elif config.loss_type == LossType.VOLATILITY_QUANTILE_LOSS:
            return LossFactory._create_volatility_quantile_loss(config)
        elif config.loss_type == LossType.GARCH_LIKELIHOOD_LOSS:
            return LossFactory._create_garch_likelihood_loss(config)
        # Advanced trading losses
        elif config.loss_type == LossType.PROFIT_MAXIMIZATION_LOSS:
            return LossFactory._create_profit_maximization_loss(config)
        elif config.loss_type == LossType.DRAWDOWN_CONSTRAINED_LOSS:
            return LossFactory._create_drawdown_constrained_loss(config)
        elif config.loss_type == LossType.TRANSACTION_COST_AWARE_LOSS:
            return LossFactory._create_transaction_cost_aware_loss(config)
        else:
            raise ValueError(f"Unsupported loss type: {config.loss_type}")

    @staticmethod
    def _create_weighted_mse(config: LossConfig) -> WeightedMSE:
        """Create weighted MSE loss."""
        params = {
            "weight_factor": config.params.get("weight_factor", 2.0),
            **config.params
        }
        return WeightedMSE(**params)

    @staticmethod
    def _create_asymmetric_mse(config: LossConfig) -> AsymmetricMSE:
        """Create asymmetric MSE loss."""
        params = {
            "alpha": config.alpha,
            **config.params
        }
        return AsymmetricMSE(**params)

    @staticmethod
    def _create_threshold_mse(config: LossConfig) -> ThresholdMSE:
        """Create threshold MSE loss."""
        params = {
            "threshold": config.threshold,
            **config.params
        }
        return ThresholdMSE(**params)

    @staticmethod
    def _create_hybrid_loss(config: LossConfig) -> HybridLoss:
        """Create hybrid loss."""
        params = {
            "mse_weight": config.mse_weight,
            "trading_weight": config.trading_weight,
            "holding_period": config.holding_period or 4,
            "trading_fee": config.trading_fee,
            **config.params
        }
        return HybridLoss(**params)

    @staticmethod
    def _create_trading_strategy_loss(config: LossConfig) -> TradingStrategyLoss:
        """Create trading strategy loss."""
        params = {
            "holding_period": config.holding_period or 4,
            "trading_fee": config.trading_fee,
            **config.params
        }
        return TradingStrategyLoss(**params)

    @staticmethod
    def _create_ohlcv_long_strategy_loss(config: LossConfig) -> OHLCVLongStrategyLoss:
        """Create OHLCV long strategy loss."""
        params = {
            "holding_period": config.holding_period or 4,
            "trading_fee": config.trading_fee,
            **config.params
        }
        return OHLCVLongStrategyLoss(**params)

    @staticmethod
    def _create_ohlcv_sharpe_ratio_loss(config: LossConfig) -> OHLCVSharpeRatioLoss:
        """Create OHLCV Sharpe ratio loss."""
        params = {
            "holding_period": config.holding_period or 4,
            "trading_fee": config.trading_fee,
            **config.params
        }
        return OHLCVSharpeRatioLoss(**params)

    @staticmethod
    def _create_qlike_loss(config: LossConfig):
        """Create QLIKE loss."""
        from ..training.volatility_losses import QLIKELoss

        params = {
            "epsilon": config.params.get("epsilon", 1e-8)
        }
        return QLIKELoss(**params)

    @staticmethod
    def _create_log_likelihood_loss(config: LossConfig):
        """Create log-likelihood loss."""
        from ..training.volatility_losses import LogLikelihoodLoss

        params = {
            "epsilon": config.params.get("epsilon", 1e-8)
        }
        return LogLikelihoodLoss(**params)

    @staticmethod
    def _create_volatility_quantile_loss(config: LossConfig):
        """Create volatility quantile loss."""
        from ..training.volatility_losses import VolatilityQuantileLoss

        params = {
            "quantiles": config.params.get("quantiles", [0.1, 0.5, 0.9])
        }
        return VolatilityQuantileLoss(**params)

    @staticmethod
    def _create_garch_likelihood_loss(config: LossConfig):
        """Create GARCH likelihood loss."""
        from ..training.volatility_losses import GARCHLikelihoodLoss

        params = {
            "distribution": config.params.get("distribution", "normal"),
            "epsilon": config.params.get("epsilon", 1e-8)
        }
        return GARCHLikelihoodLoss(**params)

    @staticmethod
    def _create_profit_maximization_loss(config: LossConfig):
        """Create profit maximization loss."""
        from ..training.advanced_loss_functions import ComprehensiveTradingLoss

        params = {
            "profit_weight": config.params.get("profit_weight", 1.0),
            "risk_weight": config.params.get("risk_weight", 0.1),
            "transaction_cost": config.trading_fee
        }
        return ComprehensiveTradingLoss(**params)

    @staticmethod
    def _create_drawdown_constrained_loss(config: LossConfig):
        """Create drawdown constrained loss."""
        from ..training.advanced_loss_functions import DrawdownConstraintLoss

        params = {
            "max_drawdown": config.params.get("max_drawdown", 0.1)
        }
        return DrawdownConstraintLoss(**params)

    @staticmethod
    def _create_transaction_cost_aware_loss(config: LossConfig):
        """Create transaction cost aware loss."""
        from ..training.advanced_loss_functions import TransactionCostAwareLoss

        params = {
            "transaction_cost": config.trading_fee,
            "cost_weight": config.params.get("cost_weight", 1.0)
        }
        return TransactionCostAwareLoss(**params)


class OptimizerFactory:
    """Factory for creating optimizers."""

    @staticmethod
    def create_optimizer(model: nn.Module, config: TrainingConfig) -> optim.Optimizer:
        """Create an optimizer based on configuration."""
        optimizer_type = config.params.get("optimizer", "adam").lower()

        if optimizer_type == "adam":
            return optim.Adam(
                model.parameters(),
                lr=config.stage1_lr,
                weight_decay=config.weight_decay
            )
        elif optimizer_type == "sgd":
            return optim.SGD(
                model.parameters(),
                lr=config.stage1_lr,
                momentum=config.params.get("momentum", 0.9),
                weight_decay=config.weight_decay
            )
        elif optimizer_type == "adamw":
            return optim.AdamW(
                model.parameters(),
                lr=config.stage1_lr,
                weight_decay=config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")

    @staticmethod
    def create_scheduler(optimizer: optim.Optimizer, config: TrainingConfig) -> Optional[object]:
        """Create a learning rate scheduler."""
        if config.scheduler_type == "reduce_on_plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=config.scheduler_factor,
                patience=config.scheduler_patience
            )
        elif config.scheduler_type == "step":
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.params.get("step_size", 10),
                gamma=config.scheduler_factor
            )
        elif config.scheduler_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.stage1_epochs
            )
        elif config.scheduler_type is None or config.scheduler_type == "none":
            return None
        else:
            raise ValueError(f"Unsupported scheduler: {config.scheduler_type}")


class DataLoaderFactory:
    """Factory for creating data loaders."""

    @staticmethod
    def create_dataloaders(config: DataConfig) -> Tuple[Any, Any, Dict[str, Any]]:
        """Create data loaders based on configuration."""
        logger.info(f"Creating data loaders: {config.loader_type.value}")

        if config.loader_type == DataLoaderType.OHLCV_TRADING:
            return DataLoaderFactory._create_ohlcv_trading_loaders(config)
        elif config.loader_type == DataLoaderType.FLEXIBLE_TIME_SERIES:
            return DataLoaderFactory._create_flexible_time_series_loaders(config)
        else:
            raise ValueError(f"Unsupported data loader type: {config.loader_type}")

    @staticmethod
    def _create_ohlcv_trading_loaders(config: DataConfig) -> Tuple[Any, Any, Dict[str, Any]]:
        """Create OHLCV trading data loaders."""
        # Load dataset
        dataset, metadata = load_ohlcv_trading_data(
            volatility_file=config.volatility_file,
            lags=config.lags,
            holding_period=config.holding_period,
            debug=config.debug
        )

        # Create subset if specified
        if config.subset_size is not None:
            from ..scripts.garch_pipeline_strategy import SmallDataSubset
            dataset = SmallDataSubset(dataset, subset_size=config.subset_size)

        # Create data loaders
        train_loader, val_loader, split_info = create_ohlcv_dataloaders(
            dataset,
            train_ratio=config.train_ratio,
            batch_size=8,  # Will be updated by training config
            shuffle=True
        )

        return train_loader, val_loader, {**metadata, **split_info}

    @staticmethod
    def _create_flexible_time_series_loaders(config: DataConfig) -> Tuple[Any, Any, Dict[str, Any]]:
        """Create flexible time series data loaders."""
        # Import and use the flexible dataloader
        from ..flexible_dataloader import create_flexible_dataloaders

        train_loader, val_loader, metadata = create_flexible_dataloaders(
            data_file=config.volatility_file,
            lags=config.lags,
            train_ratio=config.train_ratio,
            batch_size=8,  # Will be updated by training config
            normalization=config.normalization
        )

        return train_loader, val_loader, metadata


class TrainingStrategyFactory:
    """Factory for creating training strategies."""

    @staticmethod
    def create_strategy(config: TrainingConfig):
        """Create a training strategy based on configuration."""
        logger.info(f"Creating training strategy: {config.approach.value}")

        if config.approach == TrainingApproach.SINGLE_STAGE:
            from .training_strategies import SingleStageStrategy
            return SingleStageStrategy(config)
        elif config.approach == TrainingApproach.TWO_STAGE:
            from .training_strategies import TwoStageStrategy
            return TwoStageStrategy(config)
        elif config.approach == TrainingApproach.PROFIT_MAXIMIZATION:
            from .training_strategies import ProfitMaximizationStrategy
            return ProfitMaximizationStrategy(config)
        elif config.approach == TrainingApproach.GARCH_PIPELINE:
            from .training_strategies import GARCHPipelineStrategy
            return GARCHPipelineStrategy(config)
        elif config.approach in [TrainingApproach.MULTI_MODEL_SEQUENTIAL, TrainingApproach.VOLATILITY_THEN_TRADING]:
            from .training_strategies import MultiModelSequentialStrategy
            return MultiModelSequentialStrategy(config)
        else:
            raise ValueError(f"Unsupported training approach: {config.approach}")


# Utility functions for component creation
def create_all_components(config):
    """Create all components from a complete experiment configuration."""
    from .experiment_config import ExperimentConfig

    if not isinstance(config, ExperimentConfig):
        raise ValueError("Expected ExperimentConfig instance")

    # Create model
    model = ModelFactory.create_model(config.model, config.device)

    # Create loss functions
    loss_fn = LossFactory.create_loss(config.loss)
    stage2_loss_fn = None
    if config.training.stage2_loss is not None:
        stage2_loss_fn = LossFactory.create_loss(config.training.stage2_loss)

    # Create data loaders
    train_loader, val_loader, metadata = DataLoaderFactory.create_dataloaders(config.data)

    # Update batch size in loaders if needed
    if hasattr(train_loader, 'batch_size'):
        train_loader.batch_size = config.training.batch_size
        val_loader.batch_size = config.training.batch_size

    # Create training strategy
    training_strategy = TrainingStrategyFactory.create_strategy(config.training)

    components = {
        'model': model,
        'loss_fn': loss_fn,
        'stage2_loss_fn': stage2_loss_fn,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'metadata': metadata,
        'training_strategy': training_strategy
    }

    logger.info("All components created successfully")
    return components
