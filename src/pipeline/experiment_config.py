"""
Experiment Configuration System for Flexible Training Pipeline.

This module provides a comprehensive configuration system that allows easy experimentation
with different combinations of models, loss functions, training approaches, and data loaders.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
from enum import Enum
import torch


class ModelType(Enum):
    """Available model architectures."""
    FLEXIBLE_GSPHAR = "flexible_gsphar"
    GSPHAR = "gsphar"
    GARCH = "garch"
    EGARCH = "egarch"


class LossType(Enum):
    """Available loss functions."""
    MSE = "mse"
    WEIGHTED_MSE = "weighted_mse"
    ASYMMETRIC_MSE = "asymmetric_mse"
    THRESHOLD_MSE = "threshold_mse"
    HYBRID_LOSS = "hybrid_loss"
    TRADING_STRATEGY_LOSS = "trading_strategy_loss"
    OHLCV_LONG_STRATEGY_LOSS = "ohlcv_long_strategy_loss"
    OHLCV_SHARPE_RATIO_LOSS = "ohlcv_sharpe_ratio_loss"


class TrainingApproach(Enum):
    """Available training approaches."""
    SINGLE_STAGE = "single_stage"
    TWO_STAGE = "two_stage"
    PROFIT_MAXIMIZATION = "profit_maximization"
    OHLCV_BASED = "ohlcv_based"
    GARCH_PIPELINE = "garch_pipeline"


class DataLoaderType(Enum):
    """Available data loader types."""
    FLEXIBLE_TIME_SERIES = "flexible_time_series"
    OHLCV_TRADING = "ohlcv_trading"
    CUSTOM = "custom"


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    model_type: ModelType
    params: Dict[str, Any] = field(default_factory=dict)
    
    # FlexibleGSPHAR specific parameters
    n_assets: Optional[int] = None
    A: Optional[int] = None
    dropout_rate: float = 0.1
    
    # GARCH specific parameters
    param1: float = 2.0  # Volatility discount multiplier
    n_hold: int = 4      # Holding period


@dataclass
class LossConfig:
    """Configuration for loss functions."""
    loss_type: LossType
    params: Dict[str, Any] = field(default_factory=dict)
    
    # Common loss parameters
    weight: float = 1.0
    
    # Trading loss specific parameters
    holding_period: Optional[int] = None
    trading_fee: float = 0.0002
    
    # Asymmetric loss parameters
    alpha: float = 0.5
    
    # Threshold loss parameters
    threshold: float = 0.02
    
    # Hybrid loss parameters
    mse_weight: float = 0.5
    trading_weight: float = 0.5


@dataclass
class TrainingConfig:
    """Configuration for training approach."""
    approach: TrainingApproach
    
    # Stage 1 parameters (supervised learning)
    stage1_epochs: int = 20
    stage1_lr: float = 0.001
    stage1_loss: Optional[LossConfig] = None
    
    # Stage 2 parameters (trading optimization)
    stage2_epochs: int = 10
    stage2_lr: float = 0.0005
    stage2_loss: Optional[LossConfig] = None
    
    # General training parameters
    batch_size: int = 8
    validation_split: float = 0.2
    early_stopping: bool = True
    patience: int = 5
    
    # Regularization
    weight_decay: float = 1e-6
    dropout_rate: float = 0.1
    
    # Scheduler parameters
    scheduler_type: str = "reduce_on_plateau"
    scheduler_patience: int = 2
    scheduler_factor: float = 0.7


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    loader_type: DataLoaderType
    
    # Data file paths
    volatility_file: str = "data/crypto_rv1h_38_20200822_20250116.csv"
    price_file: Optional[str] = None
    
    # Time series parameters
    lags: List[int] = field(default_factory=lambda: [1, 4, 24])
    holding_period: int = 4
    
    # Data preprocessing
    normalization: bool = True
    subset_size: Optional[int] = None
    
    # Validation parameters
    train_ratio: float = 0.8
    debug: bool = False


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    name: str
    description: str = ""
    
    # Component configurations
    model: ModelConfig
    loss: LossConfig
    training: TrainingConfig
    data: DataConfig
    
    # Experiment parameters
    device: str = "auto"
    seed: int = 42
    output_dir: str = "experiments"
    save_model: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Auto-detect device if needed
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Set default loss configurations based on training approach
        if self.training.stage1_loss is None:
            self.training.stage1_loss = LossConfig(loss_type=LossType.MSE)
        
        if self.training.stage2_loss is None and self.training.approach == TrainingApproach.TWO_STAGE:
            self.training.stage2_loss = LossConfig(
                loss_type=LossType.OHLCV_LONG_STRATEGY_LOSS,
                holding_period=self.data.holding_period,
                trading_fee=0.0002
            )
        
        # Validate compatibility
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration compatibility."""
        # Check model-data compatibility
        if self.model.model_type in [ModelType.GARCH, ModelType.EGARCH]:
            if self.data.loader_type != DataLoaderType.OHLCV_TRADING:
                raise ValueError("GARCH models require OHLCV trading data loader")
        
        # Check training-loss compatibility
        if self.training.approach == TrainingApproach.TWO_STAGE:
            if self.training.stage2_loss is None:
                raise ValueError("Two-stage training requires stage2_loss configuration")
        
        # Check loss-data compatibility
        trading_losses = [
            LossType.TRADING_STRATEGY_LOSS,
            LossType.OHLCV_LONG_STRATEGY_LOSS,
            LossType.OHLCV_SHARPE_RATIO_LOSS
        ]
        
        if (self.loss.loss_type in trading_losses or 
            (self.training.stage2_loss and self.training.stage2_loss.loss_type in trading_losses)):
            if self.data.loader_type != DataLoaderType.OHLCV_TRADING:
                raise ValueError("Trading losses require OHLCV trading data loader")


# Predefined experiment configurations
def get_flexible_gsphar_config() -> ExperimentConfig:
    """Standard FlexibleGSPHAR configuration."""
    return ExperimentConfig(
        name="flexible_gsphar_standard",
        description="Standard FlexibleGSPHAR with two-stage training",
        model=ModelConfig(
            model_type=ModelType.FLEXIBLE_GSPHAR,
            n_assets=38,
            A=10,
            dropout_rate=0.1
        ),
        loss=LossConfig(loss_type=LossType.MSE),
        training=TrainingConfig(
            approach=TrainingApproach.TWO_STAGE,
            stage1_epochs=20,
            stage2_epochs=10
        ),
        data=DataConfig(
            loader_type=DataLoaderType.OHLCV_TRADING,
            lags=[1, 4, 24],
            holding_period=4
        )
    )


def get_garch_config() -> ExperimentConfig:
    """GARCH-based trading configuration."""
    return ExperimentConfig(
        name="garch_trading",
        description="GARCH-based trading strategy",
        model=ModelConfig(
            model_type=ModelType.GARCH,
            param1=2.0,
            n_hold=4
        ),
        loss=LossConfig(loss_type=LossType.OHLCV_LONG_STRATEGY_LOSS),
        training=TrainingConfig(
            approach=TrainingApproach.GARCH_PIPELINE,
            stage1_epochs=1,  # GARCH fitting
            stage2_epochs=1   # Signal generation
        ),
        data=DataConfig(
            loader_type=DataLoaderType.OHLCV_TRADING
        )
    )


def get_profit_maximization_config() -> ExperimentConfig:
    """Profit maximization configuration."""
    return ExperimentConfig(
        name="profit_maximization",
        description="Direct profit maximization training",
        model=ModelConfig(
            model_type=ModelType.FLEXIBLE_GSPHAR,
            n_assets=38,
            A=10
        ),
        loss=LossConfig(loss_type=LossType.OHLCV_SHARPE_RATIO_LOSS),
        training=TrainingConfig(
            approach=TrainingApproach.PROFIT_MAXIMIZATION,
            stage1_epochs=30
        ),
        data=DataConfig(
            loader_type=DataLoaderType.OHLCV_TRADING
        )
    )


def get_hybrid_loss_config() -> ExperimentConfig:
    """Hybrid loss configuration."""
    return ExperimentConfig(
        name="hybrid_loss",
        description="Hybrid MSE + Trading loss",
        model=ModelConfig(
            model_type=ModelType.FLEXIBLE_GSPHAR,
            n_assets=38,
            A=10
        ),
        loss=LossConfig(
            loss_type=LossType.HYBRID_LOSS,
            mse_weight=0.3,
            trading_weight=0.7
        ),
        training=TrainingConfig(
            approach=TrainingApproach.SINGLE_STAGE,
            stage1_epochs=30
        ),
        data=DataConfig(
            loader_type=DataLoaderType.OHLCV_TRADING
        )
    )


# Configuration registry for easy access
EXPERIMENT_CONFIGS = {
    "flexible_gsphar": get_flexible_gsphar_config,
    "garch": get_garch_config,
    "profit_max": get_profit_maximization_config,
    "hybrid": get_hybrid_loss_config,
}


def load_config(config_name: str) -> ExperimentConfig:
    """Load a predefined configuration by name."""
    if config_name not in EXPERIMENT_CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(EXPERIMENT_CONFIGS.keys())}")
    
    return EXPERIMENT_CONFIGS[config_name]()


def create_custom_config(**kwargs) -> ExperimentConfig:
    """Create a custom configuration with specified parameters."""
    # Start with a base configuration
    base_config = get_flexible_gsphar_config()
    
    # Update with custom parameters
    for key, value in kwargs.items():
        if hasattr(base_config, key):
            setattr(base_config, key, value)
        else:
            # Handle nested updates
            if '.' in key:
                obj, attr = key.split('.', 1)
                if hasattr(base_config, obj):
                    nested_obj = getattr(base_config, obj)
                    if hasattr(nested_obj, attr):
                        setattr(nested_obj, attr, value)
    
    return base_config
