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
    # Volatility-specific models
    EGARCH_VOLATILITY = "egarch_volatility"
    GARCH_VOLATILITY = "garch_volatility"
    LSTM_VOLATILITY = "lstm_volatility"
    # Trading-specific models
    LINEAR_TRADING_AGENT = "linear_trading_agent"
    NEURAL_TRADING_AGENT = "neural_trading_agent"
    TRANSFORMER_TRADING_AGENT = "transformer_trading_agent"


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
    # Volatility-specific loss functions
    QLIKE_LOSS = "qlike_loss"
    LOG_LIKELIHOOD_LOSS = "log_likelihood_loss"
    VOLATILITY_QUANTILE_LOSS = "volatility_quantile_loss"
    GARCH_LIKELIHOOD_LOSS = "garch_likelihood_loss"
    # Advanced trading loss functions
    PROFIT_MAXIMIZATION_LOSS = "profit_maximization_loss"
    DRAWDOWN_CONSTRAINED_LOSS = "drawdown_constrained_loss"
    TRANSACTION_COST_AWARE_LOSS = "transaction_cost_aware_loss"


class TrainingApproach(Enum):
    """Available training approaches."""
    SINGLE_STAGE = "single_stage"
    TWO_STAGE = "two_stage"
    PROFIT_MAXIMIZATION = "profit_maximization"
    OHLCV_BASED = "ohlcv_based"
    GARCH_PIPELINE = "garch_pipeline"
    # Multi-model approaches
    MULTI_MODEL_SEQUENTIAL = "multi_model_sequential"
    MULTI_MODEL_JOINT = "multi_model_joint"
    VOLATILITY_THEN_TRADING = "volatility_then_trading"


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

    # Volatility model specific parameters
    volatility_window: int = 24  # Lookback window for volatility
    volatility_horizon: int = 1  # Forecast horizon

    # Trading agent specific parameters
    input_features: int = 10     # Number of input features
    hidden_dim: int = 64         # Hidden layer dimension
    output_dim: int = 1          # Output dimension (e.g., trading ratio)


@dataclass
class MultiModelConfig:
    """Configuration for multi-model architecture with separate volatility and trading models."""
    volatility_model: ModelConfig
    trading_model: ModelConfig

    # Integration parameters
    volatility_features: List[str] = field(default_factory=lambda: ['vol_pred', 'vol_confidence'])
    additional_features: List[str] = field(default_factory=lambda: ['returns', 'volume'])

    # Model interaction
    freeze_volatility_after_training: bool = True
    use_volatility_uncertainty: bool = True


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

    # Component configurations (single model)
    model: Optional[ModelConfig] = None
    loss: Optional[LossConfig] = None
    training: TrainingConfig = None
    data: DataConfig = None

    # Multi-model configurations
    multi_model: Optional[MultiModelConfig] = None
    volatility_loss: Optional[LossConfig] = None
    trading_loss: Optional[LossConfig] = None

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

        # Determine if this is a multi-model configuration
        is_multi_model = (
            self.multi_model is not None or
            self.training.approach in [
                TrainingApproach.MULTI_MODEL_SEQUENTIAL,
                TrainingApproach.MULTI_MODEL_JOINT,
                TrainingApproach.VOLATILITY_THEN_TRADING
            ]
        )

        if is_multi_model:
            self._setup_multi_model_defaults()
        else:
            self._setup_single_model_defaults()

        # Validate compatibility
        self._validate_config()

    def _setup_single_model_defaults(self):
        """Setup defaults for single model configuration."""
        if self.training.stage1_loss is None:
            self.training.stage1_loss = LossConfig(loss_type=LossType.MSE)

        if self.training.stage2_loss is None and self.training.approach == TrainingApproach.TWO_STAGE:
            self.training.stage2_loss = LossConfig(
                loss_type=LossType.OHLCV_LONG_STRATEGY_LOSS,
                holding_period=self.data.holding_period,
                trading_fee=0.0002
            )

    def _setup_multi_model_defaults(self):
        """Setup defaults for multi-model configuration."""
        # Set default volatility loss
        if self.volatility_loss is None:
            self.volatility_loss = LossConfig(loss_type=LossType.QLIKE_LOSS)

        # Set default trading loss
        if self.trading_loss is None:
            self.trading_loss = LossConfig(
                loss_type=LossType.OHLCV_SHARPE_RATIO_LOSS,
                holding_period=self.data.holding_period if self.data else 4,
                trading_fee=0.0002
            )

        # Create default multi-model config if not provided
        if self.multi_model is None:
            self.multi_model = MultiModelConfig(
                volatility_model=ModelConfig(
                    model_type=ModelType.EGARCH_VOLATILITY,
                    n_assets=self.data.subset_size if self.data else 20
                ),
                trading_model=ModelConfig(
                    model_type=ModelType.LINEAR_TRADING_AGENT,
                    input_features=10,
                    hidden_dim=64,
                    output_dim=1
                )
            )

    def _validate_config(self):
        """Validate configuration compatibility."""
        # Check model-data compatibility (only for single model configs)
        if self.model is not None and self.model.model_type in [ModelType.GARCH, ModelType.EGARCH]:
            if self.data and self.data.loader_type != DataLoaderType.OHLCV_TRADING:
                raise ValueError("GARCH models require OHLCV trading data loader")

        # Check training-loss compatibility (only for single model configs)
        if self.training and self.training.approach == TrainingApproach.TWO_STAGE:
            if self.training.stage2_loss is None:
                raise ValueError("Two-stage training requires stage2_loss configuration")

        # Check loss-data compatibility (only for single model configs)
        if self.loss is not None and self.data is not None:
            trading_losses = [
                LossType.TRADING_STRATEGY_LOSS,
                LossType.OHLCV_LONG_STRATEGY_LOSS,
                LossType.OHLCV_SHARPE_RATIO_LOSS
            ]

            if (self.loss.loss_type in trading_losses or
                (self.training and self.training.stage2_loss and self.training.stage2_loss.loss_type in trading_losses)):
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


def get_multi_model_config() -> ExperimentConfig:
    """Multi-model configuration with separate volatility and trading models."""
    return ExperimentConfig(
        name="multi_model_egarch_linear",
        description="EGARCH volatility model + Linear trading agent",
        multi_model=MultiModelConfig(
            volatility_model=ModelConfig(
                model_type=ModelType.EGARCH_VOLATILITY,
                n_assets=20,
                params={"p": 1, "q": 1}
            ),
            trading_model=ModelConfig(
                model_type=ModelType.LINEAR_TRADING_AGENT,
                input_features=5,  # vol_pred, vol_uncertainty, returns, volume, price
                output_dim=1,
                params={"activation": "sigmoid"}
            )
        ),
        volatility_loss=LossConfig(loss_type=LossType.QLIKE_LOSS),
        trading_loss=LossConfig(
            loss_type=LossType.OHLCV_SHARPE_RATIO_LOSS,
            trading_fee=0.0002
        ),
        training=TrainingConfig(
            approach=TrainingApproach.MULTI_MODEL_SEQUENTIAL,
            stage1_epochs=15,  # Volatility model training
            stage2_epochs=10   # Trading agent training
        ),
        data=DataConfig(
            loader_type=DataLoaderType.OHLCV_TRADING,
            subset_size=20
        )
    )


def get_advanced_multi_model_config() -> ExperimentConfig:
    """Advanced multi-model configuration with neural trading agent."""
    return ExperimentConfig(
        name="multi_model_lstm_neural",
        description="LSTM volatility model + Neural trading agent",
        multi_model=MultiModelConfig(
            volatility_model=ModelConfig(
                model_type=ModelType.LSTM_VOLATILITY,
                input_features=5,
                hidden_dim=64,
                output_dim=1,
                params={"num_layers": 2, "bidirectional": False}
            ),
            trading_model=ModelConfig(
                model_type=ModelType.NEURAL_TRADING_AGENT,
                input_features=8,
                hidden_dim=64,
                output_dim=1,
                params={
                    "hidden_dims": [64, 32, 16],
                    "activation": "relu",
                    "output_activation": "sigmoid",
                    "batch_norm": True
                }
            )
        ),
        volatility_loss=LossConfig(
            loss_type=LossType.LOG_LIKELIHOOD_LOSS,
            params={"epsilon": 1e-8}
        ),
        trading_loss=LossConfig(
            loss_type=LossType.PROFIT_MAXIMIZATION_LOSS,
            trading_fee=0.0002,
            params={"profit_weight": 1.0, "risk_weight": 0.2}
        ),
        training=TrainingConfig(
            approach=TrainingApproach.VOLATILITY_THEN_TRADING,
            stage1_epochs=20,
            stage2_epochs=15,
            stage1_lr=0.001,
            stage2_lr=0.0005
        ),
        data=DataConfig(
            loader_type=DataLoaderType.OHLCV_TRADING,
            subset_size=20
        )
    )


# Configuration registry for easy access
EXPERIMENT_CONFIGS = {
    "flexible_gsphar": get_flexible_gsphar_config,
    "garch": get_garch_config,
    "profit_max": get_profit_maximization_config,
    "hybrid": get_hybrid_loss_config,
    "multi_model": get_multi_model_config,
    "advanced_multi_model": get_advanced_multi_model_config,
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
