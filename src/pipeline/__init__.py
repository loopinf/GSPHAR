"""
Flexible Training Pipeline Package.

This package provides a comprehensive framework for flexible experimentation
with different combinations of models, loss functions, training approaches,
and data loaders for the GSPHAR trading strategy.
"""

from .experiment_config import (
    ModelType, LossType, TrainingApproach, DataLoaderType,
    ModelConfig, LossConfig, TrainingConfig, DataConfig, ExperimentConfig,
    get_predefined_config, create_custom_config
)

from .component_factory import (
    ModelFactory, LossFactory, OptimizerFactory, DataLoaderFactory,
    TrainingStrategyFactory, create_all_components
)

from .training_strategies import (
    BaseTrainingStrategy, SingleStageStrategy, TwoStageStrategy,
    ProfitMaximizationStrategy, OHLCVBasedStrategy, GARCHPipelineStrategy,
    TrainingResult
)

from .flexible_training_pipeline import (
    FlexibleTrainingPipeline, ExperimentTracker,
    run_quick_experiment, run_comparison_study
)

__all__ = [
    # Configuration
    'ModelType', 'LossType', 'TrainingApproach', 'DataLoaderType',
    'ModelConfig', 'LossConfig', 'TrainingConfig', 'DataConfig', 'ExperimentConfig',
    'get_predefined_config', 'create_custom_config',
    
    # Factories
    'ModelFactory', 'LossFactory', 'OptimizerFactory', 'DataLoaderFactory',
    'TrainingStrategyFactory', 'create_all_components',
    
    # Training Strategies
    'BaseTrainingStrategy', 'SingleStageStrategy', 'TwoStageStrategy',
    'ProfitMaximizationStrategy', 'OHLCVBasedStrategy', 'GARCHPipelineStrategy',
    'TrainingResult',
    
    # Main Pipeline
    'FlexibleTrainingPipeline', 'ExperimentTracker',
    'run_quick_experiment', 'run_comparison_study'
]

# Version info
__version__ = "1.0.0"
__author__ = "GSPHAR Research Team"
__description__ = "Flexible Training Pipeline for GSPHAR Trading Strategy"
