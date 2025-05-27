# Flexible Training Pipeline for GSPHAR

A comprehensive, modular training pipeline that allows easy experimentation with different combinations of model architectures, loss functions, training approaches, and data loaders for the GSPHAR trading strategy.

## Overview

The Flexible Training Pipeline provides a systematic way to experiment with various components of the GSPHAR model training process. It implements a factory pattern and configuration-driven approach that makes it easy to:

- **Model Architectures**: FlexibleGSPHAR, GARCH, EGARCH
- **Loss Functions**: MSE, WeightedMSE, AsymmetricMSE, ThresholdMSE, HybridLoss, TradingStrategyLoss, OHLCVLoss
- **Training Approaches**: Single-stage, Two-stage, Profit maximization, OHLCV-based, GARCH pipeline
- **Data Loaders**: OHLCV trading data, flexible time series, custom datasets

## Architecture

```
src/pipeline/
├── __init__.py                    # Package initialization
├── experiment_config.py           # Configuration classes and presets
├── component_factory.py           # Factory classes for component creation
├── training_strategies.py         # Training strategy implementations
└── flexible_training_pipeline.py  # Main pipeline orchestrator
```

## Key Components

### 1. Configuration System (`experiment_config.py`)

Provides a type-safe configuration system with enums and dataclasses:

```python
from pipeline import ExperimentConfig, ModelType, LossType, TrainingApproach

config = ExperimentConfig(
    model=ModelConfig(model_type=ModelType.FLEXIBLE_GSPHAR),
    loss=LossConfig(loss_type=LossType.WEIGHTED_MSE),
    training=TrainingConfig(approach=TrainingApproach.TWO_STAGE),
    data=DataConfig(loader_type=DataLoaderType.OHLCV_TRADING)
)
```

### 2. Component Factories (`component_factory.py`)

Factory classes that create components based on configuration:

- `ModelFactory`: Creates model instances
- `LossFactory`: Creates loss functions
- `OptimizerFactory`: Creates optimizers and schedulers
- `DataLoaderFactory`: Creates data loaders
- `TrainingStrategyFactory`: Creates training strategies

### 3. Training Strategies (`training_strategies.py`)

Implements different training approaches:

- `SingleStageStrategy`: Standard training
- `TwoStageStrategy`: Pre-training + fine-tuning
- `ProfitMaximizationStrategy`: Trading-focused training
- `OHLCVBasedStrategy`: OHLCV-specific training
- `GARCHPipelineStrategy`: GARCH model training

### 4. Main Pipeline (`flexible_training_pipeline.py`)

The main orchestrator that coordinates all components:

- `FlexibleTrainingPipeline`: Main pipeline class
- `ExperimentTracker`: Manages experiment results and comparisons
- Utility functions for quick experiments and comparison studies

## Usage Examples

### Quick Start

```python
from pipeline import run_quick_experiment

# Run a simple experiment
result = run_quick_experiment(
    model_type="flexible_gsphar",
    loss_type="weighted_mse",
    training_approach="single_stage",
    n_epochs=50
)
```

### Custom Configuration

```python
from pipeline import (
    FlexibleTrainingPipeline, ExperimentConfig,
    ModelConfig, LossConfig, TrainingConfig, DataConfig,
    ModelType, LossType, TrainingApproach, DataLoaderType
)

# Create custom configuration
config = ExperimentConfig(
    model=ModelConfig(
        model_type=ModelType.FLEXIBLE_GSPHAR,
        n_assets=38,
        A=15,
        dropout_rate=0.2
    ),
    loss=LossConfig(
        loss_type=LossType.ASYMMETRIC_MSE,
        alpha=0.7
    ),
    training=TrainingConfig(
        approach=TrainingApproach.TWO_STAGE,
        n_epochs=100,
        learning_rate=0.001,
        early_stopping_patience=10
    ),
    data=DataConfig(
        loader_type=DataLoaderType.OHLCV_TRADING,
        volatility_file="data/rv5_sqrt_24.csv",
        lags=5,
        holding_period=4
    )
)

# Run experiment
pipeline = FlexibleTrainingPipeline()
result = pipeline.run_experiment(config, "my_experiment")
```

### Predefined Configurations

```python
from pipeline import FlexibleTrainingPipeline, get_predefined_config

pipeline = FlexibleTrainingPipeline()

# Use predefined configurations
result1 = pipeline.run_predefined_experiment("flexible_gsphar")
result2 = pipeline.run_predefined_experiment("profit_maximization")
result3 = pipeline.run_predefined_experiment("garch")
```

### Comparison Studies

```python
from pipeline import run_comparison_study

# Compare different loss functions
comparison_df = run_comparison_study(
    model_types=["flexible_gsphar"],
    loss_types=["mse", "weighted_mse", "asymmetric_mse"],
    training_approaches=["single_stage"],
    n_epochs=50
)

print(comparison_df)
```

### Experiment Suite

```python
from pipeline import FlexibleTrainingPipeline

pipeline = FlexibleTrainingPipeline()

# Create multiple configurations
configs = [
    get_predefined_config("flexible_gsphar"),
    get_predefined_config("profit_maximization"),
    get_predefined_config("garch")
]

# Run all experiments
results = pipeline.run_experiment_suite(configs, "model_comparison")

# Generate comparison report
successful_experiments = [k for k, v in results.items() if v is not None]
comparison_df = pipeline.compare_experiments(successful_experiments)
```

## Available Components

### Model Types
- `FLEXIBLE_GSPHAR`: Main GSPHAR model with flexible architecture
- `GARCH`: Traditional GARCH model
- `EGARCH`: Exponential GARCH model

### Loss Functions
- `MSE`: Standard Mean Squared Error
- `WEIGHTED_MSE`: Weighted MSE for imbalanced data
- `ASYMMETRIC_MSE`: Asymmetric loss for directional bias
- `THRESHOLD_MSE`: Threshold-based MSE
- `HYBRID_LOSS`: Combination of MSE and trading loss
- `TRADING_STRATEGY_LOSS`: Pure trading strategy loss
- `OHLCV_LONG_STRATEGY_LOSS`: OHLCV-based long strategy loss
- `OHLCV_SHARPE_RATIO_LOSS`: OHLCV-based Sharpe ratio loss

### Training Approaches
- `SINGLE_STAGE`: Standard single-stage training
- `TWO_STAGE`: Pre-training with MSE + fine-tuning with target loss
- `PROFIT_MAXIMIZATION`: Trading profit-focused training
- `OHLCV_BASED`: OHLCV data-specific training
- `GARCH_PIPELINE`: GARCH model-specific pipeline

### Data Loaders
- `OHLCV_TRADING`: OHLCV trading dataset loader
- `FLEXIBLE_TIME_SERIES`: General time series data loader
- `CUSTOM`: Custom data loader interface

## Predefined Configurations

The pipeline includes several predefined configurations:

1. **flexible_gsphar**: Standard FlexibleGSPHAR with WeightedMSE
2. **garch**: GARCH model with MSE loss
3. **profit_maximization**: FlexibleGSPHAR with HybridLoss and profit focus
4. **hybrid**: Two-stage approach with HybridLoss

## Experiment Tracking

The pipeline includes comprehensive experiment tracking:

```python
from pipeline import ExperimentTracker

tracker = ExperimentTracker("my_experiments")
pipeline = FlexibleTrainingPipeline(tracker)

# Run experiments
result = pipeline.run_experiment(config, "experiment_001")

# List all experiments
experiments = tracker.list_experiments()

# Load specific results
results = tracker.load_results("experiment_001")

# Create comparison report
comparison_df = tracker.create_comparison_report(["exp_001", "exp_002"])
```

## Running Examples

### Basic Demo
```bash
cd examples
python flexible_pipeline_demo.py
```

### Practical Experiments
```bash
cd scripts
python run_flexible_experiments.py
```

## File Structure

After running experiments, the following structure is created:

```
experiments/
├── configs/           # Experiment configurations (JSON)
├── results/           # Training results and metrics (JSON)
├── models/            # Saved model states (PyTorch)
├── logs/              # Training logs
└── *_comparison.csv   # Comparison reports
```

## Integration with Existing Code

The pipeline is designed to work seamlessly with your existing GSPHAR codebase:

- Uses existing model definitions (`src/models/flexible_gsphar.py`)
- Integrates with existing loss functions (`src/training/custom_losses.py`, `src/trading_loss.py`)
- Works with existing data loaders (`src/data/ohlcv_trading_dataset.py`)
- Compatible with existing training scripts

## Requirements

- PyTorch
- NumPy
- Pandas
- tqdm (for progress bars)
- Existing GSPHAR dependencies

## Configuration File Support

You can save and load configurations as JSON files:

```python
# Save configuration
config.save_to_file("my_config.json")

# Load configuration
config = ExperimentConfig.load_from_file("my_config.json")
```

## Advanced Features

### Custom Loss Functions
Add new loss functions by implementing them and updating the `LossFactory`:

```python
class MyCustomLoss(nn.Module):
    def forward(self, pred, target):
        # Custom loss logic
        return loss

# Register in LossFactory
```

### Custom Training Strategies
Implement new training strategies by extending `BaseTrainingStrategy`:

```python
class MyCustomStrategy(BaseTrainingStrategy):
    def train(self, model, train_loader, val_loader, loss_fn, optimizer, scheduler):
        # Custom training logic
        return TrainingResult(...)
```

## Best Practices

1. **Start Small**: Begin with quick experiments using `run_quick_experiment()`
2. **Use Presets**: Leverage predefined configurations for common scenarios
3. **Track Everything**: Use the experiment tracker for reproducibility
4. **Compare Systematically**: Use comparison studies to evaluate different approaches
5. **Validate Results**: Always check experiment outputs and logs

## Troubleshooting

### Common Issues

1. **Data File Not Found**: Ensure data files are in the correct location
2. **CUDA Memory Issues**: Reduce batch size or use CPU device
3. **Import Errors**: Check that all dependencies are installed
4. **Configuration Errors**: Validate configuration parameters

### Debug Mode

Enable debug mode for detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

To add new components:

1. Define the component type in `experiment_config.py`
2. Implement the component
3. Add factory method in `component_factory.py`
4. Update tests and documentation

## Future Enhancements

- Web-based experiment dashboard
- Automated hyperparameter optimization
- Distributed training support
- Real-time experiment monitoring
- Integration with MLflow/Weights & Biases
