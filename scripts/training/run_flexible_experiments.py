#!/usr/bin/env python3
"""
Practical Flexible Training Pipeline Usage.

This script shows practical usage patterns for the flexible training pipeline,
including how to set up and run real experiments for GSPHAR model development.
"""

import os
import sys
import logging
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline import (
    FlexibleTrainingPipeline, ExperimentTracker,
    ModelType, LossType, TrainingApproach, DataLoaderType,
    ModelConfig, LossConfig, TrainingConfig, DataConfig, ExperimentConfig,
    get_predefined_config
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def experiment_1_baseline_comparison():
    """Experiment 1: Compare baseline models with different loss functions."""
    logger.info("Running Experiment 1: Baseline Comparison")
    
    # Set up experiment tracker
    tracker = ExperimentTracker("baseline_experiments")
    pipeline = FlexibleTrainingPipeline(tracker)
    
    # Define configurations for comparison
    loss_types = [
        LossType.MSE,
        LossType.WEIGHTED_MSE,
        LossType.ASYMMETRIC_MSE,
        LossType.THRESHOLD_MSE
    ]
    
    results = {}
    
    for loss_type in loss_types:
        config = ExperimentConfig(
            model=ModelConfig(
                model_type=ModelType.FLEXIBLE_GSPHAR,
                n_assets=38,
                A=10,
                dropout_rate=0.1
            ),
            loss=LossConfig(
                loss_type=loss_type,
                alpha=0.6 if loss_type == LossType.ASYMMETRIC_MSE else None,
                threshold=0.05 if loss_type == LossType.THRESHOLD_MSE else None
            ),
            training=TrainingConfig(
                approach=TrainingApproach.SINGLE_STAGE,
                n_epochs=50,
                learning_rate=0.001,
                batch_size=32,
                early_stopping_patience=10
            ),
            data=DataConfig(
                loader_type=DataLoaderType.OHLCV_TRADING,
                volatility_file="data/rv5_sqrt_24.csv",
                lags=5,
                holding_period=4,
                train_ratio=0.8
            ),
            device="cuda" if os.environ.get("CUDA_AVAILABLE") else "cpu",
            seed=42
        )
        
        experiment_id = f"baseline_{loss_type.value}"
        try:
            result = pipeline.run_experiment(config, experiment_id)
            results[experiment_id] = result
            logger.info(f"Completed {experiment_id}: Best val loss = {result.best_val_loss:.6f}")
        except Exception as e:
            logger.error(f"Failed {experiment_id}: {e}")
    
    # Generate comparison report
    successful_experiments = [k for k, v in results.items() if v is not None]
    if successful_experiments:
        comparison_df = pipeline.compare_experiments(successful_experiments)
        print("\nBaseline Comparison Results:")
        print(comparison_df[['experiment_id', 'loss_type', 'best_val_loss', 'best_epoch']])
        
        # Save comparison report
        comparison_df.to_csv("baseline_comparison_results.csv", index=False)
        logger.info("Saved baseline comparison to baseline_comparison_results.csv")


def experiment_2_architecture_optimization():
    """Experiment 2: Optimize model architecture parameters."""
    logger.info("Running Experiment 2: Architecture Optimization")
    
    tracker = ExperimentTracker("architecture_experiments")
    pipeline = FlexibleTrainingPipeline(tracker)
    
    # Test different architecture parameters
    architectures = [
        {"A": 5, "dropout_rate": 0.0},
        {"A": 10, "dropout_rate": 0.1},
        {"A": 15, "dropout_rate": 0.2},
        {"A": 20, "dropout_rate": 0.1},
        {"A": 10, "dropout_rate": 0.3}
    ]
    
    results = {}
    
    for i, arch_params in enumerate(architectures):
        config = ExperimentConfig(
            model=ModelConfig(
                model_type=ModelType.FLEXIBLE_GSPHAR,
                n_assets=38,
                **arch_params
            ),
            loss=LossConfig(loss_type=LossType.WEIGHTED_MSE),
            training=TrainingConfig(
                approach=TrainingApproach.SINGLE_STAGE,
                n_epochs=30,
                learning_rate=0.001,
                batch_size=32,
                early_stopping_patience=8
            ),
            data=DataConfig(
                loader_type=DataLoaderType.OHLCV_TRADING,
                volatility_file="data/rv5_sqrt_24.csv",
                lags=5,
                holding_period=4,
                train_ratio=0.8
            ),
            device="cuda" if os.environ.get("CUDA_AVAILABLE") else "cpu",
            seed=42
        )
        
        experiment_id = f"arch_{i+1}_A{arch_params['A']}_drop{arch_params['dropout_rate']}"
        try:
            result = pipeline.run_experiment(config, experiment_id)
            results[experiment_id] = result
            logger.info(f"Completed {experiment_id}: Best val loss = {result.best_val_loss:.6f}")
        except Exception as e:
            logger.error(f"Failed {experiment_id}: {e}")
    
    # Find best architecture
    if results:
        best_experiment = min(results.items(), key=lambda x: x[1].best_val_loss if x[1] else float('inf'))
        logger.info(f"Best architecture: {best_experiment[0]} with val loss {best_experiment[1].best_val_loss:.6f}")


def experiment_3_training_strategy_comparison():
    """Experiment 3: Compare different training strategies."""
    logger.info("Running Experiment 3: Training Strategy Comparison")
    
    tracker = ExperimentTracker("strategy_experiments")
    pipeline = FlexibleTrainingPipeline(tracker)
    
    # Test different training approaches
    strategies = [
        TrainingApproach.SINGLE_STAGE,
        TrainingApproach.TWO_STAGE,
        TrainingApproach.PROFIT_MAXIMIZATION
    ]
    
    results = {}
    
    for strategy in strategies:
        config = ExperimentConfig(
            model=ModelConfig(
                model_type=ModelType.FLEXIBLE_GSPHAR,
                n_assets=38,
                A=10,
                dropout_rate=0.1
            ),
            loss=LossConfig(
                loss_type=LossType.HYBRID_LOSS if strategy == TrainingApproach.PROFIT_MAXIMIZATION else LossType.WEIGHTED_MSE,
                mse_weight=0.7 if strategy == TrainingApproach.PROFIT_MAXIMIZATION else None,
                trading_weight=0.3 if strategy == TrainingApproach.PROFIT_MAXIMIZATION else None
            ),
            training=TrainingConfig(
                approach=strategy,
                n_epochs=40,
                learning_rate=0.001,
                batch_size=32,
                early_stopping_patience=10
            ),
            data=DataConfig(
                loader_type=DataLoaderType.OHLCV_TRADING,
                volatility_file="data/rv5_sqrt_24.csv",
                lags=5,
                holding_period=4,
                train_ratio=0.8
            ),
            device="cuda" if os.environ.get("CUDA_AVAILABLE") else "cpu",
            seed=42
        )
        
        experiment_id = f"strategy_{strategy.value}"
        try:
            result = pipeline.run_experiment(config, experiment_id)
            results[experiment_id] = result
            logger.info(f"Completed {experiment_id}: Best val loss = {result.best_val_loss:.6f}")
        except Exception as e:
            logger.error(f"Failed {experiment_id}: {e}")
    
    # Compare strategies
    successful_experiments = [k for k, v in results.items() if v is not None]
    if successful_experiments:
        comparison_df = pipeline.compare_experiments(successful_experiments)
        print("\nTraining Strategy Comparison:")
        print(comparison_df[['experiment_id', 'training_approach', 'best_val_loss', 'best_epoch']])


def experiment_4_hyperparameter_grid_search():
    """Experiment 4: Grid search over hyperparameters."""
    logger.info("Running Experiment 4: Hyperparameter Grid Search")
    
    tracker = ExperimentTracker("hyperparameter_experiments")
    pipeline = FlexibleTrainingPipeline(tracker)
    
    # Define hyperparameter grid
    learning_rates = [0.0005, 0.001, 0.002]
    batch_sizes = [16, 32, 64]
    dropout_rates = [0.0, 0.1, 0.2]
    
    results = {}
    best_result = None
    best_config = None
    
    for lr in learning_rates:
        for batch_size in batch_sizes:
            for dropout in dropout_rates:
                config = ExperimentConfig(
                    model=ModelConfig(
                        model_type=ModelType.FLEXIBLE_GSPHAR,
                        n_assets=38,
                        A=10,
                        dropout_rate=dropout
                    ),
                    loss=LossConfig(loss_type=LossType.WEIGHTED_MSE),
                    training=TrainingConfig(
                        approach=TrainingApproach.SINGLE_STAGE,
                        n_epochs=20,  # Shorter for grid search
                        learning_rate=lr,
                        batch_size=batch_size,
                        early_stopping_patience=5
                    ),
                    data=DataConfig(
                        loader_type=DataLoaderType.OHLCV_TRADING,
                        volatility_file="data/rv5_sqrt_24.csv",
                        lags=5,
                        holding_period=4,
                        train_ratio=0.8
                    ),
                    device="cuda" if os.environ.get("CUDA_AVAILABLE") else "cpu",
                    seed=42
                )
                
                experiment_id = f"grid_lr{lr}_bs{batch_size}_drop{dropout}"
                try:
                    result = pipeline.run_experiment(config, experiment_id)
                    results[experiment_id] = result
                    
                    if best_result is None or result.best_val_loss < best_result.best_val_loss:
                        best_result = result
                        best_config = (lr, batch_size, dropout)
                    
                    logger.info(f"Completed {experiment_id}: Best val loss = {result.best_val_loss:.6f}")
                except Exception as e:
                    logger.error(f"Failed {experiment_id}: {e}")
    
    # Report best configuration
    if best_config:
        logger.info(f"Best hyperparameters: LR={best_config[0]}, Batch Size={best_config[1]}, Dropout={best_config[2]}")
        logger.info(f"Best validation loss: {best_result.best_val_loss:.6f}")


def main():
    """Run practical experiments."""
    logger.info("Starting Practical Flexible Training Pipeline Experiments")
    
    # Check if data file exists
    data_file = "data/rv5_sqrt_24.csv"
    if not os.path.exists(data_file):
        logger.error(f"Data file not found: {data_file}")
        logger.info("Please ensure the data file is available before running experiments.")
        return
    
    experiments = [
        experiment_1_baseline_comparison,
        experiment_2_architecture_optimization,
        experiment_3_training_strategy_comparison,
        # experiment_4_hyperparameter_grid_search,  # Commented out as it takes longer
    ]
    
    for i, experiment_func in enumerate(experiments, 1):
        try:
            logger.info(f"\n{'='*80}")
            logger.info(f"Starting Experiment {i}")
            logger.info(f"{'='*80}")
            experiment_func()
            logger.info(f"Experiment {i} completed successfully\n")
        except Exception as e:
            logger.error(f"Experiment {i} failed: {str(e)}\n")
    
    logger.info("All practical experiments completed!")


if __name__ == "__main__":
    # Set environment variables for CUDA if available
    if os.path.exists("/usr/bin/nvidia-smi"):
        os.environ["CUDA_AVAILABLE"] = "true"
    
    main()
