#!/usr/bin/env python3
"""
Flexible Training Pipeline - Comprehensive Example.

This script demonstrates how to use the flexible training pipeline to run
various experiments with different combinations of models, loss functions,
and training approaches.
"""

import os
import sys
import logging
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import (
    FlexibleTrainingPipeline, ExperimentTracker,
    ModelType, LossType, TrainingApproach, DataLoaderType,
    ModelConfig, LossConfig, TrainingConfig, DataConfig, ExperimentConfig,
    get_predefined_config, run_quick_experiment, run_comparison_study
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def example_1_quick_experiment():
    """Example 1: Run a quick experiment with minimal configuration."""
    logger.info("=== Example 1: Quick Experiment ===")
    
    # Run a quick experiment with default parameters
    result = run_quick_experiment(
        model_type="flexible_gsphar",
        loss_type="mse",
        training_approach="single_stage",
        n_epochs=5,  # Small number for demo
        learning_rate=0.001
    )
    
    logger.info(f"Quick experiment completed!")
    logger.info(f"Best validation loss: {result.best_val_loss:.6f}")
    logger.info(f"Training completed in {len(result.train_losses)} epochs")


def example_2_predefined_configurations():
    """Example 2: Use predefined experiment configurations."""
    logger.info("=== Example 2: Predefined Configurations ===")
    
    pipeline = FlexibleTrainingPipeline()
    
    # Run the flexible GSPHAR preset
    logger.info("Running flexible GSPHAR preset...")
    result1 = pipeline.run_predefined_experiment(
        "flexible_gsphar",
        n_epochs=5  # Override default epochs for demo
    )
    
    # Run the profit maximization preset
    logger.info("Running profit maximization preset...")
    result2 = pipeline.run_predefined_experiment(
        "profit_maximization",
        n_epochs=5
    )
    
    logger.info(f"Flexible GSPHAR - Best val loss: {result1.best_val_loss:.6f}")
    logger.info(f"Profit Maximization - Best val loss: {result2.best_val_loss:.6f}")


def example_3_custom_configuration():
    """Example 3: Create custom experiment configuration."""
    logger.info("=== Example 3: Custom Configuration ===")
    
    # Create custom configuration
    model_config = ModelConfig(
        model_type=ModelType.FLEXIBLE_GSPHAR,
        n_assets=38,
        A=15,  # Custom architecture parameter
        dropout_rate=0.2
    )
    
    loss_config = LossConfig(
        loss_type=LossType.ASYMMETRIC_MSE,
        alpha=0.7  # Custom asymmetry parameter
    )
    
    training_config = TrainingConfig(
        approach=TrainingApproach.TWO_STAGE,
        n_epochs=10,
        learning_rate=0.001,
        batch_size=32,
        early_stopping_patience=3
    )
    
    data_config = DataConfig(
        loader_type=DataLoaderType.OHLCV_TRADING,
        lags=5,
        holding_period=4,
        train_ratio=0.8
    )
    
    custom_config = ExperimentConfig(
        model=model_config,
        loss=loss_config,
        training=training_config,
        data=data_config,
        device="cpu",  # Use CPU for demo
        seed=42
    )
    
    # Run the custom experiment
    pipeline = FlexibleTrainingPipeline()
    result = pipeline.run_experiment(custom_config, "custom_experiment_demo")
    
    logger.info(f"Custom experiment completed!")
    logger.info(f"Best validation loss: {result.best_val_loss:.6f}")


def example_4_comparison_study():
    """Example 4: Run a comparison study across different configurations."""
    logger.info("=== Example 4: Comparison Study ===")
    
    # Compare different loss functions
    comparison_df = run_comparison_study(
        model_types=["flexible_gsphar"],
        loss_types=["mse", "weighted_mse", "asymmetric_mse"],
        training_approaches=["single_stage"],
        n_epochs=5,  # Small for demo
        batch_size=16
    )
    
    logger.info("Comparison study completed!")
    logger.info("Results summary:")
    print(comparison_df[['experiment_id', 'loss_type', 'best_val_loss', 'best_epoch']])


def example_5_experiment_suite():
    """Example 5: Run multiple experiments with different configurations."""
    logger.info("=== Example 5: Experiment Suite ===")
    
    # Create multiple configurations
    configs = []
    
    # Configuration 1: Standard MSE
    config1 = ExperimentConfig(
        model=ModelConfig(model_type=ModelType.FLEXIBLE_GSPHAR),
        loss=LossConfig(loss_type=LossType.MSE),
        training=TrainingConfig(
            approach=TrainingApproach.SINGLE_STAGE,
            n_epochs=5
        ),
        data=DataConfig(loader_type=DataLoaderType.OHLCV_TRADING)
    )
    configs.append(config1)
    
    # Configuration 2: Weighted MSE
    config2 = ExperimentConfig(
        model=ModelConfig(model_type=ModelType.FLEXIBLE_GSPHAR),
        loss=LossConfig(loss_type=LossType.WEIGHTED_MSE),
        training=TrainingConfig(
            approach=TrainingApproach.SINGLE_STAGE,
            n_epochs=5
        ),
        data=DataConfig(loader_type=DataLoaderType.OHLCV_TRADING)
    )
    configs.append(config2)
    
    # Configuration 3: Two-stage training
    config3 = ExperimentConfig(
        model=ModelConfig(model_type=ModelType.FLEXIBLE_GSPHAR),
        loss=LossConfig(loss_type=LossType.HYBRID_LOSS),
        training=TrainingConfig(
            approach=TrainingApproach.TWO_STAGE,
            n_epochs=8
        ),
        data=DataConfig(loader_type=DataLoaderType.OHLCV_TRADING)
    )
    configs.append(config3)
    
    # Run experiment suite
    pipeline = FlexibleTrainingPipeline()
    results = pipeline.run_experiment_suite(configs, "demo_suite")
    
    # Show results
    successful_runs = sum(1 for r in results.values() if r is not None)
    logger.info(f"Experiment suite completed: {successful_runs}/{len(configs)} successful")
    
    # Create comparison report
    successful_ids = [k for k, v in results.items() if v is not None]
    if successful_ids:
        comparison_df = pipeline.compare_experiments(successful_ids)
        logger.info("Suite comparison:")
        print(comparison_df[['experiment_id', 'loss_type', 'training_approach', 'best_val_loss']])


def example_6_experiment_tracking():
    """Example 6: Demonstrate experiment tracking and result management."""
    logger.info("=== Example 6: Experiment Tracking ===")
    
    # Create custom experiment tracker
    tracker = ExperimentTracker("demo_experiments")
    pipeline = FlexibleTrainingPipeline(tracker)
    
    # Run a few experiments
    configs = [
        get_predefined_config("flexible_gsphar"),
        get_predefined_config("profit_maximization")
    ]
    
    # Modify configs for quick demo
    for config in configs:
        config.training.n_epochs = 3
    
    experiment_ids = []
    for i, config in enumerate(configs):
        exp_id = f"tracking_demo_{i+1}"
        try:
            result = pipeline.run_experiment(config, exp_id)
            experiment_ids.append(exp_id)
            logger.info(f"Experiment {exp_id} completed")
        except Exception as e:
            logger.error(f"Experiment {exp_id} failed: {e}")
    
    # List all experiments
    all_experiments = tracker.list_experiments()
    logger.info(f"All experiments: {all_experiments}")
    
    # Load and compare results
    if experiment_ids:
        comparison_df = tracker.create_comparison_report(experiment_ids)
        logger.info("Tracking comparison:")
        print(comparison_df[['experiment_id', 'model_type', 'best_val_loss']])


def main():
    """Run all examples."""
    logger.info("Starting Flexible Training Pipeline Examples")
    
    examples = [
        example_1_quick_experiment,
        example_2_predefined_configurations,
        example_3_custom_configuration,
        example_4_comparison_study,
        example_5_experiment_suite,
        example_6_experiment_tracking
    ]
    
    for i, example_func in enumerate(examples, 1):
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running Example {i}")
            logger.info(f"{'='*60}")
            example_func()
            logger.info(f"Example {i} completed successfully")
        except Exception as e:
            logger.error(f"Example {i} failed: {str(e)}")
        
        logger.info("\n")
    
    logger.info("All examples completed!")


if __name__ == "__main__":
    # Set up the data path for the examples
    # You may need to adjust this path based on your data location
    data_path = "data/rv5_sqrt_24.csv"
    if not os.path.exists(data_path):
        logger.warning(f"Data file not found: {data_path}")
        logger.warning("Some examples may fail. Please ensure data files are available.")
    
    main()
