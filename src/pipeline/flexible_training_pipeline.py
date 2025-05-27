"""
Flexible Training Pipeline - Main Orchestrator.

This module provides the main pipeline orchestrator that coordinates all components
for flexible experimentation with different model architectures, loss functions,
training approaches, and data loaders.
"""

import os
import json
import torch
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import pandas as pd
from pathlib import Path

from .experiment_config import ExperimentConfig, get_predefined_config
from .component_factory import create_all_components, OptimizerFactory
from .training_strategies import TrainingResult

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Tracks and manages experiment results."""
    
    def __init__(self, base_dir: str = "experiments"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.base_dir / "configs").mkdir(exist_ok=True)
        (self.base_dir / "results").mkdir(exist_ok=True)
        (self.base_dir / "models").mkdir(exist_ok=True)
        (self.base_dir / "logs").mkdir(exist_ok=True)
    
    def save_config(self, config: ExperimentConfig, experiment_id: str):
        """Save experiment configuration."""
        config_path = self.base_dir / "configs" / f"{experiment_id}.json"
        
        # Convert config to dict for JSON serialization
        config_dict = {
            "experiment_id": experiment_id,
            "model": {
                "model_type": config.model.model_type.value,
                "n_assets": config.model.n_assets,
                "A": config.model.A,
                "dropout_rate": config.model.dropout_rate,
                "params": config.model.params
            },
            "loss": {
                "loss_type": config.loss.loss_type.value,
                "alpha": config.loss.alpha,
                "threshold": config.loss.threshold,
                "mse_weight": config.loss.mse_weight,
                "trading_weight": config.loss.trading_weight,
                "holding_period": config.loss.holding_period,
                "trading_fee": config.loss.trading_fee,
                "params": config.loss.params
            },
            "training": {
                "approach": config.training.approach.value,
                "n_epochs": config.training.n_epochs,
                "batch_size": config.training.batch_size,
                "learning_rate": config.training.learning_rate,
                "weight_decay": config.training.weight_decay,
                "early_stopping_patience": config.training.early_stopping_patience,
                "grad_clip_norm": config.training.grad_clip_norm,
                "scheduler_type": config.training.scheduler_type,
                "scheduler_factor": config.training.scheduler_factor,
                "scheduler_patience": config.training.scheduler_patience,
                "params": config.training.params
            },
            "data": {
                "loader_type": config.data.loader_type.value,
                "volatility_file": config.data.volatility_file,
                "lags": config.data.lags,
                "holding_period": config.data.holding_period,
                "train_ratio": config.data.train_ratio,
                "subset_size": config.data.subset_size,
                "normalization": config.data.normalization,
                "debug": config.data.debug
            },
            "device": config.device,
            "seed": config.seed,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Saved config to {config_path}")
    
    def save_results(self, result: TrainingResult, experiment_id: str, metadata: Dict[str, Any] = None):
        """Save experiment results."""
        results_path = self.base_dir / "results" / f"{experiment_id}.json"
        
        results_dict = {
            "experiment_id": experiment_id,
            "train_losses": result.train_losses,
            "val_losses": result.val_losses,
            "train_metrics": result.train_metrics,
            "val_metrics": result.val_metrics,
            "best_epoch": result.best_epoch,
            "best_val_loss": result.best_val_loss,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }
        
        with open(results_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"Saved results to {results_path}")
    
    def save_model(self, model: torch.nn.Module, experiment_id: str):
        """Save trained model."""
        model_path = self.base_dir / "models" / f"{experiment_id}.pt"
        torch.save(model.state_dict(), model_path)
        logger.info(f"Saved model to {model_path}")
    
    def load_results(self, experiment_id: str) -> Dict[str, Any]:
        """Load experiment results."""
        results_path = self.base_dir / "results" / f"{experiment_id}.json"
        
        if not results_path.exists():
            raise FileNotFoundError(f"Results not found: {results_path}")
        
        with open(results_path, 'r') as f:
            return json.load(f)
    
    def list_experiments(self) -> List[str]:
        """List all experiment IDs."""
        config_dir = self.base_dir / "configs"
        return [f.stem for f in config_dir.glob("*.json")]
    
    def create_comparison_report(self, experiment_ids: List[str]) -> pd.DataFrame:
        """Create a comparison report for multiple experiments."""
        comparison_data = []
        
        for exp_id in experiment_ids:
            try:
                results = self.load_results(exp_id)
                config_path = self.base_dir / "configs" / f"{exp_id}.json"
                
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                comparison_data.append({
                    "experiment_id": exp_id,
                    "model_type": config["model"]["model_type"],
                    "loss_type": config["loss"]["loss_type"],
                    "training_approach": config["training"]["approach"],
                    "data_loader": config["data"]["loader_type"],
                    "best_val_loss": results["best_val_loss"],
                    "best_epoch": results["best_epoch"],
                    "final_train_loss": results["train_losses"][-1] if results["train_losses"] else None,
                    "n_epochs": len(results["train_losses"]),
                    "learning_rate": config["training"]["learning_rate"],
                    "batch_size": config["training"]["batch_size"]
                })
            except Exception as e:
                logger.warning(f"Failed to load experiment {exp_id}: {e}")
        
        return pd.DataFrame(comparison_data)


class FlexibleTrainingPipeline:
    """Main orchestrator for flexible training experiments."""
    
    def __init__(self, experiment_tracker: Optional[ExperimentTracker] = None):
        self.tracker = experiment_tracker or ExperimentTracker()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def run_experiment(
        self, 
        config: ExperimentConfig, 
        experiment_id: Optional[str] = None,
        save_results: bool = True
    ) -> TrainingResult:
        """Run a complete training experiment."""
        if experiment_id is None:
            experiment_id = self._generate_experiment_id(config)
        
        self.logger.info(f"Starting experiment: {experiment_id}")
        
        # Set up logging for this experiment
        if save_results:
            self._setup_experiment_logging(experiment_id)
        
        try:
            # Set seed for reproducibility
            if config.seed is not None:
                torch.manual_seed(config.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(config.seed)
            
            # Create all components
            self.logger.info("Creating components...")
            components = create_all_components(config)
            
            # Extract components
            model = components['model']
            loss_fn = components['loss_fn']
            train_loader = components['train_loader']
            val_loader = components['val_loader']
            training_strategy = components['training_strategy']
            metadata = components['metadata']
            
            # Create optimizer and scheduler
            optimizer = OptimizerFactory.create_optimizer(model, config.training)
            scheduler = OptimizerFactory.create_scheduler(optimizer, config.training)
            
            # Update batch sizes in data loaders
            self._update_batch_sizes(train_loader, val_loader, config.training.batch_size)
            
            # Save configuration
            if save_results:
                self.tracker.save_config(config, experiment_id)
            
            # Run training
            self.logger.info("Starting training...")
            result = training_strategy.train(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                scheduler=scheduler
            )
            
            # Load best model state
            if result.final_model_state:
                model.load_state_dict(result.final_model_state)
            
            # Save results and model
            if save_results:
                self.tracker.save_results(result, experiment_id, metadata)
                self.tracker.save_model(model, experiment_id)
            
            self.logger.info(f"Experiment {experiment_id} completed successfully")
            self.logger.info(f"Best validation loss: {result.best_val_loss:.6f} at epoch {result.best_epoch}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Experiment {experiment_id} failed: {str(e)}")
            raise
    
    def run_experiment_suite(
        self, 
        configs: List[ExperimentConfig], 
        experiment_prefix: str = "suite"
    ) -> Dict[str, TrainingResult]:
        """Run multiple experiments in sequence."""
        results = {}
        
        for i, config in enumerate(configs):
            experiment_id = f"{experiment_prefix}_{i:03d}"
            try:
                result = self.run_experiment(config, experiment_id)
                results[experiment_id] = result
            except Exception as e:
                self.logger.error(f"Failed to run experiment {experiment_id}: {e}")
                results[experiment_id] = None
        
        # Create comparison report
        successful_experiments = [k for k, v in results.items() if v is not None]
        if successful_experiments:
            comparison_df = self.tracker.create_comparison_report(successful_experiments)
            comparison_path = self.tracker.base_dir / f"{experiment_prefix}_comparison.csv"
            comparison_df.to_csv(comparison_path, index=False)
            self.logger.info(f"Saved comparison report to {comparison_path}")
        
        return results
    
    def run_predefined_experiment(self, preset_name: str, **kwargs) -> TrainingResult:
        """Run a predefined experiment configuration."""
        config = get_predefined_config(preset_name)
        
        # Apply any parameter overrides
        if kwargs:
            config = self._apply_config_overrides(config, kwargs)
        
        experiment_id = f"{preset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return self.run_experiment(config, experiment_id)
    
    def compare_experiments(self, experiment_ids: List[str]) -> pd.DataFrame:
        """Compare multiple experiments."""
        return self.tracker.create_comparison_report(experiment_ids)
    
    def _generate_experiment_id(self, config: ExperimentConfig) -> str:
        """Generate a unique experiment ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_type = config.model.model_type.value
        loss_type = config.loss.loss_type.value
        approach = config.training.approach.value
        
        return f"{model_type}_{loss_type}_{approach}_{timestamp}"
    
    def _setup_experiment_logging(self, experiment_id: str):
        """Set up logging for a specific experiment."""
        log_file = self.tracker.base_dir / "logs" / f"{experiment_id}.log"
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to root logger
        logging.getLogger().addHandler(file_handler)
    
    def _update_batch_sizes(self, train_loader, val_loader, batch_size: int):
        """Update batch sizes in data loaders."""
        if hasattr(train_loader, 'batch_size'):
            train_loader.batch_size = batch_size
        if hasattr(val_loader, 'batch_size'):
            val_loader.batch_size = batch_size
    
    def _apply_config_overrides(self, config: ExperimentConfig, overrides: Dict[str, Any]) -> ExperimentConfig:
        """Apply parameter overrides to configuration."""
        # This is a simplified implementation
        # In practice, you might want more sophisticated override logic
        
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
            elif hasattr(config.training, key):
                setattr(config.training, key, value)
            elif hasattr(config.model, key):
                setattr(config.model, key, value)
            elif hasattr(config.loss, key):
                setattr(config.loss, key, value)
            elif hasattr(config.data, key):
                setattr(config.data, key, value)
        
        return config


# Convenience functions for easy pipeline usage
def run_quick_experiment(
    model_type: str = "flexible_gsphar",
    loss_type: str = "mse",
    training_approach: str = "single_stage",
    data_loader: str = "ohlcv_trading",
    n_epochs: int = 10,
    **kwargs
) -> TrainingResult:
    """Run a quick experiment with minimal configuration."""
    from .experiment_config import (
        ModelType, LossType, TrainingApproach, DataLoaderType,
        ModelConfig, LossConfig, TrainingConfig, DataConfig,
        ExperimentConfig
    )
    
    # Create configuration
    model_config = ModelConfig(model_type=ModelType(model_type))
    loss_config = LossConfig(loss_type=LossType(loss_type))
    training_config = TrainingConfig(
        approach=TrainingApproach(training_approach),
        n_epochs=n_epochs
    )
    data_config = DataConfig(loader_type=DataLoaderType(data_loader))
    
    config = ExperimentConfig(
        model=model_config,
        loss=loss_config,
        training=training_config,
        data=data_config
    )
    
    # Apply any additional overrides
    if kwargs:
        pipeline = FlexibleTrainingPipeline()
        config = pipeline._apply_config_overrides(config, kwargs)
    
    # Run experiment
    pipeline = FlexibleTrainingPipeline()
    return pipeline.run_experiment(config)


def run_comparison_study(
    model_types: List[str] = None,
    loss_types: List[str] = None,
    training_approaches: List[str] = None,
    **common_params
) -> pd.DataFrame:
    """Run a comparison study across different configurations."""
    from itertools import product
    from .experiment_config import (
        ModelType, LossType, TrainingApproach, DataLoaderType,
        ModelConfig, LossConfig, TrainingConfig, DataConfig,
        ExperimentConfig
    )
    
    # Default values
    model_types = model_types or ["flexible_gsphar"]
    loss_types = loss_types or ["mse", "weighted_mse"]
    training_approaches = training_approaches or ["single_stage"]
    
    configs = []
    for model_type, loss_type, approach in product(model_types, loss_types, training_approaches):
        model_config = ModelConfig(model_type=ModelType(model_type))
        loss_config = LossConfig(loss_type=LossType(loss_type))
        training_config = TrainingConfig(approach=TrainingApproach(approach))
        data_config = DataConfig(loader_type=DataLoaderType("ohlcv_trading"))
        
        config = ExperimentConfig(
            model=model_config,
            loss=loss_config,
            training=training_config,
            data=data_config
        )
        configs.append(config)
    
    # Run experiment suite
    pipeline = FlexibleTrainingPipeline()
    results = pipeline.run_experiment_suite(configs, "comparison_study")
    
    # Return comparison report
    successful_experiments = [k for k, v in results.items() if v is not None]
    return pipeline.compare_experiments(successful_experiments)
