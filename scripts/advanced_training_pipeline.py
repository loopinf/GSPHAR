#!/usr/bin/env python
"""
Advanced Training Pipeline for Enhanced Trading Agent.

This script implements the complete advanced training pipeline with:
- Multi-objective loss functions
- Curriculum learning
- Early stopping and validation
- Comprehensive monitoring
- Model checkpointing
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from pathlib import Path
import argparse
import json

# Import our modules
from src.models.enhanced_agent_model import EnhancedTradingAgentModel
from src.data.enhanced_dataloader import EnhancedOHLCVDataLoader
from src.training.advanced_trainer import AdvancedTradingTrainer
from src.training.advanced_loss_functions import create_advanced_loss_function

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('advanced_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Advanced Trading Agent Training')

    # Model parameters
    parser.add_argument('--n_assets', type=int, default=38, help='Number of assets')
    parser.add_argument('--vol_history_length', type=int, default=24, help='Volatility history length')
    parser.add_argument('--feature_dim', type=int, default=128, help='Feature dimension')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')

    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=12, help='Batch size')
    parser.add_argument('--patience', type=int, default=8, help='Early stopping patience')

    # Training strategy
    parser.add_argument('--use_curriculum', action='store_true', default=True, help='Use curriculum learning')
    parser.add_argument('--training_stage', choices=['initial', 'intermediate', 'advanced'],
                       default='intermediate', help='Training stage')
    parser.add_argument('--target_fill_rate', type=float, default=0.15, help='Target fill rate')
    parser.add_argument('--max_drawdown', type=float, default=0.1, help='Maximum drawdown')

    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--save_dir', type=str, default='models/advanced_training', help='Save directory')

    # Device
    parser.add_argument('--device', type=str, default='cpu', help='Training device')

    return parser.parse_args()


def validate_environment():
    """Validate that all required components are available."""
    logger.info("🔍 Validating training environment...")

    # Check data availability
    data_dir = Path("data")
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    ohlcv_dir = data_dir / "ohlcv_1h"
    if not ohlcv_dir.exists():
        raise FileNotFoundError(f"OHLCV data directory not found: {ohlcv_dir}")

    # Check for required OHLCV files
    required_files = [
        'crypto_close_1h_38.csv',
        'crypto_open_1h_38.csv',
        'crypto_high_1h_38.csv',
        'crypto_low_1h_38.csv',
        'crypto_volume_1h_38.csv'
    ]

    for file in required_files:
        file_path = ohlcv_dir / file
        if not file_path.exists():
            raise FileNotFoundError(f"Required OHLCV file not found: {file_path}")

    logger.info("✅ Environment validation passed")


def create_enhanced_model(args):
    """Create and initialize enhanced trading agent model."""
    logger.info("🤖 Creating enhanced trading agent model...")

    model = EnhancedTradingAgentModel(
        n_assets=args.n_assets,
        vol_history_length=args.vol_history_length,
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        init_with_vol_pred=True  # Always use vol_pred initialization
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"✅ Model created:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB")

    return model


def load_and_validate_data(args):
    """Load and validate OHLCV data."""
    logger.info("📊 Loading and validating OHLCV data...")

    device = torch.device(args.device)
    dataloader = EnhancedOHLCVDataLoader(
        data_dir=args.data_dir,
        device=device
    )

    # Get data info
    symbol_info = dataloader.get_symbol_info()
    logger.info(f"✅ Data loaded:")
    logger.info(f"  Assets: {symbol_info['n_assets']}")
    logger.info(f"  Time range: {symbol_info['time_range']}")
    logger.info(f"  Total hours: {symbol_info['total_hours']:,}")

    # Validate data quality
    quality_report = dataloader.validate_data_quality()
    logger.info("📈 Data quality summary:")
    for data_type, metrics in quality_report.items():
        missing_pct = metrics['missing_percentage']
        logger.info(f"  {data_type}: {metrics['shape']}, Missing: {missing_pct:.2f}%")

        if missing_pct > 1.0:
            logger.warning(f"High missing data percentage for {data_type}: {missing_pct:.2f}%")

    return dataloader


def run_advanced_training(args):
    """Run the complete advanced training pipeline."""
    logger.info("🚀 Starting Advanced Trading Agent Training Pipeline")
    logger.info("=" * 70)

    # Log training configuration
    logger.info("📋 Training Configuration:")
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")

    # Validate environment
    validate_environment()

    # Load data
    dataloader = load_and_validate_data(args)

    # Create model
    model = create_enhanced_model(args)

    # Initialize trainer
    device = torch.device(args.device)
    trainer = AdvancedTradingTrainer(
        model=model,
        dataloader=dataloader,
        device=device,
        save_dir=args.save_dir
    )

    logger.info(f"🎯 Training target: {args.target_fill_rate:.1%} fill rate")
    logger.info(f"🛡️ Risk constraint: {args.max_drawdown:.1%} max drawdown")

    # Start training
    start_time = datetime.now()
    logger.info(f"⏰ Training started at: {start_time}")

    try:
        training_results = trainer.train(
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            patience=args.patience,
            use_curriculum=args.use_curriculum,
            save_checkpoints=True
        )

        end_time = datetime.now()
        training_duration = end_time - start_time

        logger.info("🎉 Training completed successfully!")
        logger.info(f"⏰ Training duration: {training_duration}")
        logger.info(f"🏆 Best score: {training_results['best_score']:.4f}")
        logger.info(f"📊 Final epoch: {training_results['final_epoch']}")

        # Save training configuration and results
        save_training_summary(args, training_results, training_duration)

        return training_results

    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise


def save_training_summary(args, training_results, training_duration):
    """Save comprehensive training summary."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.save_dir)

    # Create summary dictionary
    summary = {
        'timestamp': timestamp,
        'training_duration_seconds': training_duration.total_seconds(),
        'training_duration_str': str(training_duration),
        'configuration': vars(args),
        'results': {
            'best_score': training_results['best_score'],
            'final_epoch': training_results['final_epoch'],
            'model_path': training_results['model_path']
        },
        'final_metrics': {}
    }

    # Extract final metrics from training history
    if training_results['training_history']:
        history = training_results['training_history']
        if history['epoch']:
            summary['final_metrics'] = {
                'final_train_loss': history['train_loss'][-1],
                'final_val_loss': history['val_loss'][-1],
                'final_train_fill_rate': history['train_fill_rate'][-1],
                'final_val_fill_rate': history['val_fill_rate'][-1],
                'final_train_avg_return': history['train_avg_return'][-1],
                'final_val_avg_return': history['val_avg_return'][-1],
                'final_learning_rate': history['learning_rate'][-1]
            }

    # Save summary as JSON
    summary_path = save_dir / f"training_summary_{timestamp}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"📋 Training summary saved: {summary_path}")

    # Print final summary
    logger.info("📊 TRAINING SUMMARY:")
    logger.info(f"  Duration: {training_duration}")
    logger.info(f"  Best Score: {training_results['best_score']:.4f}")
    logger.info(f"  Epochs Completed: {training_results['final_epoch']}")

    if summary['final_metrics']:
        metrics = summary['final_metrics']
        logger.info(f"  Final Fill Rate: {metrics.get('final_val_fill_rate', 0):.3f}")
        logger.info(f"  Final Avg Return: {metrics.get('final_val_avg_return', 0):.4f}")
        logger.info(f"  Final Loss: {metrics.get('final_val_loss', 0):.4f}")


def main():
    """Main training function."""
    args = parse_arguments()

    try:
        training_results = run_advanced_training(args)

        logger.info("🎉 Advanced Training Pipeline Completed Successfully!")
        logger.info("=" * 70)
        logger.info("🚀 Next Steps:")
        logger.info("  1. Review training visualizations")
        logger.info("  2. Analyze model performance metrics")
        logger.info("  3. Test on out-of-sample data")
        logger.info("  4. Deploy for live trading (if performance is satisfactory)")

        return 0

    except Exception as e:
        logger.error(f"❌ Training pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
