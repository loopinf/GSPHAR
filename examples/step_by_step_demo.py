#!/usr/bin/env python3
"""
Step-by-step pipeline example.

This script demonstrates each step of using the flexible training pipeline.
"""

import sys
import logging
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure logging to see what's happening
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def step_by_step_example():
    """Step-by-step pipeline demonstration."""
    
    print("🚀 STEP-BY-STEP FLEXIBLE TRAINING PIPELINE DEMO")
    print("=" * 60)
    
    # Step 1: Import what we need
    print("\n📦 Step 1: Importing pipeline components...")
    try:
        from pipeline.flexible_training_pipeline import FlexibleTrainingPipeline, ExperimentTracker
        from pipeline.experiment_config import (
            ExperimentConfig, ModelConfig, LossConfig, TrainingConfig, DataConfig,
            ModelType, LossType, TrainingApproach, DataLoaderType
        )
        print("✅ Imports successful!")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return
    
    # Step 2: Create experiment tracker
    print("\n📊 Step 2: Setting up experiment tracking...")
    tracker = ExperimentTracker("step_by_step_demo")
    pipeline = FlexibleTrainingPipeline(tracker)
    print("✅ Experiment tracker created!")
    
    # Step 3: Define model configuration
    print("\n🏗️ Step 3: Configuring the model...")
    model_config = ModelConfig(
        model_type=ModelType.FLEXIBLE_GSPHAR,
        n_assets=38,
        A=5,  # Small for quick demo
        dropout_rate=0.1
    )
    print(f"✅ Model config: {model_config.model_type.value} with A={model_config.A}")
    
    # Step 4: Define loss configuration
    print("\n🎯 Step 4: Configuring the loss function...")
    loss_config = LossConfig(
        loss_type=LossType.WEIGHTED_MSE
    )
    print(f"✅ Loss config: {loss_config.loss_type.value}")
    
    # Step 5: Define training configuration
    print("\n🏃 Step 5: Configuring training approach...")
    training_config = TrainingConfig(
        approach=TrainingApproach.SINGLE_STAGE,
        n_epochs=3,  # Very short for demo
        learning_rate=0.001,
        batch_size=16,  # Small batch for demo
        early_stopping_patience=2
    )
    print(f"✅ Training config: {training_config.approach.value}, {training_config.n_epochs} epochs")
    
    # Step 6: Define data configuration
    print("\n📊 Step 6: Configuring data loading...")
    data_config = DataConfig(
        loader_type=DataLoaderType.OHLCV_TRADING,
        volatility_file="data/rv5_sqrt_24.csv",
        lags=5,
        holding_period=4,
        train_ratio=0.8,
        subset_size=1000  # Small subset for demo
    )
    print(f"✅ Data config: {data_config.loader_type.value}")
    
    # Step 7: Create complete experiment configuration
    print("\n⚙️ Step 7: Creating complete experiment configuration...")
    experiment_config = ExperimentConfig(
        model=model_config,
        loss=loss_config,
        training=training_config,
        data=data_config,
        device="cpu",  # Use CPU for demo
        seed=42  # For reproducibility
    )
    print("✅ Complete experiment configuration created!")
    
    # Step 8: Run the experiment
    print("\n🚀 Step 8: Running the experiment...")
    print("This will:")
    print("  - Create the model")
    print("  - Load the data") 
    print("  - Set up loss function and optimizer")
    print("  - Train for 3 epochs")
    print("  - Save all results")
    
    try:
        result = pipeline.run_experiment(experiment_config, "step_by_step_demo")
        
        # Step 9: Examine results
        print("\n📈 Step 9: Examining results...")
        print(f"✅ Training completed successfully!")
        print(f"   - Best validation loss: {result.best_val_loss:.6f}")
        print(f"   - Best epoch: {result.best_epoch}")
        print(f"   - Total epochs completed: {len(result.train_losses)}")
        print(f"   - Final training loss: {result.train_losses[-1]:.6f}")
        
        # Step 10: Check saved files
        print("\n💾 Step 10: Checking saved files...")
        experiment_dir = Path("step_by_step_demo")
        if experiment_dir.exists():
            config_files = list((experiment_dir / "configs").glob("*.json"))
            result_files = list((experiment_dir / "results").glob("*.json"))
            model_files = list((experiment_dir / "models").glob("*.pt"))
            
            print(f"   - Configurations saved: {len(config_files)}")
            print(f"   - Results saved: {len(result_files)}")
            print(f"   - Models saved: {len(model_files)}")
            
            if config_files:
                print(f"   - Latest config: {config_files[-1].name}")
            if result_files:
                print(f"   - Latest results: {result_files[-1].name}")
        
        print("\n🎉 SUCCESS! You've successfully run your first flexible training pipeline experiment!")
        
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        print("This might be due to missing data files or other dependencies.")
        print("Check that 'data/rv5_sqrt_24.csv' exists in your workspace.")


def quick_comparison_demo():
    """Demonstrate a quick comparison between different loss functions."""
    
    print("\n\n🔄 BONUS: Quick Comparison Demo")
    print("=" * 40)
    
    try:
        from pipeline.flexible_training_pipeline import run_comparison_study
        
        print("Running comparison of 2 loss functions (very quick)...")
        
        comparison_df = run_comparison_study(
            model_types=["flexible_gsphar"],
            loss_types=["mse", "weighted_mse"],
            training_approaches=["single_stage"],
            n_epochs=2,  # Very short
            batch_size=16,
            subset_size=500  # Very small dataset
        )
        
        print("\n📊 Comparison Results:")
        if not comparison_df.empty:
            print(comparison_df[['experiment_id', 'loss_type', 'best_val_loss', 'best_epoch']])
        else:
            print("No successful experiments in comparison.")
            
    except Exception as e:
        print(f"❌ Comparison failed: {e}")


if __name__ == "__main__":
    # Check if data file exists
    data_file = Path("data/rv5_sqrt_24.csv")
    if not data_file.exists():
        print("⚠️  WARNING: Data file 'data/rv5_sqrt_24.csv' not found.")
        print("   The demo will likely fail at the training step.")
        print("   Please ensure the data file is available.")
        print()
    
    step_by_step_example()
    
    # Uncomment to run comparison demo as well
    # quick_comparison_demo()
    
    print("\n📚 Next Steps:")
    print("1. Explore the created 'step_by_step_demo' directory")
    print("2. Try modifying the configurations and re-running")
    print("3. Run the full examples with: python examples/flexible_pipeline_demo.py")
    print("4. Run practical experiments with: python scripts/run_flexible_experiments.py")
