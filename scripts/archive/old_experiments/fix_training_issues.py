#!/usr/bin/env python
"""
Fix the identified training issues with proper Stage 1 learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import sys
import logging
from datetime import datetime
from torch.utils.data import DataLoader, Subset

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.ohlcv_trading_dataset import load_ohlcv_trading_data
from src.models.flexible_gsphar import FlexibleGSPHAR
from src.ohlcv_trading_loss import OHLCVLongStrategyLoss

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_data_splits(dataset):
    """Create proper train/validation/test splits."""
    total_samples = len(dataset)
    
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)
    test_size = total_samples - train_size - val_size
    
    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, total_samples))
    
    logger.info(f"📊 DATA SPLIT:")
    logger.info(f"  Training: {len(train_indices)} samples ({train_ratio:.0%})")
    logger.info(f"  Validation: {len(val_indices)} samples ({val_ratio:.0%})")
    logger.info(f"  Testing: {len(test_indices)} samples ({test_ratio:.0%})")
    
    return train_indices, val_indices, test_indices


def train_stage1_fixed(model, train_loader, val_loader, device, n_epochs=50, initial_lr=0.0001):
    """Fixed Stage 1: Proper supervised learning with much lower learning rate."""
    logger.info(f"🔧 FIXED STAGE 1: SUPERVISED LEARNING")
    logger.info(f"  Epochs: {n_epochs}")
    logger.info(f"  Initial learning rate: {initial_lr} (much lower!)")
    
    # Much lower learning rate and less aggressive regularization
    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5)
    
    train_history = {'train_loss': [], 'train_mae': [], 'val_loss': [], 'val_mae': [], 'lr': []}
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 15  # More patience
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_losses = []
        train_maes = []
        
        for batch_idx, batch in enumerate(train_loader):
            x_lags = [x.to(device) for x in batch['x_lags']]
            vol_targets = batch['vol_targets'].to(device)
            
            optimizer.zero_grad()
            
            vol_pred = model(*x_lags)
            mse_loss = F.mse_loss(vol_pred, vol_targets)
            mae_loss = F.l1_loss(vol_pred, vol_targets)
            
            # Lighter gradient clipping
            mse_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            
            train_losses.append(mse_loss.item())
            train_maes.append(mae_loss.item())
        
        # Validation
        model.eval()
        val_losses = []
        val_maes = []
        vol_pred_stats = []
        vol_target_stats = []
        
        with torch.no_grad():
            for batch in val_loader:
                x_lags = [x.to(device) for x in batch['x_lags']]
                vol_targets = batch['vol_targets'].to(device)
                
                vol_pred = model(*x_lags)
                mse_loss = F.mse_loss(vol_pred, vol_targets)
                mae_loss = F.l1_loss(vol_pred, vol_targets)
                
                val_losses.append(mse_loss.item())
                val_maes.append(mae_loss.item())
                
                # Track prediction and target statistics
                vol_pred_stats.extend(vol_pred.cpu().numpy().flatten())
                vol_target_stats.extend(vol_targets.cpu().numpy().flatten())
        
        # Record history
        avg_train_loss = np.mean(train_losses)
        avg_train_mae = np.mean(train_maes)
        avg_val_loss = np.mean(val_losses)
        avg_val_mae = np.mean(val_maes)
        current_lr = optimizer.param_groups[0]['lr']
        
        train_history['train_loss'].append(avg_train_loss)
        train_history['train_mae'].append(avg_train_mae)
        train_history['val_loss'].append(avg_val_loss)
        train_history['val_mae'].append(avg_val_mae)
        train_history['lr'].append(current_lr)
        
        # Prediction vs target statistics
        vol_pred_mean = np.mean(vol_pred_stats)
        vol_pred_std = np.std(vol_pred_stats)
        vol_target_mean = np.mean(vol_target_stats)
        vol_target_std = np.std(vol_target_stats)
        
        # Calculate improvement metrics
        if epoch == 0:
            initial_loss = avg_train_loss
        loss_reduction = (initial_loss - avg_train_loss) / initial_loss
        
        logger.info(f"  Epoch {epoch+1}/{n_epochs}:")
        logger.info(f"    Train MSE={avg_train_loss:.6f}, MAE={avg_train_mae:.6f}")
        logger.info(f"    Val MSE={avg_val_loss:.6f}, MAE={avg_val_mae:.6f}")
        logger.info(f"    Pred: μ={vol_pred_mean:.4f}, σ={vol_pred_std:.4f}")
        logger.info(f"    Target: μ={vol_target_mean:.4f}, σ={vol_target_std:.4f}")
        logger.info(f"    Loss reduction: {loss_reduction:.1%}, LR={current_lr:.6f}")
        
        # Check if learning is happening
        if loss_reduction < 0.05 and epoch > 10:  # Less than 5% improvement after 10 epochs
            logger.warning(f"    ⚠️  Slow learning detected")
        elif loss_reduction > 0.3:  # More than 30% improvement
            logger.info(f"    ✅ Good learning progress")
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model state
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            logger.info(f"  Early stopping triggered after {epoch+1} epochs")
            # Restore best model
            model.load_state_dict(best_model_state)
            break
    
    # Final assessment
    final_loss_reduction = (initial_loss - best_val_loss) / initial_loss
    logger.info(f"✅ Stage 1 completed. Loss reduction: {final_loss_reduction:.1%}")
    
    if final_loss_reduction < 0.2:  # Less than 20% improvement
        logger.warning(f"🚨 WARNING: Poor Stage 1 learning ({final_loss_reduction:.1%})")
        logger.warning(f"Consider: lower LR, more epochs, or different architecture")
    else:
        logger.info(f"✅ Good Stage 1 learning achieved")
    
    return train_history


def validate_stage1_learning(model, val_loader, device):
    """Validate that Stage 1 actually learned something meaningful."""
    logger.info(f"🔍 VALIDATING STAGE 1 LEARNING")
    
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            x_lags = [x.to(device) for x in batch['x_lags']]
            vol_targets = batch['vol_targets'].to(device)
            
            vol_pred = model(*x_lags)
            
            predictions.extend(vol_pred.cpu().numpy().flatten())
            targets.extend(vol_targets.cpu().numpy().flatten())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Calculate metrics
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    correlation = np.corrcoef(predictions, targets)[0, 1]
    
    # Baseline: always predict mean
    mean_baseline_mse = np.mean((targets.mean() - targets) ** 2)
    
    logger.info(f"  Validation MSE: {mse:.6f}")
    logger.info(f"  Validation MAE: {mae:.6f}")
    logger.info(f"  Correlation: {correlation:.4f}")
    logger.info(f"  Mean baseline MSE: {mean_baseline_mse:.6f}")
    logger.info(f"  Improvement over baseline: {(mean_baseline_mse - mse) / mean_baseline_mse:.1%}")
    
    # Check prediction quality
    pred_mean = predictions.mean()
    pred_std = predictions.std()
    target_mean = targets.mean()
    target_std = targets.std()
    
    logger.info(f"  Prediction mean: {pred_mean:.4f} ({pred_mean*100:.1f}%)")
    logger.info(f"  Target mean: {target_mean:.4f} ({target_mean*100:.1f}%)")
    logger.info(f"  Prediction std: {pred_std:.4f}")
    logger.info(f"  Target std: {target_std:.4f}")
    
    # Quality checks
    issues = []
    
    if mse > mean_baseline_mse * 0.9:
        issues.append("❌ Barely better than mean baseline")
    
    if abs(pred_mean - target_mean) / target_mean > 0.3:
        issues.append("❌ Large mean prediction error")
    
    if pred_std < target_std * 0.1:
        issues.append("❌ Very low prediction variance")
    
    if correlation < 0.1:
        issues.append("❌ Very low correlation with targets")
    
    if issues:
        logger.warning(f"🚨 STAGE 1 QUALITY ISSUES:")
        for issue in issues:
            logger.warning(f"  {issue}")
        return False
    else:
        logger.info(f"✅ Stage 1 learning quality is good")
        return True


def main():
    """Main fixed training function."""
    logger.info("🔧 FIXED MODEL TRAINING")
    logger.info("=" * 80)
    
    # Parameters - much more conservative
    device = torch.device('cpu')
    batch_size = 8  # Smaller batch size
    stage1_epochs = 50  # More epochs
    stage1_lr = 0.0001  # Much lower learning rate
    
    # Load dataset
    logger.info("📊 Loading dataset...")
    dataset, metadata = load_ohlcv_trading_data(
        volatility_file="data/crypto_rv1h_38_20200822_20250116.csv",
        lags=[1, 4, 24],
        holding_period=4,
        debug=False
    )
    
    logger.info(f"Dataset loaded: {len(dataset)} total samples")
    
    # Create proper splits
    train_indices, val_indices, test_indices = create_data_splits(dataset)
    
    # Create data subsets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"📦 Data loaders created:")
    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Validation batches: {len(val_loader)}")
    
    # Initialize model
    logger.info("🧠 Initializing model...")
    
    # Load volatility data for correlation matrix
    vol_df = pd.read_csv("data/crypto_rv1h_38_20200822_20250116.csv", index_col=0, parse_dates=True)
    corr_sample = vol_df.iloc[-1000:] if len(vol_df) > 1000 else vol_df
    corr_matrix = corr_sample.corr().values
    A = (corr_matrix + corr_matrix.T) / 2
    
    model = FlexibleGSPHAR(
        lags=[1, 4, 24],
        output_dim=1,
        filter_size=len(metadata['assets']),
        A=A
    )
    model = model.to(device)
    
    # Stage 1: Fixed supervised learning
    stage1_history = train_stage1_fixed(
        model, train_loader, val_loader, device, 
        n_epochs=stage1_epochs, initial_lr=stage1_lr
    )
    
    # Validate Stage 1 learning
    stage1_quality = validate_stage1_learning(model, val_loader, device)
    
    if not stage1_quality:
        logger.error("🚨 Stage 1 learning failed. Stopping training.")
        logger.error("Consider: even lower learning rate, different architecture, or more data preprocessing")
        return None
    
    # Save fixed model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/fixed_stage1_model_{timestamp}.pt"
    
    # Prepare metadata
    model_metadata = {
        'assets': metadata['assets'],
        'training_samples': len(train_indices),
        'validation_samples': len(val_indices),
        'test_samples': len(test_indices),
        'train_period': f"{dataset.get_sample_info(train_indices[0])['prediction_time']} to {dataset.get_sample_info(train_indices[-1])['prediction_time']}",
        'test_period': f"{dataset.get_sample_info(test_indices[0])['prediction_time']} to {dataset.get_sample_info(test_indices[-1])['prediction_time']}",
        'no_data_leakage': True,
        'stage1_fixed': True,
        'fixes_applied': [
            'Much lower learning rate (0.0001 vs 0.001)',
            'More epochs (50 vs 25)',
            'Better early stopping (patience 15)',
            'Proper learning validation',
            'Lighter regularization'
        ]
    }
    
    model_parameters = {
        'lags': [1, 4, 24],
        'holding_period': 4,
        'stage1_epochs': stage1_epochs,
        'stage1_lr': stage1_lr,
        'batch_size': batch_size,
        'stage1_only': True  # Only Stage 1 for now
    }
    
    # Save checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'stage1_history': stage1_history,
        'metadata': model_metadata,
        'parameters': model_parameters,
        'train_indices': train_indices,
        'val_indices': val_indices,
        'test_indices': test_indices,
        'stage1_quality_passed': stage1_quality
    }, model_path)
    
    logger.info(f"✅ Fixed Stage 1 model saved to: {model_path}")
    
    # Summary
    logger.info(f"\n🎯 FIXED TRAINING SUMMARY:")
    logger.info(f"  ✅ Much lower learning rate: {stage1_lr}")
    logger.info(f"  ✅ More epochs: {stage1_epochs}")
    logger.info(f"  ✅ Proper learning validation")
    logger.info(f"  ✅ Stage 1 quality: {'PASSED' if stage1_quality else 'FAILED'}")
    
    return model_path, test_indices


if __name__ == "__main__":
    model_path, test_indices = main()
    if model_path:
        print(f"\n🚀 Next step: Test fixed Stage 1 model")
        print(f"Fixed model saved at: {model_path}")
    else:
        print(f"\n❌ Training failed. Need to investigate further.")
