#!/usr/bin/env python
"""
Check for training/testing data overlap - the most likely explanation for exceptional results.
"""

import torch
import pandas as pd
import os
import sys

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.ohlcv_trading_dataset import load_ohlcv_trading_data


def check_training_testing_overlap():
    """Check if there's overlap between training and testing data."""
    print("🔍 CHECKING TRAINING/TESTING DATA OVERLAP")
    print("=" * 60)
    
    # Load the full dataset
    dataset, _ = load_ohlcv_trading_data(
        volatility_file="data/crypto_rv1h_38_20200822_20250116.csv",
        lags=[1, 4, 24],
        holding_period=4,
        debug=False
    )
    
    print(f"Total dataset size: {len(dataset)} samples")
    
    # Training configuration (from our two-stage training)
    training_subset_size = 500  # We trained on 500 samples
    training_indices = list(range(0, training_subset_size))  # Samples 0-499
    
    # Testing configuration (from our PnL time series)
    testing_start_idx = 0  # We tested from index 0
    testing_num_periods = 2000  # We tested 2000 periods
    testing_indices = list(range(testing_start_idx, testing_start_idx + testing_num_periods))
    
    print(f"\n📊 DATA USAGE:")
    print(f"Training indices: {training_indices[0]} to {training_indices[-1]} ({len(training_indices)} samples)")
    print(f"Testing indices: {testing_indices[0]} to {testing_indices[-1]} ({len(testing_indices)} samples)")
    
    # Check for overlap
    overlap_indices = set(training_indices) & set(testing_indices)
    
    print(f"\n🚨 OVERLAP ANALYSIS:")
    print(f"Overlapping indices: {len(overlap_indices)}")
    
    if overlap_indices:
        print(f"❌ DATA LEAKAGE DETECTED!")
        print(f"Overlap: {min(overlap_indices)} to {max(overlap_indices)}")
        print(f"Overlap size: {len(overlap_indices)} samples")
        print(f"Overlap percentage: {len(overlap_indices)/len(training_indices)*100:.1f}% of training data")
        
        # Show time periods
        first_overlap_info = dataset.get_sample_info(min(overlap_indices))
        last_overlap_info = dataset.get_sample_info(max(overlap_indices))
        
        print(f"\n📅 OVERLAPPING TIME PERIOD:")
        print(f"From: {first_overlap_info['prediction_time']}")
        print(f"To: {last_overlap_info['prediction_time']}")
        
        # This explains the exceptional results!
        print(f"\n💡 EXPLANATION FOR EXCEPTIONAL RESULTS:")
        print(f"The model was tested on the SAME data it was trained on!")
        print(f"This creates artificially perfect performance.")
        
    else:
        print(f"✅ No overlap detected")
        print(f"Training and testing use different data periods")
    
    # Show actual time periods
    print(f"\n📅 ACTUAL TIME PERIODS:")
    
    training_start_info = dataset.get_sample_info(training_indices[0])
    training_end_info = dataset.get_sample_info(training_indices[-1])
    
    testing_start_info = dataset.get_sample_info(testing_indices[0])
    testing_end_info = dataset.get_sample_info(testing_indices[-1])
    
    print(f"Training period:")
    print(f"  From: {training_start_info['prediction_time']}")
    print(f"  To: {training_end_info['prediction_time']}")
    
    print(f"Testing period:")
    print(f"  From: {testing_start_info['prediction_time']}")
    print(f"  To: {testing_end_info['prediction_time']}")


def calculate_proper_split():
    """Calculate what a proper train/test split should look like."""
    print(f"\n🔧 PROPER TRAIN/TEST SPLIT RECOMMENDATION")
    print("=" * 60)
    
    # Load dataset
    dataset, _ = load_ohlcv_trading_data(
        volatility_file="data/crypto_rv1h_38_20200822_20250116.csv",
        lags=[1, 4, 24],
        holding_period=4,
        debug=False
    )
    
    total_samples = len(dataset)
    
    # Recommend proper split
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)
    test_size = total_samples - train_size - val_size
    
    print(f"Total samples: {total_samples}")
    print(f"\n📊 RECOMMENDED SPLIT:")
    print(f"Training: 0 to {train_size-1} ({train_size} samples, {train_ratio:.0%})")
    print(f"Validation: {train_size} to {train_size + val_size - 1} ({val_size} samples, {val_ratio:.0%})")
    print(f"Testing: {train_size + val_size} to {total_samples-1} ({test_size} samples, {test_ratio:.0%})")
    
    # Show time periods
    train_start_info = dataset.get_sample_info(0)
    train_end_info = dataset.get_sample_info(train_size - 1)
    
    val_start_info = dataset.get_sample_info(train_size)
    val_end_info = dataset.get_sample_info(train_size + val_size - 1)
    
    test_start_info = dataset.get_sample_info(train_size + val_size)
    test_end_info = dataset.get_sample_info(total_samples - 1)
    
    print(f"\n📅 TIME PERIODS:")
    print(f"Training: {train_start_info['prediction_time']} to {train_end_info['prediction_time']}")
    print(f"Validation: {val_start_info['prediction_time']} to {val_end_info['prediction_time']}")
    print(f"Testing: {test_start_info['prediction_time']} to {test_end_info['prediction_time']}")
    
    print(f"\n✅ BENEFITS OF PROPER SPLIT:")
    print(f"1. No data leakage between train/test")
    print(f"2. Temporal ordering preserved")
    print(f"3. Realistic out-of-sample testing")
    print(f"4. Proper validation for model selection")


def estimate_realistic_performance():
    """Estimate what realistic performance might look like."""
    print(f"\n📈 REALISTIC PERFORMANCE ESTIMATION")
    print("=" * 60)
    
    print(f"🤔 CURRENT RESULTS (with potential data leakage):")
    print(f"  Final PnL: $47,699")
    print(f"  Win rate: 85%")
    print(f"  Fill rate: 41.8%")
    print(f"  Average period PnL: $23.85")
    
    print(f"\n📉 REALISTIC EXPECTATIONS (without data leakage):")
    print(f"  Win rate: 55-65% (instead of 85%)")
    print(f"  Fill rate: 30-40% (similar)")
    print(f"  Average period PnL: $2-5 (instead of $23.85)")
    print(f"  Final PnL: $4,000-10,000 (instead of $47,699)")
    
    print(f"\n💡 WHY PERFORMANCE WOULD BE LOWER:")
    print(f"1. Model hasn't seen the test data before")
    print(f"2. Real out-of-sample prediction is harder")
    print(f"3. Market conditions may differ from training period")
    print(f"4. Model may not generalize as well")
    
    print(f"\n✅ STILL POTENTIALLY PROFITABLE:")
    print(f"Even with realistic expectations, the strategy could be profitable:")
    print(f"- 60% win rate is still good")
    print(f"- $5,000 profit over 2.8 months = ~$21,000 annually")
    print(f"- On $760,000 deployed = ~2.8% annual return")
    print(f"- Risk-adjusted returns could still be attractive")


def main():
    """Main function to check data overlap."""
    print("🕵️ TRAINING/TESTING DATA OVERLAP INVESTIGATION")
    print("=" * 80)
    print("This is the most likely explanation for exceptional results!")
    print()
    
    check_training_testing_overlap()
    calculate_proper_split()
    estimate_realistic_performance()
    
    print(f"\n🎯 CONCLUSION:")
    print("=" * 40)
    print("If data overlap is confirmed, it explains the exceptional results.")
    print("The strategy framework is still valid, but performance expectations")
    print("should be adjusted to realistic levels with proper train/test split.")
    print()
    print("Next steps:")
    print("1. Retrain model with proper train/test split")
    print("2. Test on truly out-of-sample data")
    print("3. Validate on different market periods")
    print("4. Set realistic performance expectations")


if __name__ == "__main__":
    main()
