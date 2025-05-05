import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import compute_spillover_index

@pytest.fixture
def sample_data():
    """Load test data from rv5_sqrt_24.csv"""
    return pd.read_csv('data/rv5_sqrt_24.csv', index_col=0) * 100

def plot_spillover_heatmap(matrix, title, save_path):
    """Helper function to create and save heatmap plots"""
    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def test_plot_spillover_effects(sample_data):
    """Test and visualize spillover effects with different parameters"""
    # Create results directory if it doesn't exist
    import os
    os.makedirs('test_results/spillover_plots', exist_ok=True)
    
    test_data = sample_data.iloc[:1000]  # Use first 1000 rows for testing
    
    # 1. Compare different horizons
    horizons = [1, 5, 10, 22]
    horizon_matrices = []
    
    for h in horizons:
        matrix = compute_spillover_index(
            test_data,
            horizon=h,
            lag=22,
            scarcity_prop=0.0,
            standardized=True
        )
        horizon_matrices.append(matrix)
        
        # Plot individual horizon heatmaps
        plot_spillover_heatmap(
            matrix,
            f'Spillover Index - Horizon {h}',
            f'test_results/spillover_plots/horizon_{h}_heatmap.png'
        )
    
    # Plot horizon differences
    for i in range(len(horizons)-1):
        diff_matrix = horizon_matrices[i+1] - horizon_matrices[i]
        plot_spillover_heatmap(
            diff_matrix,
            f'Spillover Difference (H{horizons[i+1]} - H{horizons[i]})',
            f'test_results/spillover_plots/horizon_diff_{horizons[i]}_{horizons[i+1]}.png'
        )
    
    # 2. Compare different scarcity levels
    scarcity_levels = [0.0, 0.3, 0.5, 0.7]
    scarcity_matrices = []
    
    for scarcity in scarcity_levels:
        matrix = compute_spillover_index(
            test_data,
            horizon=5,
            lag=22,
            scarcity_prop=scarcity,
            standardized=True
        )
        scarcity_matrices.append(matrix)
        
        # Plot individual scarcity heatmaps
        plot_spillover_heatmap(
            matrix,
            f'Spillover Index - Scarcity {scarcity}',
            f'test_results/spillover_plots/scarcity_{scarcity}_heatmap.png'
        )
    
    # 3. Compare standardized vs non-standardized
    matrix_std = compute_spillover_index(
        test_data,
        horizon=5,
        lag=22,
        scarcity_prop=0.0,
        standardized=True
    )
    
    matrix_non_std = compute_spillover_index(
        test_data,
        horizon=5,
        lag=22,
        scarcity_prop=0.0,
        standardized=False
    )
    
    # Plot standardization comparison
    plot_spillover_heatmap(
        matrix_std,
        'Spillover Index - Standardized',
        'test_results/spillover_plots/standardized_heatmap.png'
    )
    
    plot_spillover_heatmap(
        matrix_non_std,
        'Spillover Index - Non-standardized',
        'test_results/spillover_plots/non_standardized_heatmap.png'
    )
    
    # Plot standardization difference
    diff_matrix = matrix_std - matrix_non_std
    plot_spillover_heatmap(
        diff_matrix,
        'Spillover Difference (Standardized - Non-standardized)',
        'test_results/spillover_plots/standardization_diff.png'
    )
    
    # 4. Create summary plots
    plt.figure(figsize=(15, 5))
    
    # Plot average spillover by horizon
    plt.subplot(131)
    avg_spillover = [np.mean(m[~np.eye(m.shape[0], dtype=bool)]) for m in horizon_matrices]
    plt.plot(horizons, avg_spillover, marker='o')
    plt.title('Average Spillover by Horizon')
    plt.xlabel('Horizon')
    plt.ylabel('Average Spillover')
    
    # Plot average spillover by scarcity
    plt.subplot(132)
    avg_spillover = [np.mean(m[~np.eye(m.shape[0], dtype=bool)]) for m in scarcity_matrices]
    plt.plot(scarcity_levels, avg_spillover, marker='o')
    plt.title('Average Spillover by Scarcity')
    plt.xlabel('Scarcity Level')
    plt.ylabel('Average Spillover')
    
    # Plot distribution comparison
    plt.subplot(133)
    plt.hist(matrix_std.flatten(), alpha=0.5, label='Standardized', bins=30)
    plt.hist(matrix_non_std.flatten(), alpha=0.5, label='Non-standardized', bins=30)
    plt.title('Distribution Comparison')
    plt.xlabel('Spillover Value')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('test_results/spillover_plots/summary_plots.png')
    plt.close()

def test_plot_time_varying_spillover(sample_data):
    """Test and visualize time-varying spillover effects"""
    # Calculate minimum window size
    n_variables = len(sample_data.columns)
    lag = 22
    min_window_size = n_variables * lag + 1
    
    # Add buffer for stability
    window_size = min_window_size + 20
    step_size = 10
    
    print(f"\nWindow analysis parameters:")
    print(f"Minimum required size: {min_window_size}")
    print(f"Actual window size used: {window_size}")
    print(f"Step size: {step_size}")
    
    total_windows = (len(sample_data) - window_size) // step_size
    print(f"Total windows: {total_windows}")
    
    time_varying_spillover = []
    window_dates = []
    
    for i in range(200, total_windows):  # Start from 200th window for testing
        start_idx = i * step_size
        end_idx = start_idx + window_size
        window_data = sample_data.iloc[start_idx:end_idx]
        
        try:
            matrix = compute_spillover_index(
                window_data,
                horizon=5,
                lag=22,
                scarcity_prop=0.0,
                standardized=True
            )
            
            # Store average spillover and date
            time_varying_spillover.append(np.mean(matrix[~np.eye(matrix.shape[0], dtype=bool)]))
            window_dates.append(window_data.index[-1])
        except Exception as e:
            print(f"Warning: Window {i} failed: {str(e)}")
    
    # Only create plots if we have results
    if time_varying_spillover:
        # Plot time-varying spillover
        plt.figure(figsize=(15, 6))
        plt.plot(window_dates, time_varying_spillover, marker='o')
        plt.title('Time-varying Average Spillover')
        plt.xlabel('Time')
        plt.ylabel('Average Spillover')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('test_results/spillover_plots/time_varying_spillover.png')
        plt.close()

        # Create heatmap animation data
        for i in range(min(10, total_windows)):  # Save first 10 windows for visualization
            start_idx = i * step_size
            end_idx = start_idx + window_size
            window_data = sample_data.iloc[start_idx:end_idx]
            
            matrix = compute_spillover_index(
                window_data,
                horizon=5,
                lag=22,
                scarcity_prop=0.0,
                standardized=True
            )
            
            plot_spillover_heatmap(
                matrix,
                f'Spillover Index - Window {i+1}\n{window_data.index[0]} to {window_data.index[-1]}',
                f'test_results/spillover_plots/time_window_{i+1}_heatmap.png'
            )
