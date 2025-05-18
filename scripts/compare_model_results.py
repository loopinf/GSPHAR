#!/usr/bin/env python
"""
Script to compare the results of different GSPHAR models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import json
import logging
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_metrics(metrics_files):
    """
    Load metrics from JSON files.

    Args:
        metrics_files (list): List of paths to metrics JSON files.

    Returns:
        dict: Dictionary of metrics for each model.
    """
    metrics = {}
    for metrics_file in metrics_files:
        model_name = os.path.basename(os.path.dirname(metrics_file))
        with open(metrics_file, 'r') as f:
            metrics[model_name] = json.load(f)
    
    return metrics


def load_predictions(predictions_files):
    """
    Load predictions from CSV files.

    Args:
        predictions_files (list): List of paths to predictions CSV files.

    Returns:
        dict: Dictionary of predictions DataFrames for each model.
    """
    predictions = {}
    for predictions_file in predictions_files:
        model_name = os.path.basename(os.path.dirname(predictions_file))
        predictions[model_name] = pd.read_csv(predictions_file, index_col=0, parse_dates=True)
    
    return predictions


def compare_metrics(metrics, output_file=None):
    """
    Compare metrics from different models.

    Args:
        metrics (dict): Dictionary of metrics for each model.
        output_file (str, optional): Path to save the comparison table.

    Returns:
        pd.DataFrame: DataFrame containing the comparison.
    """
    # Create a DataFrame with metrics for each model
    metrics_df = pd.DataFrame(metrics).T
    
    # Add percentage improvement column for each metric
    base_model = metrics_df.index[0]
    for col in metrics_df.columns:
        if col != 'test_loss':  # For test_loss, lower is better
            metrics_df[f'{col}_improvement'] = (metrics_df[col] / metrics_df.loc[base_model, col] - 1) * 100
        else:
            metrics_df[f'{col}_improvement'] = (1 - metrics_df[col] / metrics_df.loc[base_model, col]) * 100
    
    # Save to file if requested
    if output_file:
        metrics_df.to_csv(output_file)
    
    return metrics_df


def compare_predictions(predictions, symbols, output_dir=None):
    """
    Compare predictions from different models.

    Args:
        predictions (dict): Dictionary of predictions DataFrames for each model.
        symbols (list): List of symbols to compare.
        output_dir (str, optional): Directory to save the comparison plots.

    Returns:
        None
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    model_names = list(predictions.keys())
    
    # Create plots for each symbol
    for symbol in symbols:
        # Create figure
        fig = go.Figure()
        
        # Add true values
        true_col = f"{symbol}_true"
        if true_col in predictions[model_names[0]].columns:
            fig.add_trace(
                go.Scatter(
                    x=predictions[model_names[0]].index,
                    y=predictions[model_names[0]][true_col],
                    name="True",
                    line=dict(color='black', width=2)
                )
            )
        
        # Add predictions for each model
        colors = ['blue', 'red', 'green', 'purple', 'orange']
        for i, model_name in enumerate(model_names):
            pred_col = f"{symbol}_pred"
            if pred_col in predictions[model_name].columns:
                fig.add_trace(
                    go.Scatter(
                        x=predictions[model_name].index,
                        y=predictions[model_name][pred_col],
                        name=f"{model_name} Prediction",
                        line=dict(color=colors[i % len(colors)])
                    )
                )
        
        # Update layout
        fig.update_layout(
            title=f"{symbol}: Model Predictions Comparison",
            xaxis_title="Date",
            yaxis_title="Realized Volatility",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Save figure
        if output_dir:
            fig.write_html(os.path.join(output_dir, f"{symbol}_predictions_comparison.html"))
            fig.write_image(os.path.join(output_dir, f"{symbol}_predictions_comparison.png"), width=1200, height=800)
        
        logger.info(f"Created comparison plot for {symbol}")
    
    # Create error comparison plots
    for symbol in symbols:
        # Create figure
        fig = go.Figure()
        
        # Add error for each model
        colors = ['blue', 'red', 'green', 'purple', 'orange']
        for i, model_name in enumerate(model_names):
            true_col = f"{symbol}_true"
            pred_col = f"{symbol}_pred"
            if true_col in predictions[model_name].columns and pred_col in predictions[model_name].columns:
                # Calculate error
                error = predictions[model_name][pred_col] - predictions[model_name][true_col]
                
                fig.add_trace(
                    go.Scatter(
                        x=predictions[model_name].index,
                        y=error,
                        name=f"{model_name} Error",
                        line=dict(color=colors[i % len(colors)])
                    )
                )
        
        # Update layout
        fig.update_layout(
            title=f"{symbol}: Model Prediction Errors Comparison",
            xaxis_title="Date",
            yaxis_title="Prediction Error",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Add zero line
        fig.add_hline(y=0, line=dict(color='black', width=1, dash='dash'))
        
        # Save figure
        if output_dir:
            fig.write_html(os.path.join(output_dir, f"{symbol}_errors_comparison.html"))
            fig.write_image(os.path.join(output_dir, f"{symbol}_errors_comparison.png"), width=1200, height=800)
        
        logger.info(f"Created error comparison plot for {symbol}")


def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(description='Compare the results of different GSPHAR models.')
    parser.add_argument('--metrics-files', type=str, nargs='+', required=True,
                        help='Paths to metrics JSON files.')
    parser.add_argument('--predictions-files', type=str, nargs='+', required=True,
                        help='Paths to predictions CSV files.')
    parser.add_argument('--symbols', type=str, nargs='+', default=['BTCUSDT', 'ETHUSDT', 'LTCUSDT', 'LINKUSDT'],
                        help='List of symbols to compare.')
    parser.add_argument('--output-dir', type=str, default='plots/model_comparison',
                        help='Directory to save the comparison results.')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load metrics
    metrics = load_metrics(args.metrics_files)
    
    # Compare metrics
    metrics_df = compare_metrics(metrics, os.path.join(args.output_dir, 'metrics_comparison.csv'))
    logger.info(f"Metrics comparison:\n{metrics_df}")
    
    # Load predictions
    predictions = load_predictions(args.predictions_files)
    
    # Compare predictions
    compare_predictions(predictions, args.symbols, args.output_dir)
    
    logger.info("Comparison completed successfully")


if __name__ == '__main__':
    main()
