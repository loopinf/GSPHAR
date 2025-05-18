#!/usr/bin/env python
"""
Script to plot GSPHAR predictions alongside realized volatility and percentage change.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
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


def load_data(predictions_file, pct_change_file):
    """
    Load the prediction results and percentage change data.

    Args:
        predictions_file (str): Path to the predictions CSV file.
        pct_change_file (str): Path to the percentage change CSV file.

    Returns:
        tuple: (predictions_df, pct_change_df)
    """
    logger.info(f"Loading predictions from {predictions_file}")
    predictions_df = pd.read_csv(predictions_file, index_col=0, parse_dates=True)
    
    logger.info(f"Loading percentage change data from {pct_change_file}")
    pct_change_df = pd.read_csv(pct_change_file, index_col=0, parse_dates=True)
    
    logger.info(f"Loaded predictions with shape {predictions_df.shape}")
    logger.info(f"Loaded percentage change data with shape {pct_change_df.shape}")
    
    return predictions_df, pct_change_df


def align_data(predictions_df, pct_change_df):
    """
    Align the predictions and percentage change data to have the same index.

    Args:
        predictions_df (pd.DataFrame): DataFrame containing predictions.
        pct_change_df (pd.DataFrame): DataFrame containing percentage change data.

    Returns:
        tuple: (predictions_aligned, pct_change_aligned)
    """
    logger.info("Aligning data")
    
    # Find common dates
    common_index = predictions_df.index.intersection(pct_change_df.index)
    logger.info(f"Found {len(common_index)} common timestamps")
    
    # Align data
    predictions_aligned = predictions_df.loc[common_index]
    pct_change_aligned = pct_change_df.loc[common_index]
    
    logger.info(f"Aligned data shape: {predictions_aligned.shape}")
    
    return predictions_aligned, pct_change_aligned


def create_plots(predictions_df, pct_change_df, output_dir, symbols=None, normalize=True):
    """
    Create plots showing predictions, actual values, and percentage change.

    Args:
        predictions_df (pd.DataFrame): DataFrame containing predictions.
        pct_change_df (pd.DataFrame): DataFrame containing percentage change data.
        output_dir (str): Directory to save the plots.
        symbols (list, optional): List of symbols to plot. If None, all symbols are plotted.
        normalize (bool): Whether to normalize the data for better visualization.

    Returns:
        None
    """
    logger.info("Creating plots")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract symbols from predictions DataFrame
    available_symbols = []
    for col in predictions_df.columns:
        if col.endswith('_pred'):
            symbol = col.split('_pred')[0]
            if f"{symbol}_true" in predictions_df.columns and symbol in pct_change_df.columns:
                available_symbols.append(symbol)
    
    logger.info(f"Found {len(available_symbols)} available symbols")
    
    # Filter symbols if provided
    if symbols is not None:
        symbols = [s for s in symbols if s in available_symbols]
        logger.info(f"Filtered to {len(symbols)} requested symbols")
    else:
        symbols = available_symbols
    
    # Create plots for each symbol
    for symbol in symbols:
        logger.info(f"Creating plot for {symbol}")
        
        # Extract data
        pred_col = f"{symbol}_pred"
        true_col = f"{symbol}_true"
        
        predictions = predictions_df[pred_col]
        actual_rv = predictions_df[true_col]
        pct_change = pct_change_df[symbol].abs()  # Use absolute percentage change
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        if normalize:
            # Create normalized version (z-score)
            predictions_z = (predictions - predictions.mean()) / predictions.std()
            actual_rv_z = (actual_rv - actual_rv.mean()) / actual_rv.std()
            pct_change_z = (pct_change - pct_change.mean()) / pct_change.std()
            
            # Add traces
            fig.add_trace(
                go.Scatter(x=predictions_df.index, y=predictions_z, name="Predicted RV (Z-Score)", line=dict(color='blue')),
                secondary_y=False
            )
            
            fig.add_trace(
                go.Scatter(x=predictions_df.index, y=actual_rv_z, name="Actual RV (Z-Score)", line=dict(color='green')),
                secondary_y=False
            )
            
            fig.add_trace(
                go.Scatter(x=pct_change_df.index, y=pct_change_z, name="Abs % Change (Z-Score)", line=dict(color='red')),
                secondary_y=False
            )
            
            # Set y-axis title
            fig.update_yaxes(title_text="Z-Score (Standard Deviations from Mean)", secondary_y=False)
            
            # Set title
            title = f"{symbol}: Predicted RV vs Actual RV vs Absolute % Change (Normalized)"
        else:
            # Add traces for non-normalized version
            fig.add_trace(
                go.Scatter(x=predictions_df.index, y=predictions, name="Predicted RV", line=dict(color='blue')),
                secondary_y=False
            )
            
            fig.add_trace(
                go.Scatter(x=predictions_df.index, y=actual_rv, name="Actual RV", line=dict(color='green')),
                secondary_y=False
            )
            
            fig.add_trace(
                go.Scatter(x=pct_change_df.index, y=pct_change, name="Abs % Change", line=dict(color='red')),
                secondary_y=True
            )
            
            # Set y-axes titles
            fig.update_yaxes(title_text="Realized Volatility", secondary_y=False)
            fig.update_yaxes(title_text="Absolute % Change", secondary_y=True)
            
            # Set title
            title = f"{symbol}: Predicted RV vs Actual RV vs Absolute % Change"
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Save figure
        normalized_suffix = "_normalized" if normalize else ""
        fig.write_html(os.path.join(output_dir, f"{symbol}_prediction_with_pct_change{normalized_suffix}.html"))
        
        # Also create a static image version
        fig.write_image(os.path.join(output_dir, f"{symbol}_prediction_with_pct_change{normalized_suffix}.png"), width=1200, height=800)
        
        logger.info(f"Created plot for {symbol}")
    
    logger.info("All plots created successfully")


def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(description='Plot GSPHAR predictions alongside realized volatility and percentage change.')
    parser.add_argument('--predictions-file', type=str, required=True,
                        help='Path to the predictions CSV file.')
    parser.add_argument('--pct-change-file', type=str, required=True,
                        help='Path to the percentage change CSV file.')
    parser.add_argument('--output-dir', type=str, default='plots/combined_plots',
                        help='Directory to save the plots.')
    parser.add_argument('--symbols', type=str, nargs='+', default=None,
                        help='List of symbols to plot. If not provided, all symbols are plotted.')
    parser.add_argument('--normalize', action='store_true', default=False,
                        help='Whether to normalize the data for better visualization.')
    
    args = parser.parse_args()
    
    # Load data
    predictions_df, pct_change_df = load_data(args.predictions_file, args.pct_change_file)
    
    # Align data
    predictions_aligned, pct_change_aligned = align_data(predictions_df, pct_change_df)
    
    # Create plots
    create_plots(
        predictions_aligned,
        pct_change_aligned,
        args.output_dir,
        args.symbols,
        args.normalize
    )
    
    logger.info("Plotting completed successfully")


if __name__ == '__main__':
    main()
