"""
Integration module for using date-aware predictions in notebooks.

This module provides functions to integrate the date-aware prediction functionality
into existing notebooks like gsphar_vs_garch_comparison_fixed.ipynb.
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

# Add the parent directory to the path to import from the GSPHAR package
sys.path.insert(0, os.path.abspath('..'))

# Import from local modules
from src.data import IndexMappingDataset, create_index_mapping_dataloaders, generate_index_mapped_predictions

def integrate_with_notebook(train_data, test_data, lag_list, h, batch_size, model, market_indices_list):
    """
    Integrate date-aware predictions with an existing notebook.

    Args:
        train_data (pd.DataFrame): Training data with datetime index
        test_data (pd.DataFrame): Test data with datetime index
        lag_list (list): List of lag values to use
        h (int): Forecast horizon
        batch_size (int): Batch size
        model: The GSPHAR model
        market_indices_list (list): List of market indices

    Returns:
        tuple: (pred_df, actual_df) with proper datetime indices
    """
    # Create dictionaries for dataloaders
    train_dict = {
        'data': train_data,
        'lag_list': lag_list,
        'h': h
    }
    test_dict = {
        'data': test_data,
        'lag_list': lag_list,
        'h': h
    }

    # Create dataloaders with index mapping
    train_dataloader, test_dataloader, train_dataset, test_dataset = create_index_mapping_dataloaders(
        train_dict, test_dict, batch_size
    )

    # Generate predictions with date awareness
    pred_df, actual_df = generate_index_mapped_predictions(
        model, test_dataloader, test_dataset, market_indices_list
    )

    return pred_df, actual_df

def plot_date_aware_predictions(gsphar_pred, gsphar_actual, garch_pred, garch_actual, market_index):
    """
    Plot predictions from both models for a given market index using Plotly.

    Args:
        gsphar_pred (pd.DataFrame): GSPHAR predictions with datetime index
        gsphar_actual (pd.DataFrame): GSPHAR actuals with datetime index
        garch_pred (pd.DataFrame): GARCH predictions with datetime index
        garch_actual (pd.DataFrame): GARCH actuals with datetime index
        market_index (str): Market index to plot
    """
    import plotly.graph_objects as go

    # Set Plotly as the backend for pandas plotting
    pd.options.plotting.backend = "plotly"

    # Find common date range
    gsphar_dates = gsphar_actual.index
    garch_dates = garch_actual.index
    common_dates = gsphar_dates.intersection(garch_dates)

    # Filter data to common date range
    gsphar_actual_common = gsphar_actual.loc[common_dates, market_index]
    gsphar_pred_common = gsphar_pred.loc[common_dates, market_index]
    garch_pred_common = garch_pred.loc[common_dates, market_index]

    # Create DataFrame with aligned data
    df_plot = pd.DataFrame({
        'Actual': gsphar_actual_common,
        'GSPHAR': gsphar_pred_common,
        'GARCH': garch_pred_common
    }, index=common_dates)

    # Create the plot
    fig = df_plot.plot(
        title=f'Volatility Predictions for {market_index}',
        labels=dict(index="Date", value="Volatility"),
        template="plotly_white"
    )

    # Update layout for better appearance
    fig.update_layout(
        height=600,
        width=1000,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )

    # Add grid and format x-axis
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey',
                    tickangle=45, tickformat='%Y-%m-%d')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')

    # Show the plot
    fig.show()

# Example usage in a notebook:
"""
# Replace the existing dataloader creation with:
from notebooks.date_aware_integration import integrate_with_notebook, plot_date_aware_predictions

# Create date-aware predictions
gsphar_pred_df, gsphar_actual_df = integrate_with_notebook(
    train_dataset_raw, test_dataset_raw, [1, 5, 22], h, batch_size,
    trained_model, market_indices_list
)

# Later, replace the plotting function with:
for market_index in subset_indices:
    plot_date_aware_predictions(
        gsphar_pred_df,
        gsphar_actual_df,
        garch_pred_df,
        garch_actual_df,
        market_index
    )
"""
