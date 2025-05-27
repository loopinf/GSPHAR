#!/usr/bin/env python
"""
Script to compare 1-hour percentage change with 1-hour realized volatility.
This script loads both datasets, aligns them, and creates visualizations to compare them.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import logging
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(rv_file, pct_change_file=None, calculate_pct_change=False, price_file=None):
    """
    Load the realized volatility and percentage change data.

    Args:
        rv_file (str): Path to the realized volatility file.
        pct_change_file (str, optional): Path to the percentage change file.
        calculate_pct_change (bool): Whether to calculate percentage change from price data.
        price_file (str, optional): Path to the price data file if calculating percentage change.

    Returns:
        tuple: (rv_df, pct_change_df) - DataFrames containing the data.
    """
    logger.info(f"Loading realized volatility data from {rv_file}")

    # Load realized volatility data
    if rv_file.endswith('.csv'):
        rv_df = pd.read_csv(rv_file, index_col=0, parse_dates=True)
    elif rv_file.endswith('.parquet'):
        rv_df = pd.read_parquet(rv_file)
    else:
        raise ValueError("Unsupported RV file format. Please use .csv or .parquet")

    # Load or calculate percentage change data
    if pct_change_file is not None:
        logger.info(f"Loading percentage change data from {pct_change_file}")

        if pct_change_file.endswith('.csv'):
            pct_change_df = pd.read_csv(pct_change_file, index_col=0, parse_dates=True)
        elif pct_change_file.endswith('.parquet'):
            pct_change_df = pd.read_parquet(pct_change_file)
        else:
            raise ValueError("Unsupported percentage change file format. Please use .csv or .parquet")

    elif calculate_pct_change and price_file is not None:
        logger.info(f"Calculating percentage change from price data in {price_file}")

        # Load price data
        if price_file.endswith('.csv'):
            price_df = pd.read_csv(price_file, index_col=0, parse_dates=True)
        elif price_file.endswith('.parquet'):
            price_df = pd.read_parquet(price_file)
        else:
            raise ValueError("Unsupported price file format. Please use .csv or .parquet")

        # Calculate 1-hour percentage change
        pct_change_df = price_df.pct_change(freq='1h') * 100

    else:
        raise ValueError("Either provide a percentage change file or set calculate_pct_change=True and provide a price file")

    logger.info(f"Loaded realized volatility data with shape {rv_df.shape}")
    logger.info(f"Loaded/calculated percentage change data with shape {pct_change_df.shape}")

    return rv_df, pct_change_df


def align_data(rv_df, pct_change_df):
    """
    Align the realized volatility and percentage change data to have the same index and columns.

    Args:
        rv_df (pd.DataFrame): DataFrame containing realized volatility data.
        pct_change_df (pd.DataFrame): DataFrame containing percentage change data.

    Returns:
        tuple: (rv_aligned, pct_change_aligned) - Aligned DataFrames.
    """
    logger.info("Aligning data")

    # Find common columns
    common_columns = sorted(list(set(rv_df.columns) & set(pct_change_df.columns)))
    logger.info(f"Found {len(common_columns)} common symbols")

    # Find common dates
    common_index = rv_df.index.intersection(pct_change_df.index)
    logger.info(f"Found {len(common_index)} common timestamps")

    # Align data
    rv_aligned = rv_df.loc[common_index, common_columns]
    pct_change_aligned = pct_change_df.loc[common_index, common_columns]

    logger.info(f"Aligned data shape: {rv_aligned.shape}")

    return rv_aligned, pct_change_aligned


def calculate_statistics(rv_df, pct_change_df):
    """
    Calculate statistics to compare realized volatility and percentage change.

    Args:
        rv_df (pd.DataFrame): DataFrame containing realized volatility data.
        pct_change_df (pd.DataFrame): DataFrame containing percentage change data.

    Returns:
        pd.DataFrame: DataFrame containing statistics.
    """
    logger.info("Calculating statistics")

    stats = {}

    # Calculate statistics for each symbol
    for column in rv_df.columns:
        rv = rv_df[column]
        pct = pct_change_df[column].abs()  # Use absolute percentage change for comparison

        # Calculate correlation
        correlation = rv.corr(pct)

        # Calculate ratio of means
        mean_ratio = rv.mean() / pct.mean() if pct.mean() != 0 else np.nan

        # Calculate ratio of standard deviations
        std_ratio = rv.std() / pct.std() if pct.std() != 0 else np.nan

        # Calculate quantile ratios
        q25_ratio = rv.quantile(0.25) / pct.quantile(0.25) if pct.quantile(0.25) != 0 else np.nan
        q50_ratio = rv.quantile(0.50) / pct.quantile(0.50) if pct.quantile(0.50) != 0 else np.nan
        q75_ratio = rv.quantile(0.75) / pct.quantile(0.75) if pct.quantile(0.75) != 0 else np.nan
        q90_ratio = rv.quantile(0.90) / pct.quantile(0.90) if pct.quantile(0.90) != 0 else np.nan
        q95_ratio = rv.quantile(0.95) / pct.quantile(0.95) if pct.quantile(0.95) != 0 else np.nan
        q99_ratio = rv.quantile(0.99) / pct.quantile(0.99) if pct.quantile(0.99) != 0 else np.nan

        # Store statistics
        stats[column] = {
            'correlation': correlation,
            'mean_ratio': mean_ratio,
            'std_ratio': std_ratio,
            'q25_ratio': q25_ratio,
            'q50_ratio': q50_ratio,
            'q75_ratio': q75_ratio,
            'q90_ratio': q90_ratio,
            'q95_ratio': q95_ratio,
            'q99_ratio': q99_ratio,
            'rv_mean': rv.mean(),
            'pct_mean': pct.mean(),
            'rv_std': rv.std(),
            'pct_std': pct.std()
        }

    # Convert to DataFrame
    stats_df = pd.DataFrame.from_dict(stats, orient='index')

    logger.info("Statistics calculation completed")
    return stats_df


def create_visualizations(rv_df, pct_change_df, stats_df, output_dir, top_n=5, normalize=True):
    """
    Create visualizations to compare realized volatility and percentage change.

    Args:
        rv_df (pd.DataFrame): DataFrame containing realized volatility data.
        pct_change_df (pd.DataFrame): DataFrame containing percentage change data.
        stats_df (pd.DataFrame): DataFrame containing statistics.
        output_dir (str): Directory to save the visualizations.
        top_n (int): Number of top symbols to visualize.
        normalize (bool): Whether to normalize the data to align scales.

    Returns:
        None
    """
    logger.info("Creating visualizations")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Select top symbols by correlation
    top_symbols = stats_df.sort_values('correlation', ascending=False).head(top_n).index.tolist()

    # Create time series plots for top symbols
    for symbol in top_symbols:
        # Create two versions of the plot: one with dual y-axes and one with normalized data

        # 1. Dual y-axes plot (original)
        fig_dual = make_subplots(specs=[[{"secondary_y": True}]])

        # Add realized volatility
        fig_dual.add_trace(
            go.Scatter(x=rv_df.index, y=rv_df[symbol], name="Realized Volatility", line=dict(color='blue')),
            secondary_y=False
        )

        # Add absolute percentage change
        fig_dual.add_trace(
            go.Scatter(x=pct_change_df.index, y=pct_change_df[symbol].abs(), name="Absolute % Change", line=dict(color='red')),
            secondary_y=True
        )

        # Set titles and labels
        fig_dual.update_layout(
            title=f"{symbol}: Realized Volatility vs Absolute Percentage Change (Dual Y-Axes)",
            xaxis_title="Date",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        fig_dual.update_yaxes(title_text="Realized Volatility", secondary_y=False)
        fig_dual.update_yaxes(title_text="Absolute % Change", secondary_y=True)

        # Save figure
        fig_dual.write_html(os.path.join(output_dir, f"{symbol}_rv_vs_pct_change_dual.html"))

        # 2. Normalized plot (same scale)
        fig_norm = go.Figure()

        # Get the data
        rv_series = rv_df[symbol]
        pct_series = pct_change_df[symbol].abs()

        # Normalize both series to 0-1 range
        rv_norm = (rv_series - rv_series.min()) / (rv_series.max() - rv_series.min())
        pct_norm = (pct_series - pct_series.min()) / (pct_series.max() - pct_series.min())

        # Add normalized realized volatility
        fig_norm.add_trace(
            go.Scatter(x=rv_df.index, y=rv_norm, name="Realized Volatility (Normalized)", line=dict(color='blue'))
        )

        # Add normalized absolute percentage change
        fig_norm.add_trace(
            go.Scatter(x=pct_change_df.index, y=pct_norm, name="Absolute % Change (Normalized)", line=dict(color='red'))
        )

        # Set titles and labels
        fig_norm.update_layout(
            title=f"{symbol}: Normalized Realized Volatility vs Absolute Percentage Change",
            xaxis_title="Date",
            yaxis_title="Normalized Value (0-1 Scale)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        # Save figure
        fig_norm.write_html(os.path.join(output_dir, f"{symbol}_rv_vs_pct_change_normalized.html"))

        # 3. Z-score normalized plot (standardized)
        fig_z = go.Figure()

        # Standardize both series (z-score normalization)
        rv_z = (rv_series - rv_series.mean()) / rv_series.std()
        pct_z = (pct_series - pct_series.mean()) / pct_series.std()

        # Add z-score normalized realized volatility
        fig_z.add_trace(
            go.Scatter(x=rv_df.index, y=rv_z, name="Realized Volatility (Z-Score)", line=dict(color='blue'))
        )

        # Add z-score normalized absolute percentage change
        fig_z.add_trace(
            go.Scatter(x=pct_change_df.index, y=pct_z, name="Absolute % Change (Z-Score)", line=dict(color='red'))
        )

        # Set titles and labels
        fig_z.update_layout(
            title=f"{symbol}: Z-Score Normalized Realized Volatility vs Absolute Percentage Change",
            xaxis_title="Date",
            yaxis_title="Z-Score (Standard Deviations from Mean)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        # Save figure
        fig_z.write_html(os.path.join(output_dir, f"{symbol}_rv_vs_pct_change_zscore.html"))

        logger.info(f"Created time series plots for {symbol}")

    # Create scatter plots for top symbols
    for symbol in top_symbols:
        # Original scatter plot
        fig_scatter = px.scatter(
            x=rv_df[symbol],
            y=pct_change_df[symbol].abs(),
            title=f"{symbol}: Realized Volatility vs Absolute Percentage Change",
            labels={"x": "Realized Volatility", "y": "Absolute % Change"},
            trendline="ols"
        )

        # Save figure
        fig_scatter.write_html(os.path.join(output_dir, f"{symbol}_scatter.html"))

        # Normalized scatter plot
        rv_series = rv_df[symbol]
        pct_series = pct_change_df[symbol].abs()

        # Normalize both series to 0-1 range
        rv_norm = (rv_series - rv_series.min()) / (rv_series.max() - rv_series.min())
        pct_norm = (pct_series - pct_series.min()) / (pct_series.max() - pct_series.min())

        fig_scatter_norm = px.scatter(
            x=rv_norm,
            y=pct_norm,
            title=f"{symbol}: Normalized Realized Volatility vs Absolute Percentage Change",
            labels={"x": "Realized Volatility (Normalized 0-1)", "y": "Absolute % Change (Normalized 0-1)"},
            trendline="ols"
        )

        # Save figure
        fig_scatter_norm.write_html(os.path.join(output_dir, f"{symbol}_scatter_normalized.html"))

        logger.info(f"Created scatter plots for {symbol}")

    # Create heatmap of correlations
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        stats_df[['correlation']].sort_values('correlation', ascending=False),
        annot=True,
        cmap='coolwarm',
        vmin=-1,
        vmax=1
    )
    plt.title('Correlation between Realized Volatility and Absolute Percentage Change')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=300)
    plt.close()

    logger.info("Created correlation heatmap")

    # Create summary statistics plot
    plt.figure(figsize=(14, 8))

    # Plot mean ratio
    plt.subplot(2, 2, 1)
    sns.histplot(stats_df['mean_ratio'].dropna(), kde=True)
    plt.title('Distribution of Mean Ratios (RV/|%Change|)')
    plt.axvline(stats_df['mean_ratio'].median(), color='red', linestyle='--')

    # Plot std ratio
    plt.subplot(2, 2, 2)
    sns.histplot(stats_df['std_ratio'].dropna(), kde=True)
    plt.title('Distribution of Std Dev Ratios (RV/|%Change|)')
    plt.axvline(stats_df['std_ratio'].median(), color='red', linestyle='--')

    # Plot correlation
    plt.subplot(2, 2, 3)
    sns.histplot(stats_df['correlation'].dropna(), kde=True)
    plt.title('Distribution of Correlations')
    plt.axvline(stats_df['correlation'].median(), color='red', linestyle='--')

    # Plot quantile ratios
    plt.subplot(2, 2, 4)
    quantiles = ['q25_ratio', 'q50_ratio', 'q75_ratio', 'q90_ratio', 'q95_ratio', 'q99_ratio']
    medians = [stats_df[q].median() for q in quantiles]
    plt.bar(range(len(quantiles)), medians)
    plt.xticks(range(len(quantiles)), [q.split('_')[0] for q in quantiles], rotation=45)
    plt.title('Median Quantile Ratios (RV/|%Change|)')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_statistics.png'), dpi=300)
    plt.close()

    logger.info("Created summary statistics plot")

    # Save statistics to CSV
    stats_df.to_csv(os.path.join(output_dir, 'rv_vs_pct_change_stats.csv'))

    logger.info(f"Visualizations saved to {output_dir}")


def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(description='Compare 1-hour realized volatility with 1-hour percentage change.')
    parser.add_argument('--rv-file', type=str, required=True,
                        help='Path to the realized volatility file.')
    parser.add_argument('--pct-change-file', type=str, default=None,
                        help='Path to the percentage change file.')
    parser.add_argument('--price-file', type=str, default=None,
                        help='Path to the price file if calculating percentage change.')
    parser.add_argument('--calculate-pct-change', action='store_true',
                        help='Whether to calculate percentage change from price data.')
    parser.add_argument('--output-dir', type=str, default='plots/rv_vs_pct_change',
                        help='Directory to save the visualizations.')
    parser.add_argument('--top-n', type=int, default=5,
                        help='Number of top symbols to visualize.')
    parser.add_argument('--normalize', action='store_true', default=True,
                        help='Whether to normalize the data to align scales.')

    args = parser.parse_args()

    # Load data
    rv_df, pct_change_df = load_data(
        args.rv_file,
        args.pct_change_file,
        args.calculate_pct_change,
        args.price_file
    )

    # Align data
    rv_aligned, pct_change_aligned = align_data(rv_df, pct_change_df)

    # Calculate statistics
    stats_df = calculate_statistics(rv_aligned, pct_change_aligned)

    # Create visualizations
    create_visualizations(
        rv_aligned,
        pct_change_aligned,
        stats_df,
        args.output_dir,
        args.top_n,
        args.normalize
    )

    logger.info("Comparison completed successfully")


if __name__ == '__main__':
    main()
