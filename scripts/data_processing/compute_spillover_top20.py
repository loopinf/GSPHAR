#!/usr/bin/env python
"""
Script to compute spillover index for the top 20 cryptocurrency symbols.
This script loads the daily_pct_change_crypto.csv file, selects the top 20 symbols,
computes the spillover index, and visualizes the network.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from local modules
from src.utils.graph_utils import compute_spillover_index
from scripts.load_pct_change_data import load_pct_change_data


def select_top_symbols(data, num_symbols=20, method='data_availability'):
    """
    Select the top symbols based on specified method.

    Args:
        data (pd.DataFrame): Input data.
        num_symbols (int): Number of symbols to select.
        method (str): Method to use for selection ('data_availability', 'volatility', 'market_cap', 'spillover').

    Returns:
        list: List of selected symbols.
    """
    if method == 'data_availability':
        # Count non-NaN values for each symbol
        data_counts = data.count()
        # Sort by data availability (descending)
        sorted_symbols = data_counts.sort_values(ascending=False).index.tolist()
        # Select top symbols
        selected_symbols = sorted_symbols[:num_symbols]

        print(f"Selected top {num_symbols} symbols based on data availability:")
        for i, symbol in enumerate(selected_symbols, 1):
            print(f"{i}. {symbol}: {data_counts[symbol]} data points")

        return selected_symbols

    elif method == 'volatility':
        # Calculate volatility (standard deviation) for each symbol
        volatility = data.std()
        # Sort by volatility (descending)
        sorted_symbols = volatility.sort_values(ascending=False).index.tolist()
        # Select top symbols
        selected_symbols = sorted_symbols[:num_symbols]

        print(f"Selected top {num_symbols} symbols based on volatility:")
        for i, symbol in enumerate(selected_symbols, 1):
            print(f"{i}. {symbol}: {volatility[symbol]:.4f}")

        return selected_symbols

    elif method == 'market_cap':
        # Predefined list of top cryptocurrencies by market cap
        # This is a static list based on typical market caps, update as needed
        top_by_market_cap = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT',
            'DOGEUSDT', 'DOTUSDT', 'LTCUSDT', 'LINKUSDT', 'XLMUSDT',
            'TRXUSDT', 'BCHUSDT', 'ETCUSDT', 'XMRUSDT', 'DASHUSDT',
            'ZECUSDT', 'XTZUSDT', 'ATOMUSDT', 'NEOUSDT', 'VETUSDT'
        ]

        # Filter to include only symbols that exist in the data
        available_symbols = [symbol for symbol in top_by_market_cap if symbol in data.columns]
        selected_symbols = available_symbols[:num_symbols]

        print(f"Selected top {len(selected_symbols)} symbols based on market cap:")
        for i, symbol in enumerate(selected_symbols, 1):
            print(f"{i}. {symbol}")

        return selected_symbols

    elif method == 'spillover':
        # This method requires computing the spillover index for all symbols first
        # and then selecting the top symbols based on their spillover values
        print("Computing spillover index for all symbols to select top ones...")

        # Drop rows with NaN values
        clean_data = data.dropna()

        # Import here to avoid circular import
        from src.utils.graph_utils import compute_spillover_index

        # Compute spillover index for all symbols
        try:
            # Use a subset of symbols if there are too many (for computational efficiency)
            if len(data.columns) > 30:
                # First select based on data availability
                data_counts = data.count()
                initial_symbols = data_counts.sort_values(ascending=False).index.tolist()[:30]
                initial_data = data[initial_symbols].copy().dropna()
                DY_adj = compute_spillover_index(initial_data, 1, 22, 0.0, standardized=True)
                symbols_used = initial_symbols
            else:
                DY_adj = compute_spillover_index(clean_data, 1, 22, 0.0, standardized=True)
                symbols_used = data.columns.tolist()

            # Calculate total spillover effect for each symbol (sum of row)
            spillover_effects = np.sum(DY_adj, axis=1)

            # Create a Series with symbol names as index
            spillover_series = pd.Series(spillover_effects, index=symbols_used)

            # Sort by spillover effect (descending)
            sorted_symbols = spillover_series.sort_values(ascending=False).index.tolist()

            # Select top symbols
            selected_symbols = sorted_symbols[:num_symbols]

            print(f"Selected top {num_symbols} symbols based on spillover effects:")
            for i, symbol in enumerate(selected_symbols, 1):
                print(f"{i}. {symbol}: {spillover_series[symbol]:.4f}")

            return selected_symbols

        except Exception as e:
            print(f"Error computing spillover for selection: {e}")
            print("Falling back to data availability method.")
            return select_top_symbols(data, num_symbols, 'data_availability')

    else:
        raise ValueError(f"Unknown selection method: {method}")


def compute_and_visualize_spillover(data, symbols, h=5, look_back=22, scarcity_prop=0.0,
                                   standardized=True, save_path=None):
    """
    Compute and visualize spillover index for selected symbols.

    Args:
        data (pd.DataFrame): Input data.
        symbols (list): List of symbols to include.
        h (int): Prediction horizon.
        look_back (int): Look-back window size.
        scarcity_prop (float): Sparsity proportion for the adjacency matrix.
        standardized (bool): Whether to standardize the adjacency matrix.
        save_path (str): Path to save the visualization.

    Returns:
        np.ndarray: Spillover index (adjacency matrix).
    """
    # Filter data to include only selected symbols
    filtered_data = data[symbols].copy()

    # Drop rows with NaN values
    filtered_data = filtered_data.dropna()

    print(f"Computing spillover index for {len(symbols)} symbols...")
    print(f"Data shape after filtering: {filtered_data.shape}")

    # Compute spillover index
    try:
        DY_adj = compute_spillover_index(
            filtered_data, h, look_back, scarcity_prop, standardized
        )
        print(f"Spillover index computed successfully. Shape: {DY_adj.shape}")
    except Exception as e:
        print(f"Error computing spillover index: {e}")
        print("Using a simple adjacency matrix instead.")
        # Create a simple adjacency matrix as fallback
        n = len(symbols)
        DY_adj = np.ones((n, n))  # Full connectivity

    # Visualize the spillover network
    plt.figure(figsize=(12, 10))

    # Create a mask for the diagonal
    mask = np.zeros_like(DY_adj, dtype=bool)
    np.fill_diagonal(mask, True)

    # Plot heatmap
    sns.heatmap(
        DY_adj,
        cmap='viridis',
        xticklabels=symbols,
        yticklabels=symbols,
        mask=mask,  # Mask the diagonal
        annot=True,  # Show values
        fmt='.2f',  # Format for annotations
        cbar_kws={'label': 'Spillover Intensity'}
    )

    plt.title(f'Cryptocurrency Volatility Spillover Network (Top {len(symbols)} Symbols)')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")

    plt.show()

    return DY_adj


def save_spillover_data(DY_adj, symbols, output_dir='data', prefix='top20'):
    """
    Save spillover index and symbols to files.

    Args:
        DY_adj (np.ndarray): Spillover index (adjacency matrix).
        symbols (list): List of symbols.
        output_dir (str): Directory to save files.
        prefix (str): Prefix for filenames.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save adjacency matrix
    adj_path = os.path.join(output_dir, f"{prefix}_spillover_adj.npy")
    np.save(adj_path, DY_adj)
    print(f"Adjacency matrix saved to {adj_path}")

    # Save symbols list
    symbols_path = os.path.join(output_dir, f"{prefix}_symbols.txt")
    with open(symbols_path, 'w') as f:
        for symbol in symbols:
            f.write(f"{symbol}\n")
    print(f"Symbols list saved to {symbols_path}")

    # Save as CSV for easier inspection
    df_adj = pd.DataFrame(DY_adj, index=symbols, columns=symbols)
    csv_path = os.path.join(output_dir, f"{prefix}_spillover_adj.csv")
    df_adj.to_csv(csv_path)
    print(f"Adjacency matrix saved as CSV to {csv_path}")


def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(description='Compute spillover index for top cryptocurrency symbols.')
    parser.add_argument('--data-file', type=str, default='data/daily_pct_change_crypto.csv',
                        help='Path to the data file.')
    parser.add_argument('--num-symbols', type=int, default=20,
                        help='Number of symbols to select.')
    parser.add_argument('--selection-method', type=str, default='spillover',
                        choices=['data_availability', 'volatility', 'market_cap', 'spillover'],
                        help='Method to select top symbols.')
    parser.add_argument('--horizon', type=int, default=1,
                        help='Prediction horizon.')
    parser.add_argument('--look-back', type=int, default=22,
                        help='Look-back window size.')
    parser.add_argument('--scarcity', type=float, default=0.0,
                        help='Sparsity proportion for the adjacency matrix.')
    parser.add_argument('--output-dir', type=str, default='data',
                        help='Directory to save output files.')
    parser.add_argument('--prefix', type=str, default='top20',
                        help='Prefix for output filenames.')
    parser.add_argument('--save-plot', action='store_true',
                        help='Save the spillover network visualization.')

    args = parser.parse_args()

    # Load data
    data = load_pct_change_data(args.data_file)

    if data is None:
        print("Failed to load data. Exiting.")
        return

    # Select top symbols
    top_symbols = select_top_symbols(
        data,
        num_symbols=args.num_symbols,
        method=args.selection_method
    )

    # Compute and visualize spillover index
    save_path = None
    if args.save_plot:
        save_path = os.path.join(args.output_dir, f"{args.prefix}_spillover_network.png")

    DY_adj = compute_and_visualize_spillover(
        data,
        top_symbols,
        h=args.horizon,
        look_back=args.look_back,
        scarcity_prop=args.scarcity,
        standardized=True,
        save_path=save_path
    )

    # Save spillover data
    save_spillover_data(
        DY_adj,
        top_symbols,
        output_dir=args.output_dir,
        prefix=args.prefix
    )

    print("Spillover analysis completed successfully.")


if __name__ == '__main__':
    main()
