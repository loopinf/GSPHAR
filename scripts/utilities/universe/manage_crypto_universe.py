#!/usr/bin/env python
"""
Script to manage cryptocurrency universes based on spillover impacts.
This script creates, updates, and manages lists of cryptocurrencies with high spillover impacts.
"""

import pandas as pd
import numpy as np
import os
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define constants
UNIVERSE_DIR = os.path.join('data', 'universe')
DEFAULT_METADATA_FILE = os.path.join(UNIVERSE_DIR, 'crypto_universe_metadata.json')


def ensure_directories():
    """
    Ensure that the necessary directories exist.
    """
    os.makedirs(UNIVERSE_DIR, exist_ok=True)
    logger.info(f"Ensured directory exists: {UNIVERSE_DIR}")


def load_spillover_data(spillover_file):
    """
    Load spillover data from a CSV file.

    Args:
        spillover_file (str): Path to the spillover data file.

    Returns:
        pd.DataFrame: DataFrame containing spillover data.
    """
    logger.info(f"Loading spillover data from {spillover_file}")

    if spillover_file.endswith('.csv'):
        # Read the CSV file
        df = pd.read_csv(spillover_file, index_col=0)

        # Ensure all values are numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    elif spillover_file.endswith('.npy'):
        # If it's a numpy array, we need to convert it to a DataFrame
        spillover_matrix = np.load(spillover_file)

        # Try to find a symbols file
        symbols_file = os.path.join(os.path.dirname(spillover_file), 'top20_symbols.txt')
        if os.path.exists(symbols_file):
            with open(symbols_file, 'r') as f:
                symbols = [line.strip() for line in f.readlines()]

            # Create a DataFrame with the symbols
            df = pd.DataFrame(spillover_matrix, index=symbols, columns=symbols)
        else:
            # If no symbols file, use generic names
            n = spillover_matrix.shape[0]
            symbols = [f"Symbol_{i}" for i in range(n)]
            df = pd.DataFrame(spillover_matrix, index=symbols, columns=symbols)
    else:
        raise ValueError("Unsupported file format. Please use .csv or .npy")

    logger.info(f"Loaded spillover data with shape {df.shape}")
    return df


def calculate_spillover_metrics(spillover_df):
    """
    Calculate various spillover metrics for each cryptocurrency.

    Args:
        spillover_df (pd.DataFrame): DataFrame containing spillover data.

    Returns:
        pd.DataFrame: DataFrame with spillover metrics.
    """
    logger.info("Calculating spillover metrics")

    # Calculate total spillover impact (sum of row)
    spillover_to_others = spillover_df.sum(axis=1)

    # Calculate total spillover received (sum of column)
    spillover_from_others = spillover_df.sum(axis=0)

    # Calculate net spillover (impact - received)
    net_spillover = spillover_to_others - spillover_from_others

    # Create a DataFrame with the metrics
    metrics_df = pd.DataFrame({
        'symbol': spillover_df.index,
        'spillover_to_others': spillover_to_others.values,
        'spillover_from_others': spillover_from_others.values,
        'net_spillover': net_spillover.values,
        'total_spillover': (spillover_to_others + spillover_from_others).values
    })

    logger.info(f"Calculated spillover metrics for {len(metrics_df)} symbols")
    return metrics_df


def calculate_pairwise_metrics(spillover_df):
    """
    Calculate pairwise spillover metrics between cryptocurrencies.

    Args:
        spillover_df (pd.DataFrame): DataFrame containing spillover data.

    Returns:
        pd.DataFrame: DataFrame with pairwise spillover metrics.
    """
    logger.info("Calculating pairwise spillover metrics")

    symbols = spillover_df.index.tolist()
    pairs = []

    for i, symbol1 in enumerate(symbols):
        for j, symbol2 in enumerate(symbols):
            if i < j:  # Avoid duplicates and self-pairs
                # Get spillover values in both directions
                spillover_1_to_2 = spillover_df.loc[symbol1, symbol2]
                spillover_2_to_1 = spillover_df.loc[symbol2, symbol1]

                # Calculate metrics
                bidirectional_sum = spillover_1_to_2 + spillover_2_to_1
                bidirectional_product = spillover_1_to_2 * spillover_2_to_1
                bidirectional_min = min(spillover_1_to_2, spillover_2_to_1)
                bidirectional_max = max(spillover_1_to_2, spillover_2_to_1)
                bidirectional_ratio = bidirectional_max / bidirectional_min if bidirectional_min > 0 else float('inf')
                bidirectional_balance = 1 - abs(spillover_1_to_2 - spillover_2_to_1) / bidirectional_sum if bidirectional_sum > 0 else 0

                # Add to pairs list
                pairs.append({
                    'symbol1': symbol1,
                    'symbol2': symbol2,
                    'spillover_1_to_2': spillover_1_to_2,
                    'spillover_2_to_1': spillover_2_to_1,
                    'bidirectional_sum': bidirectional_sum,
                    'bidirectional_product': bidirectional_product,
                    'bidirectional_min': bidirectional_min,
                    'bidirectional_max': bidirectional_max,
                    'bidirectional_ratio': bidirectional_ratio,
                    'bidirectional_balance': bidirectional_balance
                })

    # Create DataFrame from pairs list
    pairs_df = pd.DataFrame(pairs)

    logger.info(f"Calculated pairwise metrics for {len(pairs_df)} cryptocurrency pairs")
    return pairs_df


def create_top_pairs(pairs_df, top_n=10, method='bidirectional_sum'):
    """
    Create a list of top cryptocurrency pairs based on pairwise metrics.

    Args:
        pairs_df (pd.DataFrame): DataFrame with pairwise spillover metrics.
        top_n (int): Number of pairs to include.
        method (str): Method to rank pairs.
            Options: 'bidirectional_sum', 'bidirectional_product', 'bidirectional_min', 'bidirectional_balance'

    Returns:
        pd.DataFrame: DataFrame containing the top pairs.
    """
    logger.info(f"Creating top {top_n} pairs using method: {method}")

    # Select the ranking method
    if method not in ['bidirectional_sum', 'bidirectional_product', 'bidirectional_min', 'bidirectional_balance']:
        raise ValueError("Unsupported method. Use 'bidirectional_sum', 'bidirectional_product', 'bidirectional_min', or 'bidirectional_balance'")

    # Sort by the selected method and take the top N
    top_pairs_df = pairs_df.sort_values(by=method, ascending=False).head(top_n).reset_index(drop=True)

    logger.info(f"Created list of top {len(top_pairs_df)} cryptocurrency pairs")
    return top_pairs_df


def create_universe(metrics_df, top_n=20, method='to_others'):
    """
    Create a universe of cryptocurrencies based on spillover metrics.

    Args:
        metrics_df (pd.DataFrame): DataFrame with spillover metrics.
        top_n (int): Number of cryptocurrencies to include in the universe.
        method (str): Method to rank cryptocurrencies.
            Options: 'to_others', 'from_others', 'net', 'total'

    Returns:
        pd.DataFrame: DataFrame containing the universe.
    """
    logger.info(f"Creating universe with top {top_n} symbols using method: {method}")

    # Select the ranking method
    if method == 'to_others':
        ranking_col = 'spillover_to_others'
    elif method == 'from_others':
        ranking_col = 'spillover_from_others'
    elif method == 'net':
        ranking_col = 'net_spillover'
    elif method == 'total':
        ranking_col = 'total_spillover'
    else:
        raise ValueError("Unsupported method. Use 'to_others', 'from_others', 'net', or 'total'")

    # Sort by the selected method and take the top N
    universe_df = metrics_df.sort_values(by=ranking_col, ascending=False).head(top_n).reset_index(drop=True)

    logger.info(f"Created universe with {len(universe_df)} symbols")
    return universe_df


def save_universe(universe_df, output_file, metadata=None):
    """
    Save the universe to a file and update metadata.

    Args:
        universe_df (pd.DataFrame): DataFrame containing the universe.
        output_file (str): Path to save the universe.
        metadata (dict, optional): Additional metadata to save.

    Returns:
        str: Path to the saved universe file.
    """
    logger.info(f"Saving universe to {output_file}")

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save the universe
    universe_df.to_csv(output_file, index=False)

    # Create a symbols-only text file
    symbols_file = os.path.splitext(output_file)[0] + '_symbols.txt'
    with open(symbols_file, 'w') as f:
        for symbol in universe_df['symbol']:
            f.write(f"{symbol}\n")

    # Update metadata
    update_metadata(output_file, universe_df, metadata)

    logger.info(f"Saved universe to {output_file} and symbols to {symbols_file}")
    return output_file


def save_pairs(pairs_df, output_file, metadata=None):
    """
    Save the cryptocurrency pairs to a file and update metadata.

    Args:
        pairs_df (pd.DataFrame): DataFrame containing the pairs.
        output_file (str): Path to save the pairs.
        metadata (dict, optional): Additional metadata to save.

    Returns:
        str: Path to the saved pairs file.
    """
    logger.info(f"Saving pairs to {output_file}")

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save the pairs
    pairs_df.to_csv(output_file, index=False)

    # Create a pairs-only text file
    pairs_file = os.path.splitext(output_file)[0] + '_list.txt'
    with open(pairs_file, 'w') as f:
        for _, row in pairs_df.iterrows():
            f.write(f"{row['symbol1']}-{row['symbol2']}\n")

    # Update metadata
    pairs_metadata = {
        'file_path': output_file,
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'num_pairs': len(pairs_df),
        'pairs': [f"{row['symbol1']}-{row['symbol2']}" for _, row in pairs_df.iterrows()]
    }

    # Add additional metadata if provided
    if metadata:
        pairs_metadata.update(metadata)

    # Save the metadata
    metadata_file = DEFAULT_METADATA_FILE

    # Load existing metadata if it exists
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            all_metadata = json.load(f)
    else:
        all_metadata = {}

    # Update the metadata
    universe_name = os.path.basename(output_file)
    all_metadata[universe_name] = pairs_metadata

    # Save the metadata
    with open(metadata_file, 'w') as f:
        json.dump(all_metadata, f, indent=2)

    logger.info(f"Saved pairs to {output_file} and pairs list to {pairs_file}")
    return output_file


def update_metadata(universe_file, universe_df, metadata=None, metadata_file=DEFAULT_METADATA_FILE):
    """
    Update the metadata for the universe.

    Args:
        universe_file (str): Path to the universe file.
        universe_df (pd.DataFrame): DataFrame containing the universe.
        metadata (dict, optional): Additional metadata to save.
        metadata_file (str, optional): Path to the metadata file.

    Returns:
        None
    """
    logger.info(f"Updating metadata in {metadata_file}")

    # Load existing metadata if it exists
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            all_metadata = json.load(f)
    else:
        all_metadata = {}

    # Create metadata for this universe
    universe_name = os.path.basename(universe_file)
    universe_metadata = {
        'file_path': universe_file,
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'num_symbols': len(universe_df),
        'symbols': universe_df['symbol'].tolist()
    }

    # Add additional metadata if provided
    if metadata:
        universe_metadata.update(metadata)

    # Update the metadata
    all_metadata[universe_name] = universe_metadata

    # Save the metadata
    with open(metadata_file, 'w') as f:
        json.dump(all_metadata, f, indent=2)

    logger.info(f"Updated metadata for {universe_name}")


def list_universes(metadata_file=DEFAULT_METADATA_FILE):
    """
    List all available universes.

    Args:
        metadata_file (str, optional): Path to the metadata file.

    Returns:
        dict: Dictionary containing universe metadata.
    """
    if not os.path.exists(metadata_file):
        logger.warning(f"Metadata file {metadata_file} does not exist")
        return {}

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    logger.info(f"Found {len(metadata)} universes")
    return metadata


def main():
    """
    Main function to manage cryptocurrency universes.
    """
    parser = argparse.ArgumentParser(description='Manage cryptocurrency universes based on spillover impacts.')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Create universe command
    create_parser = subparsers.add_parser('create', help='Create a new universe')
    create_parser.add_argument('--spillover-file', type=str, required=True,
                              help='Path to the spillover data file.')
    create_parser.add_argument('--output-file', type=str, default=None,
                              help='Path to save the universe. If not provided, a default name will be used.')
    create_parser.add_argument('--top-n', type=int, default=20,
                              help='Number of cryptocurrencies to include in the universe.')
    create_parser.add_argument('--method', type=str, default='to_others',
                              choices=['to_others', 'from_others', 'net', 'total'],
                              help='Method to rank cryptocurrencies.')
    create_parser.add_argument('--description', type=str, default=None,
                              help='Description of the universe.')

    # Create pairs command
    pairs_parser = subparsers.add_parser('pairs', help='Create a list of top cryptocurrency pairs')
    pairs_parser.add_argument('--spillover-file', type=str, required=True,
                             help='Path to the spillover data file.')
    pairs_parser.add_argument('--output-file', type=str, default=None,
                             help='Path to save the pairs. If not provided, a default name will be used.')
    pairs_parser.add_argument('--top-n', type=int, default=10,
                             help='Number of cryptocurrency pairs to include.')
    pairs_parser.add_argument('--method', type=str, default='bidirectional_sum',
                             choices=['bidirectional_sum', 'bidirectional_product', 'bidirectional_min', 'bidirectional_balance'],
                             help='Method to rank cryptocurrency pairs.')
    pairs_parser.add_argument('--description', type=str, default=None,
                             help='Description of the pairs list.')

    # List universes command
    list_parser = subparsers.add_parser('list', help='List all available universes')

    # Parse arguments
    args = parser.parse_args()

    # Ensure directories exist
    ensure_directories()

    # Execute the command
    if args.command == 'create':
        # Load spillover data
        spillover_df = load_spillover_data(args.spillover_file)

        # Calculate spillover metrics
        metrics_df = calculate_spillover_metrics(spillover_df)

        # Create universe
        universe_df = create_universe(metrics_df, top_n=args.top_n, method=args.method)

        # Generate output file name if not provided
        if args.output_file is None:
            output_file = os.path.join(UNIVERSE_DIR, f"crypto_spillover_top{args.top_n}_{args.method}.csv")
        else:
            output_file = args.output_file

        # Create metadata
        metadata = {
            'description': args.description or f"Top {args.top_n} cryptocurrencies by {args.method} spillover",
            'method': args.method,
            'top_n': args.top_n,
            'spillover_file': args.spillover_file
        }

        # Save universe
        save_universe(universe_df, output_file, metadata)

        logger.info(f"Universe creation completed successfully")

    elif args.command == 'pairs':
        # Load spillover data
        spillover_df = load_spillover_data(args.spillover_file)

        # Calculate pairwise metrics
        pairs_df = calculate_pairwise_metrics(spillover_df)

        # Create top pairs
        top_pairs_df = create_top_pairs(pairs_df, top_n=args.top_n, method=args.method)

        # Generate output file name if not provided
        if args.output_file is None:
            output_file = os.path.join(UNIVERSE_DIR, f"crypto_pairs_top{args.top_n}_{args.method}.csv")
        else:
            output_file = args.output_file

        # Create metadata
        metadata = {
            'description': args.description or f"Top {args.top_n} cryptocurrency pairs by {args.method}",
            'method': args.method,
            'top_n': args.top_n,
            'spillover_file': args.spillover_file
        }

        # Save pairs
        save_pairs(top_pairs_df, output_file, metadata)

        logger.info(f"Pairs creation completed successfully")

        # Print the top pairs
        print("\nTop Cryptocurrency Pairs:")
        print("=" * 80)
        for i, (_, row) in enumerate(top_pairs_df.iterrows()):
            print(f"{i+1}. {row['symbol1']} ⟷ {row['symbol2']}")
            print(f"   {row['symbol1']} → {row['symbol2']}: {row['spillover_1_to_2']:.4f}")
            print(f"   {row['symbol2']} → {row['symbol1']}: {row['spillover_2_to_1']:.4f}")
            print(f"   Sum: {row['bidirectional_sum']:.4f}, Product: {row['bidirectional_product']:.4f}")
            print(f"   Balance: {row['bidirectional_balance']:.4f} (1.0 = perfectly balanced)")
            print("-" * 80)

    elif args.command == 'list':
        # List universes
        universes = list_universes()

        if universes:
            print("\nAvailable Cryptocurrency Universes and Pairs:")
            print("=" * 80)
            for name, metadata in universes.items():
                print(f"Name: {name}")
                print(f"  Path: {metadata['file_path']}")
                print(f"  Created: {metadata['created_at']}")

                if 'num_symbols' in metadata:
                    print(f"  Symbols: {metadata['num_symbols']}")
                elif 'num_pairs' in metadata:
                    print(f"  Pairs: {metadata['num_pairs']}")

                if 'description' in metadata:
                    print(f"  Description: {metadata['description']}")
                print("-" * 80)
        else:
            print("\nNo cryptocurrency universes or pairs found.")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
