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

    elif args.command == 'list':
        # List universes
        universes = list_universes()

        if universes:
            print("\nAvailable Cryptocurrency Universes:")
            print("=" * 80)
            for name, metadata in universes.items():
                print(f"Name: {name}")
                print(f"  Path: {metadata['file_path']}")
                print(f"  Created: {metadata['created_at']}")
                print(f"  Symbols: {metadata['num_symbols']}")
                if 'description' in metadata:
                    print(f"  Description: {metadata['description']}")
                print("-" * 80)
        else:
            print("\nNo cryptocurrency universes found.")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
