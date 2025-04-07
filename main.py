import argparse
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import warnings
import os
from pathlib import Path
from typing import Tuple, Optional
warnings.filterwarnings('ignore')

# Import project modules
from data_utils import (load_and_split_data, compute_spillover_index,
                       prepare_lagged_features, create_model_input_dicts)
from model import GSPHAR, GSPHAR_Dataset
from train_utils import train_eval_model, load_model, predict_and_evaluate
from volatility_utils import calculate_volatility
from config.model_config import ModelConfig

def setup_args() -> argparse.Namespace:
    """Setup and parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, 
                       default='data/df_cl_5m.parquet',
                       help='Path to the input data file close prices')
    parser.add_argument('--n_symbols', type=int, default=38,
                       help='Number of symbols/assets to process')
    parser.add_argument('--vol_method', type=str, default='pct_change',
                       choices=['realized', 'pct_change'],
                       help='Volatility calculation method')
    parser.add_argument('--train', action='store_true',
                       help='Train new model instead of loading existing one')
    return parser.parse_args()

def setup_directories() -> None:
    """Create necessary directories if they don't exist"""
    for directory in ['cache', 'results', 'models']:
        Path(directory).mkdir(parents=True, exist_ok=True)

def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def process_raw_data(data_path: str, vol_method: str, n_symbols: int) -> pd.DataFrame:
    """Load and process raw data"""
    print(f"Loading data from {data_path}")
    raw_data = pd.read_parquet(data_path)
    raw_data = raw_data.loc['2021-09-01': ]   # too many NaNs before this date
    
    print(f"Calculating {vol_method} volatility...")
    volatility = calculate_volatility(raw_data, method=vol_method, freq='1H')
    
    # Handle NaN values
    print("Handling NaN values...")
    print(f"Original shape: {volatility.shape}")
    print(f"NaN count before cleaning: {volatility.isna().sum().sum()}")
    
    # Forward fill then backward fill for any remaining NaNs
    volatility = volatility.fillna(method='ffill').fillna(method='bfill')
    
    # Optional: Remove any rows that still have NaNs
    volatility = volatility.dropna()
    
    print(f"Final shape: {volatility.shape}")
    print(f"NaN count after cleaning: {volatility.isna().sum().sum()}")
    
    output_filename = f'data/volatility_{vol_method}_{n_symbols}.csv'
    volatility.to_csv(output_filename)
    print(f"Saved volatility data to {output_filename}")
    
    return volatility

def prepare_model_data(volatility: pd.DataFrame, config: ModelConfig) -> Tuple:
    """Prepare data for model training/evaluation"""
    # Split volatility data into train and test
    train_size = int(len(volatility) * 0.7)  # 70% for training
    train_df = volatility.iloc[:train_size]
    test_df = volatility.iloc[train_size:]
    
    market_indices_list = train_df.columns.tolist()
    
    # Store test dates before any processing
    test_dates = test_df.index
    
    # Compute spillover index using volatility data
    DY_adj = compute_spillover_index(
        train_df, config.h, config.look_back_window, 0.0, standardized=True
    )
    
    # Prepare lagged features
    train_df, test_df, market_indices_list = prepare_lagged_features(
        train_df, test_df, config.look_back_window, config.h
    )
    
    # Create model input dictionaries
    train_dict, test_dict, y_columns = create_model_input_dicts(
        train_df, test_df, market_indices_list
    )
    
    return (train_dict, test_dict, DY_adj, y_columns, 
            test_dates, market_indices_list)

def create_dataloaders(train_dict: dict, test_dict: dict, 
                      batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """Create PyTorch DataLoaders"""
    dataset_train = GSPHAR_Dataset(train_dict)
    dataset_test = GSPHAR_Dataset(test_dict)
    
    return (
        DataLoader(dataset_train, batch_size=batch_size, shuffle=True),
        DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    )

def save_results(results_df: pd.DataFrame, rv_hat: pd.DataFrame, 
                rv_true: pd.DataFrame, h: int) -> None:
    """Save model predictions and true values"""
    results_df.to_csv(f'results/predictions_h{h}.csv')
    rv_hat.to_csv(f'results/predictions_only_h{h}.csv')
    rv_true.to_csv(f'results/true_values_h{h}.csv')
    
    print(f"\nResults saved to:")
    print(f"- results/predictions_h{h}.csv (both predictions and true values)")
    print(f"- results/predictions_only_h{h}.csv (only predictions)")
    print(f"- results/true_values_h{h}.csv (only true values)")

def load_processed_data(cache_path: str) -> Tuple:
    """Load processed data from cache if it exists"""
    if os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        data = torch.load(cache_path)
        return data
    return (None,) * 6  # Return 6 None values to match the expected tuple size

def save_processed_data(train_dict: dict, test_dict: dict, 
                       DY_adj: np.ndarray, y_columns: list,
                       test_dates: pd.DatetimeIndex, 
                       market_indices_list: list,
                       cache_path: str) -> None:
    """Save processed data to cache"""
    print(f"Saving processed data to {cache_path}")
    data = (train_dict, test_dict, DY_adj, y_columns, 
            test_dates, market_indices_list)
    torch.save(data, cache_path)

def main():
    # Setup
    args = setup_args()
    config = ModelConfig()
    setup_directories()
    set_random_seeds()
    
    # Process data
    volatility = process_raw_data(args.data_path, args.vol_method, args.n_symbols)
    
    # Define cache path
    cache_path = f'cache/processed_data_h{config.h}_n{args.n_symbols}_{args.vol_method}.pt'
    
    # Try to load cached data
    data = load_processed_data(cache_path)
    if data[0] is None:
        print("Processing data and creating input dictionaries...")
        data = prepare_model_data(volatility, config)
        save_processed_data(*data, cache_path)
    
    # Unpack data
    (train_dict, test_dict, DY_adj, y_columns, 
     test_dates, market_indices_list) = data
    
    # Create dataloaders
    dataloader_train, dataloader_test = create_dataloaders(
        train_dict, test_dict, config.batch_size
    )
    
    # Initialize model
    model = GSPHAR(config.input_dim, config.output_dim, args.n_symbols, DY_adj)
    
    # Train or load model
    if args.train:
        print("Training new model...")
        valid_loss, *_ = train_eval_model(
            model, dataloader_train, dataloader_test, 
            config.num_epochs, config.lr, config.h
        )
    
    # Load best model
    trained_model, mae_loss = load_model(config.model_name, model)
    
    # Predict and evaluate
    results_df, rv_hat, rv_true = predict_and_evaluate(
        trained_model, dataloader_test, market_indices_list,
        test_dates=test_dates
    )
    
    # Save results
    save_results(results_df, rv_hat, rv_true, config.h)
    print(f"MAE loss: {mae_loss}")
    
    return results_df, rv_hat, rv_true, mae_loss

if __name__ == '__main__':
    main()
