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
    parser.add_argument('--continue_training', action='store_true',
                       help='Continue training existing model')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
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
    # multiply by 100 to convert to percentage
    volatility = volatility * 100
    
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
    if not os.path.exists(cache_path):
        return (None,) * 6
    
    print(f"Loading cached data from {cache_path}")
    try:
        # Try loading with torch first
        data = torch.load(cache_path)
        return data
    except Exception as e:
        print(f"Error loading with torch: {str(e)}")
        try:
            # Try loading with pickle as fallback
            print("Attempting to load using pickle...")
            import pickle
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            return data
        except Exception as e:
            print(f"Failed to load data: {str(e)}")
            return (None,) * 6

def save_processed_data(train_dict: dict, test_dict: dict, 
                       DY_adj: np.ndarray, y_columns: list,
                       test_dates: pd.DatetimeIndex, 
                       market_indices_list: list,
                       cache_path: str) -> None:
    """Save processed data to cache with error handling"""
    print(f"Saving processed data to {cache_path}")
    
    # Create directory if it doesn't exist
    cache_dir = os.path.dirname(cache_path)
    os.makedirs(cache_dir, exist_ok=True)
    
    # First save to a temporary file
    temp_path = cache_path + '.tmp'
    try:
        data = (train_dict, test_dict, DY_adj, y_columns, 
                test_dates, market_indices_list)
        
        # Convert numpy arrays to torch tensors if needed
        if isinstance(DY_adj, np.ndarray):
            DY_adj = torch.from_numpy(DY_adj)
        
        # Save with error handling
        torch.save(data, temp_path)
        
        # If save was successful, rename temp file to final file
        if os.path.exists(cache_path):
            os.remove(cache_path)
        os.rename(temp_path, cache_path)
        print("Data saved successfully")
        
    except Exception as e:
        print(f"Error saving data: {str(e)}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        # Try alternative saving method using pickle
        try:
            print("Attempting to save using pickle...")
            import pickle
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            print("Data saved successfully using pickle")
        except Exception as e:
            print(f"Failed to save using pickle: {str(e)}")
            raise

def save_training_history(train_losses, valid_losses, h):
    """Save training history to a CSV file"""
    history_df = pd.DataFrame({
        'epoch': range(1, len(train_losses) + 1),
        'train_loss': train_losses,
        'valid_loss': valid_losses
    })
    history_df.to_csv(f'results/training_history_h{h}.csv', index=False)
    print(f"Training history saved to results/training_history_h{h}.csv")

def main():
    # Setup
    args = setup_args()
    config = ModelConfig()
    config.num_epochs = args.epochs  # Update config with command-line argument
    setup_directories()
    set_random_seeds()
    # perpetual_futures = [ "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "LINKUSDT", "MATICUSDT", "DOTUSDT" ]
    tmp_symbols = ['LINKUSDT', 'XRPUSDT', 'DOGEUSDT', 'ETHUSDT', 'DOTUSDT', 'BNBUSDT', 'ADAUSDT', 'BTCUSDT'] 
    # Define cache path
    cache_path = f'cache/processed_data_h{config.h}_n{args.n_symbols}_{args.vol_method}.pt'
    cache_path_test = f'cache_small/processed_data_h{config.h}_n{args.n_symbols}_{args.vol_method}.pt'
    SMALL = True
    if SMALL: cache_path = cache_path_test
    
    # Try to load cached data
    data = load_processed_data(cache_path)
    if data[0] is None:
        # Process data
        volatility = process_raw_data(args.data_path, args.vol_method, args.n_symbols)
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
    
    # Train, continue training, or load model
    if args.train and args.continue_training:
        print("Error: Cannot specify both --train and --continue_training")
        return
        
    if args.continue_training:
        print(f"Loading existing model...")
        model_name = f'GSPHAR_24_magnet_dynamic_h{config.h}'
        model, previous_loss = load_model(model_name, model)
        print(f"Previous loss: {previous_loss}")
        print(f"Continuing training for {args.epochs} more epochs...")
        
    if args.train or args.continue_training:
        best_loss_val, train_losses, valid_losses = train_eval_model(
            model, 
            dataloader_train, 
            dataloader_test, 
            num_epochs=args.epochs,
            lr=config.lr, 
            h=config.h
        )
        print(f"Final validation loss: {best_loss_val}")
        
        # Save training history
        save_training_history(train_losses, valid_losses, config.h)
        loss = best_loss_val
    else:
        # Just load the model without training
        model_name = f'GSPHAR_24_magnet_dynamic_h{config.h}'
        model, loss = load_model(model_name, model)
    
    # Predict and evaluate
    results_df, rv_hat, rv_true = predict_and_evaluate(
        model, dataloader_test, market_indices_list,
        test_dates=test_dates
    )
    
    # Save results
    save_results(results_df, rv_hat, rv_true, config.h)
    print(f"Loss: {loss}")
    
    return results_df, rv_hat, rv_true, loss

if __name__ == '__main__':
    main()
