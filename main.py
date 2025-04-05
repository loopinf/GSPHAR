import argparse
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import warnings
import os
warnings.filterwarnings('ignore')

# Import project modules
from data_utils import (load_and_split_data, compute_spillover_index,
                       prepare_lagged_features, create_model_input_dicts)
from model import GSPHAR, GSPHAR_Dataset
from train_utils import train_eval_model, load_model, predict_and_evaluate

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='data/sample_1h_rv5_sqrt_38.csv', 
                    help='Path to the input data file')
parser.add_argument('--n_symbols', type=int, default=38,
                    help='Number of symbols/assets to process')
args = parser.parse_args()

# Set random seeds for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

def save_processed_data(train_dict, test_dict, DY_adj, y_columns, test_dates, market_indices_list, cache_path):
    """Save processed dictionaries and DY_adj matrix to disk"""
    if not os.path.exists('cache/'):
        os.makedirs('cache/')
    
    save_dict = {
        'train_dict': train_dict,
        'test_dict': test_dict,
        'DY_adj': DY_adj,
        'y_columns': y_columns,
        'test_dates': test_dates,
        'market_indices_list': market_indices_list
    }
    
    torch.save(save_dict, cache_path)
    print(f"Saved processed data to {cache_path}")

def load_processed_data(cache_path):
    """Load cached dictionaries and DY_adj matrix from disk"""
    if os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        cache = torch.load(cache_path)
        return (cache['train_dict'], cache['test_dict'], 
                cache['DY_adj'], cache['y_columns'],
                cache['test_dates'], cache['market_indices_list'])
    return None, None, None, None, None, None

def main():
    # Configuration
    h = 1  # forecasting horizon
    data_path = args.data_path
    n_symbols = args.n_symbols
    look_back_window = 24  # 24 hours
    input_dim = 3
    output_dim = 1
    filter_size = n_symbols
    num_epochs = 1  # Set to 500 for full training
    lr = 0.01
    batch_size = 32
    
    # Define cache path
    cache_path = f'cache/processed_data_h{h}_n{n_symbols}.pt'
    
    # Try to load cached data
    train_dict, test_dict, DY_adj, y_columns, test_dates, market_indices_list = load_processed_data(cache_path)
    
    if train_dict is None:  # If cached data is missing
        print("Processing data and creating input dictionaries...")
        # Load and split data
        raw_data, train_df, test_df = load_and_split_data(data_path)
        market_indices_list = train_df.columns.tolist()
        
        # Store test dates before any processing
        test_dates = test_df.index
        
        # Compute spillover index
        DY_adj = compute_spillover_index(train_df, h, look_back_window, 0.0, standardized=True)
        
        # Prepare lagged features
        train_df, test_df, market_indices_list = prepare_lagged_features(
            train_df, test_df, look_back_window, h)
            
        # Create model input dictionaries
        train_dict, test_dict, y_columns = create_model_input_dicts(
            train_df, test_df, market_indices_list)
            
        # Cache the processed data
        save_processed_data(train_dict, test_dict, DY_adj, y_columns, test_dates, market_indices_list, cache_path)
    
    # Create PyTorch datasets
    dataset_train = GSPHAR_Dataset(train_dict)
    dataset_test = GSPHAR_Dataset(test_dict)
    
    # Create PyTorch dataloaders
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    
    # Create and train model
    model = GSPHAR(input_dim, output_dim, filter_size, DY_adj)
    # valid_loss, final_conv1d_lag4_weights, final_conv1d_lag24_weights = train_eval_model(
    #     model, dataloader_train, dataloader_test, num_epochs, lr, h)
    
    # Load best model
    model_name = f'GSPHAR_24_magnet_dynamic_h{h}'
    trained_model, mae_loss = load_model(model_name, model)
    
    # Predict and evaluate using the stored test_dates
    results_df, rv_hat, rv_true = predict_and_evaluate(
        trained_model, 
        dataloader_test, 
        market_indices_list,
        test_dates=test_dates
    )
    
    # Ensure the results directory exists
    if not os.path.exists('results/'):
        os.makedirs('results/')
    
    # Save results with time index
    results_df.to_csv(f'results/predictions_h{h}.csv')
    rv_hat.to_csv(f'results/predictions_only_h{h}.csv')
    rv_true.to_csv(f'results/true_values_h{h}.csv')
    
    print(f"Training and evaluation complete. Results saved to:")
    print(f"- results/predictions_h{h}.csv (both predictions and true values)")
    print(f"- results/predictions_only_h{h}.csv (only predictions)")
    print(f"- results/true_values_h{h}.csv (only true values)")
    print(f"MAE loss: {mae_loss}")
    
    return results_df, rv_hat, rv_true, mae_loss

if __name__ == '__main__':
    main()
