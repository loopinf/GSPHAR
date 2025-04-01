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
parser.add_argument('--data_path', type=str, default='data/sample_1h_rv5_sqrt_38.csv', help='Path to the input data file')
args = parser.parse_args()

# Set random seeds for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

def main():
    # Configuration
    h = 1  # forecasting horizon
    data_path = args.data_path
    look_back_window = 24  # 24 hours
    input_dim = 3
    output_dim = 1
    filter_size = 38 
    num_epochs = 1 # Set to 500 for full training
    lr = 0.01
    batch_size = 32
    
    # Load and split data
    data, train_dataset, test_dataset = load_and_split_data(data_path)
    market_indices_list = train_dataset.columns.tolist()
    
    # Compute spillover index
    DY_adj = compute_spillover_index(train_dataset, h, look_back_window, 0.0, standardized=True)
    
    # Prepare lagged features
    train_dataset, test_dataset, market_indices_list = prepare_lagged_features(
        train_dataset, test_dataset, look_back_window, h)
    
    # Create model input dictionaries
    train_dict, test_dict, y_columns = create_model_input_dicts(
        train_dataset, test_dataset, market_indices_list)
    
    # Create datasets and dataloaders
    dataset_train = GSPHAR_Dataset(train_dict)
    dataset_test = GSPHAR_Dataset(test_dict)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    
    # Create and train model
    model = GSPHAR(input_dim, output_dim, filter_size, DY_adj)
    valid_loss, final_conv1d_lag4_weights, final_conv1d_lag24_weights = train_eval_model(
        model, dataloader_train, dataloader_test, num_epochs, lr, h)
    
    # Load best model
    model_name = f'GSPHAR_24_magnet_dynamic_h{h}'
    trained_model, mae_loss = load_model(model_name, model)
    
    # Predict and evaluate
    results_df, rv_hat, rv_true = predict_and_evaluate(
        trained_model, dataloader_test, market_indices_list)
    
    # Ensure the results directory exists
    if not os.path.exists('results/'):
        os.makedirs('results/')
    
    # Save results
    results_df.to_csv(f'results/predictions_h{h}.csv')
    
    print(f"Training and evaluation complete. Results saved to results/predictions_h{h}.csv")
    print(f"MAE loss: {mae_loss}")
    
    return results_df, rv_hat, rv_true, mae_loss

if __name__ == '__main__':
    main()
