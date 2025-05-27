## Import packages
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch.nn as nn

import pandas as pd
import numpy as np
from scipy.linalg import sqrtm
from scipy.linalg import eig

import statsmodels.api as sm
import os
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm # Add tqdm import

# Import from new modules
import config
from data_utils import load_data, split_data, create_lagged_features, prepare_data_dict, create_dataloaders
from src.models import GSPHAR
from src.utils import compute_spillover_index, save_model, load_model

torch.manual_seed(config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)  # if using multi-GPU


from scipy.sparse import linalg

def train_eval_model(model, dataloader_train, dataloader_test, num_epochs = 200, lr = 0.01):
    best_loss_val = 1000000
    patience = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = lr,
                                                   steps_per_epoch=len(dataloader_train), epochs = num_epochs,
                                                   three_phase=True)
    model.to(device)
    criterion = nn.MSELoss()
    criterion = criterion.to(device)
    train_loss_list = []
    test_loss_list = []
    # Wrap epochs loop with tqdm
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        model.train() # Set model to training mode at the start of each epoch
        epoch_loss = 0.0
        # Wrap dataloader loop with tqdm
        for x_lag1, x_lag5, x_lag22, y in tqdm(dataloader_train, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            x_lag1 = x_lag1.to(device)
            x_lag5 = x_lag5.to(device)
            x_lag22 = x_lag22.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            output, conv1d_lag5_weights, conv1d_lag22_weights = model(x_lag1, x_lag5, x_lag22)
            loss = criterion(output, y)
            loss.backward()

            # Update parameters
            optimizer.step()

            # Update scheduler: this scheduler is designed to be updated after each batch.
            scheduler.step()
            epoch_loss += loss.item() # Accumulate loss for the epoch

        # Evaluate model
        valid_loss = evaluate_model(model, dataloader_test)
        avg_epoch_loss = epoch_loss / len(dataloader_train)
        train_loss_list.append(avg_epoch_loss)
        test_loss_list.append(valid_loss)

        # Use tqdm.write for logging to avoid interfering with progress bars
        tqdm.write(f"Epoch {epoch+1}: Train Loss: {avg_epoch_loss:.4f}, Val Loss: {valid_loss:.4f}")

        if valid_loss < best_loss_val:
            best_loss_val = valid_loss
            final_conv1d_lag5_weights = conv1d_lag5_weights.detach().cpu().numpy()
            final_conv1d_lag22_weights = conv1d_lag22_weights.detach().cpu().numpy()
            patience = 0
            # Use config for model save pattern components
            model_save_name = config.MODEL_SAVE_NAME_PATTERN.format(filter_size=config.FILTER_SIZE, h=config.PREDICTION_HORIZON)
            save_model(model_save_name, model, None, best_loss_val)
            tqdm.write(f"Epoch {epoch+1}: Validation loss improved to {best_loss_val:.4f}. Saving model.") # Log improvement
        else:
            patience = patience + 1
            # Use config for patience
            if patience >= config.PATIENCE:
                tqdm.write(f'Early stopping at epoch {epoch+1}.') # Use tqdm.write
                break
            else:
                pass
    # Ensure weights are returned even if early stopping doesn't happen in the first epoch
    if 'final_conv1d_lag5_weights' not in locals():
        # Need to handle case where training loop doesn't run even once (e.g., num_epochs=0)
        # Or if the last batch didn't update the weights variables (shouldn't happen with current logic)
        try:
            final_conv1d_lag5_weights = conv1d_lag5_weights.detach().cpu().numpy()
            final_conv1d_lag22_weights = conv1d_lag22_weights.detach().cpu().numpy()
        except NameError: # Handle case where loop didn't run
            final_conv1d_lag5_weights = None
            final_conv1d_lag22_weights = None
            tqdm.write("Warning: Training loop did not run. Returning None for weights.")

    return best_loss_val, final_conv1d_lag5_weights, final_conv1d_lag22_weights


# Evaluate model
def evaluate_model(model, dataloader_test):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.L1Loss()
    criterion = criterion.to(device)
    valid_loss = 0
    model.eval()
    with torch.no_grad():
        for x_lag1, x_lag5, x_lag22, y in dataloader_test:
            x_lag1 = x_lag1.to(device)
            x_lag5 = x_lag5.to(device)
            x_lag22 = x_lag22.to(device)
            y = y.to(device)
            output, _, _ = model(x_lag1, x_lag5, x_lag22)
            loss = criterion(output, y)
            valid_loss = valid_loss + loss.item()
    valid_loss = valid_loss/len(dataloader_test)
    return valid_loss


# Use config constants
h = config.PREDICTION_HORIZON
data = load_data(config.DATA_FILE)
# date_list = data.index.tolist() # Keep if needed elsewhere, otherwise remove
train_dataset_raw, test_dataset_raw = split_data(data, config.TRAIN_SPLIT_RATIO)

market_indices_list = train_dataset_raw.columns.tolist()

# Compute spillover index (keep here for now, might move later if it's purely data prep)
DY_adj = compute_spillover_index(train_dataset_raw, h, config.LOOK_BACK_WINDOW, 0.0, standardized=True)

# Create lagged features using data_utils function
# Rename variables back
train_dataset = create_lagged_features(train_dataset_raw, market_indices_list, h, config.LOOK_BACK_WINDOW)
test_dataset = create_lagged_features(test_dataset_raw, market_indices_list, h, config.LOOK_BACK_WINDOW)

# --- Start: Restore original data preparation ---
# Remove column group definitions
# y_cols = market_indices_list
# lag1_cols = [f"{idx}_1" for idx in market_indices_list]
# lag5_cols = [f"{idx}_{lag}" for idx in market_indices_list for lag in range(1, min(6, config.LOOK_BACK_WINDOW + 1))]
# lag22_cols = [f"{idx}_{lag}" for idx in market_indices_list for lag in range(1, config.LOOK_BACK_WINDOW + 1)]

# Prepare data dictionaries using data_utils function
train_dict = prepare_data_dict(train_dataset, market_indices_list, config.LOOK_BACK_WINDOW)
test_dict = prepare_data_dict(test_dataset, market_indices_list, config.LOOK_BACK_WINDOW)

# Create dataset and dataloader using data_utils function
dataloader_train, dataloader_test = create_dataloaders(train_dict, test_dict, config.BATCH_SIZE)
# --- End: Restore original data preparation ---


# Use config constants for model and training parameters
input_dim = config.INPUT_DIM
output_dim = config.OUTPUT_DIM
filter_size = config.FILTER_SIZE
num_epochs = config.NUM_EPOCHS
lr = config.LEARNING_RATE

GSPHAR_RV = GSPHAR(input_dim,output_dim, filter_size, DY_adj)
valid_loss, final_conv1d_lag5_weights, final_conv1d_lag22_weights = train_eval_model(GSPHAR_RV, dataloader_train, dataloader_test, num_epochs, lr)

# Use config for model load pattern components
model_load_name = config.MODEL_SAVE_NAME_PATTERN.format(filter_size=config.FILTER_SIZE, h=config.PREDICTION_HORIZON)
trained_GSPHAR, mae_GSPHAR  = load_model(model_load_name, GSPHAR_RV)

y_hat_list = []
y_list = []

trained_GSPHAR.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu' # Define device for inference loop
trained_GSPHAR.to(device) # Move model to device for inference

with torch.no_grad():
    for x_lag1, x_lag5, x_lag22, y in dataloader_test:
        # Move data to device inside the loop
        x_lag1 = x_lag1.to(device)
        x_lag5 = x_lag5.to(device)
        x_lag22 = x_lag22.to(device)
        # y is not needed on device for inference unless loss is calculated here
        y_hat, _, _ = trained_GSPHAR(x_lag1, x_lag5, x_lag22)

       # Append the predicted and actual values to their respective lists
        y_hat_list.append(y_hat.cpu().numpy())
        y_list.append(y.cpu().numpy()) # y is already a tensor, move to cpu

y_hat_concatenated = np.concatenate(y_hat_list, axis=0)
y_concatenated = np.concatenate(y_list, axis=0)

rv_hat_GSPHAR_dynamic = pd.DataFrame(data = y_hat_concatenated, columns = market_indices_list)
rv_true = pd.DataFrame(data = y_concatenated, columns = market_indices_list)

pred_GSPHAR_dynamic_df = pd.DataFrame()
for market_index in market_indices_list:
    pred_column = market_index+'_rv_forecast'
    true_column = market_index+'_rv_true'
    pred_GSPHAR_dynamic_df[pred_column] = rv_hat_GSPHAR_dynamic[market_index]
    pred_GSPHAR_dynamic_df[true_column] = rv_true[market_index]

# Optional: Save the results
# pred_GSPHAR_dynamic_df.to_csv('predictions_GSPHAR_dynamic.csv')
# print("Predictions saved to predictions_GSPHAR_dynamic.csv")


