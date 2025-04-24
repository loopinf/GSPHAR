
## Import packages
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch.nn as nn

import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from scipy.linalg import sqrtm
from scipy.linalg import eig

import statsmodels.api as sm
import os
import warnings
warnings.filterwarnings('ignore')


torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)  # if using multi-GPU


from scipy.sparse import linalg
from src.models import GSPHAR
from src.utils import compute_spillover_index, save_model, load_model
from src.dataset import GSPHAR_Dataset

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
    model.train()
    train_loss_list = []
    test_loss_list = []
    for epoch in range(num_epochs):
        for x_lag1, x_lag5, x_lag22, y in dataloader_train:
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
        
        # Evaluate model
        valid_loss = evaluate_model(model, dataloader_test)

        if valid_loss < best_loss_val:
            best_loss_val = valid_loss
            final_conv1d_lag5_weights = conv1d_lag5_weights.detach().cpu().numpy()
            final_conv1d_lag22_weights = conv1d_lag22_weights.detach().cpu().numpy()
            patience = 0
            save_model(f'GSPHAR_24_magnet_dynamic_h{h}', model, None, best_loss_val)
        else:
            patience = patience + 1
            if patience >= 200:
                print(f'early stopping at epoch {epoch+1}.')
                break
            else:
                pass
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


h = 5
data = pd.read_csv('rv5_sqrt_24.csv', index_col = 0)*100
date_list = data.index.tolist()
train_end_idx = int(len(date_list)*0.7)
train_dataset = data.iloc[0:train_end_idx,:]
test_dataset = data.iloc[train_end_idx:,:]

market_indices_list = train_dataset.columns.tolist()

DY_adj = compute_spillover_index(train_dataset, h, 22, 0.0, standardized=True)

look_back_window = 22
for market_index in market_indices_list:
    for lag in range(look_back_window):
        train_dataset[market_index + f'_{lag+1}'] = train_dataset[market_index].shift(lag+h)
        test_dataset[market_index + f'_{lag+1}'] = test_dataset[market_index].shift(lag+h)

train_dataset = train_dataset.dropna()
test_dataset = test_dataset.dropna()

columns_lag1 = [x for x in train_dataset.columns.tolist() if x[-2:] == '_1']
columns_lag5 = [x for x in train_dataset.columns.tolist() if (x[-2]=='_') and (float(x[-1]) in range(1,6))]
columns_lag22 = [x for x in train_dataset.columns.tolist() if '_' in x]
x_columns = columns_lag1 + columns_lag5 + columns_lag22
y_columns = [x for x in train_dataset.columns.tolist() if x not in x_columns]
row_index_order = market_indices_list
column_index_order_5 = [f'lag_{i}' for i in range(1,6)]
column_index_order_22 = [f'lag_{i}' for i in range(1,23)]


train_dict = {}
for date in train_dataset.index:
    y = train_dataset.loc[date,y_columns]
    
    x_lag1 = train_dataset.loc[date,columns_lag1]
    new_index = [ind[:-2] for ind in x_lag1.index.tolist()]
    x_lag1.index = new_index
    
    x_lag5 = train_dataset.loc[date,columns_lag5]

    # Split the index into market indices and lags
    data_lag5 = {
    'Market': [index.split('_')[0] for index in x_lag5.index],
    'Lag': [f'lag_{index.split("_")[1]}' for index in x_lag5.index],
    'Value': x_lag5.values
    }
    # Convert to DataFrame
    df_lag5 = pd.DataFrame(data_lag5)
    # Pivot the DataFrame
    df_lag5 = df_lag5.pivot(index='Market', columns='Lag', values='Value')

    x_lag22 = train_dataset.loc[date,columns_lag22]
    # Split the index into market indices and lags
    data_lag22 = {
    'Market': [index.split('_')[0] for index in x_lag22.index],
    'Lag': [f'lag_{index.split("_")[1]}' for index in x_lag22.index],
    'Value': x_lag22.values
    }
    # Convert to DataFrame
    df_lag22 = pd.DataFrame(data_lag22)
    # Pivot the DataFrame
    df_lag22 = df_lag22.pivot(index='Market', columns='Lag', values='Value')

    x_lag1 = x_lag1.reindex(row_index_order)
    df_lag5 = df_lag5.reindex(row_index_order)
    df_lag22= df_lag22.reindex(row_index_order)
    df_lag5 = df_lag5[column_index_order_5]
    df_lag22 = df_lag22[column_index_order_22]
    
    dfs_dict = {
        'y': y,
        'x_lag1': x_lag1,
        'x_lag5': df_lag5,
        'x_lag22': df_lag22
    }
    train_dict[date] = dfs_dict

test_dict = {}
for date in test_dataset.index:
    y = test_dataset.loc[date,y_columns]
    
    x_lag1 = test_dataset.loc[date,columns_lag1]
    new_index = [ind[:-2] for ind in x_lag1.index.tolist()]
    x_lag1.index = new_index
    
    x_lag5 = test_dataset.loc[date,columns_lag5]
    # Split the index into market indices and lags
    data_lag5 = {
    'Market': [index.split('_')[0] for index in x_lag5.index],
    'Lag': [f'lag_{index.split("_")[1]}' for index in x_lag5.index],
    'Value': x_lag5.values
    }
    # Convert to DataFrame
    df_lag5 = pd.DataFrame(data_lag5)
    # Pivot the DataFrame
    df_lag5 = df_lag5.pivot(index='Market', columns='Lag', values='Value')

    x_lag22 = test_dataset.loc[date,columns_lag22]
    # Split the index into market indices and lags
    data_lag22 = {
    'Market': [index.split('_')[0] for index in x_lag22.index],
    'Lag': [f'lag_{index.split("_")[1]}' for index in x_lag22.index],
    'Value': x_lag22.values
    }
    # Convert to DataFrame
    df_lag22 = pd.DataFrame(data_lag22)
    # Pivot the DataFrame
    df_lag22 = df_lag22.pivot(index='Market', columns='Lag', values='Value')

    x_lag1 = x_lag1.reindex(row_index_order)
    df_lag5 = df_lag5.reindex(row_index_order)
    df_lag22= df_lag22.reindex(row_index_order)
    df_lag5 = df_lag5[column_index_order_5]
    df_lag22 = df_lag22[column_index_order_22]
    
    dfs_dict = {
        'y': y,
        'x_lag1': x_lag1,
        'x_lag5': df_lag5,
        'x_lag22': df_lag22
    }
    test_dict[date] = dfs_dict

# Create dataset and dataloader
dataset_train = GSPHAR_Dataset(train_dict)
dataset_test = GSPHAR_Dataset(test_dict)

dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=False)

input_dim = 3
output_dim = 1
filter_size = 24
num_epochs = 500
lr = 0.01

GSPHAR_RV = GSPHAR(input_dim,output_dim, filter_size, DY_adj)
valid_loss, final_conv1d_lag5_weights, final_conv1d_lag22_weights = train_eval_model(GSPHAR_RV, dataloader_train, dataloader_test, num_epochs, lr)
trained_GSPHAR, mae_GSPHAR  = load_model(f'GSPHAR_24_magnet_dynamic_h{h}',GSPHAR_RV) 

y_hat_list = []
y_list = []

trained_GSPHAR.eval()

with torch.no_grad():
    for x_lag1, x_lag5, x_lag22, y in dataloader_test:
        y_hat, _, _ = trained_GSPHAR(x_lag1, x_lag5, x_lag22)
        
       # Append the predicted and actual values to their respective lists
        y_hat_list.append(y_hat.cpu().numpy())
        y_list.append(y.cpu().numpy())

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


