import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import numpy as np
import pandas as pd
from src.dataset import GSPHAR_Dataset
from src.dataset import NewGSPHAR_Dataset
from torch.utils.data import DataLoader

def create_train_dict_like_original(data):
    """Recreate train_dict exactly as in d-GSPHAR.py but with smaller data"""
    # Define column groups exactly as in d-GSPHAR.py
    market_indices_list = data.columns.tolist()
    row_index_order = market_indices_list
    column_index_order_5 = [f'lag_{i}' for i in range(1,6)]
    column_index_order_22 = [f'lag_{i}' for i in range(1,23)]

    # Create lag columns lists
    columns_lag1 = [x for x in data.columns.tolist() if x[-2:] == '_1']
    columns_lag5 = [x for x in data.columns.tolist() if (x[-2]=='_') and (float(x[-1]) in range(1,6))]
    columns_lag22 = [x for x in data.columns.tolist() if '_' in x]
    x_columns = columns_lag1 + columns_lag5 + columns_lag22
    y_columns = [x for x in data.columns.tolist() if x not in x_columns]

    train_dict = {}
    for date in data.index:
        y = data.loc[date,y_columns]
        
        x_lag1 = data.loc[date,columns_lag1]
        new_index = [ind[:-2] for ind in x_lag1.index.tolist()]
        x_lag1.index = new_index
        
        x_lag5 = data.loc[date,columns_lag5]
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

        x_lag22 = data.loc[date,columns_lag22]
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
    
    return train_dict

def test_dataset_equivalence():
    # 1. Create small test data
    test_size = 30
    n_markets = 3
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create sample data exactly like in d-GSPHAR.py
    test_data = pd.DataFrame(
        np.random.randn(test_size, n_markets), 
        columns=[f'market_{i}' for i in range(n_markets)]
    )
    
    # Add lagged columns exactly like in d-GSPHAR.py
    look_back_window = 22
    h = 5  # Same as in d-GSPHAR.py
    market_indices_list = test_data.columns.tolist()
    
    for market_index in market_indices_list:
        for lag in range(look_back_window):
            test_data[market_index + f'_{lag+1}'] = test_data[market_index].shift(lag+h)
    
    test_data = test_data.dropna()
    
    # Define column groups
    y_columns = [col for col in test_data.columns if '_' not in col]
    columns_lag1 = [x for x in test_data.columns if x[-2:] == '_1']
    columns_lag5 = [x for x in test_data.columns if (x[-2]=='_') and (float(x[-1]) in range(1,6))]
    columns_lag22 = [x for x in test_data.columns if '_' in x]
    
    # 2. Process with original implementation
    old_dict = create_train_dict_like_original(test_data)
    old_dataset = GSPHAR_Dataset(old_dict)
    old_dataloader = DataLoader(old_dataset, batch_size=2, shuffle=False)
    
    # 3. Process with new implementation
    new_dataset = NewGSPHAR_Dataset(
        test_data,
        y_columns,
        columns_lag1,
        columns_lag5,
        columns_lag22,
        market_indices_list
    )
    new_dataloader = DataLoader(new_dataset, batch_size=2, shuffle=False)
    
    # 4. Compare outputs
    for batch_idx, ((old_x1, old_x5, old_x22, old_y), (new_x1, new_x5, new_x22, new_y)) in enumerate(zip(old_dataloader, new_dataloader)):
        print(f"\nBatch {batch_idx}:")
        print(f"old_x1 shape: {old_x1.shape}, new_x1 shape: {new_x1.shape}")
        print(f"old_x5 shape: {old_x5.shape}, new_x5 shape: {new_x5.shape}")
        print(f"old_x22 shape: {old_x22.shape}, new_x22 shape: {new_x22.shape}")
        print(f"old_y shape: {old_y.shape}, new_y shape: {new_y.shape}")
        
        assert torch.allclose(old_x1, new_x1, rtol=1e-5), "x_lag1 mismatch"
        assert torch.allclose(old_x5, new_x5, rtol=1e-5), "x_lag5 mismatch"
        assert torch.allclose(old_x22, new_x22, rtol=1e-5), "x_lag22 mismatch"
        assert torch.allclose(old_y, new_y, rtol=1e-5), "y mismatch"
    
    print("\nAll tests passed! Old and new implementations produce identical results.")

if __name__ == "__main__":
    test_dataset_equivalence()
