import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import numpy as np
import pandas as pd
from src.dataset import GSPHAR_Dataset
from src.dataset import NewGSPHAR_Dataset
from torch.utils.data import DataLoader

def create_old_dict(data):
    """Recreate dictionary in the same format as the original implementation"""
    result_dict = {}
    
    # Define column groups
    market_indices_list = [col for col in data.columns if '_' not in col]
    y_columns = market_indices_list.copy()
    
    # Create lag column lists
    columns_lag1 = [f"{market}_1" for market in market_indices_list]
    columns_lag5 = [f"{market}_{i}" for market in market_indices_list for i in range(1, 6)]
    columns_lag22 = [f"{market}_{i}" for market in market_indices_list for i in range(1, 23)]
    
    for date in data.index:
        y = data.loc[date, y_columns]
        
        # Process lag1 data
        x_lag1 = data.loc[date, columns_lag1]
        x_lag1.index = [ind[:-2] for ind in x_lag1.index]
        
        # Process lag5 data
        x_lag5 = data.loc[date, columns_lag5]
        df_lag5 = pd.DataFrame({
            'Market': [index.split('_')[0] for index in x_lag5.index],
            'Lag': [f'lag_{index.split("_")[1]}' for index in x_lag5.index],
            'Value': x_lag5.values
        })
        df_lag5 = df_lag5.pivot(index='Market', columns='Lag', values='Value')
        
        # Process lag22 data
        x_lag22 = data.loc[date, columns_lag22]
        df_lag22 = pd.DataFrame({
            'Market': [index.split('_')[0] for index in x_lag22.index],
            'Lag': [f'lag_{index.split("_")[1]}' for index in x_lag22.index],
            'Value': x_lag22.values
        })
        df_lag22 = df_lag22.pivot(index='Market', columns='Lag', values='Value')
        
        # Store in dictionary
        result_dict[date] = {
            'y': y,
            'x_lag1': x_lag1,
            'x_lag5': df_lag5,
            'x_lag22': df_lag22
        }
    
    return result_dict

def test_dataset_equivalence():
    # 1. Create small test data
    test_size = 10
    n_markets = 3
    
    # Create sample data
    test_data = pd.DataFrame(
        np.random.randn(test_size, n_markets), 
        columns=[f'market_{i}' for i in range(n_markets)]
    )
    
    # Add lagged columns
    market_indices_list = test_data.columns.tolist()
    for market_index in market_indices_list:
        for lag in range(22):
            test_data[market_index + f'_{lag+1}'] = test_data[market_index].shift(lag)
    
    test_data = test_data.dropna()
    
    # Define column groups for new implementation
    y_columns = market_indices_list
    columns_lag1 = [f"{market}_1" for market in market_indices_list]
    columns_lag5 = [f"{market}_{i}" for market in market_indices_list for i in range(1, 6)]
    columns_lag22 = [f"{market}_{i}" for market in market_indices_list for i in range(1, 23)]
    
    # 2. Process with old implementation
    old_dict = create_old_dict(test_data)
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
    for (old_x1, old_x5, old_x22, old_y), (new_x1, new_x5, new_x22, new_y) in zip(old_dataloader, new_dataloader):
        assert torch.allclose(old_x1, new_x1, rtol=1e-5), "x_lag1 mismatch"
        assert torch.allclose(old_x5, new_x5, rtol=1e-5), "x_lag5 mismatch"
        assert torch.allclose(old_x22, new_x22, rtol=1e-5), "x_lag22 mismatch"
        assert torch.allclose(old_y, new_y, rtol=1e-5), "y mismatch"
    
    print("All tests passed! Old and new implementations produce identical results.")

if __name__ == "__main__":
    test_dataset_equivalence()
