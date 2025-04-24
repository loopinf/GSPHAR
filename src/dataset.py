import torch
from torch.utils.data import Dataset

class GSPHAR_Dataset(Dataset):
    def __init__(self, dic):
        self.dict = dic

    def __len__(self):
        return len(self.dict.keys())

    def __getitem__(self, idx):
        date = list(self.dict.keys())[idx]
        dfs_dict = self.dict[date]
        y = dfs_dict['y'].values
        x_lag1 = dfs_dict['x_lag1'].values
        x_lag5 = dfs_dict['x_lag5'].values
        x_lag22 = dfs_dict['x_lag22'].values
        
        y_tensor = torch.tensor(y, dtype=torch.float32)
        x_lag1_tensor = torch.tensor(x_lag1, dtype=torch.float32)
        x_lag5_tensor = torch.tensor(x_lag5, dtype=torch.float32)
        x_lag22_tensor = torch.tensor(x_lag22, dtype=torch.float32)
        return x_lag1_tensor, x_lag5_tensor, x_lag22_tensor, y_tensor


class NewGSPHAR_Dataset(Dataset):
    """New implementation"""
    def __init__(self, dataset, y_cols, lag1_cols, lag5_cols, lag22_cols, market_indices):
        self.dataset = dataset
        self.y_cols = y_cols
        self.lag1_cols = lag1_cols
        self.lag5_cols = lag5_cols
        self.lag22_cols = lag22_cols
        self.market_indices = market_indices
        self.n_markets = len(market_indices)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        date = self.dataset.index[idx]
        
        # Get y values
        y = self.dataset.loc[date, self.y_cols].values
        
        # Get lag1 values
        x_lag1 = self.dataset.loc[date, self.lag1_cols].values.reshape(-1)
        
        # Get lag5 values
        x_lag5 = self.dataset.loc[date, self.lag5_cols].values
        x_lag5 = x_lag5.reshape(self.n_markets, 5)
        
        # Get lag22 values
        x_lag22 = self.dataset.loc[date, self.lag22_cols].values
        x_lag22 = x_lag22.reshape(self.n_markets, 22)
        
        return (torch.tensor(x_lag1, dtype=torch.float32),
                torch.tensor(x_lag5, dtype=torch.float32),
                torch.tensor(x_lag22, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32))