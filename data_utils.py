import pandas as pd
import numpy as np
from scipy.sparse import linalg
from statsmodels.tsa.api import VAR
from scipy import stats
from tqdm import tqdm

def compute_spillover_index(data, horizon, lag, scarcity_prop, standardized=True):
    """
    Compute the Diebold-Yilmaz spillover index to create an adjacency matrix
    for the network model.
    
    Args:
        data: pandas DataFrame with assets as columns
        horizon: forecast horizon for variance decomposition
        lag: number of lags in VAR model
        scarcity_prop: proportion of smallest values to set to zero
        standardized: whether to standardize the output matrix
    
    Returns:
        numpy array: Adjacency matrix where A[i,j] represents spillover from i to j
    """
    # 1. Fit VAR model
    data_array = data.values
    model = VAR(data_array)
    results = model.fit(maxlags=lag)
    
    # 2. Compute Forecast Error Variance Decomposition (FEVD)
    Sigma = results.sigma_u  # Residual covariance matrix
    A = results.orth_ma_rep(maxn=horizon - 1)  # Moving average coefficients
    
    # 3. Calculate numerator and denominator for FEVD
    Sigma_A = []
    A_Sigma_A = []
    for h in range(horizon):
        # Numerator: normalized shock impact
        Sigma_A_h = (A[h] @ Sigma @ np.linalg.inv(np.diag(np.sqrt(np.diag(Sigma))))) ** 2
        Sigma_A.append(Sigma_A_h)
        
        # Denominator: total forecast variance
        A_Sigma_A_h = A[h] @ Sigma @ A[h].T
        A_Sigma_A.append(A_Sigma_A_h)
    
    # 4. Compute Generalized FEVD
    num = np.cumsum(Sigma_A, axis=0)
    den = np.cumsum(A_Sigma_A, axis=0)
    gfevd = np.array([num[h] / np.diag(den[h])[:, None] for h in range(horizon)])
    
    # 5. Standardize if requested
    if standardized:
        gfevd = np.array([fevd / fevd.sum(axis=1, keepdims=True) for fevd in gfevd])
    
    # 6. Create final spillover matrix
    spillover_matrix = gfevd[-1]  # Use last horizon
    spillover_matrix = spillover_matrix.T  # Transpose for adjacency matrix format
    spillover_matrix *= 100  # Convert to percentage
    
    # 7. Apply sparsity threshold
    results_df = pd.DataFrame(spillover_matrix, 
                            columns=results.names, 
                            index=results.names)
    vsp_df_sparse = results_df.copy()
    if scarcity_prop > 0:
        threshold = pd.Series(results_df.values.flatten()).quantile(scarcity_prop)
        vsp_df_sparse[vsp_df_sparse < threshold] = 0
    
    # 8. Final processing
    vsp_np_sparse = vsp_df_sparse.values
    np.fill_diagonal(vsp_np_sparse, 0)  # Remove self-loops
    
    if standardized:
        vsp_np_sparse = vsp_np_sparse / spillover_matrix.shape[0]
        
    return vsp_np_sparse

def compute_crypto_spillover_index(data, horizon=1, lag=24, scarcity_prop=0.0):
    """
    Compute spillover index optimized for 1-hour crypto data
    
    Args:
        data: DataFrame with 1-hour crypto price data
        horizon: forecast horizon (1 period = 1 hour)
        lag: lookback window (24 periods = 24 hours)
        scarcity_prop: sparsity threshold
    """
    # Use original compute_spillover_index with crypto-optimized parameters
    return compute_spillover_index(
        data=data,
        horizon=horizon,  # 1 hour ahead
        lag=lag,         # 24 hour history
        scarcity_prop=scarcity_prop,
        standardized=True
    )

def prepare_lagged_features(train_dataset, test_dataset, look_back_window, h):
    """Create lagged features for training and testing datasets"""
    market_indices_list = train_dataset.columns.tolist()
    
    # Create progress bar for market indices
    progress_bar = tqdm(market_indices_list, desc="Creating lagged features", leave=True)
    
    # Create lagged features
    for market_index in progress_bar:
        progress_bar.set_postfix({'Market': market_index})
        for lag in range(look_back_window):
            train_dataset[market_index + f'_{lag+1}'] = train_dataset[market_index].shift(lag+h)
            test_dataset[market_index + f'_{lag+1}'] = test_dataset[market_index].shift(lag+h)
    
    # Remove rows with NaN values
    train_dataset = train_dataset.dropna()
    test_dataset = test_dataset.dropna()
    
    return train_dataset, test_dataset, market_indices_list

def create_model_input_dicts(train_dataset, test_dataset, market_indices_list):
    """Create dictionary of model inputs for training and testing"""
    train_dict = {}
    test_dict = {}
    y_columns = market_indices_list
    
    # Get all dates
    train_dates = train_dataset.index
    test_dates = test_dataset.index
    
    # Create nested progress bars
    with tqdm(total=len(train_dates) + len(test_dates), desc="Overall Progress", position=0) as pbar:
        # Process training data
        train_progress = tqdm(train_dates, 
                            desc="making train_dict", 
                            position=1, 
                            leave=False)
        
        for date in train_progress:
            train_dict[date] = process_single_date(train_dataset, date, market_indices_list)
            train_progress.set_postfix({
                'Date': date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else date,
                'Markets': len(market_indices_list)
            })
            pbar.update(1)
        
        # Process test data
        test_progress = tqdm(test_dates, 
                           desc="Test data  ", 
                           position=1, 
                           leave=False)
        
        for date in test_progress:
            test_dict[date] = process_single_date(test_dataset, date, market_indices_list)
            test_progress.set_postfix({
                'Date': date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else date,
                'Markets': len(market_indices_list)
            })
            pbar.update(1)
    
    return train_dict, test_dict, y_columns

def process_single_date(dataset, date, market_indices_list):
    """Process data for a single date"""
    # Prepare column names for different lags
    columns_lag1 = [f"{market}_1" for market in market_indices_list]
    columns_lag4 = [f"{market}_{i}" for market in market_indices_list for i in range(1, 5)]
    columns_lag24 = [f"{market}_{i}" for market in market_indices_list for i in range(1, 25)]
    
    # Create consistent index orders
    row_index_order = market_indices_list
    column_index_order_4 = [f"lag_{i}" for i in range(1, 5)]
    column_index_order_24 = [f"lag_{i}" for i in range(1, 25)]
    
    # Process target variables with progress tracking
    y = dataset.loc[date, market_indices_list]
    
    # Process lag-1 data
    x_lag1 = dataset.loc[date, columns_lag1]
    new_index = [ind[:-2] for ind in x_lag1.index.tolist()]
    x_lag1.index = new_index
    
    # Process lag-4 data with DataFrame operations
    x_lag4 = dataset.loc[date, columns_lag4]
    data_lag4 = {
        'Market': [index.split('_')[0] for index in x_lag4.index],
        'Lag': [f'lag_{index.split("_")[1]}' for index in x_lag4.index],
        'Value': x_lag4.values
    }
    df_lag4 = pd.DataFrame(data_lag4).pivot(index='Market', columns='Lag', values='Value')
    
    # Process lag-24 data with DataFrame operations
    x_lag24 = dataset.loc[date, columns_lag24]
    data_lag24 = {
        'Market': [index.split('_')[0] for index in x_lag24.index],
        'Lag': [f'lag_{index.split("_")[1]}' for index in x_lag24.index],
        'Value': x_lag24.values
    }
    df_lag24 = pd.DataFrame(data_lag24).pivot(index='Market', columns='Lag', values='Value')
    
    # Ensure consistent ordering
    x_lag1 = x_lag1.reindex(row_index_order)
    df_lag4 = df_lag4.reindex(row_index_order)[column_index_order_4]
    df_lag24 = df_lag24.reindex(row_index_order)[column_index_order_24]
    
    return {
        'y': y,
        'x_lag1': x_lag1,
        'x_lag4': df_lag4,
        'x_lag24': df_lag24
    }

def load_and_split_data(file_path, scale_factor=100, train_ratio=0.7):
    """Load data from CSV and split into train/test sets"""
    data = pd.read_csv(file_path, index_col=0) * scale_factor
    date_list = data.index.tolist()
    train_end_idx = int(len(date_list) * train_ratio)
    train_dataset = data.iloc[0:train_end_idx, :]
    test_dataset = data.iloc[train_end_idx:, :]
    
    return data, train_dataset, test_dataset
