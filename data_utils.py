import pandas as pd
import numpy as np
from scipy.sparse import linalg
from statsmodels.tsa.api import VAR
from scipy import stats

def compute_spillover_index(data, horizon, lag, scarcity_prop, standardized=True):
    # Input data should be np.array
    data_array = data.values
    # Fit VAR model
    model = VAR(data_array)
    results = model.fit(maxlags=lag)
    
    # Manually Compute Forecast Error Variance Decomposition (FEVD)
    Sigma = results.sigma_u
    A = results.orth_ma_rep(maxn=horizon - 1)
    
    Sigma_A = []
    A_Sigma_A = []
    
    for h in range(horizon):
        # Numerator
        Sigma_A_h = (A[h] @ Sigma @ np.linalg.inv(np.diag(np.sqrt(np.diag(Sigma))))) ** 2
        Sigma_A.append(Sigma_A_h)
        
        # Denominator
        A_Sigma_A_h = A[h] @ Sigma @ A[h].T
        A_Sigma_A.append(A_Sigma_A_h)
    
    # Numerator: cumulative sum of Sigma_A
    num = np.cumsum(Sigma_A, axis=0)
    
    # Denominator: cumulative sum of A_Sigma_A
    den = np.cumsum(A_Sigma_A, axis=0)
    
    # Generalized FEVD
    gfevd = np.array([num[h] / np.diag(den[h])[:, None] for h in range(horizon)])
    
    if standardized:
        # Standardize each FEVD matrix so that each row sums to 1
        gfevd = np.array([fevd / fevd.sum(axis=1, keepdims=True) for fevd in gfevd])
    
    # Aggregate results over n_ahead steps
    spillover_matrix = gfevd[-1]
    
    # VSP from row to column so can be used as adjacency matrix
    spillover_matrix = spillover_matrix.T ## row --> column: if node i --> node j, A_{ij} != 0
    
    # Convert to percentage
    spillover_matrix *= 100      
    
    # Calculate 'to' and 'from' others
    K = spillover_matrix.shape[0]
    
    # Create results DataFrame
    results_df = pd.DataFrame(spillover_matrix, columns=results.names, index=results.names)
    
    # Increase sparcity
    vsp_df_sparse = results_df.copy()
    threshold = pd.Series(results_df.values.flatten()).quantile(scarcity_prop)
    vsp_df_sparse[vsp_df_sparse < threshold] = 0
    vsp_np_sparse = vsp_df_sparse.values
    np.fill_diagonal(vsp_np_sparse, 0)

    if standardized:
        vsp_np_sparse = vsp_np_sparse / K
        return vsp_np_sparse
    else:
        return vsp_np_sparse # for each train_x batch, dim(results_array) = [num_node, num_node]

def prepare_lagged_features(train_dataset, test_dataset, look_back_window, h):
    """Create lagged features for training and testing datasets"""
    market_indices_list = train_dataset.columns.tolist()
    
    # Create lagged features
    for market_index in market_indices_list:
        for lag in range(look_back_window):
            train_dataset[market_index + f'_{lag+1}'] = train_dataset[market_index].shift(lag+h)
            test_dataset[market_index + f'_{lag+1}'] = test_dataset[market_index].shift(lag+h)
    
    # Remove rows with NaN values
    train_dataset = train_dataset.dropna()
    test_dataset = test_dataset.dropna()
    
    return train_dataset, test_dataset, market_indices_list

def create_model_input_dicts(train_dataset, test_dataset, market_indices_list):
    """Create dictionary of model inputs for training and testing"""
    # Identify column types
    columns_lag1 = [x for x in train_dataset.columns.tolist() if x[-2:] == '_1']
    columns_lag5 = [x for x in train_dataset.columns.tolist() if (x[-2]=='_') and (float(x[-1]) in range(1,6))]
    columns_lag22 = [x for x in train_dataset.columns.tolist() if '_' in x]
    x_columns = columns_lag1 + columns_lag5 + columns_lag22
    y_columns = [x for x in train_dataset.columns.tolist() if x not in x_columns]
    
    # Define output order
    row_index_order = market_indices_list
    column_index_order_5 = [f'lag_{i}' for i in range(1,6)]
    column_index_order_22 = [f'lag_{i}' for i in range(1,23)]
    
    # Process training data
    train_dict = {}
    for date in train_dataset.index:
        train_dict[date] = process_single_date(train_dataset, date, y_columns, columns_lag1, columns_lag5, 
                                              columns_lag22, row_index_order, column_index_order_5, column_index_order_22)
    
    # Process testing data
    test_dict = {}
    for date in test_dataset.index:
        test_dict[date] = process_single_date(test_dataset, date, y_columns, columns_lag1, columns_lag5, 
                                             columns_lag22, row_index_order, column_index_order_5, column_index_order_22)
    
    return train_dict, test_dict, y_columns

def process_single_date(dataset, date, y_columns, columns_lag1, columns_lag5, columns_lag22, 
                       row_index_order, column_index_order_5, column_index_order_22):
    """Process data for a single date"""
    y = dataset.loc[date, y_columns]
    
    # Process lag-1 data
    x_lag1 = dataset.loc[date, columns_lag1]
    new_index = [ind[:-2] for ind in x_lag1.index.tolist()]
    x_lag1.index = new_index
    
    # Process lag-5 data
    x_lag5 = dataset.loc[date, columns_lag5]
    data_lag5 = {
        'Market': [index.split('_')[0] for index in x_lag5.index],
        'Lag': [f'lag_{index.split("_")[1]}' for index in x_lag5.index],
        'Value': x_lag5.values
    }
    df_lag5 = pd.DataFrame(data_lag5).pivot(index='Market', columns='Lag', values='Value')
    
    # Process lag-22 data
    x_lag22 = dataset.loc[date, columns_lag22]
    data_lag22 = {
        'Market': [index.split('_')[0] for index in x_lag22.index],
        'Lag': [f'lag_{index.split("_")[1]}' for index in x_lag22.index],
        'Value': x_lag22.values
    }
    df_lag22 = pd.DataFrame(data_lag22).pivot(index='Market', columns='Lag', values='Value')
    
    # Reindex to ensure consistent order
    x_lag1 = x_lag1.reindex(row_index_order)
    df_lag5 = df_lag5.reindex(row_index_order)
    df_lag22 = df_lag22.reindex(row_index_order)
    df_lag5 = df_lag5[column_index_order_5]
    df_lag22 = df_lag22[column_index_order_22]
    
    return {
        'y': y,
        'x_lag1': x_lag1,
        'x_lag5': df_lag5,
        'x_lag22': df_lag22
    }

def load_and_split_data(file_path, scale_factor=100, train_ratio=0.7):
    """Load data from CSV and split into train/test sets"""
    data = pd.read_csv(file_path, index_col=0) * scale_factor
    date_list = data.index.tolist()
    train_end_idx = int(len(date_list) * train_ratio)
    train_dataset = data.iloc[0:train_end_idx, :]
    test_dataset = data.iloc[train_end_idx:, :]
    
    return data, train_dataset, test_dataset