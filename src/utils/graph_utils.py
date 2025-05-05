"""
Graph utility functions for GSPHAR.
This module contains functions for graph-related operations.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR


def compute_spillover_index(data, horizon, lag, scarcity_prop, standardized=True):
    """
    Compute the spillover index.
    
    Args:
        data (pd.DataFrame): Input data.
        horizon (int): Forecast horizon.
        lag (int): Number of lags.
        scarcity_prop (float): Sparsity proportion.
        standardized (bool, optional): Whether to standardize the result. Defaults to True.
        
    Returns:
        numpy.ndarray: Spillover matrix.
    """
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
    
    # Compute GFEVD
    gfevd = []
    for h in range(horizon):
        # Compute GFEVD for horizon h
        gfevd_h = np.zeros_like(Sigma_A[0])
        for i in range(h + 1):
            gfevd_h += Sigma_A[i] / np.diag(A_Sigma_A[i])[:, np.newaxis]
        gfevd_h /= (h + 1)
        gfevd.append(gfevd_h)
    
    # Aggregate results over n_ahead steps
    spillover_matrix = gfevd[-1]
    
    # VSP from row to column so can be used as adjacency matrix
    spillover_matrix = spillover_matrix.T  # row --> column: if node i --> node j, A_{ij} != 0
    
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
        return vsp_np_sparse  # for each train_x batch, dim(results_array) = [num_node, num_node]
