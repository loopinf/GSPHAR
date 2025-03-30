import torch
import os
import numpy as np
import pandas as pd
from scipy.sparse import linalg
from statsmodels.tsa.api import VAR
from scipy import stats
from scipy.linalg import sqrtm
from scipy.linalg import eig

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

def save_model(name, model, num_L = None, best_loss_val = None):
    if not os.path.exists('models/'):
            os.makedirs('models/')
    # Prepare the model state dictionary
    config = {
        'model_state_dict': model.state_dict(),
        'layer': num_L,
        'loss': best_loss_val
    }
    # Save the model state dictionary
    torch.save(config, f'models/{name}.tar')
    return

def load_model(name, model):
    checkpoint = torch.load(f'models/{name}.tar', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    num_L = checkpoint['layer']
    mae_loss = checkpoint['loss']
    print(f"Loaded model: {name}")
    print(f"MAE loss: {mae_loss}")
    return model, mae_loss
