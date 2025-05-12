"""
Original GSPHAR implementation extracted from d-GSPHAR.ipynb.
This file is used for validation purposes.
"""

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


## Save model
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

## Load model
def load_model(name, model):
    checkpoint = torch.load(f'models/{name}.tar', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    num_L = checkpoint['layer']
    mae_loss = checkpoint['loss']
    print(f"Loaded model: {name}")
    print(f"MAE loss: {mae_loss}")
    return model, mae_loss


class GSPHAR(nn.Module):
    # linear transformation
    def __init__ (self, input_dim, output_dim, filter_size, A):
        super(GSPHAR,self).__init__()
        self.A = torch.from_numpy(A)
        self.filter_size = filter_size
        self.conv1d_lag5 = nn.Conv1d(in_channels = filter_size, out_channels = filter_size, kernel_size = 5, groups = filter_size, bias = False) # groups=1 markets share similarity
        nn.init.constant_(self.conv1d_lag5.weight, 1.0 / 5)
        self.conv1d_lag22 = nn.Conv1d(in_channels = filter_size, out_channels = filter_size, kernel_size = 22, groups = filter_size, bias = False) # groups=1
        nn.init.constant_(self.conv1d_lag22.weight, 1.0 / 22)
        self.spatial_process = nn.Sequential(
            nn.Linear(2, 2 * 8),
            nn.ReLU(),
            nn.Linear(2 * 8, 2 * 8),
            nn.ReLU(),
            nn.Linear(2 * 8, 1),
            nn.ReLU()
        )
        self.linear_output = nn.Linear(input_dim, output_dim, bias=True)
    
    def nomalized_magnet_laplacian(self, A, q, norm = True):
        A_s = (A + A.T)/2
        D_s = np.diag(np.sum(A_s, axis=1))
        pi = np.pi
        theta_q = 2 * pi * q * (A - A.T)
        H_q = A_s * np.exp(1j*theta_q)
        if norm == True:
            D_s_inv = np.linalg.inv(D_s)
            D_s_inv_sqrt = sqrtm(D_s_inv)
            L = np.eye(len(D_s)) - (D_s_inv_sqrt @ A_s @ D_s_inv_sqrt) * np.exp(1j*theta_q)
        else:
            L = D_s - H_q

        return L
    
    def dynamic_magnet_Laplacian(self, A, x_lag_p, x_lag_q): # p < q
        A = (A - A.T)
        A[A < 0] = 0
        
        # Center the data by subtracting the mean along the last dimension (axis=2)
        mean_p = x_lag_p.mean(dim=2, keepdim=True)
        x_lag_p_centered = x_lag_p - mean_p
        # Compute the covariance matrices: (batch_size, num_features, num_features)
        cov_matrices_p = torch.bmm(x_lag_p_centered, x_lag_p_centered.transpose(1, 2)) / (x_lag_p.shape[2] - 1)
        # Compute the standard deviations: (batch_size, num_features)
        stddev_p = x_lag_p_centered.std(dim=2, unbiased=True)
        # Avoid division by zero
        stddev_p[stddev_p == 0] = 1
        # Outer product of the standard deviations to get a matrix of standard deviations for normalization
        stddev_matrix_p = stddev_p.unsqueeze(2) * stddev_p.unsqueeze(1)
        # Compute the correlation matrices by normalizing the covariance matrices
        correlation_matrices_p = cov_matrices_p / stddev_matrix_p
        abs_correlation_matrices_p = abs(correlation_matrices_p)

        # Center the data by subtracting the mean along the last dimension (axis=2)
        mean_q = x_lag_q.mean(dim=2, keepdim=True)
        x_lag_q_centered = x_lag_q - mean_q
        # Compute the covariance matrices: (batch_size, num_features, num_features)
        cov_matrices_q = torch.bmm(x_lag_q_centered, x_lag_q_centered.transpose(1, 2)) / (x_lag_q.shape[2] - 1)
        # Compute the standard deviations: (batch_size, num_features)
        stddev_q = x_lag_q_centered.std(dim=2, unbiased=True)
        # Avoid division by zero
        stddev_q[stddev_q == 0] = 1
        # Outer product of the standard deviations to get a matrix of standard deviations for normalization
        stddev_matrix_q = stddev_q.unsqueeze(2) * stddev_q.unsqueeze(1)
        # Compute the correlation matrices by normalizing the covariance matrices
        correlation_matrices_q = cov_matrices_q / stddev_matrix_q
        abs_correlation_matrices_q = abs(correlation_matrices_q)

        alpha = 0.5
        
        A_p = abs_correlation_matrices_p * A
        A_q = abs_correlation_matrices_q * A
        
        A =  alpha * (A_p) +  (1.-alpha) * A_q

        # normalization
        A = F.softmax(A, dim=1)
        
        U_dega_list = []
        U_list = []
        for i in range(len(A)):
            sub_A = A[i]
            sub_L = self.nomalized_magnet_laplacian(sub_A.cpu().numpy(), 0.25)
            eigenvalues, eigenvectors = eig(sub_L)
            sub_Lambda = eigenvalues.real
            sub_U_dega = eigenvectors
            sub_U = eigenvectors.T.conj()
            sorted_indices = np.argsort(sub_Lambda)  # Sort in ascending order
            sub_U_dega_sorted = sub_U_dega[sorted_indices, :]
            sub_U_sorted = sub_U[:, sorted_indices] 
            sub_U_dega =  torch.complex(torch.tensor(sub_U_dega.real, dtype=torch.float32), torch.tensor(sub_U_dega.imag, dtype=torch.float32))
            sub_U =  torch.complex(torch.tensor(sub_U.real, dtype=torch.float32), torch.tensor(sub_U.imag, dtype=torch.float32))
            U_dega_list.append(sub_U_dega)
            U_list.append(sub_U)
        
        U_dega = torch.stack(U_dega_list)
        U = torch.stack(U_list)
        
        return U_dega, U 
        
    def forward(self, x_lag1, x_lag5, x_lag22):
        # Ensure all items are on the same device as the input x
        device = x_lag1.device
        A = self.A.to(device)
        self.conv1d_lag5 = self.conv1d_lag5.to(device)
        self.conv1d_lag22 = self.conv1d_lag22.to(device)
        self.spatial_process = self.spatial_process.to(device)
        self.linear_output_real = self.linear_output.to(device)
        self.linear_output_imag = self.linear_output.to(device)
        
        # Compute dynamic adj_mx
        U_dega, U = self.dynamic_magnet_Laplacian(A,x_lag5, x_lag22)
        U_dega = U_dega.to(device)
        U = U.to(device)
    
        # Convert RV to complex domain
        x_lag1 = torch.complex(x_lag1, torch.zeros_like(x_lag1))
        x_lag5 = torch.complex(x_lag5, torch.zeros_like(x_lag5))
        x_lag22 = torch.complex(x_lag22, torch.zeros_like(x_lag22))

        # Spectral domain operations on lag-5
        x_lag5 = torch.matmul(U_dega, x_lag5)
        exp_param_5 = torch.exp(self.conv1d_lag5.weight)
        sum_exp_param_5 = torch.sum(exp_param_5, dim=-1, keepdim=True)
        softmax_param_5 = exp_param_5/sum_exp_param_5
        x_lag5_real = F.conv1d(input=x_lag5.real, weight=softmax_param_5, bias=None, groups=self.filter_size)
        x_lag5_imag = F.conv1d(input=x_lag5.imag, weight=softmax_param_5, bias=None, groups=self.filter_size)
        x_lag5 = torch.complex(x_lag5_real, x_lag5_imag)
        x_lag5 = x_lag5.squeeze(-1)

        # Spectral domain operations on lag-22
        x_lag22 = torch.matmul(U_dega, x_lag22)
        exp_param_22 = torch.exp(self.conv1d_lag22.weight)
        sum_exp_param_22 = torch.sum(exp_param_22, dim=-1, keepdim=True)
        softmax_param_22 = exp_param_22/sum_exp_param_22
        x_lag22_real = F.conv1d(input=x_lag22.real, weight=softmax_param_22, bias=None, groups=self.filter_size)
        x_lag22_imag = F.conv1d(input=x_lag22.imag, weight=softmax_param_22, bias=None, groups=self.filter_size)
        x_lag22 = torch.complex(x_lag22_real, x_lag22_imag)
        x_lag22 = x_lag22.squeeze(-1)

        # Lag-1 processing
        x_lag1 = torch.matmul(U_dega, x_lag1.unsqueeze(-1))
        x_lag1 = x_lag1.squeeze(-1)

        # Combine lagged responses in the spectral domain
        lagged_rv_spectral = torch.stack((x_lag1, x_lag5, x_lag22), dim=-1)
        
        # Apply linear transformation separately to the real and imaginary parts
        y_hat_real = self.linear_output_real(lagged_rv_spectral.real)
        y_hat_imag = self.linear_output_imag(lagged_rv_spectral.imag)
        y_hat_spectral = torch.complex(y_hat_real, y_hat_imag)

        # Back to the spatial domain
        y_hat = torch.matmul(U, y_hat_spectral)

        y_hat_real = y_hat.real # [batch_size, num_markets, 1]
        y_hat_imag = y_hat.imag # [batch_size, num_markets, 1]
        y_hat_real = y_hat_real.squeeze(-1)
        y_hat_imag = y_hat_imag.squeeze(-1)
        
        y_hat_spatial = torch.stack((y_hat_real, y_hat_imag), dim=-1)
        
        # Apply linear transformation separately to the real and imaginary parts
        y_hat = self.spatial_process(y_hat_spatial)
        
        return y_hat.squeeze(-1), softmax_param_5, softmax_param_22


class GSPHAR_Dataset(Dataset):
    def __init__(self, dict):
        self.dict = dict

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
    final_conv1d_lag5_weights = None
    final_conv1d_lag22_weights = None
    
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
            save_model(f'GSPHAR_24_magnet_dynamic_h5', model, None, best_loss_val)
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
