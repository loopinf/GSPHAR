import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from scipy.linalg import sqrtm, eig

class GSPHAR(nn.Module):
    def __init__(self, input_dim, output_dim, n_nodes, A):
        """
        Initialize GSPHAR model
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            n_nodes: Number of nodes in the network (number of assets)
            A: Adjacency matrix (should be n_nodes x n_nodes)
        """
        super(GSPHAR, self).__init__()
        
        # Validate input dimensions
        if A.shape != (n_nodes, n_nodes):
            raise ValueError(f"Adjacency matrix shape {A.shape} does not match n_nodes {n_nodes}")
            
        self.A = torch.from_numpy(A)
        self.n_nodes = n_nodes
        self.conv1d_lag4 = nn.Conv1d(in_channels=n_nodes, out_channels=n_nodes, 
                                    kernel_size=4, groups=n_nodes, bias=False)
        nn.init.constant_(self.conv1d_lag4.weight, 1.0 / 4)
        self.conv1d_lag24 = nn.Conv1d(in_channels=n_nodes, out_channels=n_nodes, 
                                     kernel_size=24, groups=n_nodes, bias=False)
        nn.init.constant_(self.conv1d_lag24.weight, 1.0 / 24)
        self.spatial_process = nn.Sequential(
            nn.Linear(2, 2 * 8),
            nn.ReLU(),
            nn.Linear(2 * 8, 2 * 8),
            nn.ReLU(),
            nn.Linear(2 * 8, 1),
            nn.ReLU()
        )
        # Replace single linear_output with separate real and imaginary components
        self.linear_output_real = nn.Linear(input_dim, output_dim, bias=True)
        self.linear_output_imag = nn.Linear(input_dim, output_dim, bias=True)
    
    def nomalized_magnet_laplacian(self, A, q, norm=True):
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
    
    def dynamic_magnet_Laplacian(self, A, x_lag_p, x_lag_q):
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
        
    def forward(self, x_lag1, x_lag4, x_lag24):
        # Ensure all items are on the same device as the input x
        device = x_lag1.device
        A = self.A.to(device)
        self.conv1d_lag4 = self.conv1d_lag4.to(device)
        self.conv1d_lag24 = self.conv1d_lag24.to(device)
        self.spatial_process = self.spatial_process.to(device)
        self.linear_output_real = self.linear_output_real.to(device)
        self.linear_output_imag = self.linear_output_imag.to(device)
        
        # Compute dynamic adj_mx
        U_dega, U = self.dynamic_magnet_Laplacian(A,x_lag4, x_lag24)
        U_dega = U_dega.to(device)
        U = U.to(device)
    
        # Convert RV to complex domain
        x_lag1 = torch.complex(x_lag1, torch.zeros_like(x_lag1))
        x_lag4 = torch.complex(x_lag4, torch.zeros_like(x_lag4))
        x_lag24 = torch.complex(x_lag24, torch.zeros_like(x_lag24))

        # Spectral domain operations on lag-4
        x_lag4 = torch.matmul(U_dega, x_lag4)
        exp_param_4 = torch.exp(self.conv1d_lag4.weight)
        sum_exp_param_4 = torch.sum(exp_param_4, dim=-1, keepdim=True)
        softmax_param_4 = exp_param_4/sum_exp_param_4
        x_lag4_real = F.conv1d(input=x_lag4.real, weight=softmax_param_4, bias=None, groups=self.n_nodes)
        x_lag4_imag = F.conv1d(input=x_lag4.imag, weight=softmax_param_4, bias=None, groups=self.n_nodes)
        x_lag4 = torch.complex(x_lag4_real, x_lag4_imag)
        x_lag4 = x_lag4.squeeze(-1)

        # Spectral domain operations on lag-24
        x_lag24 = torch.matmul(U_dega, x_lag24)
        exp_param_24 = torch.exp(self.conv1d_lag24.weight)
        sum_exp_param_24 = torch.sum(exp_param_24, dim=-1, keepdim=True)
        softmax_param_24 = exp_param_24/sum_exp_param_24
        x_lag24_real = F.conv1d(input=x_lag24.real, weight=softmax_param_24, bias=None, groups=self.n_nodes)
        x_lag24_imag = F.conv1d(input=x_lag24.imag, weight=softmax_param_24, bias=None, groups=self.n_nodes)
        x_lag24 = torch.complex(x_lag24_real, x_lag24_imag)
        x_lag24 = x_lag24.squeeze(-1)

        # Lag-1 processing
        x_lag1 = torch.matmul(U_dega, x_lag1.unsqueeze(-1))
        x_lag1 = x_lag1.squeeze(-1)

        # Combine lagged responses in the spectral domain
        lagged_rv_spectral = torch.stack((x_lag1, x_lag4, x_lag24), dim=-1)
        
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
        
        return y_hat.squeeze(-1), softmax_param_4, softmax_param_24


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
        x_lag4 = dfs_dict['x_lag4'].values
        x_lag24 = dfs_dict['x_lag24'].values
        
        y_tensor = torch.tensor(y, dtype=torch.float32)
        x_lag1_tensor = torch.tensor(x_lag1, dtype=torch.float32)
        x_lag4_tensor = torch.tensor(x_lag4, dtype=torch.float32)
        x_lag24_tensor = torch.tensor(x_lag24, dtype=torch.float32)
        return x_lag1_tensor, x_lag4_tensor, x_lag24_tensor, y_tensor
