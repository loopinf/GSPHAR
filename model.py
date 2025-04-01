import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from scipy.linalg import sqrtm, eig

class GSPHAR(nn.Module):
    def __init__(self, input_dim, output_dim, filter_size, A):
        super(GSPHAR, self).__init__()
        self.A = torch.from_numpy(A)
        self.filter_size = filter_size
        self.conv1d_lag5 = nn.Conv1d(in_channels=filter_size, out_channels=filter_size, 
                                    kernel_size=5, groups=filter_size, bias=False)
        nn.init.constant_(self.conv1d_lag5.weight, 1.0 / 5)
        self.conv1d_lag22 = nn.Conv1d(in_channels=filter_size, out_channels=filter_size, 
                                     kernel_size=22, groups=filter_size, bias=False)
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