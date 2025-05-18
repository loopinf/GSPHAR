"""
GSPHAR model implementation.
This module contains the exact Graph Signal Processing for Heterogeneous Autoregressive (GSPHAR) model
from the original notebook without any modifications.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.linalg import sqrtm, eig


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
        # The input dimension for the linear layers is 3 (for lag1, lag5, lag22)
        # not the input_dim parameter which is the sequence length
        self.linear_output_real = nn.Linear(3, output_dim, bias=True)
        self.linear_output_imag = nn.Linear(3, output_dim, bias=True)

        print(f"GSPHAR model initialized with:")
        print(f"  filter_size: {filter_size}")
        print(f"  input_dim: {input_dim}")
        print(f"  output_dim: {output_dim}")
        print(f"  A shape: {A.shape}")

    def nomalized_magnet_laplacian(self, A, q, norm=True):
        A_s = (A + A.T) / 2
        D_s = np.diag(np.sum(A_s, axis=1))
        pi = np.pi
        theta_q = 2 * pi * q * (A - A.T)
        H_q = A_s * np.exp(1j * theta_q)

        if norm:
            D_s_inv = np.linalg.inv(D_s)
            D_s_inv_sqrt = sqrtm(D_s_inv)
            L = np.eye(len(D_s)) - (D_s_inv_sqrt @ A_s @ D_s_inv_sqrt) * np.exp(1j * theta_q)
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

        A = alpha * (A_p) + (1. - alpha) * A_q

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
            sub_U_dega = torch.complex(
                torch.tensor(sub_U_dega.real, dtype=torch.float32),
                torch.tensor(sub_U_dega.imag, dtype=torch.float32)
            )
            sub_U = torch.complex(
                torch.tensor(sub_U.real, dtype=torch.float32),
                torch.tensor(sub_U.imag, dtype=torch.float32)
            )
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
        self.linear_output_real = self.linear_output_real.to(device)
        self.linear_output_imag = self.linear_output_imag.to(device)

        # Compute dynamic adj_mx
        U_dega, U = self.dynamic_magnet_Laplacian(A, x_lag5, x_lag22)
        U_dega = U_dega.to(device)
        U = U.to(device)

        # Convert RV to complex domain
        x_lag1 = torch.complex(x_lag1, torch.zeros_like(x_lag1))
        x_lag5 = torch.complex(x_lag5, torch.zeros_like(x_lag5))
        x_lag22 = torch.complex(x_lag22, torch.zeros_like(x_lag22))

        # Transform to spectral domain
        # Process each batch item separately
        batch_size = x_lag1.shape[0]
        x_lag1_spectral_list = []
        x_lag5_spectral_list = []
        x_lag22_spectral_list = []

        for i in range(batch_size):
            # For each batch, get the corresponding eigenvectors
            U_dega_i = U_dega[i]  # Shape: [filter_size, filter_size]

            # Get the corresponding input tensors
            x_lag1_i = x_lag1[i]  # Shape: [filter_size, 1]
            x_lag5_i = x_lag5[i]  # Shape: [filter_size, 5]
            x_lag22_i = x_lag22[i]  # Shape: [filter_size, 22]

            # Perform the matrix multiplication
            x_lag1_spectral_i = torch.matmul(U_dega_i, x_lag1_i)  # Shape: [filter_size, 1]
            x_lag5_spectral_i = torch.matmul(U_dega_i, x_lag5_i)  # Shape: [filter_size, 5]
            x_lag22_spectral_i = torch.matmul(U_dega_i, x_lag22_i)  # Shape: [filter_size, 22]

            # Append to the lists
            x_lag1_spectral_list.append(x_lag1_spectral_i)
            x_lag5_spectral_list.append(x_lag5_spectral_i)
            x_lag22_spectral_list.append(x_lag22_spectral_i)

        # Stack the results to get tensors of shape [batch_size, filter_size, sequence_length]
        x_lag1_spectral = torch.stack(x_lag1_spectral_list)
        x_lag5_spectral = torch.stack(x_lag5_spectral_list)
        x_lag22_spectral = torch.stack(x_lag22_spectral_list)

        # Apply 1D convolution to lag5 and lag22
        # Reshape to match the expected input shape for Conv1d
        # Conv1d expects input of shape [batch_size, channels, sequence_length]
        # Our data is [batch_size, sequence_length, channels], so we need to permute
        x_lag5_spectral_real = x_lag5_spectral.real.permute(0, 1, 2)  # [batch_size, filter_size, 5]
        x_lag5_spectral_imag = x_lag5_spectral.imag.permute(0, 1, 2)  # [batch_size, filter_size, 5]

        x_lag22_spectral_real = x_lag22_spectral.real.permute(0, 1, 2)  # [batch_size, filter_size, 22]
        x_lag22_spectral_imag = x_lag22_spectral.imag.permute(0, 1, 2)  # [batch_size, filter_size, 22]

        # Apply 1D convolution to lag5 and lag22
        x_lag5_conv_real = self.conv1d_lag5(x_lag5_spectral_real)
        x_lag5_conv_imag = self.conv1d_lag5(x_lag5_spectral_imag)

        x_lag22_conv_real = self.conv1d_lag22(x_lag22_spectral_real)
        x_lag22_conv_imag = self.conv1d_lag22(x_lag22_spectral_imag)

        # Combine the real and imaginary parts
        x_lag5_conv = torch.complex(x_lag5_conv_real, x_lag5_conv_imag)
        x_lag22_conv = torch.complex(x_lag22_conv_real, x_lag22_conv_imag)

        # No need to permute back since we're already in the right shape
        # The output of conv1d is [batch_size, filter_size, 1]

        # Concatenate the lag1, lag5, and lag22 features
        # Ensure all tensors have the same shape before concatenation
        x_lag1_spectral_squeezed = x_lag1_spectral.squeeze(-1)  # [batch_size, filter_size]
        x_lag5_conv_squeezed = x_lag5_conv.squeeze(-1)  # [batch_size, filter_size]
        x_lag22_conv_squeezed = x_lag22_conv.squeeze(-1)  # [batch_size, filter_size]

        # Stack them along a new dimension to create [batch_size, filter_size, 3]
        lagged_rv_spectral = torch.stack([
            x_lag1_spectral_squeezed,
            x_lag5_conv_squeezed,
            x_lag22_conv_squeezed
        ], dim=-1)

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

        # Reshape to match the target shape [batch_size, filter_size, horizon]
        y_hat = y_hat.view(y_hat.shape[0], y_hat.shape[1], 1)
        return y_hat
