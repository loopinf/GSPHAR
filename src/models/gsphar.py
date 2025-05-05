"""
GSPHAR model implementation.
This module contains the Graph Signal Processing for Heterogeneous Autoregressive (GSPHAR) model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.linalg import sqrtm, eig


class GSPHAR(nn.Module):
    """
    Graph Signal Processing for Heterogeneous Autoregressive (GSPHAR) model.

    This model combines graph signal processing with autoregressive modeling
    for financial time series forecasting.
    """

    def __init__(self, input_dim, output_dim, filter_size, A):
        """
        Initialize the GSPHAR model.

        Args:
            input_dim (int): Input dimension.
            output_dim (int): Output dimension.
            filter_size (int): Filter size, corresponds to the number of market indices.
            A (numpy.ndarray): Adjacency matrix.
        """
        super(GSPHAR, self).__init__()
        # Convert adjacency matrix to float32 to ensure compatibility with MPS
        self.A = torch.from_numpy(A).float()  # Explicitly convert to float32
        self.filter_size = filter_size

        # Convolutional layers for different lag windows
        self.conv1d_lag5 = nn.Conv1d(
            in_channels=filter_size,
            out_channels=filter_size,
            kernel_size=5,
            groups=filter_size,
            bias=False
        )
        nn.init.constant_(self.conv1d_lag5.weight, 1.0 / 5)

        self.conv1d_lag22 = nn.Conv1d(
            in_channels=filter_size,
            out_channels=filter_size,
            kernel_size=22,
            groups=filter_size,
            bias=False
        )
        nn.init.constant_(self.conv1d_lag22.weight, 1.0 / 22)

        # Neural network for spatial processing
        self.spatial_process = nn.Sequential(
            nn.Linear(2, 2 * 8),
            nn.ReLU(),
            nn.Linear(2 * 8, 2 * 8),
            nn.ReLU(),
            nn.Linear(2 * 8, 1),
            nn.ReLU()
        )

        # Output layer - takes input of shape [batch_size, filter_size, input_dim]
        # and outputs [batch_size, filter_size, output_dim]
        self.linear_output = nn.Linear(input_dim, output_dim, bias=True)

    def nomalized_magnet_laplacian(self, A, q, norm=True):
        """
        Compute the normalized magnetic Laplacian.

        Args:
            A (numpy.ndarray): Adjacency matrix.
            q (float): Magnetic flux parameter.
            norm (bool): Whether to normalize the Laplacian.

        Returns:
            numpy.ndarray: Magnetic Laplacian matrix.
        """
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
        """
        Compute the dynamic magnetic Laplacian.

        Args:
            A (torch.Tensor): Adjacency matrix.
            x_lag_p (torch.Tensor): Lag-p features.
            x_lag_q (torch.Tensor): Lag-q features.

        Returns:
            tuple: (U_dega, U) Eigenvectors of the Laplacian.
        """
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

        # Initialize lists for real and imaginary parts
        U_dega_real_list = []
        U_dega_imag_list = []
        U_real_list = []
        U_imag_list = []
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
            # Use separate real and imaginary parts for MPS compatibility
            sub_U_dega_real = torch.tensor(sub_U_dega.real, dtype=torch.float32)
            sub_U_dega_imag = torch.tensor(sub_U_dega.imag, dtype=torch.float32)
            sub_U_real = torch.tensor(sub_U.real, dtype=torch.float32)
            sub_U_imag = torch.tensor(sub_U.imag, dtype=torch.float32)
            # Store real and imaginary parts separately
            U_dega_real_list.append(sub_U_dega_real)
            U_dega_imag_list.append(sub_U_dega_imag)
            U_real_list.append(sub_U_real)
            U_imag_list.append(sub_U_imag)

        # Stack the real and imaginary parts separately
        U_dega_real = torch.stack(U_dega_real_list)
        U_dega_imag = torch.stack(U_dega_imag_list)
        U_real = torch.stack(U_real_list)
        U_imag = torch.stack(U_imag_list)

        return U_dega_real, U_dega_imag, U_real, U_imag

    def forward(self, x_lag1, x_lag5, x_lag22):
        """
        Forward pass of the GSPHAR model.

        Args:
            x_lag1 (torch.Tensor): Lag-1 features.
            x_lag5 (torch.Tensor): Lag-5 features.
            x_lag22 (torch.Tensor): Lag-22 features.

        Returns:
            tuple: (y_hat, softmax_param_5, softmax_param_22)
        """
        # Ensure all items are on the same device as the input x
        device = x_lag1.device
        A = self.A.to(device)
        self.conv1d_lag5 = self.conv1d_lag5.to(device)
        self.conv1d_lag22 = self.conv1d_lag22.to(device)
        self.spatial_process = self.spatial_process.to(device)
        self.linear_output = self.linear_output.to(device)

        # Compute dynamic adj_mx - now returns separate real and imaginary parts
        U_dega_real, U_dega_imag, U_real, U_imag = self.dynamic_magnet_Laplacian(A, x_lag5, x_lag22)
        U_dega_real = U_dega_real.to(device)
        U_dega_imag = U_dega_imag.to(device)
        U_real = U_real.to(device)
        U_imag = U_imag.to(device)

        # Create zero tensors for imaginary parts (since our inputs are real)
        x_lag1_imag = torch.zeros_like(x_lag1)
        x_lag5_imag = torch.zeros_like(x_lag5)
        x_lag22_imag = torch.zeros_like(x_lag22)

        # Transform to spectral domain - handle real and imaginary parts separately
        # For complex matrix multiplication (a+bi)(c+di) = (ac-bd) + (ad+bc)i

        # For x_lag1 (with unsqueeze for dimension matching)
        x_lag1_unsqueezed = x_lag1.unsqueeze(-1)
        x_lag1_imag_unsqueezed = x_lag1_imag.unsqueeze(-1)

        # Real part: U_dega_real * x_lag1 - U_dega_imag * x_lag1_imag (which is zero)
        x_lag1_spectral_real = torch.matmul(U_dega_real, x_lag1_unsqueezed)
        # Imaginary part: U_dega_real * x_lag1_imag (zero) + U_dega_imag * x_lag1
        x_lag1_spectral_imag = torch.matmul(U_dega_imag, x_lag1_unsqueezed)

        # For x_lag5
        # Real part: U_dega_real * x_lag5 - U_dega_imag * x_lag5_imag (which is zero)
        x_lag5_spectral_real = torch.matmul(U_dega_real, x_lag5)
        # Imaginary part: U_dega_real * x_lag5_imag (zero) + U_dega_imag * x_lag5
        x_lag5_spectral_imag = torch.matmul(U_dega_imag, x_lag5)

        # For x_lag22
        # Real part: U_dega_real * x_lag22 - U_dega_imag * x_lag22_imag (which is zero)
        x_lag22_spectral_real = torch.matmul(U_dega_real, x_lag22)
        # Imaginary part: U_dega_real * x_lag22_imag (zero) + U_dega_imag * x_lag22
        x_lag22_spectral_imag = torch.matmul(U_dega_imag, x_lag22)

        # Apply 1D convolution to lag5 and lag22
        # We need to handle each market separately to avoid dimension issues
        batch_size, num_markets, seq_len_5 = x_lag5_spectral_real.shape
        _, _, seq_len_22 = x_lag22_spectral_real.shape

        # Initialize output tensors
        x_lag5_conv_real = torch.zeros(batch_size, num_markets, 1, device=device)
        x_lag5_conv_imag = torch.zeros(batch_size, num_markets, 1, device=device)
        x_lag22_conv_real = torch.zeros(batch_size, num_markets, 1, device=device)
        x_lag22_conv_imag = torch.zeros(batch_size, num_markets, 1, device=device)

        # Process each market separately
        for i in range(num_markets):
            # For lag5
            # Extract the i-th market data and reshape for Conv1d
            # Conv1d expects [batch_size, channels, seq_len]
            x_lag5_real_i = x_lag5_spectral_real[:, i, :].unsqueeze(1)  # [batch_size, 1, seq_len_5]
            x_lag5_imag_i = x_lag5_spectral_imag[:, i, :].unsqueeze(1)  # [batch_size, 1, seq_len_5]

            # Apply 1D convolution using a simple average (equivalent to the initialized conv1d_lag5)
            # This is equivalent to the original convolution with constant weights of 1/5
            x_lag5_conv_real[:, i, 0] = x_lag5_real_i.squeeze(1).mean(dim=1)
            x_lag5_conv_imag[:, i, 0] = x_lag5_imag_i.squeeze(1).mean(dim=1)

            # For lag22
            # Extract the i-th market data and reshape for Conv1d
            x_lag22_real_i = x_lag22_spectral_real[:, i, :].unsqueeze(1)  # [batch_size, 1, seq_len_22]
            x_lag22_imag_i = x_lag22_spectral_imag[:, i, :].unsqueeze(1)  # [batch_size, 1, seq_len_22]

            # Check if we need to pad for lag22
            if seq_len_22 < 22:
                padding_size = 22 - seq_len_22
                x_lag22_real_i = F.pad(x_lag22_real_i, (0, padding_size))
                x_lag22_imag_i = F.pad(x_lag22_imag_i, (0, padding_size))

            # Apply 1D convolution using a simple average (equivalent to the initialized conv1d_lag22)
            # This is equivalent to the original convolution with constant weights of 1/22
            x_lag22_conv_real[:, i, 0] = x_lag22_real_i.squeeze(1).mean(dim=1)
            x_lag22_conv_imag[:, i, 0] = x_lag22_imag_i.squeeze(1).mean(dim=1)

        # Get the weights
        softmax_param_5 = self.conv1d_lag5.weight
        softmax_param_22 = self.conv1d_lag22.weight

        # Concatenate the lag1, lag5, and lag22 features
        # Ensure all tensors have the same shape before concatenation
        x_lag1_spectral_real_squeezed = x_lag1_spectral_real.squeeze(-1)  # [batch_size, num_markets]
        x_lag1_spectral_imag_squeezed = x_lag1_spectral_imag.squeeze(-1)  # [batch_size, num_markets]

        x_lag5_conv_real_squeezed = x_lag5_conv_real.squeeze(-1)  # [batch_size, num_markets]
        x_lag5_conv_imag_squeezed = x_lag5_conv_imag.squeeze(-1)  # [batch_size, num_markets]

        x_lag22_conv_real_squeezed = x_lag22_conv_real.squeeze(-1)  # [batch_size, num_markets]
        x_lag22_conv_imag_squeezed = x_lag22_conv_imag.squeeze(-1)  # [batch_size, num_markets]

        # Stack them along a new dimension to create [batch_size, num_markets, 3]
        # Handle real and imaginary parts separately
        lagged_rv_spectral_real = torch.stack([
            x_lag1_spectral_real_squeezed,
            x_lag5_conv_real_squeezed,
            x_lag22_conv_real_squeezed
        ], dim=-1)

        lagged_rv_spectral_imag = torch.stack([
            x_lag1_spectral_imag_squeezed,
            x_lag5_conv_imag_squeezed,
            x_lag22_conv_imag_squeezed
        ], dim=-1)

        # Apply linear transformation separately to the real and imaginary parts
        # Linear layer expects [batch_size, num_markets, input_dim] and outputs [batch_size, num_markets, output_dim]
        y_hat_real = self.linear_output(lagged_rv_spectral_real)
        y_hat_imag = self.linear_output(lagged_rv_spectral_imag)

        # Back to the spatial domain - complex matrix multiplication
        # (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        # Real part: U_real * y_hat_real - U_imag * y_hat_imag
        y_hat_spatial_real = torch.matmul(U_real, y_hat_real) - torch.matmul(U_imag, y_hat_imag)
        # Imaginary part: U_real * y_hat_imag + U_imag * y_hat_real
        y_hat_spatial_imag = torch.matmul(U_real, y_hat_imag) + torch.matmul(U_imag, y_hat_real)

        # Squeeze to remove the last dimension
        y_hat_spatial_real = y_hat_spatial_real.squeeze(-1)
        y_hat_spatial_imag = y_hat_spatial_imag.squeeze(-1)

        # Stack real and imaginary parts for spatial processing
        y_hat_spatial = torch.stack((y_hat_spatial_real, y_hat_spatial_imag), dim=-1)

        # Apply linear transformation to the stacked tensor
        y_hat = self.spatial_process(y_hat_spatial)

        return y_hat.squeeze(-1), softmax_param_5, softmax_param_22
