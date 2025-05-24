"""
Flexible GSPHAR model that can handle custom lags.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.linalg import sqrtm, eig


class FlexibleGSPHAR(nn.Module):
    """
    Flexible GSPHAR model that can handle custom lags.

    Args:
        lags (list): List of lag values to use (e.g., [1, 4, 24]).
        output_dim (int): Output dimension (prediction horizon).
        filter_size (int): Number of features/symbols.
        A (np.ndarray): Adjacency matrix.
    """
    def __init__(self, lags, output_dim, filter_size, A):
        super(FlexibleGSPHAR, self).__init__()
        self.A = torch.from_numpy(A)
        self.filter_size = filter_size
        self.lags = sorted(lags)  # Sort lags in ascending order
        self.num_lags = len(lags)

        # Create convolution layers for each lag (except lag 1)
        self.conv_layers = nn.ModuleDict()
        for lag in self.lags:
            if lag > 1:  # Skip lag 1 as it doesn't need convolution
                conv = nn.Conv1d(
                    in_channels=filter_size,
                    out_channels=filter_size,
                    kernel_size=lag,
                    groups=filter_size,
                    bias=False
                )
                # Initialize with equal weights (average)
                nn.init.constant_(conv.weight, 1.0 / lag)
                self.conv_layers[f'conv_lag{lag}'] = conv

        # Spatial processing layers
        self.spatial_process = nn.Sequential(
            nn.Linear(2, 2 * 8),
            nn.ReLU(),
            nn.Linear(2 * 8, 2 * 8),
            nn.ReLU(),
            nn.Linear(2 * 8, 1),
            nn.ReLU()
        )

        # Linear output layers
        # The input dimension is the number of lags
        self.linear_output_real = nn.Linear(self.num_lags, output_dim, bias=True)
        self.linear_output_imag = nn.Linear(self.num_lags, output_dim, bias=True)

        print(f"FlexibleGSPHAR model initialized with:")
        print(f"  lags: {self.lags}")
        print(f"  num_lags: {self.num_lags}")
        print(f"  filter_size: {filter_size}")
        print(f"  output_dim: {output_dim}")
        print(f"  A shape: {A.shape}")

    def nomalized_magnet_laplacian(self, A, q, norm=True):
        """
        Compute the normalized magnetic Laplacian.

        Args:
            A (np.ndarray): Adjacency matrix.
            q (float): Magnetic flux parameter.
            norm (bool): Whether to normalize the Laplacian.

        Returns:
            np.ndarray: Magnetic Laplacian.
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
            x_lag_p (torch.Tensor): First lag tensor.
            x_lag_q (torch.Tensor): Second lag tensor.

        Returns:
            tuple: (U_dega, U) - Eigenvectors and their conjugate transpose.
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

    def forward(self, *x_lags):
        """
        Forward pass of the FlexibleGSPHAR model.

        Args:
            *x_lags: Variable number of lag tensors.

        Returns:
            torch.Tensor: Predictions.
        """
        # Check if the number of inputs matches the number of lags
        # If it's one more, assume the last element is the target and exclude it
        if len(x_lags) == self.num_lags + 1:
            x_lags = x_lags[:-1]

        # Ensure we have the correct number of lags
        assert len(x_lags) == self.num_lags, f"Expected {self.num_lags} lags, got {len(x_lags)}"

        # Ensure all items are on the same device as the first input
        device = x_lags[0].device
        A = self.A.to(device)

        # Move all modules to the same device
        for name, module in self.named_modules():
            module.to(device)

        # Use the two largest lags for the dynamic Laplacian
        # (similar to how the original model used lag5 and lag22)
        p_idx = -2  # Second-to-last lag (second largest)
        q_idx = -1  # Last lag (largest)

        # Compute dynamic adj_mx
        U_dega, U = self.dynamic_magnet_Laplacian(A, x_lags[p_idx], x_lags[q_idx])
        U_dega = U_dega.to(device)
        U = U.to(device)

        # Convert all lags to complex domain
        x_lags_complex = [torch.complex(x_lag, torch.zeros_like(x_lag)) for x_lag in x_lags]

        # Transform to spectral domain
        # Process each batch item separately
        batch_size = x_lags[0].shape[0]
        x_lags_spectral_lists = [[] for _ in range(self.num_lags)]

        for i in range(batch_size):
            # For each batch, get the corresponding eigenvectors
            U_dega_i = U_dega[i]  # Shape: [filter_size, filter_size]

            # Process each lag
            for j, x_lag_complex in enumerate(x_lags_complex):
                # Get the corresponding input tensor
                x_lag_i = x_lag_complex[i]  # Shape: [filter_size, lag_length]

                # Perform the matrix multiplication
                x_lag_spectral_i = torch.matmul(U_dega_i, x_lag_i)  # Shape: [filter_size, lag_length]

                # Append to the list
                x_lags_spectral_lists[j].append(x_lag_spectral_i)

        # Stack the results to get tensors of shape [batch_size, filter_size, lag_length]
        x_lags_spectral = [torch.stack(x_lag_list) for x_lag_list in x_lags_spectral_lists]

        # Apply 1D convolution to lags > 1
        x_lags_conv = []
        for j, lag in enumerate(self.lags):
            if lag == 1:
                # For lag 1, no convolution needed
                x_lags_conv.append(x_lags_spectral[j])
            else:
                # For other lags, apply convolution
                conv_name = f'conv_lag{lag}'
                conv_layer = self.conv_layers[conv_name]

                # Prepare input for convolution
                x_lag_spectral_real = x_lags_spectral[j].real.permute(0, 1, 2)  # [batch_size, filter_size, lag_length]
                x_lag_spectral_imag = x_lags_spectral[j].imag.permute(0, 1, 2)  # [batch_size, filter_size, lag_length]

                # Apply convolution
                x_lag_conv_real = conv_layer(x_lag_spectral_real)
                x_lag_conv_imag = conv_layer(x_lag_spectral_imag)

                # Combine real and imaginary parts
                x_lag_conv = torch.complex(x_lag_conv_real, x_lag_conv_imag)
                x_lags_conv.append(x_lag_conv)

        # Squeeze and stack the convolved lags
        x_lags_squeezed = []
        for x_lag_conv in x_lags_conv:
            x_lag_squeezed = x_lag_conv.squeeze(-1)  # [batch_size, filter_size]
            x_lags_squeezed.append(x_lag_squeezed)

        # Stack along a new dimension to create [batch_size, filter_size, num_lags]
        lagged_rv_spectral = torch.stack(x_lags_squeezed, dim=-1)

        # Apply linear transformation separately to the real and imaginary parts
        y_hat_real = self.linear_output_real(lagged_rv_spectral.real)
        y_hat_imag = self.linear_output_imag(lagged_rv_spectral.imag)
        y_hat_spectral = torch.complex(y_hat_real, y_hat_imag)

        # Back to the spatial domain
        y_hat = torch.matmul(U, y_hat_spectral)

        y_hat_real = y_hat.real  # [batch_size, num_markets, 1]
        y_hat_imag = y_hat.imag  # [batch_size, num_markets, 1]
        y_hat_real = y_hat_real.squeeze(-1)
        y_hat_imag = y_hat_imag.squeeze(-1)

        y_hat_spatial = torch.stack((y_hat_real, y_hat_imag), dim=-1)

        # Apply spatial processing
        y_hat = self.spatial_process(y_hat_spatial)

        # Reshape to match the target shape [batch_size, filter_size, horizon]
        y_hat = y_hat.view(y_hat.shape[0], y_hat.shape[1], 1)
        return y_hat
