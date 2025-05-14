#!/usr/bin/env python
"""
Standalone script to visualize the GSPHAR model architecture using TorchView.
This script creates a simplified version of the GSPHAR model and visualizes it.
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import argparse


class SimplifiedGSPHAR(nn.Module):
    """
    Simplified version of the GSPHAR model for visualization purposes.
    """
    def __init__(self, input_dim=3, output_dim=1, filter_size=20):
        super(SimplifiedGSPHAR, self).__init__()

        # Create convolutional layers
        self.conv1d_lag5 = nn.Conv1d(filter_size, filter_size, kernel_size=5, groups=filter_size, bias=False)
        self.conv1d_lag22 = nn.Conv1d(filter_size, filter_size, kernel_size=22, groups=filter_size, bias=False)

        # Create spatial process
        self.spatial_process = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.ReLU()
        )

        # Create output layers
        self.linear_output_real = nn.Linear(3, output_dim)
        self.linear_output_imag = nn.Linear(3, output_dim)

    def forward(self, x_lag1, x_lag5, x_lag22):
        batch_size, filter_size, input_dim = x_lag1.shape

        # Simulate FFT transformation
        x_lag1_real = x_lag1
        x_lag1_imag = torch.zeros_like(x_lag1)

        x_lag5_real = x_lag5
        x_lag5_imag = torch.zeros_like(x_lag5)

        x_lag22_real = x_lag22
        x_lag22_imag = torch.zeros_like(x_lag22)

        # Reshape for Conv1D (N, C, L)
        x_lag5_real = x_lag5_real.permute(0, 1, 2)  # (batch_size, filter_size, input_dim)
        x_lag5_imag = x_lag5_imag.permute(0, 1, 2)

        x_lag22_real = x_lag22_real.permute(0, 1, 2)
        x_lag22_imag = x_lag22_imag.permute(0, 1, 2)

        # Apply Conv1D
        x_lag5_conv_real = self.conv1d_lag5(x_lag5_real)
        x_lag5_conv_imag = self.conv1d_lag5(x_lag5_imag)

        x_lag22_conv_real = self.conv1d_lag22(x_lag22_real)
        x_lag22_conv_imag = self.conv1d_lag22(x_lag22_imag)

        # Spatial process
        spatial_input = torch.cat([x_lag1_real[:, :, 0:1], x_lag1_imag[:, :, 0:1]], dim=2)
        spatial_output = self.spatial_process(spatial_input)

        # Concatenate features
        features_real = torch.cat([
            spatial_output,
            x_lag5_conv_real[:, :, 0:1],
            x_lag22_conv_real[:, :, 0:1]
        ], dim=2)

        features_imag = torch.cat([
            spatial_output,
            x_lag5_conv_imag[:, :, 0:1],
            x_lag22_conv_imag[:, :, 0:1]
        ], dim=2)

        # Linear output
        output_real = self.linear_output_real(features_real)
        output_imag = self.linear_output_imag(features_imag)

        # Combine real and imaginary parts
        output = torch.complex(output_real, output_imag)

        return output


def visualize_with_torchview(model, batch_size=32, filter_size=20, input_dim=22,
                            output_dir='plots', filename='gsphar_torchview'):
    """
    Visualize the model architecture using TorchView.

    Args:
        model (torch.nn.Module): Model to visualize.
        batch_size (int): Batch size for the input.
        filter_size (int): Filter size for the model.
        input_dim (int): Input dimension.
        output_dir (str): Directory to save the visualization.
        filename (str): Filename for the visualization.

    Returns:
        None
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Import torchview
    try:
        from torchview import draw_graph
    except ImportError:
        print("TorchView is not installed. Installing it now...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torchview"])
        from torchview import draw_graph

    # Create dummy inputs
    x_lag1 = torch.randn(batch_size, filter_size, input_dim)
    x_lag5 = torch.randn(batch_size, filter_size, input_dim)
    x_lag22 = torch.randn(batch_size, filter_size, input_dim)

    # Draw the model graph
    try:
        print("Generating model visualization...")
        model_graph = draw_graph(
            model,
            input_data=(x_lag1, x_lag5, x_lag22),
            save_graph=True,
            directory=output_dir,
            filename=filename,
            expand_nested=True,  # Expand nested modules
            show_shapes=True,
            depth=3,  # Increased depth
            hide_inner_tensors=False  # Show inner tensors for more detail
        )
        print(f"Model visualization saved to {os.path.join(output_dir, filename)}.png")
    except Exception as e:
        print(f"Error generating model visualization: {e}")
        print("Saving model summary as fallback...")
        summary_path = os.path.join(output_dir, f"{filename}_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(str(model))
        print(f"Model summary saved to {summary_path}")


def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(description='Visualize the GSPHAR model architecture using TorchView.')
    parser.add_argument('--filter-size', type=int, default=20,
                        help='Filter size for the model.')
    parser.add_argument('--input-dim', type=int, default=22,
                        help='Input dimension.')
    parser.add_argument('--output-dim', type=int, default=1,
                        help='Output dimension.')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for the input.')
    parser.add_argument('--output-dir', type=str, default='plots',
                        help='Directory to save the visualization.')
    parser.add_argument('--filename', type=str, default='gsphar_torchview',
                        help='Filename for the visualization.')

    args = parser.parse_args()

    # Create model instance
    model = SimplifiedGSPHAR(
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        filter_size=args.filter_size
    )

    # Visualize model with TorchView
    visualize_with_torchview(
        model=model,
        batch_size=args.batch_size,
        filter_size=args.filter_size,
        input_dim=args.input_dim,
        output_dir=args.output_dir,
        filename=args.filename
    )

    print("Model visualization completed.")


if __name__ == '__main__':
    main()
