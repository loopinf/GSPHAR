# GSPHAR Wrapper Usage Guide

This guide explains how to use the GSPHAR wrapper to handle input shape issues with the GSPHAR model.

## Background

The GSPHAR (Graph Signal Processing for Heterogeneous Autoregressive) model has specific input shape requirements that can be difficult to satisfy in practice. The model expects:

1. Input tensors with matching dimensions
2. Filter size that matches the input dimension
3. Specific kernel sizes for the convolution layers

The GSPHAR wrapper provides a solution to these issues by adapting input tensors to be compatible with the model without modifying the model itself.

## Installation

The GSPHAR wrapper is included in the `src/utils` directory of the project. No additional installation is required.

## Usage

### Basic Usage

```python
from src.utils.gsphar_wrapper import GSPHARWrapper
import torch
import numpy as np

# Create adjacency matrix
input_dim = 5
adj_matrix = np.ones((input_dim, input_dim))  # Example adjacency matrix

# Create model
model = GSPHARWrapper(
    input_dim=input_dim,
    output_dim=input_dim,
    filter_size=input_dim,
    adj_matrix=adj_matrix
)

# Create input tensors
batch_size = 32
x_lag1 = torch.randn(batch_size, 1, input_dim)  # 1 time step
x_lag5 = torch.randn(batch_size, 5, input_dim)  # 5 time steps
x_lag22 = torch.randn(batch_size, 22, input_dim)  # 22 time steps

# Forward pass
output, softmax_param_5, softmax_param_22 = model(x_lag1, x_lag5, x_lag22)

# Print output shape
print(f"Output shape: {output.shape}")  # Should be [batch_size, output_dim]
```

### Advanced Usage

The GSPHAR wrapper can handle various input shapes and dimensions:

#### Different Time Steps

```python
# Create input tensors with different time steps
x_lag1 = torch.randn(batch_size, 1, input_dim)  # 1 time step
x_lag5 = torch.randn(batch_size, 5, input_dim)  # 5 time steps
x_lag22 = torch.randn(batch_size, 22, input_dim)  # 22 time steps

# Forward pass
output, _, _ = model(x_lag1, x_lag5, x_lag22)
```

#### Different Input Dimensions

```python
# Create input tensors with different dimensions
x_lag1 = torch.randn(batch_size, 5, 10)  # 10 features
x_lag5 = torch.randn(batch_size, 5, 10)  # 10 features
x_lag22 = torch.randn(batch_size, 5, 10)  # 10 features

# Forward pass
output, _, _ = model(x_lag1, x_lag5, x_lag22)
```

#### Different Filter Size

```python
# Create model with a different filter size
model = GSPHARWrapper(
    input_dim=input_dim,
    output_dim=input_dim,
    filter_size=10,  # Different filter size
    adj_matrix=adj_matrix
)

# Create input tensors
x_lag1 = torch.randn(batch_size, 5, input_dim)
x_lag5 = torch.randn(batch_size, 5, input_dim)
x_lag22 = torch.randn(batch_size, 5, input_dim)

# Forward pass
output, _, _ = model(x_lag1, x_lag5, x_lag22)
```

## How It Works

The GSPHAR wrapper handles input shape issues in several ways:

1. **Adjacency Matrix Adaptation**: The wrapper adapts the adjacency matrix to match the input dimension.

2. **Input Tensor Adaptation**: The wrapper adapts input tensors to have compatible shapes.

3. **Kernel Size Modification**: The wrapper modifies the convolution layers to use smaller kernel sizes.

4. **Batch Processing**: The wrapper processes each batch element separately to avoid batch dimension issues.

## Limitations

The GSPHAR wrapper has some limitations:

1. **Performance**: Processing each batch element separately can be slower than processing the entire batch at once.

2. **Accuracy**: The wrapper uses smaller kernel sizes, which may affect the model's accuracy.

3. **Memory Usage**: The wrapper may use more memory than the original model due to the batch processing approach.

## Troubleshooting

If you encounter issues with the GSPHAR wrapper, try the following:

1. **Check Input Shapes**: Make sure your input tensors have the correct shapes.

2. **Check Adjacency Matrix**: Make sure your adjacency matrix has the correct dimensions.

3. **Check Device**: Make sure all tensors are on the same device (CPU or GPU).

4. **Check Batch Size**: Try using a smaller batch size if you encounter memory issues.

## Conclusion

The GSPHAR wrapper provides a convenient way to use the GSPHAR model with various input shapes and dimensions. It handles the model's specific input shape requirements without modifying the model itself.
