from model import GSPHAR
import torch
import numpy as np
your_adjacency_matrix = np.eye(24) # Replace with your adjacency matrix

# Create model instance with same parameters
model = GSPHAR(input_dim=3, output_dim=1, n_nodes=24, A=your_adjacency_matrix)

print("\n=== Current Model Layers ===")
for name, param in model.named_parameters():
    print(f"Layer: {name}, Shape: {param.shape}")