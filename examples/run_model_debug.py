import sys
import os

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import numpy as np
# Adjust the import path based on where this script is saved and how the project is structured.
# If this script is in the root of the GSPHAR project, this should work.
from src.models.gsphar_exact import GSPHAR
# from torchviz import make_dot # Removed torchviz
from torchview import draw_graph # Added torchview

# Define model parameters (these are examples, adjust as needed)
num_assets = 10 # Example number of assets
input_dim = 3  # Corresponds to the number of lagged RVs (lag1, lag5, lag22)
output_dim = 1 # Predicting the next RV
filter_size = num_assets # Typically the number of assets/nodes in the graph

# Create a dummy adjacency matrix A
# For the GSPHAR model, A is expected to be a NumPy array during initialization
A_np = np.random.rand(num_assets, num_assets)

# Instantiate the model
model = GSPHAR(input_dim=input_dim, output_dim=output_dim, filter_size=filter_size, A=A_np)

# Create dummy input tensors
batch_size = 4 # Example batch size
# x_lag1: [batch_size, num_assets, 1] (RV for lag 1)
x_lag1 = torch.rand(batch_size, num_assets, 1)
# x_lag5: [batch_size, num_assets, 5] (RVs for lag 5)
x_lag5 = torch.rand(batch_size, num_assets, 5)
# x_lag22: [batch_size, num_assets, 22] (RVs for lag 22)
x_lag22 = torch.rand(batch_size, num_assets, 22)

# Perform a forward pass
# Make sure to set the model to evaluation mode if you're not training
model.eval()
with torch.no_grad(): # Disable gradient calculations for inference
    print("Running model forward pass...")
    y_hat, softmax_param_5, softmax_param_22 = model(x_lag1, x_lag5, x_lag22)
    print("Model forward pass complete.")

print(f"Output y_hat shape: {y_hat.shape}")
print(f"Softmax param 5 shape: {softmax_param_5.shape}") # Assuming these are returned
print(f"Softmax param 22 shape: {softmax_param_22.shape}") # Assuming these are returned

print("\\nDebug prints from GSPHAR.forward() should appear above this line if the model was called correctly.")

# Generate model visualization with torchview
print("\\nGenerating model visualization with torchview...")

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
x_lag1, x_lag5, x_lag22 = x_lag1.to(device), x_lag5.to(device), x_lag22.to(device)

# It's good practice to ensure the model is in eval mode for visualization
model.eval()

# torchview requires a tuple of inputs if there are multiple inputs
model_inputs = (x_lag1, x_lag5, x_lag22)

# Generate the graph
# You might need to adjust graph_name and other parameters as needed.
# `expand_nested` can be useful for complex models with submodules.
# `show_shapes` and `show_params` add useful information to the graph.
try:
    # Note: torchview expects input_data as a tuple or a single tensor.
    # If your model's forward method expects multiple arguments, pass them as a tuple.
    model_graph = draw_graph(
        model,
        input_data=model_inputs, # Pass inputs as a tuple
        graph_name='gsphar_model_torchview',
        save_graph=True, # Saves the graph to a file
        filename='gsphar_model_torchview', # Output filename (will be .pdf)
        expand_nested=True,
        show_shapes=True,
        # show_params=True # Can be verbose, enable if needed
    )
    # The graph is saved as gsphar_model_torchview.pdf by default with save_graph=True
    print("Model visualization saved to gsphar_model_torchview.pdf")
except Exception as e:
    print(f"Could not render model visualization with torchview. Error: {e}")
    print("Ensure Graphviz is installed and in your PATH.")