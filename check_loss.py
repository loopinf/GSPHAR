import torch

# Load the checkpoint
checkpoint = torch.load('checkpoints/GSPHAR_24_magnet_dynamic_h1.tar', map_location='cpu')

# Print the best loss achieved during training
print(f"Best loss achieved: {checkpoint['loss']}")