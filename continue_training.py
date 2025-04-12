import torch
from model import GSPHAR, GSPHAR_Dataset
from train_utils import train_eval_model, load_model
from torch.utils.data import DataLoader
from config.model_config import ModelConfig
from plot_training_history import list_training_histories, plot_saved_history

# Load cached processed data
cache_path = 'cache/processed_data_h1_n38_pct_change.pt'  # adjust path based on your settings
data = torch.load(cache_path)
train_dict, test_dict, DY_adj, y_columns, test_dates, market_indices_list = data

# Create dataloaders
batch_size = 32
dataset_train = GSPHAR_Dataset(train_dict)
dataset_test = GSPHAR_Dataset(test_dict)
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
DY_adj = DY_adj

# Initialize model and load existing weights
config = ModelConfig()
model = GSPHAR(config.input_dim, config.output_dim, n_nodes=38, A=DY_adj)  # adjust n_nodes as needed
model_name = f'GSPHAR_24_magnet_dynamic_h{config.h}'
model, previous_mae = load_model(model_name, model)
print(f"Loaded model with previous MAE: {previous_mae}")

# Set training parameters
additional_epochs = 50  # Set how many more epochs you want
lr = 0.01

# Continue training
print(f"Training for {additional_epochs} more epochs...")
valid_loss, train_losses, valid_losses = train_eval_model(
    model,
    dataloader_train,
    dataloader_test,
    num_epochs=additional_epochs,
    lr=lr,
    h=config.h
)

# Load and print final results
trained_model, final_mae = load_model(model_name, model)
print(f"Final MAE: {final_mae}")


# Show training history info
latest_runs = list_training_histories()
latest_timestamp = latest_runs.iloc[-1]['timestamp']
print("\nPlotting training history...")
plot_saved_history(timestamp=latest_timestamp)