import torch
import torch.nn as nn
import os
from tqdm import tqdm
import numpy as np
import pandas as pd

def save_model(name, model, num_L=None, best_loss_val=None):
    if not os.path.exists('checkpoints/'):
        os.makedirs('checkpoints/')
    # Prepare the model state dictionary
    config = {
        'model_state_dict': model.state_dict(),
        'layer': num_L,
        'loss': best_loss_val
    }
    # Save the model state dictionary
    torch.save(config, f'checkpoints/{name}.tar')
    return

def load_model(name, model):
    checkpoint = torch.load(f'checkpoints/{name}.tar', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    num_L = checkpoint['layer']
    loss = checkpoint['loss']
    print(f"Loaded model: {name}")
    print(f"Loss: {loss}")
    return model, loss

def train_eval_model(model, dataloader_train, dataloader_test, num_epochs=200, lr=0.01, h=5, progress_callback=None):
    from tqdm import tqdm
    
    best_loss_val = 1000000
    patience = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, 
                                                   steps_per_epoch=len(dataloader_train), 
                                                   epochs=num_epochs,
                                                   three_phase=True)
    model.to(device)
    criterion = nn.MSELoss()
    criterion = criterion.to(device)
    model.train()
    train_loss_list = []
    test_loss_list = []

    # Main training loop with progress bar for epochs
    epoch_progress = tqdm(range(num_epochs), desc="Training Progress", leave=True)
    for epoch in epoch_progress:
        running_loss = 0.0
        batch_count = 0
        
        # Add progress bar for batches
        batch_progress = tqdm(dataloader_train, 
                            desc=f"Epoch {epoch+1}/{num_epochs}", 
                            leave=False)
        
        for x_lag1, x_lag4, x_lag24, y in batch_progress:
            optimizer.zero_grad()
            
            # Move data to device
            x_lag1 = x_lag1.to(device)
            x_lag4 = x_lag4.to(device)
            x_lag24 = x_lag24.to(device)
            y = y.to(device)
            
            # Forward pass
            outputs, _, _ = model(x_lag1, x_lag4, x_lag24)
            loss = criterion(outputs, y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Update metrics
            running_loss += loss.item()
            batch_count += 1
            
            # Update batch progress bar
            batch_progress.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{running_loss/batch_count:.4f}'
            })
        
        # Calculate average loss for the epoch
        avg_train_loss = running_loss / batch_count if batch_count > 0 else 0
        train_loss_list.append(avg_train_loss)
        
        # Evaluate model
        valid_loss = evaluate_model(model, dataloader_test)
        test_loss_list.append(valid_loss)
        
        # Update epoch progress bar
        epoch_progress.set_postfix({
            'Train Loss': f'{avg_train_loss:.4f}',
            'Valid Loss': f'{valid_loss:.4f}',
            'Best': f'{best_loss_val:.4f}',
            'Patience': patience
        })
        
        if valid_loss < best_loss_val:
            best_loss_val = valid_loss
            patience = 0
            save_model(f'GSPHAR_24_magnet_dynamic_h{h}', model, None, best_loss_val)
            epoch_progress.set_postfix(
                Train_Loss=f'{avg_train_loss:.4f}',
                Valid_Loss=f'{valid_loss:.4f}',
                Best=f'{best_loss_val:.4f}',
                Patience=patience,
                Status='Saved ✓'
            )
        else:
            patience = patience + 1
            if patience >= 200:
                epoch_progress.set_description(f"Early stopping at epoch {epoch+1}")
                break
        
        # Call progress callback if provided
        if progress_callback:
            progress_callback()
    
    # Save training history
    save_training_history(train_loss_list, test_loss_list, h)
    
    return best_loss_val, train_loss_list, test_loss_list

def evaluate_model(model, dataloader_test):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.MSELoss()
    criterion = criterion.to(device)
    valid_loss = 0
    model.eval()
    with torch.no_grad():
        for x_lag1, x_lag4, x_lag24, y in dataloader_test:
            x_lag1 = x_lag1.to(device)
            x_lag4 = x_lag4.to(device)
            x_lag24 = x_lag24.to(device)
            y = y.to(device)
            output, _, _ = model(x_lag1, x_lag4, x_lag24)
            loss = criterion(output, y)
            valid_loss = valid_loss + loss.item()
    valid_loss = valid_loss/len(dataloader_test)
    return valid_loss

def predict_and_evaluate(model, dataloader_test, market_indices_list, test_dates=None):
    """Generate predictions and evaluation metrics
    
    Args:
        model: The trained model
        dataloader_test: PyTorch DataLoader for test data
        market_indices_list: List of market indices
        test_dates: Index timestamps for the predictions (optional)
    """
    y_hat_list = []
    y_list = []
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        for x_lag1, x_lag4, x_lag24, y in dataloader_test:
            x_lag1 = x_lag1.to(device)
            x_lag4 = x_lag4.to(device)
            x_lag24 = x_lag24.to(device)
            
            y_hat, _, _ = model(x_lag1, x_lag4, x_lag24)
            y_hat_list.append(y_hat.cpu().numpy())
            y_list.append(y.cpu().numpy())
    
    y_hat_concatenated = np.concatenate(y_hat_list, axis=0)
    y_concatenated = np.concatenate(y_list, axis=0)
    
    # Create DataFrames with time index if available
    look_back_window = 24
    if len(test_dates)  == len(y_hat_concatenated)+ look_back_window:
        test_dates = test_dates[look_back_window:]
    # if test_dates is not None and len(test_dates) == len(y_hat_concatenated):
        rv_hat = pd.DataFrame(
            data=y_hat_concatenated, 
            columns=market_indices_list,
            index=test_dates
        )
        rv_true = pd.DataFrame(
            data=y_concatenated, 
            columns=market_indices_list,
            index=test_dates
        )
    else:
        rv_hat = pd.DataFrame(data=y_hat_concatenated, columns=market_indices_list)
        rv_true = pd.DataFrame(data=y_concatenated, columns=market_indices_list)
    
    # Create results DataFrame with predictions and true values
    results_df = pd.DataFrame(index=rv_hat.index)
    for market_index in market_indices_list:
        results_df[f'{market_index}_rv_forecast'] = rv_hat[market_index]
        results_df[f'{market_index}_rv_true'] = rv_true[market_index]
    
    return results_df, rv_hat, rv_true

def save_training_history(train_losses, valid_losses, h, name=None):
    """Save training and validation losses to a file with timestamp
    
    Args:
        train_losses (list): List of training losses
        valid_losses (list): List of validation losses
        h (int): Forecasting horizon
        name (str, optional): Model name prefix. Defaults to 'GSPHAR'
    """
    from datetime import datetime
    
    if not os.path.exists('results/training_history/'):
        os.makedirs('results/training_history/')
    
    model_name = name if name else 'GSPHAR'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    history_df = pd.DataFrame({
        'epoch': range(1, len(train_losses) + 1),
        'train_loss': train_losses,
        'valid_loss': valid_losses
    })
    
    filename = f'results/training_history/{model_name}_h{h}_{timestamp}.csv'
    history_df.to_csv(filename, index=False)
    print(f"Training history saved to: {filename}")
    
    # Also save a reference to the latest history file
    latest_ref = {
        'timestamp': timestamp,
        'filename': filename,
        'h': h,
        'model_name': model_name,
        'final_train_loss': train_losses[-1],
        'final_valid_loss': valid_losses[-1],
        'num_epochs': len(train_losses)
    }
    
    latest_file = 'results/training_history/latest_runs.csv'
    if os.path.exists(latest_file):
        latest_df = pd.read_csv(latest_file)
        latest_df = pd.concat([latest_df, pd.DataFrame([latest_ref])], ignore_index=True)
    else:
        latest_df = pd.DataFrame([latest_ref])
    
    latest_df.to_csv(latest_file, index=False)
