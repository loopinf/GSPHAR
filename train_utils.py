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
    mae_loss = checkpoint['loss']
    print(f"Loaded model: {name}")
    print(f"MAE loss: {mae_loss}")
    return model, mae_loss

def train_eval_model(model, dataloader_train, dataloader_test, num_epochs=200, lr=0.01, h=5):
    best_loss_val = 1000000
    patience = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, 
                                                   steps_per_epoch=len(dataloader_train), epochs=num_epochs,
                                                   three_phase=True)
    model.to(device)
    criterion = nn.MSELoss()
    criterion = criterion.to(device)
    model.train()
    train_loss_list = []
    test_loss_list = []
    
    # Add progress bar for epochs
    epoch_progress = tqdm(range(num_epochs), desc="Training Progress", leave=True)
    
    for epoch in epoch_progress:
        running_loss = 0.0
        batch_count = 0
        
        # Process batches
        for x_lag1, x_lag4, x_lag24, y in dataloader_train:
            x_lag1 = x_lag1.to(device)
            x_lag4 = x_lag4.to(device)
            x_lag24 = x_lag24.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            output, conv1d_lag4_weights, conv1d_lag24_weights = model(x_lag1, x_lag4, x_lag24)
            loss = criterion(output, y)
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            # Update scheduler: this scheduler is designed to be updated after each batch.
            scheduler.step()
            
            # Track running loss
            running_loss += loss.item()
            batch_count += 1
        
        # Calculate average loss for the epoch
        avg_train_loss = running_loss / batch_count if batch_count > 0 else 0
        train_loss_list.append(avg_train_loss)
        
        # Evaluate model
        valid_loss = evaluate_model(model, dataloader_test)
        test_loss_list.append(valid_loss)
        
        # Update progress bar description with metrics
        epoch_progress.set_postfix({
            'Train Loss': f'{avg_train_loss:.4f}', 
            'Valid Loss': f'{valid_loss:.4f}',
            'Best': f'{best_loss_val:.4f}',
            'Patience': patience,
        })

        if valid_loss < best_loss_val:
            best_loss_val = valid_loss
            final_conv1d_lag4_weights = conv1d_lag4_weights.detach().cpu().numpy()
            final_conv1d_lag24_weights = conv1d_lag24_weights.detach().cpu().numpy()
            patience = 0
            save_model(f'GSPHAR_24_magnet_dynamic_h{h}', model, None, best_loss_val)
            # epoch_progress.set_postfix({**epoch_progress.postfix, 'Status': 'Saved ✓'})
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
    
    return best_loss_val, final_conv1d_lag4_weights, final_conv1d_lag24_weights

def evaluate_model(model, dataloader_test):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.L1Loss()
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

def predict_and_evaluate(model, dataloader_test, market_indices_list):
    """Generate predictions and evaluation metrics"""
    y_hat_list = []
    y_list = []
    
    model.eval()
    with torch.no_grad():
        for x_lag1, x_lag4, x_lag24, y in dataloader_test:
            y_hat, _, _ = model(x_lag1, x_lag4, x_lag24)
            y_hat_list.append(y_hat.cpu().numpy())
            y_list.append(y.cpu().numpy())
    
    y_hat_concatenated = np.concatenate(y_hat_list, axis=0)
    y_concatenated = np.concatenate(y_list, axis=0)
    
    rv_hat = pd.DataFrame(data=y_hat_concatenated, columns=market_indices_list)
    rv_true = pd.DataFrame(data=y_concatenated, columns=market_indices_list)
    
    # Create results dataframe
    results_df = pd.DataFrame()
    for market_index in market_indices_list:
        pred_column = market_index + '_rv_forecast'
        true_column = market_index + '_rv_true'
        results_df[pred_column] = rv_hat[market_index]
        results_df[true_column] = rv_true[market_index]
    
    return results_df, rv_hat, rv_true