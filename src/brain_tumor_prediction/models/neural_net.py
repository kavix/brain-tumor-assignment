import os
import copy
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class BrainTumorDataset(Dataset):
    """
    Custom PyTorch Dataset for tabular brain tumor data.
    """
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

class BrainTumorMLP(nn.Module):
    """
    Feed-Forward Neural Network (Multilayer Perceptron) for Brain Tumor Type Classification.
    Includes Linear layers, Batch Normalization, ReLU activations, and Dropout.
    """
    def __init__(self, input_dim=50, hidden_dims=[64, 32], output_dim=3, dropout_rate=0.2, use_batch_norm=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm

        layers = []
        prev_dim = input_dim
        for h_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def save_model_hdf5(model, filepath):
    """
    Saves PyTorch model architecture and weights to an HDF5 file (.h5).
    """
    with h5py.File(filepath, 'w') as f:
        f.attrs['input_dim'] = model.input_dim
        f.attrs['hidden_dims'] = model.hidden_dims
        f.attrs['output_dim'] = model.output_dim
        f.attrs['dropout_rate'] = model.dropout_rate
        f.attrs['use_batch_norm'] = model.use_batch_norm
        
        g = f.create_group('state_dict')
        for key, tensor in model.state_dict().items():
            g.create_dataset(key, data=tensor.cpu().numpy())
    print(f"Model saved successfully to {filepath}")

def load_model_hdf5(filepath):
    """
    Loads a PyTorch model from an HDF5 file (.h5).
    """
    with h5py.File(filepath, 'r') as f:
        input_dim = f.attrs['input_dim']
        hidden_dims = list(f.attrs['hidden_dims'])
        output_dim = f.attrs['output_dim']
        dropout_rate = f.attrs['dropout_rate']
        use_batch_norm = f.attrs.get('use_batch_norm', False)
        
        model = BrainTumorMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm
        )
        
        state_dict = {}
        g = f['state_dict']
        for key in g.keys():
            state_dict[key] = torch.tensor(g[key][()])
            
        model.load_state_dict(state_dict)
    print(f"Model loaded successfully from {filepath}")
    return model

def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, weight_decay=1e-4, patience=10, device='cpu'):
    """
    Trains the PyTorch model with early stopping.
    Returns the training history (losses and accuracies).
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    
    for epoch in range(1, epochs + 1):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            
        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total
        
        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                running_val_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()
                
        epoch_val_loss = running_val_loss / val_total
        epoch_val_acc = val_correct / val_total
        
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        # Check early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break
            
    # Load best weights
    model.load_state_dict(best_model_wts)
    return history
