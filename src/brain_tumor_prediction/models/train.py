import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import h5py

def set_seed(seed=42):
    """Set random seed for reproducibility across python, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class BrainTumorDataset(Dataset):
    """PyTorch Dataset wrapper for brain tumor tabular features and targets."""
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long) if y is not None else None
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate_loader(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                epochs=100, device='cpu', early_stopping_patience=10):
    """
    Train a model with early stopping.
    Returns a dictionary of history (loss and accuracy lists) and the best model state.
    """
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_loader(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Check early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_praise_patience(early_stopping_patience):
                break
                
    # Load best weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    return history

def early_stopping_praise_patience(patience):
    # Helper to return the patience value
    return patience

def get_predictions(model, dataloader, device):
    """Runs prediction and returns probability scores, predicted classes, and true classes (if available)."""
    model.eval()
    all_probs = []
    all_preds = []
    all_targets = []
    
    has_targets = True
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch
                all_targets.extend(targets.numpy())
            else:
                inputs = batch
                has_targets = False
                
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
    if has_targets:
        return np.array(all_probs), np.array(all_preds), np.array(all_targets)
    return np.array(all_probs), np.array(all_preds)

def save_model_h5(model, filepath):
    """Saves PyTorch model state_dict weights to HDF5 file."""
    with h5py.File(filepath, 'w') as f:
        for name, param in model.state_dict().items():
            f.create_dataset(name, data=param.cpu().numpy())

def load_model_h5(model, filepath):
    """Loads PyTorch model state_dict weights from HDF5 file."""
    new_state_dict = {}
    with h5py.File(filepath, 'r') as f:
        for name in model.state_dict().keys():
            if name in f:
                # Load weight as torch tensor
                new_state_dict[name] = torch.tensor(f[name][:])
            else:
                raise KeyError(f"Weight parameter {name} not found in model HDF5 file!")
    model.load_state_dict(new_state_dict)
    return model
