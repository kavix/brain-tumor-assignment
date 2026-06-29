import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from sklearn.inspection import permutation_importance
import joblib

# Add src to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from brain_tumor_prediction.features.preprocessing import build_preprocessing_pipeline
from brain_tumor_prediction.models.neural_net import BrainTumorMLP
from brain_tumor_prediction.models.train import (
    set_seed, BrainTumorDataset, train_model, get_predictions, save_model_h5
)
from brain_tumor_prediction.models.baseline import BaselineClassifier

def main():
    # Set seed
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Data
    data_path = 'data/raw/brain_tumor_data.csv'
    if not os.path.exists(data_path):
        data_path = '../../data/raw/brain_tumor_data.csv'
        
    df = pd.read_csv(data_path)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Check target distribution
    target_dist = df['tumor_type'].value_counts()
    print("\nClass distribution:")
    for cls, val in target_dist.items():
        print(f"  {cls}: {val} ({val/len(df)*100:.2f}%)")
        
    # Extract target and features
    X = df.drop(columns=['tumor_type'])
    y_raw = df['tumor_type']
    
    # Map target labels to integers
    target_mapping = {'Glioma': 0, 'Meningioma': 1, 'Pituitary': 2}
    y = y_raw.map(target_mapping).values
    
    # 2. Partition dataset: 70% Train, 15% Val, 15% Test (Stratified)
    X_train_raw, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    
    print(f"\nPartitions:")
    print(f"  Train: {X_train_raw.shape[0]} samples")
    print(f"  Val:   {X_val_raw.shape[0]} samples")
    print(f"  Test:  {X_test_raw.shape[0]} samples")
    
    # 3. Fit preprocessing on training partition only
    pipeline = build_preprocessing_pipeline()
    X_train = pipeline.fit_transform(X_train_raw)
    X_val = pipeline.transform(X_val_raw)
    X_test = pipeline.transform(X_test_raw)
    
    # Retrieve feature names out of column transformer
    col_transformer = pipeline.named_steps['preprocessor']
    feature_names = list(col_transformer.get_feature_names_out())
    print(f"Preprocessed features shape: {X_train.shape[1]}")
    
    # Save the preprocessing pipeline
    os.makedirs('models', exist_ok=True)
    joblib.dump(pipeline, 'models/preprocessing_pipeline.joblib')
    joblib.dump(pipeline, 'preprocessing_pipeline.joblib')
    print("Saved preprocessing pipeline to preprocessing_pipeline.joblib")
    
    # 4. Hyperparameter tuning grid search
    print("\nStarting hyperparameter tuning...")
    tuning_results = []
    
    # Tuning parameter grid
    hidden_configs = [[64, 32], [128, 64], [32, 16]]
    learning_rates = [0.005, 0.001, 0.0005]
    dropout_rates = [0.1, 0.2, 0.4]
    batch_sizes = [32, 64]
    
    # Assemble configurations
    configs_to_run = []
    for h in hidden_configs:
        for lr in learning_rates:
            for dr in dropout_rates:
                for bs in batch_sizes:
                    configs_to_run.append((h, lr, dr, bs))
                    
    # Select 12 representative configurations
    np.random.seed(42)
    selected_indices = np.random.choice(len(configs_to_run), 12, replace=False)
    selected_configs = [configs_to_run[i] for i in selected_indices]
    
    # Ensure standard config is present
    default_config = ([64, 32], 0.001, 0.2, 64)
    if default_config not in selected_configs:
        selected_configs[0] = default_config
        
    for idx, (h, lr, dr, bs) in enumerate(selected_configs):
        train_ds = BrainTumorDataset(X_train, y_train)
        val_ds = BrainTumorDataset(X_val, y_val)
        
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)
        
        model = BrainTumorMLP(
            input_dim=X_train.shape[1],
            hidden_dims=h,
            output_dim=3,
            dropout_rate=dr
        ).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            epochs=40, # fast tuning run
            device=device,
            early_stopping_patience=6
        )
        
        best_val_loss = min(history['val_loss'])
        best_val_loss_idx = history['val_loss'].index(best_val_loss)
        best_val_acc = history['val_acc'][best_val_loss_idx]
        
        tuning_results.append({
            'Config ID': idx + 1,
            'Hidden Layers': str(h),
            'Learning Rate': lr,
            'Dropout Rate': dr,
            'Batch Size': bs,
            'Best Val Loss': f"{best_val_loss:.4f}",
            'Val Accuracy': f"{best_val_acc*100:.2f}%",
            'Val Acc Float': best_val_acc,
            'val_loss_raw': best_val_loss
        })
        
        print(f"Config {idx+1}/12: Hidden={h}, LR={lr}, Dropout={dr}, Batch={bs} -> Val Acc: {best_val_acc*100:.2f}%")
        
    tuning_df = pd.DataFrame(tuning_results)
    print("\nTuning Results Summary:")
    print(tuning_df.drop(columns=['Val Acc Float', 'val_loss_raw']).to_string(index=False))
    
    # Save tuning summary
    os.makedirs('logs', exist_ok=True)
    tuning_df.to_csv('logs/tuning_summary.csv', index=False)
    
    # Select best model based on validation performance
    best_config_idx = tuning_df['Val Acc Float'].astype(float).idxmax()
    best_config_row = tuning_df.loc[best_config_idx]
    print(f"\nBest Config: ID {best_config_row['Config ID']} with Val Acc: {best_config_row['Val Accuracy']}")
    
    best_h = eval(best_config_row['Hidden Layers'])
    best_lr = float(best_config_row['Learning Rate'])
    best_dr = float(best_config_row['Dropout Rate'])
    best_bs = int(best_config_row['Batch Size'])
    
    # 5. Train Final Best Model
    print("\nTraining final model using the best hyperparameters...")
    set_seed(42)
    final_model = BrainTumorMLP(
        input_dim=X_train.shape[1],
        hidden_dims=best_h,
        output_dim=3,
        dropout_rate=best_dr
    ).to(device)
    
    train_loader = DataLoader(BrainTumorDataset(X_train, y_train), batch_size=best_bs, shuffle=True)
    val_loader = DataLoader(BrainTumorDataset(X_val, y_val), batch_size=best_bs, shuffle=False)
    test_loader = DataLoader(BrainTumorDataset(X_test, y_test), batch_size=best_bs, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(final_model.parameters(), lr=best_lr, weight_decay=1e-4)
    
    history = train_model(
        model=final_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=120,
        device=device,
        early_stopping_patience=15
    )
    
    # Save history
    history_df = pd.DataFrame(history)
    history_df.to_csv('logs/training_history.csv', index=False)
    print("Saved training history to logs/training_history.csv")
    
    # Save model weights to HDF5
    save_model_h5(final_model, 'models/studentID_model.h5')
    save_model_h5(final_model, 'studentID_model.h5')
    print("Saved final model weights to studentID_model.h5")
    
    # 6. Final Evaluation
    test_probs, test_preds, test_targets = get_predictions(final_model, test_loader, device)
    test_acc = accuracy_score(test_targets, test_preds)
    
    print(f"\n================ FINAL TEST EVALUATION ================")
    print(f"Overall Accuracy: {test_acc*100:.2f}%")
    
    report = classification_report(
        test_targets, test_preds, 
        target_names=['Glioma', 'Meningioma', 'Pituitary'], 
        digits=4
    )
    print("\nClassification Report:")
    print(report)
    
    macro_f1 = f1_score(test_targets, test_preds, average='macro')
    weighted_f1 = f1_score(test_targets, test_preds, average='weighted')
    print(f"Macro F1-Score: {macro_f1:.4f}")
    print(f"Weighted F1-Score: {weighted_f1:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(test_targets, test_preds)
    print("\nConfusion Matrix:")
    print("              Predicted")
    print("              Glioma  Meningioma  Pituitary")
    print(f"True Glioma    {cm[0,0]:<7d} {cm[0,1]:<11d} {cm[0,2]:<10d}")
    print(f"Meningioma     {cm[1,0]:<7d} {cm[1,1]:<11d} {cm[1,2]:<10d}")
    print(f"Pituitary      {cm[2,0]:<7d} {cm[2,1]:<11d} {cm[2,2]:<10d}")
    
    np.savetxt('logs/confusion_matrix.txt', cm, fmt='%d')
    
    # 7. Confidence Analysis
    test_max_probs = test_probs.max(axis=1)
    sorted_prob_indices = np.argsort(test_max_probs)
    
    low_confidence_indices = sorted_prob_indices[:2]
    high_confidence_indices = sorted_prob_indices[-2:]
    
    class_names = ['Glioma', 'Meningioma', 'Pituitary']
    
    print("\n--- Confidence Analysis ---")
    print("\nHigh Confidence Cases:")
    for idx in high_confidence_indices:
        print(f"  Test Case Index: {idx}")
        print(f"    True Class: {class_names[test_targets[idx]]}")
        print(f"    Predicted:  {class_names[test_preds[idx]]}")
        print(f"    Probabilities: Glioma={test_probs[idx, 0]:.4f}, Meningioma={test_probs[idx, 1]:.4f}, Pituitary={test_probs[idx, 2]:.4f}")
        
    print("\nLow Confidence Cases:")
    for idx in low_confidence_indices:
        print(f"  Test Case Index: {idx}")
        print(f"    True Class: {class_names[test_targets[idx]]}")
        print(f"    Predicted:  {class_names[test_preds[idx]]}")
        print(f"    Probabilities: Glioma={test_probs[idx, 0]:.4f}, Meningioma={test_probs[idx, 1]:.4f}, Pituitary={test_probs[idx, 2]:.4f}")
        
    # 8. Baseline Comparison
    print("\nTraining Baseline Models...")
    rf_baseline = BaselineClassifier(model_type='random_forest', random_state=42)
    rf_baseline.fit(X_train, y_train)
    rf_eval = rf_baseline.evaluate(X_test, y_test)
    
    lr_baseline = BaselineClassifier(model_type='logistic_regression', random_state=42)
    lr_baseline.fit(X_train, y_train)
    lr_eval = lr_baseline.evaluate(X_test, y_test)
    
    print("\nBaseline Comparison:")
    baseline_comparison = {
        'Model': ['Neural Network', 'Random Forest', 'Logistic Regression'],
        'Test Accuracy': [f"{test_acc*100:.2f}%", f"{rf_eval['accuracy']*100:.2f}%", f"{lr_eval['accuracy']*100:.2f}%"],
        'F1 Macro': [f"{macro_f1:.4f}", f"{rf_eval['f1_macro']:.4f}", f"{lr_eval['f1_macro']:.4f}"],
        'F1 Weighted': [f"{weighted_f1:.4f}", f"{rf_eval['f1_weighted']:.4f}", f"{lr_eval['f1_weighted']:.4f}"]
    }
    comparison_df = pd.DataFrame(baseline_comparison)
    print(comparison_df.to_string(index=False))
    comparison_df.to_csv('logs/baseline_comparison.csv', index=False)
    
    # 9. Feature Importance
    print("\nComputing Permutation Feature Importance for the Neural Network...")
    class SklearnPyTorchWrapper:
        def __init__(self, pt_model, dev):
            self.model = pt_model
            self.device = dev
            self._estimator_type = "classifier"
            self.classes_ = [0, 1, 2]
            
        def fit(self, X, y=None):
            return self
            
        def predict(self, X):
            self.model.eval()
            with torch.no_grad():
                inputs = torch.tensor(X, dtype=torch.float32).to(self.device)
                outputs = self.model(inputs)
                _, preds = outputs.max(1)
            return preds.cpu().numpy()
            
        def score(self, X, y):
            y_pred = self.predict(X)
            return accuracy_score(y, y_pred)
            
    wrapper = SklearnPyTorchWrapper(final_model, device)
    
    result = permutation_importance(
        wrapper, X_val, y_val, n_repeats=5, random_state=42, n_jobs=1
    )
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance Mean': result.importances_mean,
        'Importance Std': result.importances_std
    }).sort_values(by='Importance Mean', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10).to_string(index=False))
    
    print("\nLeast Important Features (uninformative):")
    print(importance_df.tail(5).to_string(index=False))
    
    importance_df.to_csv('logs/feature_importances.csv', index=False)
    print("\nFeature importances saved to logs/feature_importances.csv")
    print("\nPipeline execution complete!")

if __name__ == '__main__':
    main()
