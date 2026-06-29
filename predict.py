import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib

# Add src to python path to resolve package imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from brain_tumor_prediction.models.neural_net import BrainTumorMLP
from brain_tumor_prediction.models.train import load_model_h5

def main():
    parser = argparse.ArgumentParser(description="Run inference with the trained Brain Tumor MLP classifier.")
    parser.add_argument('data_path', type=str, nargs='?', default='data/raw/brain_tumor_data.csv',
                        help="Path to the input CSV file in brain_tumor_data.csv format.")
    parser.add_argument('--output', type=str, default='predictions.csv',
                        help="Path to save the output predictions CSV.")
    parser.add_argument('--model_weights', type=str, default='studentID_model.h5',
                        help="Path to the HDF5 weights file.")
    parser.add_argument('--pipeline_path', type=str, default='preprocessing_pipeline.joblib',
                        help="Path to the preprocessor joblib file.")
    
    args = parser.parse_args()
    
    # 1. Check file existence
    if not os.path.exists(args.data_path):
        print(f"Error: Input data file '{args.data_path}' not found.")
        sys.exit(1)
        
    if not os.path.exists(args.pipeline_path):
        # Fallback to models/
        args.pipeline_path = os.path.join('models', 'preprocessing_pipeline.joblib')
        if not os.path.exists(args.pipeline_path):
            print(f"Error: Preprocessing pipeline file not found.")
            sys.exit(1)
            
    if not os.path.exists(args.model_weights):
        # Fallback to models/
        args.model_weights = os.path.join('models', 'studentID_model.h5')
        if not os.path.exists(args.model_weights):
            print(f"Error: Model weights file '{args.model_weights}' not found.")
            sys.exit(1)
            
    print(f"Loading preprocessing pipeline from: {args.pipeline_path}")
    pipeline = joblib.load(args.pipeline_path)
    
    print(f"Loading input data from: {args.data_path}")
    df = pd.read_csv(args.data_path)
    
    # Check if target column is in data (if so, we ignore it for prediction)
    if 'tumor_type' in df.columns:
        df_feats = df.drop(columns=['tumor_type'])
    else:
        df_feats = df.copy()
        
    # 2. Run Preprocessing
    print("Preprocessing data...")
    try:
        X_trans = pipeline.transform(df_feats)
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        print("Please ensure your CSV matches the exact formatting requirements (columns and types).")
        sys.exit(1)
        
    input_dim = X_trans.shape[1]
    
    # 3. Load MLP Model Architecture and Weights
    print(f"Initializing model with input dimension: {input_dim}")
    # Initialize architecture corresponding to the best tuned configuration (ID 12: [64, 32], lr=0.001, dropout=0.2, batch=32)
    model = BrainTumorMLP(
        input_dim=input_dim,
        hidden_dims=[64, 32],
        output_dim=3,
        dropout_rate=0.2
    )
    
    print(f"Loading model weights from: {args.model_weights}")
    model = load_model_h5(model, args.model_weights)
    model.eval()
    
    # 4. Perform Inference
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    inputs_tensor = torch.tensor(X_trans, dtype=torch.float32).to(device)
    
    print("Running inference...")
    with torch.no_grad():
        outputs = model(inputs_tensor)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        
    # 5. Output Results
    class_names = ['Glioma', 'Meningioma', 'Pituitary']
    predicted_classes = [class_names[pred] for pred in predictions]
    
    output_df = pd.DataFrame({
        'patient_id': df['patient_id'] if 'patient_id' in df.columns else range(len(df)),
        'predicted_class': predicted_classes,
        'probability_glioma': probabilities[:, 0],
        'probability_meningioma': probabilities[:, 1],
        'probability_pituitary': probabilities[:, 2]
    })
    
    output_df.to_csv(args.output, index=False)
    print(f"Predictions successfully saved to {args.output}")
    print("\nFirst 10 predictions:")
    print(output_df[['patient_id', 'predicted_class']].head(10).to_string(index=False))

if __name__ == '__main__':
    main()
