import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import joblib

# Add src/ to PYTHONPATH so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from brain_tumor_prediction.models.neural_net import BrainTumorMLP
from brain_tumor_prediction.models.train import load_model_h5

def main():
    parser = argparse.ArgumentParser(description="Predict brain tumor types from clinical data CSV.")
    parser.add_argument('--data', type=str, default='data/raw/brain_tumor_data.csv',
                        help="Path to the input CSV file containing patient records.")
    parser.add_argument('--output', type=str, default='predictions.csv',
                        help="Path to save the predicted classes CSV.")
    parser.add_argument('--model', type=str, default='studentID_model.h5',
                        help="Path to the trained model H5 file.")
    parser.add_argument('--pipeline', type=str, default='preprocessing_pipeline.joblib',
                        help="Path to the fitted preprocessing pipeline joblib file.")
    args = parser.parse_args()

    # Verify input file exists
    if not os.path.exists(args.data):
        print(f"Error: Input data file {args.data} does not exist.")
        sys.exit(1)

    # Resolve model path (fall back to student_id_model.h5 if studentID_model.h5 is missing)
    model_path = args.model
    if not os.path.exists(model_path):
        fallback_path = 'student_id_model.h5'
        if os.path.exists(fallback_path):
            model_path = fallback_path
        else:
            print(f"Error: Model file {args.model} (and fallback {fallback_path}) does not exist.")
            sys.exit(1)

    # Verify pipeline exists
    if not os.path.exists(args.pipeline):
        fallback_pipeline = 'models/preprocessing_pipeline.joblib'
        if os.path.exists(fallback_pipeline):
            pipeline_path = fallback_pipeline
        else:
            print(f"Error: Preprocessing pipeline file {args.pipeline} does not exist.")
            sys.exit(1)
    else:
        pipeline_path = args.pipeline

    print(f"Loading preprocessing pipeline from: {pipeline_path}")
    preprocessor = joblib.load(pipeline_path)

    # Hardcoded parameters of the best trained configuration:
    # Config 12: Hidden=[64, 32], LR=0.001, Dropout=0.2, Batch=32
    input_dim = 50
    hidden_dims = [64, 32]
    output_dim = 3
    dropout_rate = 0.2
    use_batch_norm = False # Default

    print(f"Instantiating model with: input_dim={input_dim}, hidden_dims={hidden_dims}, output_dim={output_dim}, dropout_rate={dropout_rate}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BrainTumorMLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        dropout_rate=dropout_rate,
        use_batch_norm=use_batch_norm
    ).to(device)

    print(f"Loading neural network weights from: {model_path}")
    model = load_model_h5(model, model_path).to(device)
    model.eval()

    # Load input data
    print(f"Loading input data from: {args.data}")
    df_raw = pd.read_csv(args.data)
    
    # Store patient_id for output reference
    patient_ids = df_raw['patient_id'] if 'patient_id' in df_raw.columns else pd.Series([f"PT_{i}" for i in range(len(df_raw))])

    # Preprocess
    print("Preprocessing input features...")
    X_processed = preprocessor.transform(df_raw)

    # Predict
    print("Running predictions through neural network...")
    X_tensor = torch.tensor(X_processed, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

    # Map predictions back to class names
    inverse_mapping = {0: 'Glioma', 1: 'Meningioma', 2: 'Pituitary'}
    pred_labels = [inverse_mapping[p] for p in preds]

    # Save to output file
    output_df = pd.DataFrame({
        'patient_id': patient_ids,
        'predicted_class': pred_labels,
        'glioma_prob': probs[:, 0],
        'meningioma_prob': probs[:, 1],
        'pituitary_prob': probs[:, 2]
    })
    
    output_df.to_csv(args.output, index=False)
    print(f"Predictions saved successfully to {args.output}")

    # Output first 10 predicted classes to stdout
    print("\nFirst 10 predictions:")
    print(output_df[['patient_id', 'predicted_class']].head(10).to_string(index=False))

if __name__ == '__main__':
    main()
