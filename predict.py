#!/usr/bin/env python3
"""
predict.py
==========
Standalone inference script for brain tumor classification.

Loads raw data in the same format as brain_tumor_data.csv,
applies the saved preprocessing pipeline (fitted on training data),
runs the trained neural network, and outputs predicted classes with probabilities.

Usage:
    python predict.py <input.csv> <output.csv>

Example:
    python predict.py data/brain_tumor_data.csv output/predictions.csv
"""

import sys
import pandas as pd
import numpy as np
import joblib
import json
from tensorflow import keras


def predict(input_csv_path, output_csv_path):
    """
    Load raw patient data, apply preprocessing, and predict tumor types.

    CRITICAL: This function uses PRE-FITTED preprocessing artifacts.
    It does NOT fit anything on the new data - this prevents data leakage
    and ensures predictions use the exact same transformations as training.
    """

    # -------------------------------------------------------------------------
    # 1. Load configuration metadata
    # -------------------------------------------------------------------------
    with open('models/config.json', 'r') as f:
        config = json.load(f)

    # -------------------------------------------------------------------------
    # 2. Load raw data with same missing value handling as training
    # -------------------------------------------------------------------------
    missing_markers = ['NA', 'N/A', 'na', 'n/a', 'null', 'NULL', 
                       'None', 'none', '-', '--', 'missing', 'Missing',
                       'unknown', 'Unknown', ' ', '', 'NULL']

    df = pd.read_csv(input_csv_path, na_values=missing_markers)

    # Store patient IDs for output
    if 'patient_id' in df.columns:
        patient_ids = df['patient_id'].copy()
    else:
        patient_ids = pd.Series(range(len(df)), name='patient_id')

    # -------------------------------------------------------------------------
    # 3. Prepare features (drop ID and target if present)
    # -------------------------------------------------------------------------
    X = df.drop(columns=['patient_id'], errors='ignore')

    # If target column exists (e.g., for evaluation), drop it but do NOT use it
    if 'tumor_type' in X.columns:
        X = X.drop(columns=['tumor_type'])

    # -------------------------------------------------------------------------
    # 4. Apply SAME cleaning as training (no fitting, just transformation)
    # -------------------------------------------------------------------------
    # Clean gender lowercase (data dictionary noted this issue)
    if 'gender' in X.columns:
        X['gender'] = X['gender'].str.strip().str.capitalize()

    # Force numeric conversion for columns that may have loaded as text
    for col in config['numeric_cols']:
        if col in X.columns and X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce')

    # -------------------------------------------------------------------------
    # 5. Load pre-fitted artifacts and transform data
    # -------------------------------------------------------------------------
    # Load preprocessing pipeline (fitted on training data only)
    preprocessor = joblib.load('models/preprocessing_pipeline.pkl')

    # Load label encoder to map integers back to class names
    label_encoder = joblib.load('models/label_encoder.pkl')

    # Load trained neural network
    model = keras.models.load_model('models/studentID_model.h5')

    # Transform features - NO fitting, only transform!
    X_processed = preprocessor.transform(X)

    # -------------------------------------------------------------------------
    # 6. Predict
    # -------------------------------------------------------------------------
    # Get probability distribution over 3 classes
    probabilities = model.predict(X_processed, verbose=0)

    # Predicted class = index of highest probability
    predicted_indices = np.argmax(probabilities, axis=1)

    # Convert back to class names
    predicted_classes = label_encoder.inverse_transform(predicted_indices)

    # Confidence = probability of predicted class
    confidence = np.max(probabilities, axis=1)

    # -------------------------------------------------------------------------
    # 7. Save results
    # -------------------------------------------------------------------------
    results = pd.DataFrame({
        'patient_id': patient_ids,
        'predicted_tumor_type': predicted_classes,
        'confidence': np.round(confidence, 4)
    })

    # Add probability for each class
    for i, cls in enumerate(label_encoder.classes_):
        results[f'prob_{cls}'] = np.round(probabilities[:, i], 4)

    results.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to: {output_csv_path}")
    print(f"Total cases processed: {len(results)}")
    print(f"\nPrediction distribution:")
    print(results['predicted_tumor_type'].value_counts())


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python predict.py <input.csv> <output.csv>")
        print("Example: python predict.py data/new_patients.csv output/predictions.csv")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    predict(input_path, output_path)
