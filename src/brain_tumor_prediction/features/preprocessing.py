import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

class BrainTumorDataCleaner(BaseEstimator, TransformerMixin):
    """
    Custom transformer to clean the brain tumor dataset.
    - Standardizes the case for the 'gender' column.
    - Maps inconsistent missing value markers ('?' and 'Unknown') to NaN.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        # Standardize gender to 'Male' or 'Female' (handle case inconsistencies)
        if 'gender' in X.columns:
            X['gender'] = X['gender'].astype(str).str.strip().str.capitalize()
            X['gender'] = X['gender'].apply(lambda val: val if val in ['Male', 'Female'] else np.nan)
            
        # Convert custom missing value markers to NaN
        if 'alcohol_consumption' in X.columns:
            X['alcohol_consumption'] = X['alcohol_consumption'].replace('Unknown', np.nan)
            
        if 'genetic_marker_status' in X.columns:
            X['genetic_marker_status'] = X['genetic_marker_status'].replace('?', np.nan)
            
        return X

def build_preprocessing_pipeline():
    """
    Builds and returns the complete preprocessing pipeline.
    Fits all preprocessing steps (imputation, scaling, encoding) on training data only.
    """
    # Columns lists matching data_dictionary.md
    numeric_cols = [
        'age', 'bmi', 'tumor_size_mm', 'tumor_growth_rate', 'headache_severity',
        'mri_intensity', 'ct_density', 'edema_grade', 'ki67_index',
        'bp_systolic', 'bp_diastolic', 'wbc_count', 'crp_level'
    ]
    
    # Ordinal features and their ordered categories
    ordinal_mappings = {
        'alcohol_consumption': ['None', 'Moderate', 'Heavy'],
        'contrast_enhancement': ['None', 'Mild', 'Moderate', 'Strong']
    }
    
    # Nominal categorical features
    nominal_cols = [
        'gender', 'ethnicity', 'region', 'smoking_status', 'family_history',
        'tumor_location', 'nausea', 'vision_problems', 'seizures',
        'memory_loss', 'balance_issues', 'genetic_marker_status'
    ]
    
    # Preprocessors for each data type
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Build ordinal preprocessors dynamically
    ordinal_transformers = []
    for col, categories in ordinal_mappings.items():
        transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(categories=[categories])),
            ('scaler', StandardScaler())
        ])
        ordinal_transformers.append((f'ordinal_{col}', transformer, [col]))
        
    nominal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            *ordinal_transformers,
            ('nom', nominal_transformer, nominal_cols)
        ]
    )
    
    # Complete pipeline
    pipeline = Pipeline(steps=[
        ('cleaner', BrainTumorDataCleaner()),
        ('preprocessor', preprocessor)
    ])
    
    return pipeline
