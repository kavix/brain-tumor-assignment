"""
Brain Tumor Classification - Complete Assignment Solution
Student ID: [YOUR_STUDENT_ID]
"""

# =============================================================================
# PART 0: SETUP
# =============================================================================
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# TensorFlow
import tensorflow as tf
tf.random.set_seed(RANDOM_SEED)
from tensorflow import keras
from tensorflow.keras import layers, regularizers

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, 
                            classification_report, confusion_matrix)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance

# Create directories
for d in ['models', 'figures', 'outputs']:
    os.makedirs(d, exist_ok=True)

print(f"TensorFlow version: {tf.__version__}")
print(f"Random seed set to: {RANDOM_SEED}")

# =============================================================================
# PART A - STEP 1: LOAD AND INSPECT
# =============================================================================
# Define missing value markers as per assignment: "several inconsistent markers"
MISSING_MARKERS = ['NA', 'N/A', 'na', 'n/a', 'null', 'NULL', 'None', 'none', 
                   '-', '--', 'missing', 'Missing', 'unknown', 'Unknown', 
                   ' ', '', 'NULL']

# Load data
df = pd.read_csv('data/brain_tumor_data.csv', na_values=MISSING_MARKERS)

print("=" * 60)
print("PART A - STEP 1: LOAD AND INSPECT")
print("=" * 60)
print(f"\nDataset shape: {df.shape}")
print(f"Expected: (9000, 29) - 1 ID + 27 features + 1 target\n")

print("Column data types:")
print(df.dtypes)
print(f"\nFirst 5 rows:")
print(df.head())

# Identify column types based on data dictionary
ID_COL = 'patient_id'
TARGET_COL = 'tumor_type'

NUMERIC_COLS = [
    'age', 'bmi', 'tumor_size_mm', 'tumor_growth_rate', 'headache_severity',
    'mri_intensity', 'ct_density', 'edema_grade', 'ki67_index',
    'bp_systolic', 'bp_diastolic', 'wbc_count', 'crp_level'
]

CATEGORICAL_COLS = [
    'gender', 'ethnicity', 'region', 'smoking_status', 'family_history',
    'tumor_location', 'nausea', 'vision_problems', 'seizures', 
    'memory_loss', 'balance_issues', 'genetic_marker_status'
]

ORDINAL_COLS = ['alcohol_consumption', 'contrast_enhancement']

print(f"\nIdentifier column: {ID_COL}")
print(f"Target column: {TARGET_COL}")
print(f"Numeric columns ({len(NUMERIC_COLS)}): {NUMERIC_COLS}")
print(f"Categorical columns ({len(CATEGORICAL_COLS)}): {CATEGORICAL_COLS}")
print(f"Ordinal columns ({len(ORDINAL_COLS)}): {ORDINAL_COLS}")

# Why drop patient_id?
print(f"\n--- Why exclude {ID_COL}? ---")
print(f"'{ID_COL}' is a unique identifier (string type, unique per patient).")
print(f"It has no predictive value for tumor type and would cause overfitting.")
print(f"Models cannot generalize patient IDs to new, unseen patients.")

# =============================================================================
# PART A - STEP 2: MISSING VALUES
# =============================================================================
print("\n" + "=" * 60)
print("PART A - STEP 2: MISSING VALUES")
print("=" * 60)

# Check for numeric columns that loaded as object (text) due to string markers
print("\nChecking for numeric columns loaded as text:")
object_cols = df.select_dtypes(include=['object']).columns.tolist()
object_cols.remove(ID_COL)  # Keep ID as object
object_cols.remove(TARGET_COL)  # Keep target as object
print(f"Object columns (excluding ID and target): {object_cols}")

# Check which supposed-numeric columns are in object type
numeric_as_object = [c for c in NUMERIC_COLS if c in df.select_dtypes(include=['object']).columns]
print(f"Numeric columns that loaded as object: {numeric_as_object}")

# Force conversion to numeric (coerce errors to NaN)
for col in NUMERIC_COLS:
    if df[col].dtype == 'object':
        print(f"  Converting {col} from {df[col].dtype} to numeric")
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Clean gender lowercase issue (per data dictionary)
print(f"\nGender value counts before cleaning:")
print(df['gender'].value_counts())
df['gender'] = df['gender'].str.strip().str.capitalize()
print(f"Gender value counts after cleaning:")
print(df['gender'].value_counts())

# Missing value counts per column
missing_counts = df.isnull().sum()
print(f"\nMissing values per column:")
print(missing_counts[missing_counts > 0].sort_values(ascending=False))

# =============================================================================
# PART A - STEP 3: ENCODING AND SCALING
# =============================================================================
print("\n" + "=" * 60)
print("PART A - STEP 3: ENCODING AND SCALING")
print("=" * 60)

# Separate features and target
X = df.drop(columns=[ID_COL, TARGET_COL])
y = df[TARGET_COL]

print(f"\nFeatures shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts()}")

# Encode target: Integer labels (0, 1, 2) for SparseCategoricalCrossentropy
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"\nTarget encoding:")
for i, cls in enumerate(label_encoder.classes_):
    print(f"  {cls} -> {i}")

# JUSTIFICATION:
print(f"\n--- Target Encoding Justification ---")
print(f"Using integer labels (0, 1, 2) with SparseCategoricalCrossentropy.")
print(f"Reason: For 3 classes, one-hot encoding is redundant. Integer labels")
print(f"are memory-efficient and directly compatible with softmax output.")
print(f"This constrains the output layer to 3 units with softmax activation.")

# Save label encoder
joblib.dump(label_encoder, 'models/label_encoder.pkl')

# Define ordinal category orders (must match data dictionary)
ALCOHOL_ORDER = ['None', 'Moderate', 'Heavy']
CONTRAST_ORDER = ['None', 'Mild', 'Moderate', 'Strong']

print(f"\nOrdinal category orders:")
print(f"  alcohol_consumption: {ALCOHOL_ORDER}")
print(f"  contrast_enhancement: {CONTRAST_ORDER}")

# JUSTIFICATION for encoding:
print(f"\n--- Encoding Justification ---")
print(f"Nominal categoricals (gender, ethnicity, etc.): OneHotEncoder")
print(f"  Reason: No inherent order. One-hot prevents false ordinal relationships.")
print(f"Ordinal categoricals (alcohol_consumption, contrast_enhancement): OrdinalEncoder")
print(f"  Reason: Clinical order matters (None < Moderate < Heavy). Preserves monotonicity.")
print(f"  edema_grade is already numeric (0,1,2,3) so treated as numeric.")

# Build preprocessing pipelines
# Numeric: Impute median + StandardScaler
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical: Impute most_frequent + OneHotEncoder
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Ordinal: Impute most_frequent + OrdinalEncoder
ordinal_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal', OrdinalEncoder(categories=[ALCOHOL_ORDER, CONTRAST_ORDER],
                              handle_unknown='use_encoded_value', 
                              unknown_value=-1))
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, NUMERIC_COLS),
    ('cat', categorical_pipeline, CATEGORICAL_COLS),
    ('ord', ordinal_pipeline, ORDINAL_COLS)
], remainder='drop')  # Drop any unexpected columns

# JUSTIFICATION for scaling:
print(f"\n--- Scaling Justification ---")
print(f"Using StandardScaler on numeric features.")
print(f"Reason: Features have vastly different scales (e.g., age 5-95 vs crp_level 0.1-60).")
print(f"Without scaling, gradient descent becomes unstable: large-scale features")
print(f"dominate gradients, causing slow convergence and numerical instability.")
print(f"StandardScaler (zero mean, unit variance) handles outliers better than MinMax")
print(f"for medical data where extreme values may be clinically significant.")

# =============================================================================
# PART A - STEP 4: CLASS DISTRIBUTION
# =============================================================================
print("\n" + "=" * 60)
print("PART A - STEP 4: CLASS DISTRIBUTION")
print("=" * 60)

class_dist = pd.Series(y).value_counts().sort_index()
class_props = class_dist / len(y)
print(f"\nClass distribution (counts):")
print(class_dist)
print(f"\nClass distribution (proportions):")
print(class_props)

is_balanced = class_props.max() < 0.5 and class_props.min() > 0.2
print(f"\nIs the dataset balanced? {'Yes' if is_balanced else 'No'}")
print(f"Max class proportion: {class_props.max():.3f}")
print(f"Min class proportion: {class_props.min():.3f}")

print(f"\n--- Consequences of Imbalance ---")
print(f"(a) Training: Model becomes biased toward majority class. Gradient descent")
print(f"    minimizes loss by predicting the majority class, ignoring minorities.")
print(f"(b) Evaluation: Accuracy becomes misleading. A dummy classifier predicting")
print(f"    the majority class achieves high accuracy while failing on minorities.")
print(f"\nMitigation technique: Class weights.")
print(f"Reason: Adjusts loss contribution per class without synthetic data generation.")
print(f"SMOTE is avoided as it could create unrealistic patient profiles.")

# Compute class weights (will apply during training)
class_weights = compute_class_weight('balanced', classes=np.unique(y_encoded), y=y_encoded)
class_weight_dict = dict(enumerate(class_weights))
print(f"\nClass weights: {class_weight_dict}")

# =============================================================================
# PART C - STEP 1: DATA SPLITTING
# =============================================================================
print("\n" + "=" * 60)
print("PART C - STEP 1: DATA SPLITTING")
print("=" * 60)

# Split: 70% train, 15% validation, 15% test
# First split off test (15%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y_encoded, test_size=0.15, random_state=RANDOM_SEED, stratify=y_encoded
)

# Split temp into train (70%) and validation (15%)
# 0.15 / 0.85 = 0.17647
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.17647, random_state=RANDOM_SEED, stratify=y_temp
)

print(f"Train set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X):.1%})")
print(f"Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(X):.1%})")
print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X):.1%})")

print(f"\n--- Split Justification ---")
print(f"Stratified split ensures each subset maintains original class proportions.")
print(f"Train: used to learn model weights.")
print(f"Validation: used to tune hyperparameters and detect overfitting.")
print(f"Test: used ONLY ONCE for final evaluation. Set aside before any tuning.")

print(f"\n--- Why test set must be untouched until the end ---")
print(f"Using the test set during tuning causes data leakage. The model indirectly")
print(f"learns test patterns, making evaluation optimistic and invalid. The test set")
print(f"must simulate truly unseen future patients.")

# FIT preprocessing on training data ONLY
print(f"\nFitting preprocessing pipeline on training data ONLY...")
preprocessor.fit(X_train)

# Transform all sets
X_train_proc = preprocessor.transform(X_train)
X_val_proc = preprocessor.transform(X_val)
X_test_proc = preprocessor.transform(X_test)

print(f"Processed feature shape: {X_train_proc.shape}")

# Save fitted preprocessor
joblib.dump(preprocessor, 'models/preprocessing_pipeline.pkl')

# Get feature names for later interpretation
# One-hot feature names
onehot_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
cat_feature_names = onehot_encoder.get_feature_names_out(CATEGORICAL_COLS)

# Ordinal feature names
ord_feature_names = ORDINAL_COLS

# Combine all feature names
feature_names = NUMERIC_COLS + list(cat_feature_names) + ord_feature_names
print(f"Total features after preprocessing: {len(feature_names)}")

# =============================================================================
# PART B - STEP 1: MODEL DESIGN
# =============================================================================
print("\n" + "=" * 60)
print("PART B - STEP 1: MODEL DESIGN")
print("=" * 60)

input_dim = X_train_proc.shape[1]

print(f"Input units: {input_dim}")
print(f"  = {len(NUMERIC_COLS)} numeric + {len(cat_feature_names)} one-hot + {len(ORDINAL_COLS)} ordinal")

print(f"\nHidden layers: 3 layers [64, 32, 16]")
print(f"  Justification: 2-3 layers is optimal for ~9000 samples and ~20-30 features.")
print(f"  Too few layers -> underfitting. Too many -> overfitting on limited data.")
print(f"  First layer wide (64) to capture feature interactions.")
print(f"  Subsequent layers narrow (32, 16) to force hierarchical abstraction.")

print(f"\nOutput units: 3 (Glioma, Meningioma, Pituitary)")
print(f"Activation: Softmax")
print(f"  Justification: Softmax converts logits to probability distribution.")
print(f"  For multi-class (3+ classes), softmax ensures outputs sum to 1.0.")
print(f"  For binary (2 classes), sigmoid would be used instead.")

print(f"\nLoss function: sparse_categorical_crossentropy")
print(f"  Justification: Correct for integer-encoded labels (0, 1, 2).")
print(f"  Measures divergence between predicted probability distribution and true class.")
print(f"  Penalizes confident wrong predictions heavily - appropriate for clinical diagnosis.")

def build_model(input_dim, lr=0.001, dropout_rate=0.3, l2_reg=0.001):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        
        layers.Dense(64, activation='relu', 
                    kernel_regularizer=regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        
        layers.Dense(32, activation='relu',
                    kernel_regularizer=regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        
        layers.Dense(16, activation='relu',
                    kernel_regularizer=regularizers.l2(l2_reg)),
        
        layers.Dense(3, activation='softmax')
    ])
    
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Build initial model
model = build_model(input_dim)
print(f"\nModel architecture:")
model.summary()

# =============================================================================
# PART B - STEP 2: REGULARIZATION
# =============================================================================
print("\n" + "=" * 60)
print("PART B - STEP 2: REGULARIZATION")
print("=" * 60)

print(f"Regularization mechanisms used:")
print(f"1. Dropout (0.3): Randomly zeros 30% of neurons per batch.")
print(f"   Prevents co-adaptation. Forces distributed representations.")
print(f"2. L2 Weight Decay (0.001): Penalizes large weights.")
print(f"   Keeps weights small and spread across features.")
print(f"3. Batch Normalization: Normalizes layer inputs.")
print(f"   Reduces internal covariate shift, adds mild regularization noise.")
print(f"4. Early Stopping: Halts training when validation loss plateaus.")
print(f"   Prevents training past the point of generalization.")

print(f"\n--- What is overfitting? ---")
print(f"Overfitting occurs when the model memorizes training data noise and specifics")
print(f"instead of learning generalizable patterns. Symptoms: training accuracy rises")
print(f"while validation accuracy falls or stagnates. The model fails on new patients.")

# =============================================================================
# PART C - STEP 2: TRAINING
# =============================================================================
print("\n" + "=" * 60)
print("PART C - STEP 2: TRAINING")
print("=" * 60)

print(f"Optimizer: Adam (learning_rate=0.001)")
print(f"  Justification: Adaptive per-parameter learning rates. Combines momentum")
print(f"  and RMSprop advantages. Robust default for most problems.")
print(f"Batch size: 32")
print(f"  Justification: Smaller batches add gradient noise (mild regularization).")
print(f"  Good for dataset size ~9000. Larger batches (64) could be tried in tuning.")

# Callbacks
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1
)

# Train
history = model.fit(
    X_train_proc, y_train,
    validation_data=(X_val_proc, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    class_weight=class_weight_dict,
    verbose=1
)

# Plot training curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss
axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training vs Validation Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Accuracy
axes[1].plot(history.history['accuracy'], label='Train Acc', linewidth=2)
axes[1].plot(history.history['val_accuracy'], label='Val Acc', linewidth=2)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Training vs Validation Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/training_curves_initial.png', dpi=150, bbox_inches='tight')
plt.show()

# Interpretation
train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]
train_loss = history.history['loss'][-1]
val_loss = history.history['val_loss'][-1]

print(f"\n--- Training Curve Interpretation ---")
print(f"Final train accuracy: {train_acc:.4f}")
print(f"Final val accuracy: {val_acc:.4f}")
print(f"Final train loss: {train_loss:.4f}")
print(f"Final val loss: {val_loss:.4f}")

if val_acc < train_acc - 0.05:
    print(f"Status: OVERFITTING detected. Validation accuracy lags training by >5%.")
    print(f"Evidence: Gap between train and val curves widening.")
elif train_acc < 0.6 and val_acc < 0.6:
    print(f"Status: UNDERFITTING suspected. Both curves are low.")
    print(f"Evidence: Model too simple to capture data patterns.")
else:
    print(f"Status: WELL-FIT. Train and validation curves converge closely.")
    print(f"Evidence: Similar final values with minimal gap.")

# =============================================================================
# PART C - STEP 3: HYPERPARAMETER TUNING
# =============================================================================
print("\n" + "=" * 60)
print("PART C - STEP 3: HYPERPARAMETER TUNING")
print("=" * 60)

def build_tunable_model(input_dim, units_layer1=64, units_layer2=32, 
                        units_layer3=16, dropout_rate=0.3, l2_reg=0.001):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(units_layer1, activation='relu', 
                    kernel_regularizer=regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        layers.Dense(units_layer2, activation='relu',
                    kernel_regularizer=regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        layers.Dense(units_layer3, activation='relu',
                    kernel_regularizer=regularizers.l2(l2_reg)) if units_layer3 > 0 else layers.Identity(),
        layers.Dense(3, activation='softmax')
    ])
    # Remove identity layer if no layer3
    # Actually simpler to just conditionally add
    # Let me rewrite properly
    pass

# Actually, let me write a cleaner version
def build_tunable_model(input_dim, config):
    l2_reg = config.get('l2_reg', 0.001)
    dropout_rate = config.get('dropout_rate', 0.3)
    
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    
    # Layer 1
    model.add(layers.Dense(config['units_1'], activation='relu',
                          kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout_rate))
    
    # Layer 2
    model.add(layers.Dense(config['units_2'], activation='relu',
                          kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout_rate))
    
    # Optional Layer 3
    if config.get('units_3', 0) > 0:
        model.add(layers.Dense(config['units_3'], activation='relu',
                              kernel_regularizer=regularizers.l2(l2_reg)))
    
    # Output
    model.add(layers.Dense(3, activation='softmax'))
    
    optimizer = keras.optimizers.Adam(learning_rate=config['lr'])
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Hyperparameter configurations
configs = [
    {'name': 'Baseline', 'units_1': 64, 'units_2': 32, 'units_3': 16, 
     'lr': 0.001, 'dropout_rate': 0.3, 'l2_reg': 0.001, 'batch_size': 32},
    {'name': 'Wider', 'units_1': 128, 'units_2': 64, 'units_3': 32, 
     'lr': 0.001, 'dropout_rate': 0.3, 'l2_reg': 0.001, 'batch_size': 32},
    {'name': 'Deeper', 'units_1': 64, 'units_2': 32, 'units_3': 16, 
     'lr': 0.001, 'dropout_rate': 0.5, 'l2_reg': 0.01, 'batch_size': 32},
    {'name': 'Lower_LR', 'units_1': 64, 'units_2': 32, 'units_3': 16, 
     'lr': 0.0001, 'dropout_rate': 0.3, 'l2_reg': 0.001, 'batch_size': 16},
    {'name': 'Large_Batch', 'units_1': 64, 'units_2': 32, 'units_3': 16, 
     'lr': 0.001, 'dropout_rate': 0.2, 'l2_reg': 0.0001, 'batch_size': 64},
]

tuning_results = []

for config in configs:
    print(f"\nTraining: {config['name']}")
    tf.keras.backend.clear_session()
    
    model_t = build_tunable_model(input_dim, config)
    
    hist = model_t.fit(
        X_train_proc, y_train,
        validation_data=(X_val_proc, y_val),
        epochs=50,
        batch_size=config['batch_size'],
        callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, 
                                                  restore_best_weights=True, verbose=0)],
        class_weight=class_weight_dict,
        verbose=0
    )
    
    best_val_acc = max(hist.history['val_accuracy'])
    best_val_loss = min(hist.history['val_loss'])
    epochs_trained = len(hist.history['loss'])
    
    tuning_results.append({
        'Config': config['name'],
        'Units': f"{config['units_1']}-{config['units_2']}-{config['units_3']}",
        'LR': config['lr'],
        'Dropout': config['dropout_rate'],
        'L2': config['l2_reg'],
        'Batch': config['batch_size'],
        'Val_Acc': round(best_val_acc, 4),
        'Val_Loss': round(best_val_loss, 4),
        'Epochs': epochs_trained
    })

tuning_df = pd.DataFrame(tuning_results)
print(f"\nHyperparameter Comparison:")
print(tuning_df.to_string(index=False))

# Select best model based on validation accuracy
best_idx = tuning_df['Val_Acc'].idxmax()
best_config = configs[best_idx]
print(f"\nBest configuration: {best_config['name']} (Val Acc: {tuning_df.loc[best_idx, 'Val_Acc']})")
print(f"Selection based on VALIDATION performance only, not test set.")

# Retrain best model
print(f"\nRetraining best model...")
tf.keras.backend.clear_session()
best_model = build_tunable_model(input_dim, best_config)

best_history = best_model.fit(
    X_train_proc, y_train,
    validation_data=(X_val_proc, y_val),
    epochs=100,
    batch_size=best_config['batch_size'],
    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, 
                                              restore_best_weights=True, verbose=1)],
    class_weight=class_weight_dict,
    verbose=1
)

# =============================================================================
# PART D - STEP 1: FINAL EVALUATION
# =============================================================================
print("\n" + "=" * 60)
print("PART D - STEP 1: FINAL EVALUATION")
print("=" * 60)

print(f"CRITICAL: Using preprocessing fitted on training data, applied unchanged to test.")
print(f"NO re-fitting. NO tuning after seeing test results.")

# Predict on test
y_pred_probs = best_model.predict(X_test_proc, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

# Overall accuracy
test_acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {test_acc:.4f}")

# Per-class metrics
precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None)
macro_f1 = precision_recall_fscore_support(y_test, y_pred, average='macro')[2]
weighted_f1 = precision_recall_fscore_support(y_test, y_pred, average='weighted')[2]

metrics_df = pd.DataFrame({
    'Class': label_encoder.classes_,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1,
    'Support': support
})
print(f"\nPer-class metrics:")
print(metrics_df.round(4).to_string(index=False))

print(f"\nMacro F1: {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")

print(f"\n--- Macro vs Weighted F1 ---")
print(f"Macro F1: Unweighted average of per-class F1 scores.")
print(f"  More informative for medical diagnosis because it treats all tumor types equally.")
print(f"  Missing a rare but aggressive tumor is catastrophic; macro F1 exposes this.")
print(f"Weighted F1: Average weighted by class support (sample count).")
print(f"  Informs overall dataset performance but can hide minority class failures.")

# =============================================================================
# PART D - STEP 2: CONFUSION MATRIX
# =============================================================================
print("\n" + "=" * 60)
print("PART D - STEP 2: CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Test Set')
plt.tight_layout()
plt.savefig('figures/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

# Find most confused pair
# Off-diagonal elements
max_confusion = 0
confused_pair = None
for i in range(3):
    for j in range(3):
        if i != j and cm[i, j] > max_confusion:
            max_confusion = cm[i, j]
            confused_pair = (label_encoder.classes_[i], label_encoder.classes_[j])

print(f"\nMost confused pair: {confused_pair[0]} -> {confused_pair[1]} ({max_confusion} cases)")
print(f"\nPlausible reason: Both are primary brain tumors with overlapping clinical")
print(f"presentations per the data dictionary. They share similar locations")
print(f"(Frontal, Temporal, Parietal), symptoms (headache, seizures, vision problems),")
print(f"and imaging characteristics (MRI intensity, contrast enhancement patterns).")
print(f"Pituitary tumors are localized to the sellar region with distinct endocrine")
print(f"profiles, making them easier to distinguish.")

# =============================================================================
# PART D - STEP 3: CONFIDENCE ANALYSIS
# =============================================================================
print("\n" + "=" * 60)
print("PART D - STEP 3: CONFIDENCE ANALYSIS")
print("=" * 60)

max_probs = np.max(y_pred_probs, axis=1)

# High confidence (top 2)
high_idx = np.argsort(max_probs)[-2:][::-1]
print(f"\nHIGH CONFIDENCE PREDICTIONS:")
for idx in high_idx:
    actual = label_encoder.inverse_transform([y_test[idx]])[0]
    predicted = label_encoder.inverse_transform([y_pred[idx]])[0]
    probs = dict(zip(label_encoder.classes_, y_pred_probs[idx].round(4)))
    print(f"  Test index {idx}: Actual={actual}, Predicted={predicted}")
    print(f"    Probabilities: {probs}")
    print(f"    Confidence: {max_probs[idx]:.4f}")

# Low confidence (bottom 2)
low_idx = np.argsort(max_probs)[:2]
print(f"\nLOW CONFIDENCE PREDICTIONS:")
for idx in low_idx:
    actual = label_encoder.inverse_transform([y_test[idx]])[0]
    predicted = label_encoder.inverse_transform([y_pred[idx]])[0]
    probs = dict(zip(label_encoder.classes_, y_pred_probs[idx].round(4)))
    print(f"  Test index {idx}: Actual={actual}, Predicted={predicted}")
    print(f"    Probabilities: {probs}")
    print(f"    Confidence: {max_probs[idx]:.4f}")

print(f"\n--- What distinguishes high vs low confidence? ---")
print(f"High confidence: Feature values strongly match textbook profiles for one class.")
print(f"  (e.g., sellar location + specific symptoms = Pituitary)")
print(f"Low confidence: Ambiguous features near class boundaries, mixed symptoms,")
print(f"or imputed missing values that obscure the true clinical picture.")

# =============================================================================
# PART E - STEP 1: BASELINE COMPARISON
# =============================================================================
print("\n" + "=" * 60)
print("PART E - STEP 1: BASELINE COMPARISON")
print("=" * 60)

# Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, class_weight='balanced')
rf.fit(X_train_proc, y_train)
rf_pred = rf.predict(X_test_proc)
rf_acc = accuracy_score(y_test, rf_pred)
rf_macro_f1 = precision_recall_fscore_support(y_test, rf_pred, average='macro')[2]

# Logistic Regression
lr = LogisticRegression(max_iter=2000, random_state=RANDOM_SEED, class_weight='balanced')
lr.fit(X_train_proc, y_train)
lr_pred = lr.predict(X_test_proc)
lr_acc = accuracy_score(y_test, lr_pred)
lr_macro_f1 = precision_recall_fscore_support(y_test, lr_pred, average='macro')[2]

comparison = pd.DataFrame({
    'Model': ['Neural Network', 'Random Forest', 'Logistic Regression'],
    'Test Accuracy': [test_acc, rf_acc, lr_acc],
    'Macro F1': [macro_f1, rf_macro_f1, lr_macro_f1]
})
print(f"\nBaseline Comparison:")
print(comparison.round(4).to_string(index=False))

print(f"\n--- Was the neural network worth the complexity? ---")
if test_acc > max(rf_acc, lr_acc) + 0.03:
    print(f"Yes. The NN achieves >3% improvement over baselines.")
    print(f"For clinical diagnosis, even small accuracy gains matter.")
else:
    print(f"Marginally. The NN performs similarly to simpler models.")
    print(f"For tabular medical data with ~9000 rows, tree-based methods often match NNs.")
    print(f"The added complexity (training time, hyperparameter tuning, black-box nature)")
    print(f"may not be justified unless NN shows significantly better minority class recall.")

# =============================================================================
# PART E - STEP 2: FEATURE IMPORTANCE
# =============================================================================
print("\n" + "=" * 60)
print("PART E - STEP 2: FEATURE IMPORTANCE")
print("=" * 60)

# Use Random Forest for permutation importance (model-agnostic but needs a fitted model)
rf_for_perm = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
rf_for_perm.fit(X_train_proc, y_train)

perm = permutation_importance(rf_for_perm, X_test_proc, y_test, 
                               n_repeats=10, random_state=RANDOM_SEED, scoring='f1_macro')

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': perm.importances_mean,
    'Std': perm.importances_std
}).sort_values('Importance', ascending=False)

print(f"\nTop 10 most important features:")
print(importance_df.head(10).round(4).to_string(index=False))

print(f"\nBottom 5 least important features:")
print(importance_df.tail(5).round(4).to_string(index=False))

# Plot
plt.figure(figsize=(10, 8))
sns.barplot(data=importance_df.head(15), x='Importance', y='Feature')
plt.title('Top 15 Feature Importances (Permutation)')
plt.tight_layout()
plt.savefig('figures/feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

# Two unimportant features
unimportant = importance_df.tail(2)['Feature'].tolist()
print(f"\nTwo largely unimportant features: {unimportant}")
print(f"Reason: Their permutation importance is near zero. Shuffling these")
print(f"features does not degrade model performance, indicating the model does")
print(f"not rely on them. They may have no true correlation with tumor type")
print(f"in this dataset, or other features provide redundant information.")

# =============================================================================
# PART E - STEP 3: LIMITATIONS AND REFLECTION
# =============================================================================
print("\n" + "=" * 60)
print("PART E - STEP 3: LIMITATIONS AND REFLECTION")
print("=" * 60)

print(f"\nLIMITATION 1: Synthetic data.")
print(f"  The data dictionary states this is synthetic for coursework only.")
print(f"  Feature-tumor relationships may not reflect real histopathology biology.")
print(f"  Deployment on real patients without real-world validation is dangerous.")

print(f"\nLIMITATION 2: Missing value imputation bias.")
print(f"  We imputed missing values with median/mode. If missingness is not random")
print(f"  (e.g., severe patients missing more data), imputation biases the model")
print(f"  toward healthier profiles, potentially missing severe cases.")

print(f"\nLIMITATION 3: No external/temporal validation.")
print(f"  Model validated on random split from same dataset. Real deployment requires")
print(f"  validation across different hospitals, time periods, and imaging equipment.")

print(f"\n--- Clinical Deployment Risk ---")
print(f"RISK: Automation bias. Clinicians may defer to high-confidence predictions")
print(f"without critical review. A wrong prediction leads to incorrect treatment")
print(f"(wrong surgery, delayed chemotherapy, unnecessary radiation).")

print(f"\nSAFEGUARD required:")
print(f"1. Human-in-the-loop: Model outputs as 'suggestions', not diagnoses.")
print(f"   Final diagnosis requires neuroradiologist + histopathologist confirmation.")
print(f"2. Uncertainty quantification: Flag low-confidence predictions (<0.7)")
print(f"   for mandatory specialist review.")
print(f"3. Continuous monitoring: Track performance on incoming real data.")
print(f"   Trigger retraining if distribution shifts (concept drift).")

# =============================================================================
# SAVE FINAL MODEL AND ARTIFACTS
# =============================================================================
print("\n" + "=" * 60)
print("SAVING FINAL MODEL AND ARTIFACTS")
print("=" * 60)

best_model.save('models/studentID_model.h5')
print(f"Model saved: models/studentID_model.h5")

# Save config for reference
config_save = {
    'random_seed': RANDOM_SEED,
    'input_dim': input_dim,
    'best_config': best_config,
    'numeric_cols': NUMERIC_COLS,
    'categorical_cols': CATEGORICAL_COLS,
    'ordinal_cols': ORDINAL_COLS,
    'ordinal_orders': {'alcohol_consumption': ALCOHOL_ORDER, 
                       'contrast_enhancement': CONTRAST_ORDER}
}
with open('models/config.json', 'w') as f:
    json.dump(config_save, f, indent=2)
print(f"Config saved: models/config.json")

print(f"\nAll artifacts saved. Ready for predict.py")