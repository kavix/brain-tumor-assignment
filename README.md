# Brain Tumor Assignment

A machine learning assignment for classifying brain tumor type from tabular clinical and imaging-derived features.

## Project summary

- **Task**: multiclass classification of tumor type (`Glioma`, `Meningioma`, `Pituitary`)
- **Dataset size**: 9,000 rows
- **Feature set**: 28 input features + 1 target column (`tumor_type`)
- **Main workflow**:
  - Train/evaluate model in `notebooks/model_train.ipynb`
  - Run standalone inference with `predict.py`

## Repository structure

- `/home/runner/work/brain-tumor-assignment/brain-tumor-assignment/data/brain_tumor_data.csv` — dataset used for training/inference
- `/home/runner/work/brain-tumor-assignment/brain-tumor-assignment/notebooks/model_train.ipynb` — end-to-end training notebook
- `/home/runner/work/brain-tumor-assignment/brain-tumor-assignment/predict.py` — script for batch predictions from CSV
- `/home/runner/work/brain-tumor-assignment/brain-tumor-assignment/model/` — saved model artifacts in this repo (`.h5`, `.keras`, preprocessing pipeline, label encoder, config)
- `/home/runner/work/brain-tumor-assignment/brain-tumor-assignment/pyproject.toml` — project metadata

## Dataset columns

Key input columns include demographics, clinical markers, imaging markers, and symptoms, such as:

- Demographics: `age`, `gender`, `ethnicity`, `region`
- Clinical/lab: `bmi`, `bp_systolic`, `bp_diastolic`, `wbc_count`, `crp_level`, `ki67_index`
- Imaging: `mri_intensity`, `ct_density`, `tumor_size_mm`, `tumor_location`, `contrast_enhancement`, `edema_grade`
- Symptoms/history: `headache_severity`, `nausea`, `vision_problems`, `seizures`, `memory_loss`, `balance_issues`, `family_history`

Target column:

- `tumor_type`

## Getting started

### 1) Create a Python environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

The repository does not pin runtime packages yet, so install the required ML stack manually:

```bash
pip install numpy pandas scikit-learn tensorflow joblib jupyter
```

### 3) Run training notebook (optional)

Open and run:

- `/home/runner/work/brain-tumor-assignment/brain-tumor-assignment/notebooks/model_train.ipynb`

This notebook performs preprocessing, model training, and saves artifacts.

### 4) Run inference

```bash
python /home/runner/work/brain-tumor-assignment/brain-tumor-assignment/predict.py \
  /home/runner/work/brain-tumor-assignment/brain-tumor-assignment/data/brain_tumor_data.csv \
  /home/runner/work/brain-tumor-assignment/brain-tumor-assignment/output/predictions.csv
```

> Note: `predict.py` currently loads artifacts from a `models/` directory. In this repository, artifacts are in `model/`. Ensure artifact paths match before running inference.

## Output format

`predict.py` writes a CSV with:

- `patient_id`
- `predicted_tumor_type`
- `confidence`
- class probabilities (`prob_<class_name>`)

## Notes

- `main.py` is a placeholder entry point.
- No dedicated automated test suite is currently included in the repository.
