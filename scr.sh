#!/bin/bash

# ============================================================
# Brain Tumor Prediction - Project Scaffold Script
# Run: chmod +x create_project.sh && ./create_project.sh
# ============================================================

PROJECT_NAME="brain-tumor-prediction"
PACKAGE_NAME="brain_tumor_prediction"

echo "🚀 Creating project: $PROJECT_NAME"

# Create directory structure
mkdir -p $PROJECT_NAME/{data/{raw,processed,external},notebooks,src/$PACKAGE_NAME/{data,features,models},src/scripts,tests,models,configs,logs}

cd $PROJECT_NAME

# ============================================================
# 1. ROOT CONFIG FILES
# ============================================================

# .python-version
cat > .python-version << 'EOF'
3.11
EOF

# pyproject.toml
cat > pyproject.toml << 'EOF'
[project]
name = "brain-tumor-prediction"
version = "0.1.0"
description = "Brain tumor prediction ML pipeline"
requires-python = ">=3.11"
dependencies = [
    "pandas>=2.0",
    "numpy>=1.24",
    "scikit-learn>=1.3",
    "matplotlib>=3.7",
    "seaborn>=0.12",
    "missingno>=0.5",
    "jupyter>=1.0",
    "ipykernel>=6.0",
    "pyyaml>=6.0",
    "joblib>=1.3",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "ruff>=0.1",
    "mypy>=1.0",
]

[project.scripts]
train = "scripts.run_pipeline:main"
predict = "scripts.run_prediction:main"

[tool.ruff]
line-length = 88
target-version = "py311"

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
EOF

# .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.venv/
venv/
ENV/

# IDEs
.vscode/
.idea/
*.swp
*.swo

# Data & Models (large files)
data/raw/*
data/processed/*
data/external/*
!data/raw/.gitkeep
!data/processed/.gitkeep
!data/external/.gitkeep
models/*
!models/.gitkeep
logs/*
!logs/.gitkeep

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# OS
.DS_Store
Thumbs.db

# uv
uv.lock
EOF

# README.md
cat > README.md << 'EOF'
# Brain Tumor Prediction

ML pipeline for brain tumor type prediction using patient clinical data.

## Quick Start

```bash
# Install dependencies
uv sync

# Run full pipeline
make train

# Or run directly
uv run python -m scripts.run_pipeline