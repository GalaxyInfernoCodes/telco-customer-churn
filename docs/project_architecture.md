# Telco Customer Churn - Project Architecture

This document describes the critical parts of the project's infrastructure and how they fit together.

## Overview

The project is structured as a Python package (`telco_churn`) with executable scripts for various stages of the machine learning lifecycle. It uses a **Config-Driven Development** approach, where all parameters, paths, and feature definitions are stored in YAML files.

---

## 1. Core Components

### Configuration Management (`src/telco_churn/config/`)
The project uses **Pydantic** models to define and validate configuration schemas.

- **`ProjectConfig`**: Global settings (paths, seed, DuckDB path, and table names).
- **`FeaturesConfig`**: Column names for features and the target variable.
- **`ModelConfig`**: Hyperparameters, model types, and MLflow run names.
- **`OptunaConfig`**: Search space and settings for hyperparameter tuning.
- **`BaselineRulesConfig`**: Defines a set of logical rules (OR of ANDs) for the rule-based baseline classifier.

### Data Layer (`src/telco_churn/data/`)
- **`ingestion.py`**: Reads raw CSV data, performs a 60/20/20 Train/Val/Test split, and saves the datasets as tables in a **DuckDB** database (`telco.duckdb`).
- **`loading.py`**: Provides functions to load these tables into pandas DataFrames.
- **`exploration.py`**: Performs EDA and generates diagnostic plots.

### Features & Preprocessing (`src/telco_churn/features/`)
- **`preprocessing.py`**: Handles numeric conversions, target mapping, and categorical encoding.

### Modeling & Training (`src/telco_churn/models/`)
- **`training.py`**: A class that wraps the CatBoost training logic.
- **`core.py`**: Core modeling abstractions.
- **`baseline.py`**: Rule-based classifier for baseline performance comparison.
- **`tuning.py`**: Optuna-based hyperparameter optimization logic.
- **`wrapper.py`**: MLflow model wrapper for unified inference.

### Evaluation (`src/telco_churn/evaluation/`)
- **`metrics.py`**: Functions for calculating metrics and generating plots.
- **`testing.py`**: Logic for evaluating models on the held-out test set.

### Utilities (`src/telco_churn/utils/`)
- **`verification.py`**: Utility functions to verify data setup and model inference.

---

## 2. Orchestration & Pipelines

### Training Pipeline (`src/telco_churn/pipelines/training.py`)
The training pipeline orchestrates the end-to-end process:
1.  **Data Loading**: Fetches train/validation sets from DuckDB.
2.  **Input Logging**: Logs datasets to MLflow using `mlflow.log_input` for data lineage.
3.  **Preprocessing**: Numeric conversion and categorical feature handling.
4.  **Training**: Fits a CatBoost model using sanitized hyperparameters.
5.  **Baseline Evaluation**: (Optional) Evaluates a rule-based baseline if `baseline_rules_path` is provided in `ProjectConfig`.
6.  **Metrics & Plots**: Calculates ROC-AUC, PR-AUC, F1, etc., and logs diagnostic plots (Confusion Matrix, PR Curve).
7.  **Model Logging**: Saves the model as a custom MLflow `pyfunc` wrapper (`CatBoostWrapper`) for consistent inference.

---

## 3. Experiment Tracking & Inference

### MLflow Integration
- **Tracking URI**: Uses a local SQLite database (`sqlite:///mlflow.db`) for tracking parameters and metrics.
- **Artifact Location**: Artifacts (models, plots) are stored in `artifacts/mlruns`.
- **Custom Tagging**: Automatically logs a `balancing` tag (`none` or `class_weights`) and `mlflow.runName` for easier run filtering.

### Inference
The `CatBoostWrapper` ensures that the model can be deployed for inference with consistent preprocessing, abstracting away the underlying CatBoost specifics.

---

## 4. Execution Flow

All scripts should be executed using `uv run`.

1.  **Ingest**: `uv run scripts/ingest_data.py` (CSV -> DuckDB).
2.  **Verify Setup**: `uv run pytest tests/test_verify.py::test_verify_data_setup`.
3.  **Explore**: `uv run scripts/explore_data.py`.
4.  **Train/Tune**: `uv run scripts/train_wrapper.py` or `uv run scripts/tune.py`.
5.  **Verify Model**: `uv run pytest tests/test_verify.py::test_verify_model_inference`.
6.  **Evaluate**: `uv run scripts/evaluate_on_test.py`.
