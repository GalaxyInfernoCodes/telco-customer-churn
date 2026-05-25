# Telco Customer Churn Prediction

A machine learning project designed to predict telco customer churn using a config-driven pipeline. It leverages **CatBoost** for classification, **Optuna** for hyperparameter tuning, **DuckDB** for data management, and **MLflow** for experiment tracking.

## Overview

The project is built as a modular Python package (`telco_churn`) with a strong emphasis on **Config-Driven Development**. All parameters, paths, and feature definitions are externalized in YAML files, allowing for reproducible experiments and easy model adjustments.

## Core Features

- **Config-Driven**: Centralized configuration management using Pydantic and YAML.
- **Data Management**: CSV ingestion into a local DuckDB database (`telco.duckdb`).
- **Modeling**: Gradient boosting with CatBoost and rule-based baseline comparison.
- **Hyperparameter Tuning**: Automated search using Optuna.
- **Experiment Tracking**: Full lifecycle tracking (params, metrics, artifacts) via MLflow.
- **Verification**: Built-in scripts and tests to verify data integrity and model inference.

## Project Structure

```text
├── artifacts/          # Trained models, plots, and other outputs
├── config/             # YAML configuration files (features, model, tuning, etc.)
├── data/               # Raw data storage
├── docs/               # Technical documentation
├── scripts/            # Executable entry points for the ML pipeline
├── src/telco_churn/    # Core library (data, features, models, evaluation)
├── tests/              # Unit and integration tests
└── pyproject.toml      # Project metadata and dependencies
```

## Setup & Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

1.  **Clone the repository** (if applicable).
2.  **Install dependencies**:
    ```bash
    uv sync
    ```

## Usage

All commands should be run using `uv run`.

### 1. Data Ingestion
Ingest raw CSV data into the local DuckDB database:
```bash
uv run scripts/ingest_data.py
```

### 2. Verification
Verify that the data setup is correct:
```bash
uv run pytest tests/test_verify.py::test_verify_data_setup
```

### 3. Exploratory Data Analysis
Generate diagnostic plots and statistics:
```bash
uv run scripts/explore_data.py
```

### 4. Model Training
Train a single model with current configurations:
```bash
uv run scripts/train_wrapper.py
```

### 5. Hyperparameter Tuning
Search for optimal hyperparameters using Optuna:
```bash
uv run scripts/tune.py
```

### 6. Evaluation
Evaluate the final model on the held-out test set:
```bash
uv run scripts/evaluate_on_test.py
```

## Experiment Tracking (MLflow)

To view your experiments and training runs, start the MLflow UI:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```
Navigate to [http://127.0.0.1:5001](http://127.0.0.1:5001) in your browser.

## Testing

Run the test suite to ensure system integrity:

```bash
uv run pytest
```

## Configuration

- `config/project.yaml`: Paths, seeds, and global settings.
- `config/features.yaml`: Categorical and numerical feature definitions.
- `config/model.yaml`: Hyperparameters for the CatBoost model.
- `config/optuna.yaml`: Tuning search space and trial settings.
