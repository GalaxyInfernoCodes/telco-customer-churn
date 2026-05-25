# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management. All commands are run via `uv run`.

```bash
uv sync
```

## Common Commands

```bash
# Full pipeline (in order)
uv run scripts/ingest_data.py          # Ingest CSV → DuckDB
uv run scripts/explore_data.py         # Generate EDA plots/stats
uv run scripts/train_wrapper.py        # Train CatBoost model
uv run scripts/train_wrapper.py --experiment my_experiment  # Override MLflow experiment name
uv run scripts/tune.py                 # Optuna hyperparameter search
uv run scripts/evaluate_on_test.py     # Evaluate on held-out test set

# Tests
uv run pytest                          # Run all tests
uv run pytest tests/test_verify.py::test_verify_data_setup   # Verify DuckDB setup
uv run pytest tests/test_verify.py::test_verify_model_inference  # Verify MLflow model inference
uv run pytest -m "not integration"     # Skip integration tests (require DuckDB + MLflow runs)

# MLflow UI
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Architecture

The project is a config-driven ML pipeline with this data flow:

**CSV → DuckDB (`data/telco.duckdb`) → train/validation/test splits → CatBoost model → MLflow**

### Config Layer (`config/` + `src/telco_churn/config/`)

All parameters are externalized in YAML and validated with Pydantic models:
- `config/project.yaml` → `ProjectConfig`: DB path, table names, artifact paths, MLflow experiment, optional baseline path
- `config/features.yaml` → `FeaturesConfig`: categorical/numerical feature lists and target column
- `config/model.yaml` → `ModelConfig`: CatBoost `model_type`, `params`, optional `run_name`
- `config/optuna.yaml` → `OptunaConfig`: metric to optimize, n_trials, param search space
- `config/baseline_rules.yaml` → `BaselineRulesConfig`: OR-of-ANDs rule groups for rule-based baseline

Scripts load configs with `load_config(path, ConfigClass)` from `telco_churn.config`.

### Training Pipeline (`src/telco_churn/pipelines/training.py`)

`run_training_pipeline()` is the central orchestrator, called by `scripts/train_wrapper.py`. It:
1. Loads train/val data from DuckDB via `load_data_from_db()`
2. Preprocesses features via `preprocess_features()` (returns `X, y` DataFrames)
3. Trains CatBoost via `ModelTrainer` (handles `sanitize_catboost_params()` to drop unsupported params like `max_leaves` for non-Lossguide grow policies)
4. Optionally evaluates the rule-based baseline (`RuleBasedModel`) and logs `baseline_val_*` metrics for comparison
5. Logs params, metrics, plots, config files, and the model artifact to MLflow (SQLite backend: `mlflow.db`, artifacts in `artifacts/mlruns/`)
6. Saves the `.cbm` model to `artifacts/models/catboost_model.cbm` and registers it via `CatBoostWrapper` (a `mlflow.pyfunc` wrapper)

### Models (`src/telco_churn/models/`)

- `training.py` — `ModelTrainer`: wraps CatBoost fit
- `wrapper.py` — `CatBoostWrapper`: `mlflow.pyfunc.PythonModel` for model registry + inference
- `baseline.py` — `RuleBasedModel`: evaluates config-driven rules (OR of AND conditions); compatible with `evaluate_model()` for metric comparison

### Integration Tests

Tests in `test_verify.py` are marked `@pytest.mark.integration` and require a live DuckDB + at least one trained MLflow run. Run them explicitly after setup, not in standard CI.
