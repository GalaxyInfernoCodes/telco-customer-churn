# Project Standards

## Execution

This project uses `uv` for dependency management and execution. All Python scripts and commands should be run using `uv run`.

- **Slim Scripts**: Scripts in `scripts/` should be minimal entry points. All business logic, data processing, and model training code must be encapsulated in `src/` modules.
- **Single Responsibility**: Functions should be small and focused on a single task. Avoid "god functions" that do calculations, plotting, and data transformation all in one place. Split them into smaller helpers (e.g., `calculate_metrics`, `create_plot`).
- **Imports**: All imports must be at the top of the file. Do not import modules inside functions or loops.
- **Documentation**: Keep the overview documentation in `docs/` and update it as needed.

### Examples

**Running a script:**
```bash
uv run scripts/train_wrapper.py
```

**Running pytest:**
```bash
uv run pytest
```

**Running a python command:**
```bash
uv run python -c "import telco_churn; print(telco_churn.__version__)"
```

## Directory Structure

- `src/`: Source code for the package.
- `scripts/`: Executable scripts (training, inference, etc.).
- `notebooks/`: Jupyter notebooks for exploration.
- `artifacts/`: Data and other artifacts.
- `docs/`: Documentation.

## MLflow Configuration

- Set the tracking URI to a local SQLite database for local development:
  ```python
  mlflow.set_tracking_uri("sqlite:///mlflow.db")
  ```

## Configuration

- **Config-Driven Development**: Avoid hardcoding parameters in code. Use YAML files in `config/` directory.
  - `config/project.yaml`: General project settings (paths, seed).
  - `config/features.yaml`: Feature definitions (lists of categorical/numerical columns).
  - `config/model.yaml`: Model hyperparameters.
- Use `src/telco_churn/config.py` to define Pydantic models for validation and loading.

## Testing & Quality

- **Tests**: Unit tests live in `tests/`. Run them with `uv run pytest`. New features (e.g. sampling) should include corresponding tests.
- **Slim Scripts**: Scripts in `scripts/` must not contain business logic; they only load config and call entry points in `src/`.


## Optional Features

- **Config-Driven Toggles**: Optional components (e.g. SMOTE) are controlled by config (e.g. `use_smote: true/false`) so the pipeline can run with or without them without code changes.

## Imports

- **No Star Imports**: Prefer explicit imports (e.g. `from x import y`) over `from x import *`.

## Viewing MLflow Runs

To view the MLflow UI, run:
```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.
