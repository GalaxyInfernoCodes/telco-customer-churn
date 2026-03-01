"""
Optuna-based hyperparameter tuning. Each trial runs the full training pipeline
and logs one MLflow run per trial within the given experiment.
"""

from typing import Any, Dict

import optuna
from optuna.trial import Trial

from telco_churn.config import FeaturesConfig, ModelConfig, OptunaConfig, ProjectConfig
from telco_churn.pipelines.training import run_training_pipeline

VALID_DIRECTIONS = ("minimize", "maximize")
DEFAULT_STARTUP_TRIALS = 5


def _suggest_int(trial: Trial, name: str, spec: Dict[str, Any]) -> int:
    """Suggest a single integer from spec (low, high, optional step)."""
    low = spec["low"]
    high = spec["high"]
    step = spec.get("step")
    if step is not None:
        return trial.suggest_int(name, low, high, step=step)
    return trial.suggest_int(name, low, high)


def _suggest_float(trial: Trial, name: str, spec: Dict[str, Any]) -> float:
    """Suggest a single float from spec (low, high, optional log, step)."""
    low = spec["low"]
    high = spec["high"]
    log = spec.get("log", False)
    step = spec.get("step")
    if step is not None:
        return trial.suggest_float(name, low, high, step=step, log=log)
    return trial.suggest_float(name, low, high, log=log)


def _suggest_categorical(trial: Trial, name: str, spec: Dict[str, Any]) -> Any:
    """Suggest a single categorical value from spec (choices)."""
    return trial.suggest_categorical(name, spec["choices"])


def suggest_params(
    trial: Trial, params_config: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Suggest one value per param from the config-driven search space.
    params_config: param name -> { type: int|float|categorical, ... }.
    """
    suggested = {}
    for name, spec in params_config.items():
        kind = (spec.get("type") or "float").lower()
        if kind == "int":
            suggested[name] = _suggest_int(trial, name, spec)
        elif kind == "float":
            suggested[name] = _suggest_float(trial, name, spec)
        elif kind == "categorical":
            suggested[name] = _suggest_categorical(trial, name, spec)
        else:
            raise ValueError(f"Unknown param type: {kind} for param {name}")
    return suggested


def _build_trial_model_config(
    base_config: ModelConfig,
    suggested_params: Dict[str, Any],
    trial_number: int,
) -> ModelConfig:
    """Merge base model params with suggested params; set run name for this trial."""
    merged_params = {**base_config.params, **suggested_params}
    return ModelConfig(
        model_type=base_config.model_type,
        params=merged_params,
        run_name=f"trial_{trial_number}",
    )


def _get_objective_metric(metrics: Dict[str, float], metric_name: str) -> float:
    """Return the pipeline metric used as the Optuna objective; raise if missing."""
    value = metrics.get(metric_name)
    if value is None:
        raise KeyError(
            f"Metric '{metric_name}' not in pipeline metrics: {list(metrics.keys())}"
        )
    return float(value)


def run_optuna_study(
    project_config: ProjectConfig,
    features_config: FeaturesConfig,
    base_model_config: ModelConfig,
    optuna_config: OptunaConfig,
    experiment_name: str,
    project_config_path: str,
    features_config_path: str,
    model_config_path: str,
    seed: int = 42,
) -> optuna.Study:
    """
    Run an Optuna study: n_trials training runs, each logged as an MLflow run
    in the given experiment. Base model params are merged with suggested params
    per trial (suggested override base for overlapping keys).
    """
    if optuna_config.direction not in VALID_DIRECTIONS:
        raise ValueError(
            f"direction must be one of {VALID_DIRECTIONS}, got {optuna_config.direction}"
        )

    def objective(trial: Trial) -> float:
        suggested = suggest_params(trial, optuna_config.params)
        trial_model_config = _build_trial_model_config(
            base_model_config, suggested, trial.number
        )
        metrics = run_training_pipeline(
            project_config,
            features_config,
            trial_model_config,
            experiment_name,
            project_config_path,
            features_config_path,
            model_config_path,
        )
        return _get_objective_metric(metrics, optuna_config.metric)

    n_startup = min(DEFAULT_STARTUP_TRIALS, optuna_config.n_trials)
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_startup)
    study = optuna.create_study(direction=optuna_config.direction, sampler=sampler)
    study.optimize(
        objective,
        n_trials=optuna_config.n_trials,
        show_progress_bar=True,
    )
    return study
