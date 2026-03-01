"""
Run Optuna hyperparameter tuning: multiple trials, each as one MLflow run
in the given experiment. Search space is defined in config/optuna.yaml.
"""

import argparse
from telco_churn.config import (
    FeaturesConfig,
    ModelConfig,
    OptunaConfig,
    ProjectConfig,
    load_config,
)
from telco_churn.models.tuning import run_optuna_study

PROJECT_CONFIG_PATH = "config/project.yaml"
FEATURES_CONFIG_PATH = "config/features.yaml"
MODEL_CONFIG_PATH = "config/model.yaml"
OPTUNA_CONFIG_PATH = "config/optuna.yaml"
DEFAULT_EXPERIMENT = "telco_churn_wrapper"


def main():
    parser = argparse.ArgumentParser(
        description="Run Optuna hyperparameter tuning; each trial logs one MLflow run."
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="MLflow experiment name (overrides config).",
    )
    parser.add_argument(
        "--optuna-config",
        type=str,
        default=OPTUNA_CONFIG_PATH,
        help="Path to Optuna config YAML.",
    )
    args = parser.parse_args()

    project_config = load_config(PROJECT_CONFIG_PATH, ProjectConfig)
    features_config = load_config(FEATURES_CONFIG_PATH, FeaturesConfig)
    model_config = load_config(MODEL_CONFIG_PATH, ModelConfig)
    optuna_config = load_config(args.optuna_config, OptunaConfig)

    experiment_name = (
        args.experiment
        or (project_config.experiment_name if project_config.experiment_name else None)
        or DEFAULT_EXPERIMENT
    )

    study = run_optuna_study(
        project_config,
        features_config,
        model_config,
        optuna_config,
        experiment_name,
        PROJECT_CONFIG_PATH,
        FEATURES_CONFIG_PATH,
        MODEL_CONFIG_PATH,
        seed=project_config.seed,
    )

    print(f"Best trial: {study.best_trial.number}")
    print(f"Best {optuna_config.metric}: {study.best_value}")
    print("Best params:", study.best_params)


if __name__ == "__main__":
    main()
