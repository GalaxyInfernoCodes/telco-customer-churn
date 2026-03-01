import argparse
import mlflow
from telco_churn.config import ModelConfig, ProjectConfig, FeaturesConfig, load_config
from telco_churn.pipelines.training import run_training_pipeline

# Config Paths
PROJECT_CONFIG_PATH = "config/project.yaml"
FEATURES_CONFIG_PATH = "config/features.yaml"
MODEL_CONFIG_PATH = "config/model.yaml"
DEFAULT_EXPERIMENT = "telco_churn_wrapper"


def main():
    parser = argparse.ArgumentParser(description="Run training pipeline.")
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help=f"MLflow experiment name (overrides config). Default from config or {DEFAULT_EXPERIMENT}.",
    )
    args = parser.parse_args()

    # Load configs
    project_config = load_config(PROJECT_CONFIG_PATH, ProjectConfig)
    features_config = load_config(FEATURES_CONFIG_PATH, FeaturesConfig)
    model_config = load_config(MODEL_CONFIG_PATH, ModelConfig)

    experiment_name = (
        args.experiment
        or (project_config.experiment_name if project_config.experiment_name else None)
        or DEFAULT_EXPERIMENT
    )

    run_training_pipeline(
        project_config,
        features_config,
        model_config,
        experiment_name,
        PROJECT_CONFIG_PATH,
        FEATURES_CONFIG_PATH,
        MODEL_CONFIG_PATH,
    )


if __name__ == "__main__":
    main()
