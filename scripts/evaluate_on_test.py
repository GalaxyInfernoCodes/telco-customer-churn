"""
Evaluate a training run on the test set and log test_* metrics to that MLflow run.
Run for candidate best runs to get a fair final comparison.
"""

import argparse
from telco_churn.config import FeaturesConfig, ProjectConfig, load_config
from telco_churn.evaluation.testing import evaluate_run_on_test

PROJECT_CONFIG_PATH = "config/project.yaml"
FEATURES_CONFIG_PATH = "config/features.yaml"


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a run on test set and log test_* metrics to MLflow."
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run ID to evaluate; if omitted, use latest run in experiment.",
    )
    args = parser.parse_args()

    project_config = load_config(PROJECT_CONFIG_PATH, ProjectConfig)
    features_config = load_config(FEATURES_CONFIG_PATH, FeaturesConfig)

    evaluate_run_on_test(
        project_config,
        features_config,
        args.experiment,
        run_id=args.run_id,
    )


if __name__ == "__main__":
    main()
