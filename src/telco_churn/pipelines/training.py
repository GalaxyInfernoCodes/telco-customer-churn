import os

import mlflow
import matplotlib.pyplot as plt
import pandas as pd

from telco_churn.config import (
    ModelConfig,
    ProjectConfig,
    FeaturesConfig,
    BaselineRulesConfig,
    load_config,
)
from telco_churn.data.loading import load_data_from_db
from telco_churn.features.preprocessing import preprocess_features
from telco_churn.models.training import ModelTrainer, sanitize_catboost_params
from telco_churn.evaluation.metrics import evaluate_model, calculate_metrics_with_prefix
from telco_churn.models.baseline import RuleBasedModel
from telco_churn.models.wrapper import CatBoostWrapper


def _infer_balancing_tag(
    project_config: ProjectConfig,
    model_config: ModelConfig,
) -> str:
    """Infer MLflow 'balancing' tag from config and pipeline state."""
    if project_config.balancing is not None:
        return project_config.balancing
    has_weights = (
        (model_config.params.get("auto_class_weights") or "").lower() == "balanced"
    )
    if has_weights:
        return "class_weights"
    return "none"


def _log_run_tags(
    project_config: ProjectConfig,
    model_config: ModelConfig,
) -> None:
    """Set MLflow run name and balancing tags for filtering."""
    run_name = (
        model_config.run_name
        if model_config.run_name
        else f"catboost_{model_config.model_type}"
    )
    mlflow.set_tag("mlflow.runName", run_name)
    mlflow.set_tag("balancing", _infer_balancing_tag(project_config, model_config))


def run_training_pipeline(
    project_config: ProjectConfig,
    features_config: FeaturesConfig,
    model_config: ModelConfig,
    experiment_name: str,
    project_config_path: str,
    features_config_path: str,
    model_config_path: str
):
    """
    Run the end-to-end training pipeline.
    """
    # Set up MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    
    # Check if experiment exists, if not create with custom artifact location
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        artifact_loc = os.path.abspath("artifacts/mlruns")
        experiment_id = client.create_experiment(experiment_name, artifact_location=artifact_loc)
        mlflow.set_experiment(experiment_id=experiment_id)
    else:
        mlflow.set_experiment(experiment_name)
        
    with mlflow.start_run() as run:
        print(f"Starting run: {run.info.run_id}")
        
        # 1. Load Data
        print(f"Loading data from {project_config.db_path}...")
        train_df = load_data_from_db(project_config.db_path, project_config.tables['train'])
        val_df = load_data_from_db(project_config.db_path, project_config.tables['validation'])
        
        print(f"Train shape: {train_df.shape}")
        print(f"Validation shape: {val_df.shape}")

        # 2. Log Inputs
        train_ds = mlflow.data.from_pandas(train_df, name="train", targets=features_config.target_col)
        val_ds = mlflow.data.from_pandas(val_df, name="validation", targets=features_config.target_col)
        
        mlflow.log_input(train_ds, context="training")
        mlflow.log_input(val_ds, context="validation")
        
        # 3. Log Configs
        print("Logging config files...")
        mlflow.log_artifact(project_config_path, artifact_path="config")
        mlflow.log_artifact(features_config_path, artifact_path="config")
        mlflow.log_artifact(model_config_path, artifact_path="config")
        
        # 4. Preprocess
        print("Preprocessing features...")
        X_train, y_train = preprocess_features(
            train_df, 
            target_col=features_config.target_col,
            categorical_features=features_config.categorical_features,
            numerical_features=features_config.numerical_features
        )
        
        X_val, y_val = preprocess_features(
            val_df,
            target_col=features_config.target_col,
            categorical_features=features_config.categorical_features,
            numerical_features=features_config.numerical_features,
        )

        # 5. Train (use params actually accepted by CatBoost, e.g. drop max_leaves when not Lossguide)
        params_to_use = sanitize_catboost_params(model_config.params)
        train_config = ModelConfig(
            model_type=model_config.model_type,
            params=params_to_use,
            run_name=model_config.run_name,
        )
        print("Training model...")
        trainer = ModelTrainer(train_config)
        model = trainer.train(X_train, y_train)

        # 6. Log Params (only what was actually used by the model)
        print("Logging parameters...")
        mlflow.log_params(params_to_use)
        
        # 7. Evaluate & Log Metrics
        print("Evaluating model...")
        metrics, artifacts = evaluate_model(model, X_val, y_val)
        mlflow.log_metrics(metrics)
        print(f"Validation Metrics: {metrics}")

        # 7b. Optional: evaluate rule-based baseline for comparison
        if project_config.baseline_rules_path and os.path.isfile(project_config.baseline_rules_path):
            print("Evaluating rule-based baseline...")
            baseline_config = load_config(project_config.baseline_rules_path, BaselineRulesConfig)
            baseline_model = RuleBasedModel(baseline_config)
            y_pred_b = baseline_model.predict(X_val)
            y_prob_b = baseline_model.predict_proba(X_val)[:, 1]
            baseline_metrics = calculate_metrics_with_prefix(
                y_val,
                pd.Series(y_pred_b),
                pd.Series(y_prob_b),
                prefix="baseline_val_",
            )
            mlflow.log_metrics(baseline_metrics)
            mlflow.set_tag("baseline_rules", baseline_config.name)
            print(f"Baseline (rule) Metrics: {baseline_metrics}")
        
        # Log Artifacts (Plots)
        print("Logging plots...")
        for name, fig in artifacts.items():
            mlflow.log_figure(fig, f"plots/{name}.png")
            # Close figure to free memory
            plt.close(fig)
        
        # 8. Tags (run name, balancing) for MLflow filtering
        _log_run_tags(project_config, model_config)

        # 9. Save & Log Model
        model_path = os.path.join(project_config.paths["models"], "catboost_model.cbm")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        print(f"Saving model artifact to {model_path}...")
        model.save_model(model_path)
        
        print("Logging model to MLflow...")
        model_artifacts = {"catboost_model": model_path}

        pip_reqs = ["catboost", "pandas", "scikit-learn"]
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=CatBoostWrapper(),
            artifacts=model_artifacts,
            pip_requirements=pip_reqs,
        )

        print("Pipeline complete.")
        return metrics
