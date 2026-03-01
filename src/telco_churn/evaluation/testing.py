"""
Evaluate a logged model on the test set and log test_* metrics to the MLflow run.
Use for candidate best runs to compare approaches without validation overfitting.
"""

from pathlib import Path
from typing import Optional, Tuple

import mlflow
import pandas as pd
from catboost import CatBoostClassifier
from mlflow.tracking import MlflowClient

from telco_churn.config import FeaturesConfig, ProjectConfig
from telco_churn.data.loading import load_data_from_db
from telco_churn.evaluation.metrics import calculate_metrics_with_prefix
from telco_churn.features.preprocessing import preprocess_features


def _find_artifact(path: Path, suffix: str) -> Optional[Path]:
    """Return first file under path with given suffix, or None."""
    for f in path.rglob(f"*{suffix}"):
        if f.is_file():
            return f
    return None


def _resolve_run_id(
    client: MlflowClient, experiment_name: str, run_id: Optional[str]
) -> str:
    """Return run_id; if None, resolve to latest run in the experiment."""
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment not found: {experiment_name}")
    if run_id is not None:
        return run_id
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise ValueError(f"No runs in experiment: {experiment_name}")
    return runs.iloc[0]["run_id"]


def _load_model(
    client: MlflowClient, run_id: str
) -> CatBoostClassifier:
    """Download run artifacts and load CatBoost model."""
    download_dir = client.download_artifacts(run_id, "model")
    base = Path(download_dir)

    cbm_path = _find_artifact(base, ".cbm")
    if cbm_path is None:
        raise FileNotFoundError(f"No .cbm model found under {download_dir}")

    model = CatBoostClassifier()
    model.load_model(str(cbm_path))

    return model


def _load_and_prepare_test_data(
    project_config: ProjectConfig,
    features_config: FeaturesConfig,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load test set and preprocess; return (X_test, y_test)."""
    test_df = load_data_from_db(
        project_config.db_path, project_config.tables["test"]
    )
    X_test, y_test = preprocess_features(
        test_df,
        target_col=features_config.target_col,
        categorical_features=features_config.categorical_features,
        numerical_features=features_config.numerical_features,
    )
    return X_test, y_test


def evaluate_run_on_test(
    project_config: ProjectConfig,
    features_config: FeaturesConfig,
    experiment_name: str,
    run_id: Optional[str] = None,
    tracking_uri: str = "sqlite:///mlflow.db",
) -> None:
    """
    Load the given run's model, evaluate on test set,
    and log test_roc_auc, test_pr_auc, etc. to that run.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    resolved_run_id = _resolve_run_id(client, experiment_name, run_id)
    print(f"Evaluating run: {resolved_run_id}")

    model = _load_model(client, resolved_run_id)
    X_test, y_test = _load_and_prepare_test_data(
        project_config, features_config
    )

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    test_metrics = calculate_metrics_with_prefix(
        y_test, pd.Series(y_pred), pd.Series(y_prob), prefix="test_"
    )
    for key, value in test_metrics.items():
        client.log_metric(resolved_run_id, key, value)
    print(f"Logged test metrics: {test_metrics}")
    print("Done.")
