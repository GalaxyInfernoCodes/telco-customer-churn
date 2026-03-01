import pytest
import os
from telco_churn.utils.verification import verify_data_setup, verify_model_inference

PROJECT_CONFIG_PATH = "config/project.yaml"

@pytest.mark.integration
def test_verify_data_setup():
    """Verify that the data setup is correct (DuckDB and tables)."""
    assert verify_data_setup(PROJECT_CONFIG_PATH) is True

@pytest.mark.integration
def test_verify_model_inference():
    """Verify that the latest model in MLflow can be loaded and predict."""
    # This test might fail if no model has been trained yet.
    # In a real CI environment, we would ensure a model is trained first.
    assert verify_model_inference(PROJECT_CONFIG_PATH) is True
