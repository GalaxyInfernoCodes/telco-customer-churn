"""Tests for Optuna tuning (config-driven suggest_params)."""

import pytest
import optuna
from telco_churn.models.tuning import suggest_params


def test_suggest_params_int_float_categorical():
    """suggest_params returns one value per param in config; types and bounds respected."""
    params_config = {
        "depth": {"type": "int", "low": 4, "high": 8},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
        "loss": {"type": "categorical", "choices": ["Logloss", "CrossEntropy"]},
    }
    seen = []

    def objective(trial):
        suggested = suggest_params(trial, params_config)
        seen.append(suggested)
        assert "depth" in suggested
        assert "learning_rate" in suggested
        assert "loss" in suggested
        assert 4 <= suggested["depth"] <= 8
        assert 0.01 <= suggested["learning_rate"] <= 0.3
        assert suggested["loss"] in ("Logloss", "CrossEntropy")
        return suggested["depth"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=3)
    assert len(seen) == 3
    # Different trials can suggest different values
    assert len(set(s["depth"] for s in seen)) >= 1
