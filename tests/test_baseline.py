"""Tests for rule-based baseline model."""

import pandas as pd
import pytest

from telco_churn.config import load_config, BaselineRulesConfig
from telco_churn.models.baseline import RuleBasedModel, _condition_matches, _rule_group_matches
from telco_churn.config import BaselineRuleCondition, BaselineRuleGroup


def test_baseline_config_loads():
    cfg = load_config("config/baseline_rules.yaml", BaselineRulesConfig)
    assert cfg.name == "rule_baseline"
    assert len(cfg.rule_groups) >= 1
    assert all(len(g.conditions) >= 1 for g in cfg.rule_groups)


def test_rule_based_model_predict():
    cfg = load_config("config/baseline_rules.yaml", BaselineRulesConfig)
    model = RuleBasedModel(cfg)
    X = pd.DataFrame({
        "tenure": [5, 24, 3],
        "Contract": ["Month-to-month", "Two year", "Month-to-month"],
        "MonthlyCharges": [50.0, 40.0, 80.0],
    })
    pred = model.predict(X)
    assert pred.shape == (3,)
    assert pred[0] == 1  # tenure<=12 and Month-to-month
    assert pred[2] == 1  # MonthlyCharges >= 70 and/or tenure<=6
    proba = model.predict_proba(X)
    assert proba.shape == (3, 2)
    assert (proba[:, 1] == pred).all()


def test_condition_numeric_min_max():
    c = BaselineRuleCondition(feature="tenure", max=12)
    row = pd.Series({"tenure": 10})
    assert _condition_matches(c, row) is True
    assert _condition_matches(c, pd.Series({"tenure": 13})) is False


def test_condition_categorical_in():
    c = BaselineRuleCondition(feature="Contract", in_values=["Month-to-month"])
    assert _condition_matches(c, pd.Series({"Contract": "Month-to-month"})) is True
    assert _condition_matches(c, pd.Series({"Contract": "Two year"})) is False
