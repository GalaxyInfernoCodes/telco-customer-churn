"""
Rule-based baseline model for churn prediction.
Predicts churn using config-driven rules (e.g. feature in range or in set).
Compatible with evaluate_model() for comparison to the trained ML model.
"""

from typing import Optional

import numpy as np
import pandas as pd

from telco_churn.config import BaselineRulesConfig, BaselineRuleCondition, BaselineRuleGroup


def _condition_matches(condition: BaselineRuleCondition, row: pd.Series) -> bool:
    """Return True if the row satisfies this condition."""
    val = row.get(condition.feature)
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return False

    if condition.in_values is not None:
        return val in condition.in_values

    # Numeric: min/max (inclusive) or op + value
    try:
        v = float(val) if not isinstance(val, (int, float)) else val
    except (TypeError, ValueError):
        return False

    if condition.min is not None and v < condition.min:
        return False
    if condition.max is not None and v > condition.max:
        return False
    if condition.op is not None and condition.value is not None:
        comp = condition.value if isinstance(condition.value, (int, float)) else float(condition.value)
        if condition.op == "lt" and not (v < comp):
            return False
        if condition.op == "lte" and not (v <= comp):
            return False
        if condition.op == "gt" and not (v > comp):
            return False
        if condition.op == "gte" and not (v >= comp):
            return False
        if condition.op == "eq" and v != comp:
            return False
    return True


def _rule_group_matches(group: BaselineRuleGroup, row: pd.Series) -> bool:
    """Return True if all conditions in this rule group match (AND)."""
    return all(_condition_matches(c, row) for c in group.conditions)


class RuleBasedModel:
    """
    Baseline that predicts churn (1) if ANY rule_group matches; each rule_group
    matches when ALL its conditions match. Uses the same feature names as in X.
    """

    def __init__(self, config: BaselineRulesConfig):
        self.config = config

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict 0/1 churn labels. Churn=1 if any rule group matches."""
        pred = np.zeros(len(X), dtype=np.int64)
        for i in range(len(X)):
            row = X.iloc[i]
            for group in self.config.rule_groups:
                if _rule_group_matches(group, row):
                    pred[i] = 1
                    break
        return pred

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return (n, 2) probabilities. Rule-based has no uncertainty: use 0.0/1.0
        so ROC/PR curves and metrics are still computable.
        """
        pred = self.predict(X)
        prob_churn = pred.astype(np.float64)
        return np.column_stack([1.0 - prob_churn, prob_churn])
