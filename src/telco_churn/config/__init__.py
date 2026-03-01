import yaml
from pydantic import BaseModel, Field, ConfigDict
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

class ProjectConfig(BaseModel):
    paths: Dict[str, str]
    seed: int
    db_path: str
    tables: Dict[str, str]
    experiment_name: Optional[str] = None  # MLflow experiment (meta-level)
    balancing: Optional[str] = None  # Tag: "none" | "class_weights"
    baseline_rules_path: Optional[str] = None  # If set, evaluate rule baseline and log baseline_val_* metrics


class FeaturesConfig(BaseModel):
    categorical_features: List[str]
    numerical_features: List[str]
    target_col: str

class ModelConfig(BaseModel):
    model_type: str
    params: Dict[str, Any]
    run_name: Optional[str] = None


class OptunaConfig(BaseModel):
    """Optuna study config: metric to optimize, direction, n_trials, and param search space."""
    metric: str  # e.g. "val_pr_auc", "val_roc_auc"
    direction: str  # "maximize" or "minimize"
    n_trials: int
    params: Dict[str, Dict[str, Any]]  # param name -> { type: int|float|categorical, ... }


class BaselineRuleCondition(BaseModel):
    """Single condition: feature name + numeric range (min/max) or categorical set (in)."""
    feature: str
    min: Optional[Union[int, float]] = None
    max: Optional[Union[int, float]] = None
    op: Optional[str] = None   # "lt", "lte", "gt", "gte", "eq"
    value: Optional[Union[int, float, str]] = None
    in_values: Optional[List[Union[str, int, float]]] = Field(None, alias="in")

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class BaselineRuleGroup(BaseModel):
    """One rule = AND of conditions. Churn if ANY rule_group matches."""
    description: Optional[str] = None
    conditions: List[BaselineRuleCondition]


class BaselineRulesConfig(BaseModel):
    """Config for rule-based baseline: name + list of rule groups (OR of ANDs)."""
    name: str = "rule_baseline"
    rule_groups: List[BaselineRuleGroup]


def load_config(config_path: str, config_model: BaseModel):
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)
    return config_model(**config_data)
