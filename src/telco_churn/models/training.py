from copy import copy

from catboost import CatBoostClassifier
import pandas as pd

from telco_churn.config import ModelConfig


def sanitize_catboost_params(params: dict) -> dict:
    """
    Return a copy of params safe for CatBoost. max_leaves is only valid when
    grow_policy is Lossguide; remove max_leaves otherwise to avoid CatBoostError.
    """
    out = copy(params)
    if "max_leaves" not in out:
        return out
    grow = (out.get("grow_policy") or "").strip().lower()
    if grow and grow != "lossguide":
        out.pop("max_leaves")
    return out


class ModelTrainer:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None

    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        Trains the CatBoost model.
        """
        cat_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        params = sanitize_catboost_params(self.config.params)
        self.model = CatBoostClassifier(**params)
        
        # Train model
        self.model.fit(X, y, cat_features=cat_features, verbose=False)
        
        return self.model

    def get_model(self):
        """
        Returns the trained model.
        """
        return self.model
