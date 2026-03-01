from typing import Dict, Tuple, Any, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    accuracy_score,
    PrecisionRecallDisplay,
    confusion_matrix
)

def calculate_metrics(y_val: pd.Series, y_pred: pd.Series, y_prob: pd.Series) -> Dict[str, float]:
    """Calculate performance metrics."""
    return {
        "val_roc_auc": roc_auc_score(y_val, y_prob),
        "val_pr_auc": average_precision_score(y_val, y_prob),
        "val_precision": precision_score(y_val, y_pred),
        "val_recall": recall_score(y_val, y_pred),
        "val_f1": f1_score(y_val, y_pred),
        "val_accuracy": accuracy_score(y_val, y_pred)
    }


def calculate_metrics_with_prefix(
    y_true: pd.Series, y_pred: pd.Series, y_prob: pd.Series, prefix: str = "test_"
) -> Dict[str, float]:
    """Calculate performance metrics with a key prefix (e.g. test_ for test set)."""
    return {
        f"{prefix}roc_auc": roc_auc_score(y_true, y_prob),
        f"{prefix}pr_auc": average_precision_score(y_true, y_prob),
        f"{prefix}precision": precision_score(y_true, y_pred, zero_division=0),
        f"{prefix}recall": recall_score(y_true, y_pred, zero_division=0),
        f"{prefix}f1": f1_score(y_true, y_pred, zero_division=0),
        f"{prefix}accuracy": accuracy_score(y_true, y_pred),
    }


def plot_pr_curve(y_val: pd.Series, y_prob: pd.Series) -> plt.Figure:
    """Generate Precision-Recall curve."""
    fig, ax = plt.subplots(figsize=(8, 6))
    PrecisionRecallDisplay.from_predictions(y_val, y_prob, ax=ax)
    ax.set_title("Precision-Recall Curve")
    return fig

def plot_confusion_matrix(y_val: pd.Series, y_pred: pd.Series, normalize: Optional[str] = None, title: str = "Confusion Matrix") -> plt.Figure:
    """Generate Confusion Matrix plot."""
    cm = confusion_matrix(y_val, y_pred, normalize=normalize)
    fmt = '.2%' if normalize else 'd'
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    return fig

def evaluate_model(model, X_val: pd.DataFrame, y_val: pd.Series) -> Tuple[Dict[str, float], Dict[str, plt.Figure]]:
    """
    Evaluate the model on the validation set and return metrics and plots.
    
    Args:
        model: Trained model (must have predict and predict_proba methods).
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation target.
        
    Returns:
        Tuple[Dict[str, float], Dict[str, plt.Figure]]: Dictionary of metrics and dictionary of figures.
    """
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    
    # Calculate Metrics
    metrics = calculate_metrics(y_val, y_pred, y_prob)
    
    # Generate Plots
    artifacts = {
        "precision_recall_curve": plot_pr_curve(y_val, y_prob),
        "confusion_matrix_counts": plot_confusion_matrix(y_val, y_pred, normalize=None, title="Confusion Matrix (Counts)"),
        "confusion_matrix_percentages": plot_confusion_matrix(y_val, y_pred, normalize='true', title="Confusion Matrix (Percentages)")
    }
    
    return metrics, artifacts
