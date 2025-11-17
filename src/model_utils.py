"""
model_utils.py

Shared machine learning utilities:
- Train-test split
- Scaling & encoding pipelines
- Classification evaluation metrics
"""

from __future__ import annotations

from typing import Tuple, Dict, List, Optional

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# ---------------------------
# Dataset splitting
# ---------------------------

def split_features_target(
    df: pd.DataFrame,
    target_col: str,
    drop_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split DataFrame into features X and target y.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    target_col : str
        Target column name.
    drop_cols : list of str, optional
        Columns to drop from features.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    """
    df = df.copy()
    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col} not found in dataframe.")

    y = df[target_col]
    X = df.drop(columns=[target_col])
    return X, y


def train_test_split_stratified(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified train-test split to preserve class distribution.

    Parameters
    ----------
    X : pd.DataFrame
    y : pd.Series
    test_size : float
    random_state : int

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )


# ---------------------------
# Preprocessing pipelines
# ---------------------------

def build_preprocessor(
    X: pd.DataFrame,
    numeric_strategy: str = "standard",
    handle_unknown: str = "ignore",
) -> ColumnTransformer:
    """
    Build a ColumnTransformer that scales numeric features
    and one-hot encodes categorical features.

    Parameters
    ----------
    X : pd.DataFrame
        Feature DataFrame.
    numeric_strategy : {'standard'}
        Currently only 'standard' scaling is supported.
    handle_unknown : str
        Passed to OneHotEncoder(handle_unknown).

    Returns
    -------
    ColumnTransformer
    """
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    if numeric_strategy == "standard":
        numeric_transformer = StandardScaler()
    else:
        raise ValueError("Only 'standard' numeric_strategy is implemented.")

    categorical_transformer = OneHotEncoder(handle_unknown=handle_unknown)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor


# ---------------------------
# Evaluation
# ---------------------------

def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute common classification metrics.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted labels.
    y_proba : array-like, optional
        Predicted probabilities for positive class (for ROC-AUC).

    Returns
    -------
    dict
        Mapping of metric name to value.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics["roc_auc"] = float("nan")

    return metrics


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Compute confusion matrix.

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like
    labels : list, optional

    Returns
    -------
    np.ndarray
    """
    return confusion_matrix(y_true, y_pred, labels=labels)
