"""
fraud_utils.py

Fraud detection utilities:
- SMOTE rebalancing
- Model training (Logistic Regression, Random Forest)
- Feature importance extraction
"""

from __future__ import annotations

from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# ---------------------------
# Resampling
# ---------------------------

def apply_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE oversampling to the training set.

    Parameters
    ----------
    X_train : np.ndarray
    y_train : np.ndarray
    random_state : int

    Returns
    -------
    X_res, y_res : np.ndarray
    """
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res


# ---------------------------
# Model training
# ---------------------------

def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = 1.0,
    max_iter: int = 1000,
    class_weight: str | None = None,
) -> LogisticRegression:
    """
    Train a Logistic Regression classifier.

    Parameters
    ----------
    X_train : np.ndarray
    y_train : np.ndarray
    C : float
    max_iter : int
    class_weight : {None, 'balanced'}

    Returns
    -------
    LogisticRegression
    """
    clf = LogisticRegression(
        C=C,
        max_iter=max_iter,
        class_weight=class_weight,
        solver="lbfgs",
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 200,
    max_depth: int | None = None,
    class_weight: str | None = "balanced",
    random_state: int = 42,
) -> RandomForestClassifier:
    """
    Train a Random Forest classifier.

    Parameters
    ----------
    X_train : np.ndarray
    y_train : np.ndarray
    n_estimators : int
    max_depth : int or None
    class_weight : {None, 'balanced'}
    random_state : int

    Returns
    -------
    RandomForestClassifier
    """
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf


# ---------------------------
# Evaluation wrapper
# ---------------------------

def evaluate_fraud_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluate a fraud detection model on test data.

    Parameters
    ----------
    model : classifier with predict / predict_proba
    X_test : np.ndarray
    y_test : np.ndarray

    Returns
    -------
    dict
        Evaluation metrics.
    """
    y_pred = model.predict(X_test)

    # Use predict_proba if available; otherwise use decision_function if present
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        # Scale decision function to [0,1] for ROC-AUC
        y_proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    else:
        y_proba = None

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    if y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
        except ValueError:
            metrics["roc_auc"] = float("nan")

    return metrics


# ---------------------------
# Feature importance
# ---------------------------

def get_feature_importances(
    model,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Extract feature importances from tree-based models or
    coefficients from linear models.

    Parameters
    ----------
    model : fitted model
        Must have either `feature_importances_` or `coef_`.
    feature_names : list of str

    Returns
    -------
    pd.DataFrame
        Columns: ['feature', 'importance']
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        if coef.ndim > 1:
            coef = coef[0]
        importances = np.abs(coef)
    else:
        raise ValueError("Model does not have feature_importances_ or coef_ attributes.")

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)

    return importance_df
