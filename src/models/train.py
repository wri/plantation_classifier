from typing import Dict, Text
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


class UnsupportedClassifier(Exception):
    def __init__(self, estimator_name):
        self.msg = f"Unsupported estimator {estimator_name}"
        super().__init__(self.msg)


class UnsupportedMetric(Exception):
    def __init__(self, metric_name):
        self.msg = f"Unsupported metric {metric_name}"
        super().__init__(self.msg)


def get_supported_estimator():
    """
    Returns:
        Dict: supported classifiers
    """
    return {
        "rfc": RandomForestClassifier,
        "svm": SVC,
        "lgbm": LGBMClassifier,
        "xgb": XGBClassifier,
        "catb": CatBoostClassifier,
        "logreg": LogisticRegression,
    }


def get_supported_metrics():
    """
    Returns:
        Dict: supported evaluation metrics
    """
    return {
        "accuracy": accuracy_score,
        "balanced_accuracy": balanced_accuracy_score,
        "precision": precision_score,
        "recall": recall_score,
        "f1": f1_score,
        "roc_auc": roc_auc_score,
    }


def train(
    X_train,
    X_test,
    y_train,
    y_test,
    estimator_name,
    metric_name,
    model_params_dict,
    fit_params_dict,
    use_class_weights=False,
):
    estimators = get_supported_estimator()
    if estimator_name not in estimators.keys():
        raise UnsupportedClassifier(estimator_name)
    estimator = estimators[estimator_name]
    # Fit the model and calculate metric
    metrics = get_supported_metrics()
    if metric_name not in metrics.keys():
        raise UnsupportedMetric(metric_name)
    if use_class_weights:
        classes = np.unique(y_train)
        weights = compute_class_weight(
            class_weight="balanced", classes=classes, y=y_train
        )
        fit_params_dict[class_weights] = weights
    metric_fun = metrics[metric_name]
    model = estimator(**model_params_dict)
    model.fit(X_train, y_train, **fit_params_dict)
    metric = metric_fun(y_test, model.predict(X_test))
    return metric, model, X_test
