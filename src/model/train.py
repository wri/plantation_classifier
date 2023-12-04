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
import pickle

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
        "cat": CatBoostClassifier,
        "lr": LogisticRegression,
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

def fit_estimator(
    X_train,
    X_test,
    y_train,
    y_test,
    estimator_name,
    metric_name,
    model_params_dict,
    fit_params_dict,
    v_train_data,
    use_class_weights=False):

    '''
    Train a machine learning model, evaluate its performance, and return the results. Raises UnsupportedClassifier
    or UnsupportedMetric if the specified estimator or classifier is not supported.
        ** means receive variable arguments as dictionary
        class weights will be determined in config file?

    Parameters:
    - X_train (array-like): Training features.
    - X_test (array-like): Testing features.
    - y_train (array-like): Training labels.
    - y_test (array-like): Testing labels.
    - estimator_name (str): Name of the machine learning estimator to be used.
    - metric_name (str): Name of the evaluation metric to be used.
    - model_params_dict (dict): Dictionary of hyperparameters for the estimator.
    - fit_params_dict (dict): Additional parameters for the model fitting process.
    - use_class_weights (bool, optional): Whether to use class weights during training (default is False).

    Returns:
    - metric (float): Evaluation metric value on the testing set.
    - model: Trained machine learning model.
    - X_test: Testing features (useful for further analysis).
    '''

    estimators = get_supported_estimator()

    if estimator_name not in estimators.keys():
        raise UnsupportedClassifier(estimator_name)
    
    estimator = estimators[estimator_name]

    # Fit the model and calculate metric
    metrics = get_supported_metrics()
    if metric_name not in metrics.keys():
        raise UnsupportedMetric(metric_name)

    metric_fun = metrics[metric_name]
    model = estimator(**model_params_dict)
    model.fit(X_train, y_train, **fit_params_dict)
    metric = metric_fun(y_test, model.predict(X_test))
    
    # for now save model as pkl file
    filename = f'../models/{estimator_name}_{v_train_data}.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

    return metric, model, X_test