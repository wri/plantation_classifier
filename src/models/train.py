import hickle as hkl
import pickle
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, make_scorer, roc_auc_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import h5py
from catboost import CatBoostClassifier
from sklearn.utils.class_weight import compute_class_weight
import sys
from datetime import datetime
import rasterio as rs
import features.validate_io as validate
from typing import Dict, Text


class UnsupportedClassifier(Exception):
    def __init__(self, estimator_name):
        self.msg = f'Unsupported estimator {estimator_name}'
        super().__init__(self.msg)
class UnsupportedMetric(Exception):
    def __init__(self, metric_name):
        self.msg = f'Unsupported metric {metric_name}'
        super().__init__(self.msg)


def get_supported_estimator():
    """
    Returns:
        Dict: supported classifiers
    """
    return {
        'rfc': RandomForestClassifier,
        'svm': SVC,
        'lgbm': LGBMClassifier,
        'xgb': XGBClassifier,
        'catb': CatBoostClassifier,
        'logreg': LogisticRegression
    }

def get_supported_metrics():
    """
    Returns:
        Dict: supported evaluation metrics
    """
    return {
        'accuracy': accuracy_score,
        'balanced_accuracy': balanced_accuracy_score,
        'precision': precision_score,
        'recall': recall_score,
        'f1': f1_score,
        'roc_auc': roc_auc_score
    }

def train(X_train, 
          X_test, 
          y_train, 
          y_test, 
          estimator_name, 
          metric_name,
          model_params_dict,
          fit_params_dict):
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
    return metric, model, X_test
