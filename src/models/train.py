import hickle as hkl
import pickle
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import h5py
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.utils.class_weight import compute_class_weight
import sys
from datetime import datetime
import rasterio as rs
import features.validate_io as validate
from sklearn.metrics import f1_score, make_scorer
from typing import Dict, Text


class UnsupportedClassifier(Exception):

    def __init__(self, estimator_name):

        self.msg = f'Unsupported estimator {estimator_name}'
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
        'cat': CatBoostClassifier,
        'logreg': LogisticRegression
    }


def train(X_train, y_train,estimator_name, param_grid, cv, random_state):
    """Train model.
    Args:
        df {pandas.DataFrame}: dataset
        target_column {Text}: target column name
        estimator_name {Text}: estimator name
        param_grid {Dict}: grid parameters
        cv {int}: cross-validation value
    Returns:
        trained model
    """
    estimators = get_supported_estimator()

    if estimator_name not in estimators.keys():
        raise UnsupportedClassifier(estimator_name)

    estimator = estimators[estimator_name]()
    f1_scorer = make_scorer(f1_score, average='weighted')
    clf = GridSearchCV(estimator=estimator,
                       param_grid=param_grid,
                       cv=cv,
                       verbose=1,
                       scoring=f1_scorer)
    # Get X and Y
    clf.fit(X_train, y_train, random_state)

    return clf
