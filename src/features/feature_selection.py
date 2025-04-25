#!/usr/bin/env python

import pickle
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import shap
import model.train as trn
from utils.logs import get_logger

def least_imp_feature(model, X_test, logger):
    """
    This function uses feature importance values to explain the output 
    of the model and identify the least important feature. 
    The values are calculated for each feature in the test set, 
    and the least important feature is returned.

    Returns:
    - str: The least important feature based on feature importance
    """

    feature_importance = model.get_feature_importance()
    importance_df = pd.DataFrame(
        {"features": X_test.columns, "importance": feature_importance}
    )
    importance_df = importance_df[~importance_df["features"].str.contains("keep")]
    importance_df.sort_values(by="importance", ascending=False, inplace=True)
    logger.debug(
        f"Removing feature {importance_df.features.iloc[-1]} with importance {importance_df.importance.iloc[-1]}"
        )

    return importance_df["features"].iloc[-1]

def keep_names(index, colname, keep_list):
    if index in keep_list:
        return f"keep_{colname}"
    else:
        return colname
    
def backward_selection(
    X_train,
    X_test,
    y_train,
    y_test,
    estimator_name,
    metric_name,
    model_params_dict,
    logger,
    max_features=None,
):
    """
    This function uses Catboost's build in feature_importance method
    to incrementally remove features from the training set until the 
    provided accuracy metric no longer improves.
    This function returns a list of at most [max_features] features 
    that provide the best metric.
    """

    # establish baseline
    baseline_metric, model = trn.fit_estimator(
        X_train,
        X_test,
        y_train,
        y_test,
        estimator_name,
        metric_name,
        model_params_dict,
        logger,
    )
    logger.info(
        f"Baseline: {round(baseline_metric, 5)} with {X_train.shape[1]} features"
    )
    last_metric = baseline_metric

    # use all 94 features if max_features is not specified, otherwise
    # use max_features + 13 ARD features
    total_features = X_train.shape[1]
    if max_features is None:
        max_features = total_features - 1
    logger.info(f"Performing backward selection for {total_features} features")
    logger.info(f"{total_features - max_features} features will be dropped")

    # Drop least important feature and recalculate model peformance
    select_X_train = pd.DataFrame(X_train) 
    select_X_test = pd.DataFrame(X_test)

    # creates columns named feature_0
    select_X_train.columns = [
        f"feature_{index}" for index in range(len(select_X_train.columns))
    ]
    select_X_test.columns = [
        f"feature_{index}" for index in range(len(select_X_test.columns))
    ]
    ard_index_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # TODO: move to config
    select_X_train.columns = [
        keep_names(index, colname, ard_index_list)
        for index, colname in zip(
            range(len(select_X_train.columns)), select_X_train.columns
        )
    ]
    select_X_test.columns = [
        keep_names(index, colname, ard_index_list)
        for index, colname in zip(
            range(len(select_X_test.columns)), select_X_test.columns
        )
    ]

    for num_features in range(total_features - 1, 1, -1):
        logger.debug(f"Evaluating feature: {num_features}")
        dropped_feature = least_imp_feature(model, select_X_test, logger)
        tmp_X_train = select_X_train.drop(columns=[dropped_feature])
        tmp_X_test = select_X_test.drop(columns=[dropped_feature])
        logger.info(f"{tmp_X_train.shape[1]} features remaining")

        # Rerun modeling
        metric, model = trn.fit_estimator(
            tmp_X_train,
            tmp_X_test,
            y_train,
            y_test,
            estimator_name,
            metric_name,
            model_params_dict,
            logger,
        )
        logger.info(f"{round(metric, 5)} with {tmp_X_train.shape[1]} features")

        # removing metric < last_metric
        # because this continues fs past max feature count
        if (num_features < max_features):  
            # TODO: add a clause for min features in case it gets down to ARD feats
            top_feats = [
                int(float(i.split("_")[-1])) for i in list(select_X_train.columns)
            ]
            return top_feats

        else:
            last_metric = metric
            select_X_train = tmp_X_train
            select_X_test = tmp_X_test

    top_feats = [int(float(i.split("_")[-1])) for i in list(select_X_train.columns)]
    return top_feats

