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

        if (num_features < max_features) and (
            metric < last_metric
        ): 
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



def least_imp_feature_v2(model, X_test, logger):
    '''
    alternative using array indexing.
    '''

    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    logger.debug(f"SHAP values calculated with shape {shap_values.shape}")
    
    # calculate importance and focus on non ARD features 
    feature_importance = shap_values.abs.mean(0).mean(1).values
    feature_importance = feature_importance[13:]
    
    # get a list of indices that would sort the array in ascending order
    ranked = np.argsort(feature_importance)
    explainer = None
    shap_values = None
    return ranked[0]


def backward_selection_v2(X_train,
                        X_test,
                        y_train,
                        y_test,
                        estimator_name,
                        metric_name,
                        model_params_dict,
                        logger,
                        max_features=None,
                        ):
    
    # establish baseline
    baseline_metric, model = trn.fit_estimator(X_train,
                                        X_test,
                                        y_train,
                                        y_test,
                                        estimator_name,
                                        metric_name,
                                        model_params_dict,
                                        logger,
                                    )
    logger.info(f"Baseline: {round(baseline_metric, 5)} with {X_train.shape[1]} features")
    last_metric = baseline_metric

    # use all 94 features if max_features is not specified, otherwise
    # use max_features + 13 ARD features
    total_features = [94 if max_features is None else max_features + 13][0]
    logger.info(f"Performing backward selection for {total_features} features")
    logger.info(f"{94 - total_features} features will be dropped")

    baseline_features = np.arange(total_features)
    least_imp_list = []

    # counts down from total features to 0
    # update baseline features using index of dropped features
    for i in range(total_features - 1, -1, -1):
        dropped_feature = least_imp_feature(model, X_test, logger)
        
        # index the original location and append to a list of
        # least important feats
        original_feat_idx = baseline_features[dropped_feature]
        least_imp_list.append(original_feat_idx)
        logger.info(f"Removed feature {dropped_feature}, {X_train.shape[1]} features remaining")

        # Rerun modeling 
        metric, model = trn.fit_estimator(
            X_train,
            X_test,
            y_train,
            y_test,
            estimator_name,
            metric_name,
            model_params_dict,
            logger
        )
        logger.info(f"{round(metric, 5)} with {X_train.shape[1]} features")

        # Note: this will proceed w/o warning if max feats exceeded but not metric
        if (i < max_features) and (metric < last_metric): 
            # now remove all elements from baseline features 
            baseline_features = np.delete(baseline_features, least_imp_list)
            return baseline_features
        
        else:
            # update X_train and X_test and keep fitting
            last_metric = metric
            logger.info(f"Removing indices: {least_imp_list}")
            X_train = np.delete(X_train, least_imp_list, 1)
            X_test = np.delete(X_test, least_imp_list, 1)
            logger.info(f"X_train updated to {X_train.shape}")
            logger.info(f"X_test updated to {X_test.shape}")

    baseline_features = np.delete(baseline_features, least_imp_list)
    return baseline_features

