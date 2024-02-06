#!/usr/bin/env python

import pickle
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss
)
import json
import shap
import model.train as trn
from utils.logs import get_logger
import joblib

def feature_importance(model,
                       logger,
                       max_features=None,
                        ):

    '''
    Calculates the feature importance score for a given model
    using CatBoost's default feature importance score - FeatureImportance
    Returns df containing importance score for each feature and
    top n (feat_count) most important features
    '''
    logger.info("Calculating feature importance")
    #model = joblib.load(f'{model_path}')

    # calculate the feature importance
    df = model.get_feature_importance(prettified=True)
    df = df.astype({'Feature Id': int})

    # filter to features > index 13 (remove importance for s1, s2, dem)
    # get the indices and print in ascending order
    feats_df = df[df['Feature Id'] >= 13]
    top_feats = feats_df.sort_values(by='Importances', ascending=False)[:max_features]
    top_feats_indices = [i - 13 for i in sorted(list(top_feats['Feature Id']))]
#    print(top_feats_indices)
    return top_feats_indices


def least_imp_feature(model, X_test, logger):
    '''
    Get the least important feature based on SHAP values.

    This function uses SHAP (SHapley Additive exPlanations)
    values to explain the output of the model and identify the
    least important feature. The SHAP values are calculated for each feature
    in the test set, and the least important feature is determined
    based on the mean absolute SHAP values.

    Returns:
    - str: The least important feature based on SHAP values.
    '''
    # create the explainer interface for SHAP with 
    # my model as input
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    logger.debug(f"SHAP values calculated with shape {shap_values.shape}")
    # shap value shape will be (28420, num_feats, 4)

     # Calculate the mean absolute SHAP values across instances and features
    feature_importance = shap_values.abs.mean(0).mean(1).values
    importance_df = pd.DataFrame({"features": X_test.columns,
                                  "importance": feature_importance}
                                  )
    print(f"columns: {importance_df.features}")
    importance_df = importance_df[importance_df.features >= 13]
    importance_df.sort_values(by="importance", ascending=False, inplace=True)
    
    return importance_df["features"].iloc[-1]

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
    This function uses the SHAP importance from a model
    to incrementally remove features from the training set until the metric no longer improves.
    This function returns the dataframe with the features that give the best metric.
    Return at most max_features.
    X_test shape: (28420, 94)
    """

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
    total_features = [94 if max_features is None else max_features + 13]
    logger.info(f"Performing backward selection for {total_features} features")
    logger.info(f"{94 - total_features} features will be dropped")

    # Drop least important feature and recalculate model peformance
    select_X_train = pd.DataFrame(X_train) # removed the copy
    select_X_test = pd.DataFrame(X_test)
    for num_features in range(total_features - 1, 1, -1):
        dropped_feature = least_imp_feature(model, select_X_test, logger)
        # could tmp_X_train be a sliced array instead of df?
        # can we be sure that the original indices are maintined?
        tmp_X_train = select_X_train.drop(columns=[dropped_feature]) 
        tmp_X_test = select_X_test.drop(columns=[dropped_feature])
        logger.info(f"Removed feature {dropped_feature}, {tmp_X_train.shape[1]} features remaining")
        logger.info(f"{tmp_X_train.shape}, {select_X_train.shape}")

        # Rerun modeling 
        metric, model = trn.fit_estimator(
            tmp_X_train,
            tmp_X_test,
            y_train,
            y_test,
            estimator_name,
            metric_name,
            model_params_dict,
            logger
        )
        logger.info(f"{round(metric, 5)} with {tmp_X_train.shape[1]} features")
        
        if (num_features < max_features) and (metric < last_metric): # consider that this could exceed max featuers without warning
            # metric decreased, return last dataframe
            top_feats = [int(i) for i in list(select_X_train.columns)]
            return top_feats
        
        else:
            # metric improved, continue dropping features
            # clean up memory here?
            last_metric = metric
            select_X_train = tmp_X_train
            select_X_test = tmp_X_test
    
    top_feats = [int(i) for i in list(select_X_train.columns)]
    return top_feats


def least_imp_feature_v2(model, X_test, logger):

    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    logger.debug(f"SHAP values calculated with shape {shap_values.shape}")
    
    # calculate importance and focus on non ARD features 
    feature_importance = shap_values.abs.mean(0).mean(1).values
    feature_importance = feature_importance[13:]
    
    # get a list of indices that would sort the array in ascending order
    ranked = np.argsort(feature_importance)
    explainer = None
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
    total_features = [94 if max_features is None else max_features + 13]
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
            X_train = np.delete(X_train, dropped_feature, 1)
            X_test = np.delete(X_test, dropped_feature, 1)

    baseline_features = np.delete(baseline_features, least_imp_list)
    return baseline_features

