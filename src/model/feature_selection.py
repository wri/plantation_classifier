#!/usr/bin/env python

import pickle
import pandas as pd
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

def OLD_feature_importance(model, feat_count):
    
    '''
    Calculates the feature importance score for a given model
    using CatBoost's default feature importance score - FeatureImportance
    Returns df containing importance score for each feature and
    top n (feat_count) most important features
    '''
    
    filename = f'../models/{model}.pkl'
    with open(filename, 'rb') as file:
        model = pickle.load(file)

    # calculate the feature importance 
    df = model.get_feature_importance(prettified=True)
    df = df.astype({'Feature Id': int})

    # filter to features > index 13 (remove importance for s1, s2, dem)
    # get the indices and print in ascending order
    feats_df = df[df['Feature Id'] >= 13]
    top_feats = feats_df.sort_values(by='Importances', ascending=False)[:feat_count]
    top_feats_indices = [i - 13 for i in sorted(list(top_feats['Feature Id']))]

    return df, top_feats_indices


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
    # create shap explainer
    explainer = shap.Explainer(model)
    # logger.debug("SHAP explainer calculated")
    shap_values = explainer(X_test)
    # logger.debug("SHAP values calculated")
    # shap value shape will be (28420, num_feats, 4)
    # logger.debug(f'Shap values shape: {shap_values.shape}') 

     # Calculate the mean absolute SHAP values across instances and features
    feature_importance = shap_values.abs.mean(0).mean(1).values
    logger.debug(f'columns:{X_test.columns}')
    importance_df = pd.DataFrame({"features": X_test.columns, 
                                  "importance": feature_importance}
                                  )
    importance_df.sort_values(by="importance", ascending=False, inplace=True)

    return importance_df["features"].iloc[-1]


def backward_selection(X_train,
                        X_test,
                        y_train,
                        y_test,
                        estimator_name,
                        metric_name,
                        model_params_dict,
                        fit_params_dict,
                        logger,
                        max_features=None):
    """
    This function uses the SHAP importance from a model
    to incrementally remove features from the training set until the metric 
    no longer improves. This function returns the dataframe with the features 
    that give the best metric. Return at most max_features.

    Requires scaled, full set of 94 features in X_train, X_test
    Metric being used here is balanced accuracy
    """
    # get baseline metric
    total_features = X_train.shape[1]
    select_X_train = pd.DataFrame(X_train.copy())
    select_X_test = pd.DataFrame(X_test.copy())
    metric, model, X_test = trn.fit_estimator(
        X_train,
        X_test,
        y_train,
        y_test,
        estimator_name,
        metric_name,
        model_params_dict,
        fit_params_dict,
    )
    logger.debug(f"X_test shape: {X_test.shape}")
    logger.info(f"BASELINE: {round(metric, 4)} with {select_X_train.shape[1]} features")
    last_metric = metric

    # Drop least important feature and recalculate model peformance
    if max_features is None:
        max_features = total_features - 1

    for num_features in range(total_features - 1, 1, -1):
        # Trim features
        logger.debug(f"select X test shape: {select_X_test.shape}")
        dropped_feature = least_imp_feature(model, select_X_test, logger)
        logger.info(f"Removing feature {dropped_feature}")
        tmp_X_train = select_X_train.drop(columns=[dropped_feature])
        tmp_X_test = select_X_test.drop(columns=[dropped_feature])

        # Rerun modeling - depending on what format select x train is, this could be a separate step
        metric, model, X_test = trn.fit_estimator(
            tmp_X_train,
            tmp_X_test,
            y_train,
            y_test,
            estimator_name,
            metric_name,
            model_params_dict,
            fit_params_dict,
        )
        logger.info(f"{round(metric, 4)} with {tmp_X_train.shape[1]} features")
        # not sure this is working?
        if (num_features < max_features) and (metric < last_metric):
            # metric decreased, return last dataframe
            return select_X_train, select_X_test, model
        else:
            # metric improved, continue dropping features
            last_metric = metric
            select_X_train = tmp_X_train
            select_X_test = tmp_X_test

    # confirm which model is being returned here
    return select_X_train, select_X_test, model