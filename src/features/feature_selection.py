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


def least_imp_feature(model, X_test, logger, min_index):
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
    importance_df = importance_df[importance_df.features >= min_index]
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
    """
    total_features = X_train.shape[1] # this should be 94
    min_index = 13
    select_X_train = pd.DataFrame(X_train.copy())
    select_X_test = pd.DataFrame(X_test.copy())
    baseline_metric, model = trn.fit_estimator(X_train,
                                        X_test,
                                        y_train,
                                        y_test,
                                        estimator_name,
                                        metric_name,
                                        model_params_dict,
                                        logger,
                                    )
    logger.debug(f"X_test shape: {X_test.shape}")
    logger.info(f"Baseline: {round(baseline_metric, 5)} with {select_X_train.shape[1]} features")
    last_metric = baseline_metric

    # if max features is none, evaluate all 94 features
    if max_features is None:
        max_features = total_features - 1
    else:
        max_features += min_index - 1
    logger.info(f"Max features updated to: {max_features}")

    # Drop least important feature and recalculate model peformance
    for num_features in range(total_features - 1, 1, -1):
        dropped_feature = least_imp_feature(model, select_X_test, logger, min_index)
        tmp_X_train = select_X_train.drop(columns=[dropped_feature])
        tmp_X_test = select_X_test.drop(columns=[dropped_feature])
        logger.info(f"Removed feature {dropped_feature}, {tmp_X_train.shape[1]} features remaining")

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
        
        if (num_features < max_features) and (metric < last_metric):
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