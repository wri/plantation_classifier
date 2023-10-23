import statsmodels.api as sm
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import shap
import models.train as trn
from utils.logs import get_logger


def get_dropped_feature(model, X_test):
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    feature_importance = shap_values.abs.mean(0).values
    importance_df = pd.DataFrame(
        {"features": X_test.columns, "importance": feature_importance}
    )
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
    fit_params_dict,
    logger,
    use_class_weights,
    max_features=None,
):
    """
    This function uses the SHAP importance from a model
    to incrementally remove features from the training set until the metric no longer improves.
    This function returns the dataframe with the features that give the best metric.
    Return at most max_features.
    """
    # get baseline metric
    total_features = X_train.shape[1]
    select_X_train = pd.DataFrame(X_train.copy())
    select_X_test = pd.DataFrame(X_test.copy())
    metric, model, X_test = trn.train(
        X_train,
        X_test,
        y_train,
        y_test,
        estimator_name,
        metric_name,
        model_params_dict,
        fit_params_dict,
        use_class_weights,
    )
    logger.info(f"{metric} with {select_X_train.shape[1]} features")
    last_metric = metric

    # Drop least important feature and recalculate model peformance
    if max_features is None:
        max_features = total_features - 1

    for num_features in range(total_features - 1, 1, -1):
        # Trim features
        dropped_feature = get_dropped_feature(model, select_X_test)
        logger.info(f"Removing feature {dropped_feature}")
        tmp_X_train = select_X_train.drop(columns=[dropped_feature])
        tmp_X_test = select_X_test.drop(columns=[dropped_feature])

        # Rerun modeling
        metric, model, X_test = trn.train(
            tmp_X_train,
            tmp_X_test,
            y_train,
            y_test,
            estimator_name,
            metric_name,
            model_params_dict,
            fit_params_dict,
        )
        logger.info(f"{metric} with {tmp_X_train.shape[1]} features")
        if (num_features < max_features) and (metric < last_metric):
            # metric decreased, return last dataframe
            return select_X_train, select_X_test
        else:
            # metric improved, continue dropping features
            last_metric = metric
            select_X_train = tmp_X_train
            select_X_test = tmp_X_test
    return select_X_train, select_X_test
