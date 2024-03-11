#!/usr/bin/env python

import numpy as np
import pandas as pd
import pickle
from catboost import CatBoostClassifier
from datetime import datetime
import yaml
import model.train as train


def random_search_cat(X_train, y_train, estimator_name, metric_name, param_path):
    """
    Performs a randomized search of hyperparameters using Catboost's built in
    random search method and plots the results, then
    and saves results to a csv file

    iterations: specifies the number of boosting iterations (trees) used during training (equiv to n_estimators)
    learning_rate: controls step size at each iteration while moving toward a min of the loss function (decrease if overfitting)
    depth: Determines the max depth of the individual decision trees (equiv to max_depth (must be <= 16))
    l2_leaf_reg: Regularization term that prevents overfitting by penalizing large parameter values.
    loss_function: Specifies the loss function to be optimized during training.

    TODO: currently tailored to catboost models, not designed for other classifiers
    """
    with open(param_path) as file:
        params = yaml.safe_load(file)

    estimator_name = params["train"]["estimator_name"]
    param_dist = params["tune"]["estimators"][estimator_name]["param_grid"]
    iter_min = param_dist["iter_min"]
    iter_max = param_dist["iter_max"]
    iter_step = param_dist["iter_step"]
    depth_min = param_dist["depth_min"]
    depth_max = param_dist["depth_max"]
    depth_step = param_dist["depth_step"]
    leaf_min = param_dist["leaf_reg_min"]
    leaf_max = param_dist["leaf_reg_max"]
    leaf_step = param_dist["leaf_reg_step"]
    mdl_min = param_dist["min_data_leaf_min"]
    mdl_max = param_dist["min_data_leaf_max"]
    mdl_step = param_dist["min_data_leaf_step"]

    rs_params = {
        "iterations": [int(x) for x in np.linspace(iter_min, iter_max, iter_step)],
        "depth": [int(x) for x in np.linspace(depth_min, depth_max, depth_step)],
        "l2_leaf_reg": [int(x) for x in np.linspace(leaf_min, leaf_max, leaf_step)],
        "learning_rate": param_dist["learn_rate"],
        "min_data_in_leaf": [int(x) for x in np.linspace(mdl_min, mdl_max, mdl_step)],
    }

    # instantiate the classifier and perform Catboost built in method for random search
    cat = CatBoostClassifier(
        random_state=params["base"]["random_state"],
        loss_function=params["train"]["estimators"][estimator_name]["param_grid"]["loss_function"], 
        verbose=params["tune"]["verbose"],
    )

    randomized_search_result = cat.randomized_search(
        rs_params,
        X_train,
        y_train,
        params["tune"]["n_iter"],
        params["tune"]["cv"],
        params["tune"]["plot"],
        params["tune"]["verbose"],
    )
    return randomized_search_result
