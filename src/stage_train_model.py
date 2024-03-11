import argparse
from typing import Text
import yaml
import joblib
import pickle
import numpy as np
import json
from pathlib import Path
from utils.logs import get_logger
import model.train as train
from features import PlantationsData


def train_model(param_path: Text) -> None:
    """ 
    Per criteria in the params.yaml file, imports
    training data and parameters, filters to selected features
    """
    with open(param_path) as file:
        params = yaml.safe_load(file)
    logger = get_logger("TRAIN", log_level=params["base"]["log_level"])
    logger.info("Importing training data")
    with open(params["data_condition"]["modelData_path"], "rb") as fp:
        model_data = pickle.load(fp)
    with open(params["select"]["selected_features_path"], "r") as fp:
        selected_features = json.load(fp)
    with open(params["tune"]["best_params"], "r") as fp:
        best_params = json.load(fp)
    logger.info("All data loaded")

    estimator_name = params["train"]["estimator_name"]
    model_path = f"{params['train']['model_name']}"
    basic_params = params["train"]["estimators"][estimator_name]["param_grid"]
    # features are filtered regardless of whether fs used
    model_data.filter_features(selected_features)

    # use best params file or params in params.yaml
    if not params['train']['use_best_params']:
        logger.info("Using provided param grid in params.yaml")
        model_params = basic_params

    else:
        logger.info("Using random search param grid.")
        model_params = best_params['params']
        model_params["loss_function"] = basic_params['loss_function']
        model_params["logging_level"] = basic_params['logging_level']
    
    # make sure class weights are added for either option
    model_params["class_weights"] = model_data.class_weights
    logger.info(f"Model will be trained with the following conditions:")
    logger.info(f"Model params: {model_params}")
    logger.debug(f"X_train_reshaped: {model_data.X_train_reshaped.shape}")
    logger.debug(f"X_test_reshaped: {model_data.X_test_reshaped.shape}")
    logger.debug(f"y_train_reshaped: {model_data.y_train_reshaped.shape}")
    logger.debug(f"y_test_reshaped: {model_data.y_test_reshaped.shape}")

    metric, model = train.fit_estimator(
        model_data.X_train_reshaped,
        model_data.X_test_reshaped,
        model_data.y_train_reshaped,
        model_data.y_test_reshaped,
        estimator_name,
        params["train"]["tuning_metric"],
        model_params,
        logger,
    )
    logger.info("Saving model")
    joblib.dump(model, f"{model_path}")


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--params", dest="params", required=True)
    args = args_parser.parse_args()
    train_model(param_path=args.params)
