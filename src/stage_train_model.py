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
    Trains an ML model using parameters specified in a YAML config file.

    This function loads training data, selected features, and model parameters from the provided
    `params.yaml` file. It applies feature selection if enabled and determines whether 
    to use pre-tuned hyperparameters or a provided parameter grid for model training.
    
    The function logs various details about the dataset and model configuration before fitting
    the selected estimator. Once training is complete, the trained model is saved to disk.

    Workflow:
        1. Load parameters from `params.yaml`.
        2. Initialize logger with the specified logging level.
        3. Load training data, selected features, and hyperparameters.
        4. Apply feature selection if specified.
        5. Use either the best pre-tuned parameters or the provided parameter grid.
        6. Add class weights to the model configuration.
        7. Train the model using the selected estimator and hyperparameters.
        8. Save the trained model to disk.

    Logs:
        - Importing training data and parameter settings.
        - Details about selected features and model hyperparameters.
        - Debugging information about dataset dimensions.
        - Confirmation of successful model training and saving.

    Note:
        - Feature selection is only applied if explicitly enabled in the configuration.
    """
    with open(param_path) as file:
        params = yaml.safe_load(file)
    logger = get_logger("TRAIN", log_level=params["base"]["log_level"])
    logger.info("Importing training data")
    with open(params["data_condition"]["modelData_path"], "rb") as fp:
        model_data = pickle.load(fp)
    with open(params["select"]["selected_features_path"], "r") as fp:
        selected_features = json.load(fp)
    logger.info("All data loaded") 

    estimator_name = params["train"]["estimator_name"]
    model_path = f"{params['train']['model_name']}"
    perform_fs = params["select"]["perform_fs"]
    use_best_params = params["train"]["use_best_params"]
    tuned_params = params["train"]["estimators"][estimator_name]["param_grid"]

    logger.info(f"Model will be trained with the following conditions:") 
    if perform_fs:  
        model_data.filter_features(selected_features)
        logger.info(f"Selected features: {selected_features}.")

    model_params = None

    if use_best_params:
        model_params = tuned_params

    if model_params is None:
        model_params = {}  # Ensure we pass a valid dictionary
        model_params["loss_function"] = tuned_params['loss_function']
        model_params["logging_level"] = tuned_params['logging_level']

    model_params["class_weights"] = getattr(model_data, "class_weights", None)
    logger.info(f"Hyperparameters {model_params}")
    logger.debug(f"X_train_reshaped: {model_data.X_train_reshaped.shape}")
    logger.debug(f"X_test_reshaped: {model_data.X_test_reshaped.shape}")
    logger.debug(f"y_train_reshaped: {model_data.y_train_reshaped.shape}")
    logger.debug(f"y_test_reshaped: {model_data.y_test_reshaped.shape}")
    logger.info(f"Training..")
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
    logger.info(f"Saving model with {len(model.feature_names_)} features")
    joblib.dump(model, f"{model_path}")


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--params", dest="params", required=True)
    args = args_parser.parse_args()
    train_model(param_path=args.params)
