import argparse
from typing import Text
import yaml
import joblib
import pickle
import json
from utils.logs import get_logger
import train
import models.feature_selection as fsl



def train_model(params_path: Text) -> None:

    # load training parameters
    with open(param_path) as file:
        params = yaml.safe_load(file)
    logger = get_logger("TRAIN", log_level=params["base"]["log_level"])
    with open(params["data_condition"]["X_train"], "rb") as fp:
        X_train = pickle.load(fp)
    logger.debug(f"X_train shape: {X_train.shape}")
    with open(params["data_condition"]["y_train"], "rb") as fp:
        y_train = pickle.load(fp)
    logger.debug(f"y_train shape: {y_train.shape}")
    with open(params["data_condition"]["X_test"], "rb") as fp:
        X_test = pickle.load(fp)
    logger.debug(f"X_test shape: {X_test.shape}")
    with open(params["data_condition"]["y_test"], "rb") as fp:
        y_test = pickle.load(fp)
    logger.debug(f"y_test shape: {y_test.shape}")
    logger.info("Training and testing data loaded.")
    max_features = params["train"]["max_features"]
    logger.info(f"Max features: {max_features}")

    # if using feature selection, import fs script from models
    if params["train"]["select_features"]:
        logger.info("Starting feature selection")
        estimator_name = params["train"]["estimator_name"]
        select_X_train, select_X_test = fsl.backward_selection(
            X_train,
            X_test,
            y_train,
            y_test,
            params["train"]["estimator_name"],
            params["train"]["tuning_metric"],
            params["train"]["estimators"][estimator_name]["param_grid"],
            params["train"]["fit_params"],
            logger,
            max_features)
        logger.info(f"Feature selection completed with {select_X_train.shape[1]} features")
        with open(params["train"]["select_X_train"], "wb") as fp:
            pickle.dump(select_X_train, fp)
        with open(params["train"]["select_X_test"], "wb") as fp:
            pickle.dump(select_X_test, fp)
        X_train = select_X_train
        X_test = select_X_test
    else:
        with open(params["train"]["select_X_train"], "wb") as fp:
            pickle.dump(X_train, fp)
        with open(params["train"]["select_X_test"], "wb") as fp:
            pickle.dump(X_test, fp)
        logger.info("Using all features")
    with open(params["train"]["selected_feature_indicies"], "w") as fp:
        json.dump(
            obj={
                "feature_column_indicies": list(X_test.columns),
                "n_features": len(list(X_test.columns)),
            },
            fp=fp,
        )
    logger.info(
        f'Writing feature indicies to {params["train"]["selected_feature_indicies"]}'
    )
    if params["train"]["tune_hyperparams"]:
        logger.info("Starting hyperparameter tuning")
    #       TODO: implement hyperparameter tuning

    else:
        logger.info("Using default hyperparameters")

    logger.info("Training model...")
    metric, model, X_test = train.fit_estimator(
        X_train,
        X_test,
        y_train,
        y_test,
        estimator_name,
        params["train"]["tuning_metric"],
        params["train"]["estimators"][estimator_name]["param_grid"],
        params["train"]["fit_params"],
    )
    logger.info("Saving model")
    model_path = params["train"]["model_path"]
    joblib.dump(model, model_path)


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--params", dest="params", required=True)
    args = args_parser.parse_args()
    train_model(params_path=args.params)