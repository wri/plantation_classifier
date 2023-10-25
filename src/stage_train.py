import argparse
from typing import Text
import yaml
import joblib
import pickle
import json
from utils.logs import get_logger
import models.feature_selection as fsl
import models.train as trn


def train_model(config_path: Text) -> None:
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)
    logger = get_logger("TRAIN", log_level=config["base"]["log_level"])
    with open(config["data_condition"]["train_data_x"], "rb") as fp:
        X_train = pickle.load(fp)
    logger.debug(f"X_train shape: {X_train.shape}")
    with open(config["data_condition"]["train_data_y"], "rb") as fp:
        y_train = pickle.load(fp)
    logger.debug(f"y_train shape: {y_train.shape}")
    with open(config["data_condition"]["test_data_x"], "rb") as fp:
        X_test = pickle.load(fp)
    logger.debug(f"X_test shape: {X_test.shape}")
    with open(config["data_condition"]["test_data_y"], "rb") as fp:
        y_test = pickle.load(fp)
    logger.debug(f"y_test shape: {y_test.shape}")
    logger.info("Training and testing data loaded")
    max_features = config["train"]["max_features"]
    logger.info(f"Max features: {max_features}")

    if config["train"]["select_features"]:
        logger.info("Starting feature selection")
        estimator_name = config["train"]["estimator_name"]
        select_X_train, select_X_test = fsl.backward_selection(
            X_train,
            X_test,
            y_train,
            y_test,
            config["train"]["estimator_name"],
            config["train"]["tuning_metric"],
            config["train"]["estimators"][estimator_name]["param_grid"],
            config["train"]["fit_params"],
            logger,
            max_features,
        )
        logger.info(
            f"Feature selection completed with {select_X_train.shape[1]} features"
        )
        with open(config["train"]["selected_train_data_x"], "wb") as fp:
            pickle.dump(select_X_train, fp)
        with open(config["train"]["selected_test_data_x"], "wb") as fp:
            pickle.dump(select_X_test, fp)
        X_train = select_X_train
        X_test = select_X_test
    else:
        with open(config["train"]["selected_train_data_X"], "wb") as fp:
            pickle.dump(X_train, fp)
        with open(config["train"]["selected_test_data_X"], "wb") as fp:
            pickle.dump(X_test, fp)
        logger.info("Using all features")
    with open(config["train"]["selected_feature_indicies"], "w") as fp:
        json.dump(
            obj={
                "feature_column_indicies": list(X_test.columns),
                "n_features": len(list(X_test.columns)),
            },
            fp=fp,
        )
    logger.info(
        f'Writing feature indicies to {config["train"]["selected_feature_indicies"]}'
    )
    if config["train"]["tune_hyperparams"]:
        logger.info("Starting hyperparameter tuning")
    #       TODO: implement hyperparameter tuning
    else:
        logger.info("Using default hyperparameters")

    logger.info("Training model")
    metric, model, X_test = trn.train(
        X_train,
        X_test,
        y_train,
        y_test,
        estimator_name,
        config["train"]["tuning_metric"],
        config["train"]["estimators"][estimator_name]["param_grid"],
        config["train"]["fit_params"],
    )
    logger.info("Saving model")
    model_path = config["train"]["model_path"]
    joblib.dump(model, model_path)


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()
    train_model(config_path=args.config)
