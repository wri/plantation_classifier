import argparse
from typing import Text
import pickle
import yaml
import numpy as np
import pandas as pd
from utils.logs import get_logger
from features import create_xy, PlantationsData


def featurize(param_path: Text) -> None:
    """ """
    with open(param_path) as file:
        params = yaml.safe_load(file)

    logger = get_logger("FEATURIZE", log_level=params["base"]["log_level"])
    ceo_batch = params["data_load"]["ceo_survey"]
    logger.info(f"CEO batch: {ceo_batch}")

    X, y = create_xy.build_training_sample(
        ceo_batch,
        classes=params["data_condition"]["classes"],
        params_path=param_path,
        logger=logger,
    )

    logger.debug("X and y loaded")
    logger.debug(f"X and y shape: {X.shape, y.shape}")

    model_data = PlantationsData.PlantationsData(X, y, params)
    logger.info("Initialized ModelData object")
    model_data.split_data()
    logger.info(f"Split data")
    logger.debug(f"X_train shape: {model_data.X_train.shape}")
    logger.debug(f"X_test shape: {model_data.X_test.shape}")
    logger.debug(f"y_train shape: {model_data.y_train.shape}")
    logger.debug(f"y_test shape: {model_data.y_test.shape}")
    model_data.reshape_data_arr()
    logger.info(f"Reshaped data")
    logger.debug(f"X_train_reshaped shape: {model_data.X_train_reshaped.shape}")
    logger.debug(f"X_test_reshape shape: {model_data.X_test_reshaped.shape}")
    logger.debug(f"y_train_reshape shape: {model_data.y_train_reshaped.shape}")
    logger.debug(f"y_test_reshape shape: {model_data.y_test_reshaped.shape}")
    logger.debug(f"Class names: {model_data.class_names}")
    logger.debug(f"Class weights: {model_data.class_weights}")
    model_data.scale_X_arrays()
    logger.info("X arrays scaled for feature selection")

    # save DataModel object
    logger.info(f"Model object generated")
    with open(params["data_condition"]["modelData_path"], "wb") as fp:
        pickle.dump(model_data, fp)
    logger.info("Model object exported")


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--params", dest="params", required=True)
    args = args_parser.parse_args()
    featurize(param_path=args.params)
