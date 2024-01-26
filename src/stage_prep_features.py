import argparse
from typing import Text
import json
import pickle
import yaml
import numpy as np
import pandas as pd
from utils.logs import get_logger
from features import create_xy


def featurize(param_path: Text) -> None:
    """ """
    with open(param_path) as file:
        params = yaml.safe_load(file)

    logger = get_logger("FEATURIZE", log_level=params["base"]["log_level"])
    ceo_batch = params["data_load"]["ceo_survey"]
    logger.info(f"CEO batch: {ceo_batch}")

    #    if params['data_condition']['select_features']:
    #        logger.info("Preparing X, y with select features.")
    #        top_feats = pd.read_csv(params['data_condition']['selected_features'])
    #        fs_indices = top_feats.feature_index
    #        fs_indices = [0, 1, 5, 6, 8, 9, 11, 14, 16, 19, 20, 21, 28, 31, 33, 34, 36, 37, 39, 44, 45, 47, 48, 49, 50, 51, 52, 57, 58, 60, 61, 62, 63, 64, 72, 76, 77, 78, 79, 80]

    #    else:
    fs_indices = []

    X, y = create_xy.build_training_sample(
        ceo_batch,
        classes=params["data_condition"]["classes"],
        params_path=param_path,
        logger=logger,
    )

    logger.info("X and y loaded")
    logger.info(f"X and y shape: {X.shape, y.shape}")

    # option to subset the training data
    # by randomly selecting n random plots by index
    if params["data_condition"]["subset_fraction"] < 1.0:
        np.random.seed(params["base"]["random_state"])
        subset_idx = np.random.choice(
            X.shape[0],
            size=(int(X.shape[0] * params["data_condition"]["subset_fraction"])),
            replace=False,
        )
        X = X[subset_idx]
        y = y[subset_idx]
        logger.debug(f"X data subsetted with final dimensions: {X.shape}")
        logger.debug(f"y data subsetted with final dimensions: {y.shape}")

    (
        X_train_sc,
        X_test_sc,
        X_train,
        X_test,
        y_train,
        y_test,
    ) = create_xy.prepare_model_inputs(
        X,
        y,
        param_path,
        logger,
        fs_indices,
    )
    # save scaled and unscaled data
    # NOTE: this is not currrently saving feature selected data
    # can update per structure of yaml file.
    logger.info(f"Train and test sets generated")
    with open(params["data_condition"]["X_train"], "wb") as fp:
        pickle.dump(X_train, fp)
    with open(params["data_condition"]["X_train_scaled"], "wb") as fp:
        pickle.dump(X_train_sc, fp)
    with open(params["data_condition"]["y_train"], "wb") as fp:
        pickle.dump(y_train, fp)
    with open(params["data_condition"]["X_test"], "wb") as fp:
        pickle.dump(X_test, fp)
    with open(params["data_condition"]["X_test_scaled"], "wb") as fp:
        pickle.dump(X_test_sc, fp)
    with open(params["data_condition"]["y_test"], "wb") as fp:
        pickle.dump(y_test, fp)
    logger.info("Training and testing data exported")


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--params", dest="params", required=True)
    args = args_parser.parse_args()
    featurize(param_path=args.params)
