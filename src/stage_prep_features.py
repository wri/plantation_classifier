import argparse
from typing import Text
import json
import pickle
import yaml
import numpy as np
from utils.logs import get_logger
from features import create_xy


def featureize(param_path: Text) -> None:
    '''
    
    '''
    with open(param_path) as file:
        params = yaml.safe_load(file)

    logger = get_logger("FEATURIZE", log_level=params["base"]["log_level"])
    ceo_batch_list = params["data_load"]['ceo_survey'] # need to figure out how to write list

    X, y = create_xy.build_training_sample(
        ceo_batch_list,
        feature_select=[],
        classes=4,
        params_path=param_path,
        logger=logger,
    ) 
    logger.info("X,y loaded")

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
    
    scale_data = params["data_condition"]["scale_features"]
    X_train, X_test, y_train, y_test = create_xy.reshape_and_scale(X, 
                                                                  y, 
                                                                  scale_data,
                                                                  param_path,  
                                                                  logger)
    
    logger.info(f"Train and test set generated")
    with open(params["data_condition"]["X_train"], "wb") as fp:
        pickle.dump(X_train, fp)
    with open(params["data_condition"]["y_train"], "wb") as fp:
        pickle.dump(y_train, fp)
    with open(params["data_condition"]["X_test"], "wb") as fp:
        pickle.dump(X_test, fp)
    with open(params["data_condition"]["y_test"], "wb") as fp:
        pickle.dump(y_test, fp)
    logger.info("Training and testing data exported")


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--params", dest="params", required=True)
    args = args_parser.parse_args()
    featureize(params_path=args.params)