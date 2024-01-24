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
    '''
    '''
    with open(param_path) as file:
        params = yaml.safe_load(file)
 
    logger = get_logger("FEATURIZE", log_level=params["base"]["log_level"])
    ceo_batch = params["data_load"]['ceo_survey'] 
    logger.info(f"CEO batch: {ceo_batch}")

    if params['data_condition']['select_features']: 
        logger.info("Preparing X, y with select features.")
        top_feats = pd.read_csv(params['data_condition']['selected_features'])
        fs_indices = top_feats.feature_index
        
    else:
        fs_indices = []

    X, y = create_xy.build_training_sample(
        ceo_batch,
        classes=params['data_condition']['classes'], 
        params_path=param_path,
        logger=logger,
        feature_select=fs_indices, 
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
    
    # prepare scaled data
    X_train, X_test, y_train, y_test = create_xy.prepare_model_inputs(X, 
                                                                  y, 
                                                                  param_path,  
                                                                  logger)
    # save scaled data
    logger.info(f"Train and test sets generated")
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
    featurize(param_path=args.params)

