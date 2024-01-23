import argparse
from typing import Text
import json
import pickle
import yaml
import numpy as np
from pathlib import Path
import joblib

from utils.logs import get_logger
import model.tune as tune
import model.train as trn

def perform_tuning(param_path: Text) -> None:

    '''
    Imports training data and performs Catboost's random search of
    parameters identified in params.yaml
    Then fits and saves a model with the identified best params
    '''

    with open(param_path) as file:
        params = yaml.safe_load(file)
    logger = get_logger("TUNING", log_level=params["base"]["log_level"])
    
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
    with open(params['data_condition']['class_weights']) as fp:
        class_weights = json.load(fp)
    logger.info("Training and testing data loaded.")

    estimator_name = params["train"]["estimator_name"]
    pipe = params['base']['pipeline']
    model_path = f"{params['train']['model_dir']}{params['train']['model_name']}_{pipe}.joblib"

    logger.info(f"Starting random search with {params['tune']['n_iter']} samples")
    tuning_params, tuned_model = tune.random_search_cat(X_train,
                                                        y_train,
                                                        params["train"]["estimator_name"],
                                                        params["train"]["tuning_metric"],
                                                        param_path,
                                                        )
    
    # add class weights and save to file
    tuning_params['class_weights'] = class_weights
    with open(params["tune"]["best_params"], "w") as fp:
        json.dump(obj=tuning_params, fp=fp)
        
    logger.info(f"Fitting model with {tuning_params}")
    metric, tuned_model, X_test = trn.fit_estimator(X_train,
                                                    X_test,
                                                    y_train,
                                                    y_test,
                                                    estimator_name,
                                                    params["train"]["tuning_metric"],
                                                    tuning_params,
                                                    logger
                                                    )
    logger.info("Saving model")
    joblib.dump(tuned_model, f'{model_path}')

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--params", dest="params", required=True)
    args = args_parser.parse_args()
    perform_tuning(param_path=args.params)        