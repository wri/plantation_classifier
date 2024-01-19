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



def train_model(param_path: Text) -> None:

    '''
    ''' 
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
    with open(params['data_condition']['class_weights']) as fp:
        class_weights = json.load(fp)
    logger.info("Training and testing data loaded.")

    estimator_name = params["train"]["estimator_name"]
    model_dir = Path(params["train"]["model_dir"])
    model_path = model_dir / params['train']['model_name']
    model_params = params["train"]["estimators"][estimator_name]["param_grid"]
    model_params['class_weights'] = class_weights

    logger.info(f"Training model..")
    metric, model, X_test = train.fit_estimator(
                                        X_train,
                                        X_test,
                                        y_train,
                                        y_test, #at this stage they are ints
                                        estimator_name,
                                        params["train"]["tuning_metric"],
                                        model_params,
                                        logger
                                        )
    logger.info("Saving model")
    joblib.dump(model, f'{model_path}.joblib')
          
if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--params", dest="params", required=True)
    args = args_parser.parse_args()
    train_model(param_path=args.params)