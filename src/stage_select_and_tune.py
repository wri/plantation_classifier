import yaml
import argparse
import json
import pickle
from typing import Text
from utils.logs import get_logger
import pandas as pd
from features import feature_selection as fsl
from model import train as train
from model import tune as tune
from json.decoder import JSONDecodeError

def perform_selection_and_tuning(param_path: Text) -> None:
    '''
    References optional feature selection and tuning stage
    from the params file. If true, perfoms feature selection to identify
    and save the top [max_features] features.
    Performs optional random seach to identify and save the best parameters.
    If both options are set to false, saves empty json file for best params
    and a full list of features as top feats.
    '''
    with open(param_path) as file:
        params = yaml.safe_load(file)
    logger = get_logger("FEATURE SELECTION & TUNING",
                        log_level=params["base"]["log_level"])
    with open(params["data_condition"]["modelData_path"], "rb") as fp:
        model_data = pickle.load(fp)
    with open(params["select"]["selected_features_path"], "r") as fp:
        top_feats = json.load(fp)
    with open(params["tune"]["best_params"], "r") as fp:
        best_params = json.load(fp)
        
    perform_fs = params["select"]["select_features"]
    perform_tuning = params["tune"]["tune_hyperparameters"]
    
    if not perform_fs and not perform_tuning:
        logger.info("Skipping feature selection and tuning.")
    else:
        if perform_fs: 
            # define parameters for feature selection
            model_params = params["train"]["estimators"]["cat"]["param_grid"]
            max_features = params["select"]["max_features"]
            logger.info(f"Max features for feature selection: {max_features}")
            top_feats = fsl.backward_selection(
                                    model_data.X_train_scaled,
                                    model_data.X_test_scaled,
                                    model_data.y_train_reshaped,
                                    model_data.y_test_reshaped,
                                    "cat",
                                    params["train"]["tuning_metric"],
                                    model_params,
                                    logger,
                                    max_features)
            logger.debug(f"Top features identified: {top_feats}")
            logger.info("Writing selected features to file..")
            with open(params["select"]["selected_features_path"], "w") as fp:
                json.dump(obj=top_feats, fp=fp)
            best_params = {}
            with open(params["tune"]["best_params"], "w") as fp:
                json.dump(obj=best_params, fp=fp)
               
        if perform_tuning:
            # use model data to create feature selected and reshaped X_train
            model_data.filter_features(top_feats)
            logger.info(
                f"Starting random search with {params['tune']['n_iter']} samples"
            )
            best_params = tune.random_search_cat(
                model_data.X_train_reshaped,
                model_data.y_train_reshaped,
                params["train"]["estimator_name"],
                params["train"]["tuning_metric"],
                param_path,
                logger,
            )
            logger.debug(f"Returned params: {best_params}")
            logger.info("Writing best params to file..")
            with open(params["tune"]["best_params"], "w") as fp:
                json.dump(obj=best_params, fp=fp)

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--params", dest="params", required=True)
    args = args_parser.parse_args()
    perform_selection_and_tuning(param_path=args.params)
