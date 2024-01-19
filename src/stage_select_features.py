import argparse
from typing import Text
import json
import pickle
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from utils.logs import get_logger
from features import feature_selection as fsl
import joblib

def perform_feature_selection(param_path: Text) -> None:

    with open(param_path) as file:
        params = yaml.safe_load(file)
    logger = get_logger("FEATURE SELECTION", log_level=params["base"]["log_level"])

    # define parameters for feature selection
    estimator_name = params["train"]["estimator_name"]
    model_dir = Path(params["train"]["model_dir"])
    model_path = model_dir / params['train']['model_name']
    feature_analysis = params['select']['fs_analysis']
    max_features = params["select"]["max_features"]
    assert max_features <= 81
    logger.info(f"Max features for feature selection: {max_features}")

    # load data
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

    if feature_analysis == 'feat_imp':
        logger.info(f"Performing feature selection with CatBoost feature importance")
        top_feats = fsl.feature_importance(model_path,
                                            logger,
                                            max_features)
        logger.info(f'Writing features to file')
        df = pd.DataFrame(top_feats, columns=['feature_index'])
        df.to_csv(params["data_condition"]["selected_features"], index=False)
        
        # save dvc version
        # with open(params["data_condition"]["selected_features"], "w") as fp:
        #     json.dump(
        #         obj={"feature_index": top_feats,
        #             "n_features": len(top_feats)},
        #         fp=fp,
        #         )

    else: #this code needs to be updated
        logger.info(f"Performing feature selection with SHAP analysis")
        select_X_train, select_X_test, fs_model = fsl.backward_selection(
            X_train,
            X_test,
            y_train,
            y_test,
            params["train"]["estimator_name"],
            params["train"]["tuning_metric"],
            params["train"]["estimators"][estimator_name]["param_grid"],
            logger,
            max_features)
        logger.info(f"Feature selection completed with {select_X_train.shape[1]} features")

        # overwrite saved X_train and X_test w feature selected data
        # do we want to save this data?
        logger.info("Saving feature selected model and data.")
        joblib.dump(fs_model, f'{model_path}.joblib')
        with open(params["data_condition"]["X_train"], "wb") as fp:
            pickle.dump(select_X_train, fp)
        with open(params["data_condition"]["X_test"], "wb") as fp:
            pickle.dump(select_X_test, fp)
        
        logger.info(f'Writing features to file')
        with open(params["data_condition"]["selected_features"], "w") as fp:
            json.dump(
                obj={"feature_index": list(select_X_test.columns),
                    "n_features": len(list(select_X_test.columns))},
                fp=fp,
                )
    

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--params", dest="params", required=True)
    args = args_parser.parse_args()
    perform_feature_selection(param_path=args.params)