import argparse
from typing import Text
import yaml
import joblib
import pickle
import json
from pathlib import Path
from utils.logs import get_logger
import model.train as train
import model.feature_selection as fsl
import model.tune as tune


def train_model(param_path: Text) -> None:

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
    estimator_name = params["train"]["estimator_name"]

    # if using select features, perform backward selection
    # this will also train a model with the selected features
    # option to have backward selection return the fs model
    if params["train"]["select_features"]:
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
        
        fs_outpath = params["train"]["selected_feature_indicies"]
        logger.info(f'Writing feature indicies to {fs_outpath}')
        with open(fs_outpath, "w") as fp:
            json.dump(
                obj={"feature_column_indicies": list(X_test.columns),
                    "n_features": len(list(X_test.columns))},
                fp=fp,
            )
        # joblib.dump(model, f'{model_path}_fs.joblib')
        X_train = select_X_train
        X_test = select_X_test
    else:
        logger.info("Using all features")
        
    if params["train"]["tune_hyperparams"]:
        logger.info("Starting random search...")
        tuning_params = tune.random_search_cat(X_train, 
                                               y_train, 
                                               param_path)
        logger.info(f"Best hyperparameters: {tuning_params['params']}")
        with open(params["tune"]["best_params"], "w") as fp:
            json.dump(obj=tuning_params,
                      fp=fp,
        )
        # joblib.dump(model, f'{model_path}_tuned.joblib')

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

    model_dir = Path(params["train"]["model_dir"])
    model_path = model_dir / params['train']['model_name']
    print(f'{model_path}.joblib')
    joblib.dump(model, f'{model_path}.joblib')


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--params", dest="params", required=True)
    args = args_parser.parse_args()
    train_model(param_path=args.params)