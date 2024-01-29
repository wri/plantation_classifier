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


def perform_selection_and_tuning(param_path: Text) -> None:
    with open(param_path) as file:
        params = yaml.safe_load(file)
    logger = get_logger(
        "FEATURE SELECTION AND HYPERPARAMETER TUNING",
        log_level=params["base"]["log_level"],
    )

    # load all data
    with open(params["data_condition"]["X_train_scaled"], "rb") as fp:
        X_train_scaled = pickle.load(fp)
    logger.debug(f"X_train_scaled shape: {X_train_scaled.shape}")
    with open(params["data_condition"]["X_train"], "rb") as fp:
        X_train = pickle.load(fp)
    logger.debug(f"X_train shape: {X_train.shape}")
    with open(params["data_condition"]["y_train"], "rb") as fp:
        y_train = pickle.load(fp)
    logger.debug(f"y_train shape: {y_train.shape}")
    with open(params["data_condition"]["X_test_scaled"], "rb") as fp:
        X_test_scaled = pickle.load(fp)
    logger.debug(f"X_test_scaled shape: {X_test_scaled.shape}")
    with open(params["data_condition"]["X_test"], "rb") as fp:
        X_test = pickle.load(fp)
    logger.debug(f"X_test shape: {X_test.shape}")
    with open(params["data_condition"]["y_test"], "rb") as fp:
        y_test = pickle.load(fp)
    with open(params["data_condition"]["class_weights"]) as fp:
        class_weights = json.load(fp)
    logger.debug(f"y_test shape: {y_test.shape}")
    logger.info("Training and testing data loaded.")

    if (params["select"]["select_features"] == False) and (
        params["tune"]["tune_hyperparameters"] == False
    ):
        logger.info("Using all features and default hyperparameters")
        df = pd.DataFrame(list(range(0, (X_train.shape[1]))))
        df.to_csv(params["select"]["selected_features_path"], index=False)
        with open(params["select"]["X_train_selected"], "wb") as fp:
            pickle.dump(X_train, fp)
        with open(params["select"]["X_test_selected"], "wb") as fp:
            pickle.dump(X_test, fp)
        with open(params["tune"]["best_params"], "w") as fp:
            json.dump(obj="None", fp=fp)

    else:
        if params["select"]["select_features"]:
            estimator_name = params["train"]["estimator_name"]

            # define parameters for feature selection
            feature_analysis = params["select"]["fs_analysis"]
            max_features = params["select"]["max_features"]
            assert max_features <= 81
            logger.info(f"Max features for feature selection: {max_features}")

            if feature_analysis == "feat_imp":
                logger.info(
                    f"Performing feature selection with CatBoost feature importance"
                )
                model_params = params["train"]["estimators"]["cat"]["param_grid"]
                model = train.fit_estimator(
                    X_train_scaled,
                    X_test_scaled,
                    y_train,
                    y_test,
                    "cat",
                    params["train"]["tuning_metric"],
                    model_params,
                    logger,
                )
                top_feats = fsl.feature_importance(model, logger, max_features)
                logger.info(f"Writing features to file")
                df = pd.DataFrame(top_feats, columns=["feature_index"])
                df.to_csv(params["data_condition"]["selected_features"], index=False)
                final_hyperparams = model.get_all_params()

        if params["tune"]["tune_hyperparameters"]:
            logger.info(
                f"Starting random search with {params['tune']['n_iter']} samples"
            )
            tuning_params, tuned_model = tune.random_search_cat(
                X_train,
                y_train,
                params["train"]["estimator_name"],
                params["train"]["tuning_metric"],
                param_path,
            )
            final_hyperparams = tuned_model.get_all_params()
            logger.debug(f"Returned params: {tuning_params}")
            logger.debug(f"Final params: {final_params}")

        final_hyperparams["class_weights"] = class_weights
        with open(params["tune"]["best_params"], "w") as fp:
            json.dump(obj=final_hyperparams, fp=fp)


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--params", dest="params", required=True)
    args = args_parser.parse_args()
    perform_selection_and_tuning(param_path=args.params)
