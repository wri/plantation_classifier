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
    with open(params["data_condition"]["modelData_path"], "rb") as fp:
        model_data = pickle.load(fp)
    logger.info("Model data loaded")
    final_hyperparams = {"None": None}
    top_feats = list(range(0, (model_data.X_train_reshaped.shape[1])))

    if (params["select"]["select_features"] == False) and (
        params["tune"]["tune_hyperparameters"] == False
    ):
        logger.info("Using all features and default hyperparameters")
    else:
        if params["select"]["select_features"]:
            # define parameters for feature selection
            feature_analysis = params["select"]["fs_analysis"]
            max_features = params["select"]["max_features"]
            logger.info(f"Max features for feature selection: {max_features}")

            if feature_analysis == "feat_imp":
                logger.info(
                    f"Performing feature selection with CatBoost feature importance"
                )
                model_params = params["train"]["estimators"]["cat"]["param_grid"]
                model = train.fit_estimator(
                    model_data.X_train_scaled,
                    model_data.X_test_scaled,
                    model_data.y_train_reshaped,
                    model_data.y_test_reshaped,
                    "cat",
                    params["train"]["tuning_metric"],
                    model_params,
                    logger,
                )
                top_feats = fsl.feature_importance(model, logger, max_features)
                logger.debug(f"Top features identified, count: {len(top_feats)}")
                final_hyperparams = model.get_all_params()
        # if tuning hyperparameters
        if params["tune"]["tune_hyperparameters"]:
            # update features selection (in reshaped data arrays)
            model_data.filter_features(top_feats)
            logger.info(
                f"Starting random search with {params['tune']['n_iter']} samples"
            )
            tuning_params, tuned_model = tune.random_search_cat(
                model_data.X_train_reshaped,
                model_data.y_train_reshaped,
                params["train"]["estimator_name"],
                params["train"]["tuning_metric"],
                param_path,
            )
            final_hyperparams = tuned_model.get_all_params()
            logger.debug(f"Returned params: {tuning_params}")
            logger.debug(f"Final params: {final_params}")

    with open(params["select"]["selected_features_path"], "w") as fp:
        json.dump(obj=top_feats, fp=fp)
    with open(params["tune"]["best_params"], "w") as fp:
        json.dump(obj=final_hyperparams, fp=fp)


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--params", dest="params", required=True)
    args = args_parser.parse_args()
    perform_selection_and_tuning(param_path=args.params)
